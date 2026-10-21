/*
 * Kernel density estimate: a C port of R's stats::density() for the
 * arguments modeFinder uses (gaussian kernel, bw = "nrd0", adjust = 1,
 * cut = 3, ext = 4, old.coords = FALSE, optional weights).  The steps mirror
 * density.default() in R's src/library/stats/R/density.R, including the
 * linear binning of BinDist() (massdist.c), the FFT convolution and the
 * final linear interpolation of approx().
 */

#include <math.h>
#include <float.h>
#include <limits.h>
#include "mf_internal.h"

#define MF_1_SQRT_2PI 0.398942280401432677939946059934
#define MF_LN2        0.693147180559945309417232121458
#define MF_PI         3.141592653589793238462643383280

/* dnorm(x, 0, sigma) as in R's nmath/dnorm.c */
static double dnorm0(double x, double sigma)
{
    double x1, x2;
    x = fabs(x / sigma);
    if (x >= 2 * sqrt(DBL_MAX)) return 0.;
    if (x < 5) return MF_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;
    if (x > sqrt(-2 * MF_LN2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG))) return 0.;
    x1 = ldexp(nearbyint(ldexp(x, 16)), -16);
    x2 = x - x1;
    return MF_1_SQRT_2PI / sigma *
	(exp(-0.5 * x1 * x1) * exp((-0.5 * x2 - x1) * x2));
}

/* seq.int(from, to, length.out = n) */
static void seq_out(double from, double to, int n, double *out)
{
    int i;
    double by;
    if (n > 0) out[0] = from;
    if (n > 1) out[n - 1] = to;
    if (n > 2) {
	by = (to - from) / (double)(n - 1);
	for (i = 1; i < n - 1; i++) out[i] = from + (double)i * by;
    }
}

/* var(x), with the two-pass long double accumulation of R's cov.c */
static double var_r(const double *x, int n)
{
    long double sum = 0., tmp;
    int k;
    for (k = 0; k < n; k++) sum += x[k];
    tmp = sum / n;
    if (isfinite((double) tmp)) {
	sum = 0.;
	for (k = 0; k < n; k++) sum += (x[k] - tmp);
	tmp = tmp + sum / n;
    }
    {
	long double xm = (double) tmp;
	sum = 0.;
	for (k = 0; k < n; k++) sum += (x[k] - xm) * (x[k] - xm);
    }
    return (double)(sum / (n - 1));
}

/* quantile(x, p, type = 7) for sorted x */
static double quantile7(const double *x, int n, double p)
{
    double index = 1 + (n - 1) * p, h, qs;
    int lo = (int) floor(index), hi = (int) ceil(index);
    qs = x[lo - 1];
    if (index > lo && x[hi - 1] != qs) {
	h = index - lo;
	qs = (1 - h) * qs + h * x[hi - 1];
    }
    return qs;
}

/* bw.nrd0(x) for sorted x */
static double bw_nrd0(const double *x, int n)
{
    double hi = sqrt(var_r(x, n));
    double iqr = quantile7(x, n, 0.75) - quantile7(x, n, 0.25);
    double lo = hi < iqr / 1.34 ? hi : iqr / 1.34;
    if (lo == 0) {
	lo = hi;
	if (lo == 0) {
	    lo = fabs(x[0]);
	    if (lo == 0) lo = 1.;
	}
    }
    return 0.9 * lo * pow((double) n, -0.2);
}

/* In-place radix-2 complex FFT of length len (a power of two), unnormalized;
   sign = -1 is R's fft(z), sign = +1 is fft(z, inverse = TRUE). */
static void fft_inplace(double *re, double *im, int len, int sign,
			const double *cs, const double *sn)
{
    int i, j, k, size, half, step;
    double tr, ti, wr, wi;

    for (i = 1, j = 0; i < len; i++) {
	int bit = len >> 1;
	for (; j & bit; bit >>= 1) j ^= bit;
	j ^= bit;
	if (i < j) {
	    tr = re[i]; re[i] = re[j]; re[j] = tr;
	    ti = im[i]; im[i] = im[j]; im[j] = ti;
	}
    }
    for (size = 2; size <= len; size <<= 1) {
	half = size >> 1;
	step = len / size;
	for (i = 0; i < len; i += size) {
	    for (k = 0; k < half; k++) {
		wr = cs[k * step];
		wi = sign * sn[k * step];
		tr = re[i + k + half] * wr - im[i + k + half] * wi;
		ti = re[i + k + half] * wi + im[i + k + half] * wr;
		re[i + k + half] = re[i + k] - tr;
		im[i + k + half] = im[i + k] - ti;
		re[i + k] += tr;
		im[i + k] += ti;
	    }
	}
    }
}

/* approx(x, y, v)$y with method = "linear", rule = 1 (R's approx1()) */
static double approx1(double v, const double *x, const double *y, int n)
{
    int i = 0, j = n - 1, ij;
    if (v < x[i] || v > x[j]) return NAN;
    while (i < j - 1) {
	ij = (i + j) / 2;
	if (v < x[ij]) j = ij; else i = ij;
    }
    if (v == x[j]) return y[j];
    if (v == x[i]) return y[i];
    return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]));
}

int mf_density(const double *x, const double *wts, int nx, int n_out,
	       double *out_x, double *out_y, mf_kde_bins *bins)
{
    int n = 512, len, i, ix;
    double bw, from, to, lo, up, xdelta, xpos, fx, wi, re, im;
    double *buf, *yr, *yi, *kr, *ki, *cs, *sn, *xords, *kords;

    /* n <- max(n, 512); if (n > 512) n <- 2^ceiling(log2(n)) */
    while (n < n_out) n <<= 1;
    len = 2 * n;

    buf = (double *) malloc((size_t)(4 * len + len + 2 * n) * sizeof(double));
    if (!buf) return -1;
    yr = buf; yi = yr + len; kr = yi + len; ki = kr + len;
    cs = ki + len; sn = cs + len / 2;
    xords = sn + len / 2; kords = xords + n;

    bw = bw_nrd0(x, nx);
    from = x[0] - 3 * bw;
    to = x[nx - 1] + 3 * bw;
    lo = from - 4 * bw;
    up = to + 4 * bw;

    /* y <- BinDist(x, weights, lo, up, n), weights = rep(1/nx, nx) by default */
    for (i = 0; i < len; i++) yr[i] = yi[i] = ki[i] = 0.;
    xdelta = (up - lo) / (n - 1);
    wi = 1. / nx;
    for (i = 0; i < nx; i++) {
	if (wts) wi = wts[i];
	xpos = (x[i] - lo) / xdelta;
	if (xpos > INT_MAX || xpos < INT_MIN) continue;
	ix = (int) floor(xpos);
	fx = xpos - ix;
	if (0 <= ix && ix <= n - 2) {
	    yr[ix] += (1 - fx) * wi;
	    yr[ix + 1] += fx * wi;
	}
	else if (ix == -1) yr[0] += fx * wi;
	else if (ix == n - 1) yr[ix] += (1 - fx) * wi;
    }

    if (bins) {
	double sum = 0.;
	bins->w = (double *) malloc((size_t) n * sizeof(double));
	if (!bins->w) {
	    free(buf);
	    return -1;
	}
	for (i = 0; i < n; i++) sum += yr[i];
	for (i = 0; i < n; i++) bins->w[i] = yr[i] / sum;
	bins->n = n;
	bins->lo = lo;
	bins->up = up;
	bins->bw = bw;
    }

    /* kords <- seq.int(0, (2n-1)/(n-1) * (up-lo), length.out = 2n),
       mirrored to negative lags and evaluated with the gaussian kernel */
    seq_out(0., (2. * n - 1) / (n - 1) * (up - lo), len, kr);
    for (i = n + 1; i < len; i++) kr[i] = -kr[len - i];
    for (i = 0; i < len; i++) kr[i] = dnorm0(kr[i], bw);

    /* kords <- fft(fft(y) * Conj(fft(kords)), inverse = TRUE) */
    for (i = 0; i < len / 2; i++) {
	cs[i] = cos(2 * MF_PI * i / len);
	sn[i] = sin(2 * MF_PI * i / len);
    }
    fft_inplace(yr, yi, len, -1, cs, sn);
    fft_inplace(kr, ki, len, -1, cs, sn);
    for (i = 0; i < len; i++) {
	re = yr[i] * kr[i] + yi[i] * ki[i];
	im = yi[i] * kr[i] - yr[i] * ki[i];
	yr[i] = re;
	yi[i] = im;
    }
    fft_inplace(yr, yi, len, +1, cs, sn);

    /* kords <- pmax.int(0, Re(kords)[1L:n]/length(y)) */
    for (i = 0; i < n; i++) {
	kords[i] = yr[i] / len;
	if (kords[i] < 0) kords[i] = 0.;
    }

    seq_out(lo, up, n, xords);
    seq_out(from, to, n_out, out_x);
    for (i = 0; i < n_out; i++)
	out_y[i] = approx1(out_x[i], xords, kords, n);

    free(buf);
    return 0;
}
