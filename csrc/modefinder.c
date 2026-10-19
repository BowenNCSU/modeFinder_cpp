///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2018  Bowen Liu                                         //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //
///////////////////////////////////////////////////////////////////////////

/*
 * C port of src/emp_mode.cpp of the former modeFinder R package (see the
 * git history).  The global state and std::map coefficient caches of the
 * Rcpp version live in a per-call bern_ctx, so mf_emp_mode() is reentrant.
 */

#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "modefinder.h"
#include "mf_internal.h"


/*
AnDarl.c
$Revision: 1.1 $  $Date: 2014/06/09 07:53:21 $
Original C code by G. and J. Marsaglia
R interface by Adrian Baddeley
https://github.com/cran/goftest
*/
static double adinf(double z) {
  if(z<2.) return (
      exp(-1.2337141/z)/sqrt(z)
  )*(
      2.00012+(.247105-
      (.0649821-
      (.0347962-
      (.011672-.00168691*z)
         *z)*z)*z)*z);
  /* max |error| < .000002 for z<2, (p=.90816...) */
  return exp(
    -exp(1.0776-(2.30695-(.43424-(.082433-(.008056 -.0003146*z)
                                    *z)*z)*z)*z));
                                    /* max |error|<.0000008 for 4<z<infinity */
}

static double AD(int n,double z){
  double c,v,x;
  x=adinf(z);
  /* now x=adinf(z). Next, get v=errfix(n,x) and return x+v; */
  if(x>.8) {
    v=(-130.2137+(745.2337-(1705.091-(1950.646-(1116.360-255.7844*x)
                                        *x)*x)*x)*x)/n;
    return x+v;
  }
  c=.01265+.1757/n;
  if(x<c){
    v=x/c;
    v=sqrt(v)*(1.-v)*(49*v-102);
    return(x+v*(.0037/((double)n*n)+.00078/n+.00006)/n);
  }
  v=(x-c)/(.8-c);
  v=-.00022633+(6.54034-(14.6538-(14.458-(8.259-1.91864*v)*v)*v)*v)*v;
  return (x+v*(.04213+.01365/n)/n);
}

static double ADtest(int n, const double *x)
{ int i;
  double t,z=0;
  for(i=0;i<n;i++)   {
    t=x[i]*(1.-x[n-1-i]);
    z=z-(i+i+1)*log(t);
  }
  return AD(n,-n+z/n);
}


// Empirical approach
//
//

static void empirical(const double *data, int n, int knots, double lower,
                      double upper, mf_result *res){
  double max_diff = -1;
  int max_left_bound = -1;
  int cum_index = 0;
  double criterion = 0 * (upper - lower) + lower;
  while(cum_index < n && data[cum_index] <= criterion){
    cum_index ++ ;
  }
  double Fn_value_0 = cum_index / (double) n;

  for(int i = 0; i < knots; i++) {
    criterion = (i + 1) / (double) knots * (upper - lower) + lower;
    while(cum_index < n && data[cum_index] <= criterion){
      cum_index ++ ;
    }
    double Fn_value = cum_index / (double) n;
    double diff = Fn_value - Fn_value_0;

    if(diff > max_diff){
      max_left_bound = i;
      max_diff = diff;
    }
    Fn_value_0 = Fn_value;
  }

  res->method = MF_METHOD_EMPIRICAL;
  res->knots = knots;
  res->mode = (2 * max_left_bound + 1) / (double) knots / 2.0 * (upper - lower) + lower;
  snprintf(res->message, MF_MESSAGE_SIZE, "The mode estimate is chosen as the mid point of interval with maximum difference among knots equally-spaced intervals.\n");
}


// Kernel approach
//
//

static int mode2(const double *data, int n, int smooth_density,
                 int density_points, mf_result *res){
  double *x = (double *) malloc(density_points * sizeof(double));
  double *y = (double *) malloc(density_points * sizeof(double));
  if(!x || !y || mf_density(data, n, density_points, x, y) != 0){
    free(x); free(y);
    return MF_ERR_NOMEM;
  }
  double max_y = y[0];
  double max_x = x[0];
  for(int i = 0; i < density_points; i++){
    if(y[i] > max_y){
      max_y = y[i];
      max_x = x[i];
    }
  }
  res->method = MF_METHOD_KERNEL;
  res->mode = max_x;
  if(smooth_density){
    res->n_density = density_points;
    res->density_x = x;
    res->density_y = y;
  }else{
    free(x); free(y);
  }
  return MF_OK;
}


// Bernstein polynomials approach
//
//

/* Coefficients of one Bernstein expansion, cached per degree m (the
   std::map<int, ...> members of the Rcpp version).  coef holds log-scale
   coefficients at the point u_post; fn holds the empirical CDF values or
   their first / second differences; sign is used by the derivative only. */
typedef struct {
  int m;
  double u_post;
  double *coef;
  double *fn;
  char *sign;
} bern_cache;

typedef struct {
  bern_cache *items;
  int len, cap;
} bern_cache_list;

typedef struct {
  const double *data;   /* sorted sample rescaled to [0, 1] */
  int n;
  int m;                /* current degree */
  bern_cache_list cdf, pdf, dpdf;
  int nomem;
} bern_ctx;

static bern_cache *cache_find(bern_cache_list *l, int m){
  for(int i = 0; i < l->len; i++){
    if(l->items[i].m == m) return &l->items[i];
  }
  return NULL;
}

static bern_cache *cache_add(bern_ctx *ctx, bern_cache_list *l, int m,
                             int len, int with_sign){
  bern_cache c = {m, 0., NULL, NULL, NULL};
  if(l->len == l->cap){
    int cap = l->cap ? 2 * l->cap : 16;
    bern_cache *items = (bern_cache *) realloc(l->items, cap * sizeof(bern_cache));
    if(!items) goto nomem;
    l->items = items;
    l->cap = cap;
  }
  c.coef = (double *) malloc((len > 0 ? len : 1) * sizeof(double));
  c.fn = (double *) malloc((len > 0 ? len : 1) * sizeof(double));
  if(with_sign) c.sign = (char *) malloc(len > 0 ? len : 1);
  if(!c.coef || !c.fn || (with_sign && !c.sign)) goto nomem;
  l->items[l->len] = c;
  return &l->items[l->len++];

nomem:
  free(c.coef); free(c.fn); free(c.sign);
  ctx->nomem = 1;
  return NULL;
}

static void cache_free(bern_cache_list *l){
  for(int i = 0; i < l->len; i++){
    free(l->items[i].coef);
    free(l->items[i].fn);
    free(l->items[i].sign);
  }
  free(l->items);
  l->items = NULL;
  l->len = l->cap = 0;
}

static double mix_cdf_c(bern_ctx *ctx, double u) {
  const double *data = ctx->data;
  int n = ctx->n, m = ctx->m;
  bern_cache *c;
  if(u < 0){
    return 0;
  }else if(u >= 1){
    return 1;
  }else if(u == 0){
    int cum_index = 0;
    while(cum_index < n && data[cum_index] <= 0){
      cum_index ++ ;
    }
    double Fn_value = cum_index / (double) n;
    return Fn_value;
  }else if ( (c = cache_find(&ctx->cdf, m)) == NULL ) {
    // not found
    // coef.vec.cdf.bern <- log(Fn(0 : m / m) / (m + 1)) + dbeta_r(u, 0 : m + 1, m - (0 : m) + 1, log = TRUE)
    if((c = cache_add(ctx, &ctx->cdf, m, m + 1, 0)) == NULL) return NAN;
    double summation = 0;
    int cum_index = 0;
    double dbeta_log = m * log(1 - u);
    for(int i = 0; i < (m + 1); i++) {
      if(i > 0){
        dbeta_log += log((m - i + 1) / (double) i * u / (1 - u));
      }
      double criterion = i / (double) m;
      while(cum_index < n && data[cum_index] <= criterion){
        cum_index ++ ;
      }
      double Fn_value = cum_index / (double) n;
      c->fn[i] = Fn_value;
      c->coef[i] = log(Fn_value) + dbeta_log;
      summation += Fn_value * exp(dbeta_log);
    }
    c->u_post = u;
    return summation;
  } else {
    // found
    // sum.cdf.bern <- crossprod(coef.ls.cdf.bern[[which(m.vec.cdf.bern == m)]],
    // (u / u.post) ^ (0 : m) * ((1 - u) / (1 - u.post)) ^ (m : 0))
    double summation = 0;
    for(int i = 0; i < (m + 1); i++){
      summation += exp(c->coef[i] +
        log(u / c->u_post) * i + log((1 - u) / (1 - c->u_post)) * (m - i));
    }

    if(isnan(summation) || isinf(summation) || summation < 0 || summation > 1){
      summation = 0;
      double dbeta_log = m * log(1 - u);
      for(int i = 0; i < (m + 1); i++){
        if(i > 0){
          dbeta_log += log((m - i + 1) / (double) i * u / (1 - u));
        }
        double Fn_value = c->fn[i];
        c->coef[i] = log(Fn_value) + dbeta_log;
        summation += Fn_value * exp(dbeta_log);
      }
      c->u_post = u;
    }
    return summation;
  }
}

static double mix_pdf_c(bern_ctx *ctx, double u) {
  const double *data = ctx->data;
  int n = ctx->n, m = ctx->m;
  bern_cache *c;
  if(u * (1 - u) < 0){
    return 0;
  }else if (u == 0){
    int cum_index = 0;
    while(cum_index < n && data[cum_index] <= 0){
      cum_index ++ ;
    }
    double Fn_value_0 = cum_index / (double) n;
    while(cum_index < n && data[cum_index] <= 1 / (double) m){
      cum_index ++ ;
    }
    double Fn_value_1 = cum_index / (double) n;
    return (Fn_value_1 - Fn_value_0) * m;
  }else if (u == 1){
    int cum_index = n;
    while(cum_index > 0 && data[cum_index - 1] > 1 - 1 / (double) m){
      cum_index -- ;
    }
    double Fn_value = cum_index / (double) n;
    return (1 - Fn_value) * m;
  }else if ( (c = cache_find(&ctx->pdf, m)) == NULL ) {
    // not found
    // log(Fn((1 : m) / m) - Fn((0 : (m - 1)) / m)) + dbeta(u, 1 : m, m - (1 : m) + 1, log = TRUE)
    if((c = cache_add(ctx, &ctx->pdf, m, m, 0)) == NULL) return NAN;
    double summation = 0;
    int cum_index = 0;
    double criterion = 0;
    while(cum_index < n && data[cum_index] <= criterion){
      cum_index ++ ;
    }
    double Fn_value_0 = cum_index / (double) n;
    double dbeta_log = log(m) + (m - 1) * log(1 - u);

    for(int i = 0; i < m; i++) {
      if(i > 0){
        dbeta_log += log((m - i) / (double) i * u / (1 - u));
      }
      criterion = (i + 1) / (double) m;
      while(cum_index < n && data[cum_index] <= criterion){
        cum_index ++ ;
      }
      double Fn_value = cum_index / (double) n;
      c->fn[i] = Fn_value - Fn_value_0;
      c->coef[i] = log(Fn_value - Fn_value_0) + dbeta_log;
      summation += (Fn_value - Fn_value_0) * exp(dbeta_log);
      Fn_value_0 = Fn_value;
    }
    c->u_post = u;
    return summation;
  } else {
    // found
    double summation = 0;
    for(int i = 0; i < m; i++){
      summation += exp(c->coef[i] +
        log(u / c->u_post) * i + log((1 - u) / (1 - c->u_post)) * (m - 1 - i));
    }

    if(isnan(summation) || summation < 0){
      summation = 0;
      double dbeta_log = log(m) + (m - 1) * log(1 - u);
      for(int i = 0; i < m; i++){
        if(i > 0){
          dbeta_log += log((m - i) / (double) i * u / (1 - u));
        }
        double Fn_value = c->fn[i];
        c->coef[i] = log(Fn_value) + dbeta_log;
        summation += Fn_value * exp(dbeta_log);
      }
      c->u_post = u;
    }
    return summation;
  }
}

static double mix_dpdf_c(bern_ctx *ctx, double u) {
  const double *data = ctx->data;
  int n = ctx->n, m = ctx->m;
  bern_cache *c;
  if(u * (1 - u) < 0){
    return 0;
  }else if (u == 0){
    int cum_index = 0;
    while(cum_index < n && data[cum_index] <= 0){
      cum_index ++ ;
    }
    double Fn_value_0 = cum_index / (double) n;
    while(cum_index < n && data[cum_index] <= 1 / (double) m){
      cum_index ++ ;
    }
    double Fn_value_1 = cum_index / (double) n;
    while(cum_index < n && data[cum_index] <= 2 / (double) m){
      cum_index ++ ;
    }
    double Fn_value_2 = cum_index / (double) n;
    return (Fn_value_2 - 2 * Fn_value_1 + Fn_value_0) * m * (m - 1);
  }else if (u == 1){
    int cum_index = n;
    while(cum_index > 0 && data[cum_index - 1] > 1 - 1 / (double) m){
      cum_index -- ;
    }
    double Fn_value = cum_index / (double) n;
    while(cum_index > 0 && data[cum_index - 1] > 1 - 2 / (double) m){
      cum_index -- ;
    }
    double Fn_value_ = cum_index / (double) n;
    return (1 - 2 * Fn_value + Fn_value_) * m * (m - 1);
  }else if ( (c = cache_find(&ctx->dpdf, m)) == NULL ) {
    // not found
    // (Fn(2 : m / m) - 2 * Fn((2 : m - 1) / m) + Fn((2 : m - 2) / m)) * m * dbeta(u, 2 : m - 1, m - (2 : m) + 1)
    if((c = cache_add(ctx, &ctx->dpdf, m, m - 1, 1)) == NULL) return NAN;
    double summation = 0;
    int cum_index = 0;
    double criterion = 0;
    while(cum_index < n && data[cum_index] <= criterion){
      cum_index ++ ;
    }
    double Fn_value_0 = cum_index / (double) n;

    criterion = 1 / (double) m;
    while(cum_index < n && data[cum_index] <= criterion){
      cum_index ++ ;
    }
    double Fn_value_1 = cum_index / (double) n;
    // m * (m - 1) in double: the int product overflows for m > 46341
    double dbeta_log = log((double) m * (m - 1)) + (m - 2) * log(1 - u);

    for(int i = 0; i < (m - 1); i++) {
      if(i > 0){
        dbeta_log += log((m - i - 1) / (double) i * u / (1 - u));
      }
      criterion = (i + 2) / (double) m;
      while(cum_index < n && data[cum_index] <= criterion){
        cum_index ++ ;
      }
      double Fn_value = cum_index / (double) n;
      double diff2 = Fn_value - 2 * Fn_value_1 + Fn_value_0;
      c->fn[i] = diff2;
      c->sign[i] = diff2 > 0;
      c->coef[i] = log(fabs(diff2)) + dbeta_log;
      summation += diff2 * exp(dbeta_log);
      Fn_value_0 = Fn_value_1;
      Fn_value_1 = Fn_value;
    }
    c->u_post = u;
    return summation;
  } else {
    // found
    double summation = 0;
    for(int i = 0; i < (m - 1); i++){
      summation += (2 * c->sign[i] - 1) * exp(c->coef[i] +
        log(u / c->u_post) * i + log((1 - u) / (1 - c->u_post)) * (m - 2 - i));
    }

    if(isnan(summation)){
      summation = 0;
      double dbeta_log = log((double) m * (m - 1)) + (m - 2) * log(1 - u);
      for(int i = 0; i < (m - 1); i++){
        if(i > 0){
          dbeta_log += log((m - i - 1) / (double) i * u / (1 - u));
        }
        double Fn_value = c->fn[i];
        c->coef[i] = log(fabs(Fn_value)) + dbeta_log;
        summation += Fn_value * exp(dbeta_log);
      }
      c->u_post = u;
    }
    return summation;
  }
}

static double fminfn(int n, double *x, void *ex){
  (void) n;
  return -mix_pdf_c((bern_ctx *) ex, *x);
}

static void fmingr(int n, double *x, double *res, void *ex){
  (void) n;
  res[0] = -mix_dpdf_c((bern_ctx *) ex, *x);
}


// https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/optimize.c
/* Formerly in src/appl/fmim.c
fmin.f -- translated by f2c (version 19990503).
R's  optimize() :   function	fmin(ax,bx,f,tol)
=    ==========		~~~~~~~~~~~~~~~~~
an approximation  x  to the point where  f  attains a minimum  on
the interval  (ax,bx)  is determined.
INPUT..
ax    left endpoint of initial interval
bx    right endpoint of initial interval
f     function which evaluates  f(x, info)  for any  x
in the interval  (ax,bx)
tol   desired length of the interval of uncertainty of the final
result ( >= 0.)
OUTPUT..
fmin  abcissa approximating the point where  f  attains a minimum */

static double Brent_fmin(double ax, double bx, mf_optimfn f, void *info,
                         double tol)
{
  /*  c is the squared inverse of the golden ratio */
  const double c = (3. - sqrt(5.)) * .5;

  /* Local variables */
  double a, b, d, e, p, q, r, u, v, w, x;
  double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

  /*  eps is approximately the square root of the relative machine precision. */
  eps = DBL_EPSILON;
  tol1 = eps + 1.;/* the smallest 1.000... > 1 */
  eps = sqrt(eps);

  a = ax;
  b = bx;
  v = a + c * (b - a);
  w = v;
  x = v;

  d = 0.;/* -Wall */
  e = 0.;
  fx = f(1, &x, info);
  fv = fx;
  fw = fx;
  tol3 = tol / 3.;

  /*  main loop starts here ----------------------------------- */

  for(;;) {
    xm = (a + b) * .5;
    tol1 = eps * fabs(x) + tol3;
    t2 = tol1 * 2.;

    /* check stopping criterion */

    if (fabs(x - xm) <= t2 - (b - a) * .5) break;
    p = 0.;
    q = 0.;
    r = 0.;
    if (fabs(e) > tol1) { /* fit parabola */

      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.;
      if (q > 0.) p = -p; else q = -q;
      r = e;
      e = d;
    }

    if (fabs(p) >= fabs(q * .5 * r) ||
        p <= q * (a - x) || p >= q * (b - x)) { /* a golden-section step */

      if (x < xm) e = b - x; else e = a - x;
      d = c * e;
    }
    else { /* a parabolic-interpolation step */

      d = p / q;
      u = x + d;

      /* f must not be evaluated too close to ax or bx */

      if (u - a < t2 || b - u < t2) {
        d = tol1;
        if (x >= xm) d = -d;
      }
    }

    /* f must not be evaluated too close to x */

    if (fabs(d) >= tol1)
      u = x + d;
    else if (d > 0.)
      u = x + tol1;
    else
      u = x - tol1;

    fu = f(1, &u, info);

    /*  update  a, b, v, w, and x */

    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w;    w = x;   x = u;
      fv = fw; fw = fx; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  /* end of main loop */

  return x;
}


/* data: sorted sample, rescaled in place to [0, 1] */
static int emp_mode_bern(double *data, int n, double lower, double upper,
                         const mf_options *opt, mf_result *res){
  bern_ctx ctx;
  int subsample_size = opt->sample_size;
  int do_choose_m = opt->m_degree <= 0;
  int status = MF_OK;
  int m, len;
  double *newdata = NULL, *v = NULL;
  bern_cache *c;

  memset(&ctx, 0, sizeof ctx);
  ctx.data = data;
  ctx.n = n;

  for(int i = 0; i < n; i++){
    data[i] = (data[i] - lower) / (upper - lower);
  }

  if(n > subsample_size){
    len = subsample_size;
    newdata = (double *) malloc(len * sizeof(double));
    if(!newdata) { status = MF_ERR_NOMEM; goto done; }
    for(int i = 0 ; i < subsample_size ; i++){
      newdata[i] = data[(long long) n * (i + 1) / (subsample_size + 1)];
    }
  }else{
    len = n;
  }
  const double *sub = newdata ? newdata : data;
  v = (double *) malloc(len * sizeof(double));
  if(!v) { status = MF_ERR_NOMEM; goto done; }


  double max_pvalue = -1;
  double max_alpha = -1;
  if(do_choose_m){
    for (double alpha = 0.3; alpha <= 0.95; alpha += 0.05) {
      ctx.m = (int) ceil(pow(n, alpha));
      for(int i = 0; i < len; i++) v[i] = mix_cdf_c(&ctx, sub[i]);
      if(ctx.nomem) { status = MF_ERR_NOMEM; goto done; }
      double pvalue_curr = 1. - ADtest(len, v);
      if(pvalue_curr > max_pvalue){
        max_pvalue = pvalue_curr;
        max_alpha = alpha;
      }
      if(max_pvalue > 0.95){
        break;
      }
    }
    if(max_alpha < 0) { status = MF_ERR_NUMERIC; goto done; }
    ctx.m = (int) ceil(pow(n, max_alpha));
  }else{
    ctx.m = opt->m_degree;
    for(int i = 0; i < len; i++) v[i] = mix_cdf_c(&ctx, sub[i]);
    if(ctx.nomem) { status = MF_ERR_NOMEM; goto done; }
    max_pvalue = 1. - ADtest(len, v);
  }
  m = ctx.m;


  // gradient descent
  double max_x = -1;
  double max_density = -1;

  mix_pdf_c(&ctx, 0.5);
  if(ctx.nomem) { status = MF_ERR_NOMEM; goto done; }
  c = cache_find(&ctx.pdf, m);
  double arg_emp = 0;
  double max_diff = -1;
  for(int i = 0; i < m; ++i){
    if(c->fn[i] > max_diff){
      arg_emp = i / (double) m;
      max_diff = c->fn[i];
    }
  }

  double pdfpar_vec[] = {0, 0.5, 1, arg_emp};
  for(int i = 0; i < (int) (sizeof(pdfpar_vec) / sizeof(pdfpar_vec[0])); ++i){
    double dpar = pdfpar_vec[i];
    double lo = 0;
    double up = 1;
    int nbd = 2;
    double val = 0.0;
    int ifail = 0;
    char msg[60];
    int fncount = 0, grcount = 0;
    mf_lbfgsb(1, 5, &dpar, &lo, &up, &nbd, &val, fminfn, fmingr, &ifail,
              &ctx, 1e7, 0., &fncount, &grcount, 100, msg, 0, 10);
    if(ifail == -1 || ctx.nomem) { status = MF_ERR_NOMEM; goto done; }
    if(-val > max_density){
      max_x = dpar;
      max_density = -val;
    }
  }

  double pdfilon_vec[] = {0, 0.1, 0.5};
  for(int i = 0; i < (int) (sizeof(pdfilon_vec) / sizeof(pdfilon_vec[0])); ++i){
    double dpar = Brent_fmin(0 - pdfilon_vec[i], 1 + pdfilon_vec[i],
                             fminfn, &ctx, pow(DBL_EPSILON, 0.25));
    double val = mix_pdf_c(&ctx, dpar);
    if(val > max_density){
      max_x = dpar;
      max_density = val;
    }
  }
  if(ctx.nomem) { status = MF_ERR_NOMEM; goto done; }

  double prob_left, prob_right = 0;

  if ( (c = cache_find(&ctx.cdf, m)) == NULL ) {
    // not found
    int cum_index = 0;
    while(cum_index < n && data[cum_index] <= 2 / (double) m){
      cum_index ++ ;
    }
    prob_left = cum_index / (double) n;

    cum_index = n;
    while(cum_index > 0 && data[cum_index - 1] > 1 - 2 / (double) m){
      cum_index -- ;
    }
    prob_right = 1 - cum_index / (double) n;
  }else{
    // found
    prob_left = c->fn[2];
    prob_right = 1 - c->fn[m + 1 - 3];
  }

  int cum_index = 0;
  while(cum_index < n && data[cum_index] <= max_x - 1 / (double) m){
    cum_index ++ ;
  }
  double Fn_value_0 = cum_index / (double) n;
  while(cum_index < n && data[cum_index] <= max_x + 1 / (double) m){
    cum_index ++ ;
  }
  double Fn_value_1 = cum_index / (double) n;
  double prob_mid = Fn_value_1 - Fn_value_0;

  const char *choice;
  if(max_pvalue > 0.05 || (prob_mid > prob_left && prob_mid > prob_right)){
    choice = "The mode estimate is chosen as the maximum point of Bernstein polynomials density estimate.";
    res->mode = max_x * (upper - lower) + lower;
  }else{
    choice = "The mode estimate is chosen as a boundary point since Anderson-Darling criterion is not satisfied with p-value less than 0.05.";
    if(prob_left > prob_right){
      res->mode = 0 * (upper - lower) + lower;
    }else if(prob_left < prob_right){
      res->mode = 1 * (upper - lower) + lower;
    }else{
      res->mode = floor(rand() / (RAND_MAX + 1.0) * 2) * (upper - lower) + lower;
    }
  }
  snprintf(res->message, MF_MESSAGE_SIZE,
           "\nmaximum point: %g, probability around maximum: %g, probability of left boundary: %g, probability of right boundary: %g\n%s\n\n",
           max_x, prob_mid, prob_left, prob_right, choice);

  res->method = MF_METHOD_BERNSTEIN;
  res->degree_chosen = do_choose_m;
  res->degree = m;
  res->alpha = do_choose_m ? max_alpha : NAN;
  res->p_value = max_pvalue;
  res->max_point = max_x;
  res->prob_mid = prob_mid;
  res->prob_left = prob_left;
  res->prob_right = prob_right;

  if(opt->smooth_density){
    int n_points = opt->density_points;
    double *x_cord = (double *) malloc(n_points * sizeof(double));
    double *y_cord = (double *) malloc(n_points * sizeof(double));
    if(!x_cord || !y_cord){
      free(x_cord); free(y_cord);
      status = MF_ERR_NOMEM;
      goto done;
    }
    for(int i = 0; i < n_points; i++){
      y_cord[i] = mix_pdf_c(&ctx, i / (double) n_points);
    }
    for(int i = 0; i < n_points; i++){
      x_cord[i] = i / (double) n_points * (upper - lower) + lower;
    }
    res->n_density = n_points;
    res->density_x = x_cord;
    res->density_y = y_cord;
    if(ctx.nomem) status = MF_ERR_NOMEM;
  }

done:
  free(newdata);
  free(v);
  cache_free(&ctx.cdf);
  cache_free(&ctx.pdf);
  cache_free(&ctx.dpdf);
  return status;
}


static int cmp_double(const void *a, const void *b){
  double x = *(const double *) a, y = *(const double *) b;
  return (x > y) - (x < y);
}

void mf_default_options(mf_options *opt){
  memset(opt, 0, sizeof *opt);
  opt->smooth = 1;
  opt->smooth_option = MF_SMOOTH_BERNSTEIN;
  opt->smooth_density = 0;
  opt->density_points = 512;
  opt->m_degree = 0;
  opt->sample_size = 1000;
  opt->m_knots = 0;
}

int mf_emp_mode(const double *data_in, size_t n_in, const mf_options *opt,
                mf_result *res){
  mf_options defaults;
  double *data, lower, upper;
  int n, status;

  if(!data_in || !res) return MF_ERR_BADARG;
  memset(res, 0, sizeof *res);
  res->alpha = res->p_value = NAN;
  res->max_point = res->prob_mid = res->prob_left = res->prob_right = NAN;
  if(!opt){
    mf_default_options(&defaults);
    opt = &defaults;
  }

  if (n_in <= 38) { // larger than 3^(10/3)
    return MF_ERR_TOO_SMALL;
  }
  if (n_in > INT_MAX) return MF_ERR_BADARG;
  n = (int) n_in;
  for(int i = 0; i < n; i++){
    if(!isfinite(data_in[i])) return MF_ERR_NONFINITE;
  }
  if(opt->smooth){
    if(opt->smooth_option != MF_SMOOTH_BERNSTEIN && opt->smooth_option != MF_SMOOTH_KERNEL)
      return MF_ERR_BADARG;
    if((opt->smooth_option == MF_SMOOTH_KERNEL || opt->smooth_density) &&
       (opt->density_points < 1 || opt->density_points > INT_MAX / 8))
      return MF_ERR_BADARG;
    if(opt->smooth_option == MF_SMOOTH_BERNSTEIN &&
       (opt->sample_size < 1 || opt->m_degree == 1))
      return MF_ERR_BADARG;
  }

  data = (double *) malloc(n * sizeof(double));
  if(!data) return MF_ERR_NOMEM;
  memcpy(data, data_in, n * sizeof(double));

  // Calculate the left and right boundary
  qsort(data, n, sizeof(double), cmp_double);
  double min_data = data[0];
  double max_data = data[n - 1];
  double delta = 1 / (double) n;
  lower = opt->has_fix_lower ? opt->fix_lower : (min_data < 0 ? min_data - delta : 0);
  upper = opt->has_fix_upper ? opt->fix_upper : (max_data > 1 ? max_data + delta : 1);
  if(!(upper > lower) || !isfinite(lower) || !isfinite(upper)){
    free(data);
    return MF_ERR_BADARG;
  }

  if(opt->smooth){ // Default: BP smooth
    if(opt->smooth_option == MF_SMOOTH_BERNSTEIN){
      status = emp_mode_bern(data, n, lower, upper, opt, res);
    }else{ // Kernel density
      status = mode2(data, n, opt->smooth_density, opt->density_points, res);
    }
  }else{ // Empirical approach
    int knots = opt->m_knots <= 0 ? n : opt->m_knots;
    empirical(data, n, knots, lower, upper, res);
    status = MF_OK;
  }

  free(data);
  if(status != MF_OK) mf_result_free(res);
  return status;
}

void mf_result_free(mf_result *res){
  if(!res) return;
  free(res->density_x);
  free(res->density_y);
  res->density_x = res->density_y = NULL;
  res->n_density = 0;
}

const char *mf_strerror(int code){
  switch(code){
  case MF_OK: return "success";
  case MF_ERR_TOO_SMALL: return "The sample size is too small; it should be larger than 3^(10/3).";
  case MF_ERR_NONFINITE: return "data must not contain NaN or infinite values";
  case MF_ERR_BADARG: return "invalid argument";
  case MF_ERR_NOMEM: return "out of memory";
  case MF_ERR_NUMERIC: return "no finite Anderson-Darling p-value for any Bernstein degree";
  default: return "unknown error";
  }
}
