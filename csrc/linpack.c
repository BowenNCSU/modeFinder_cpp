/*
 * Minimal BLAS level-1 and LINPACK routines used by lbfgsb.c.
 *
 * These are C translations of the reference Fortran routines that R links
 * into its L-BFGS-B code (BLAS dcopy/ddot/dscal/daxpy, LINPACK dpofa/dtrsl,
 * the latter two as modified for R).  They keep the Fortran calling
 * convention -- scalars passed by pointer, column-major arrays -- so the
 * f2c-translated optimizer can call them unchanged.
 */

#include <math.h>
#include "mf_internal.h"

/* Start index of a strided vector, following the reference BLAS. */
#define START(n, inc) ((inc) < 0 ? (1 - (n)) * (inc) : 0)

void mf_dcopy(const int *n, const double *dx, const int *incx,
	      double *dy, const int *incy)
{
    int i, ix, iy;
    if (*n <= 0) return;
    ix = START(*n, *incx);
    iy = START(*n, *incy);
    for (i = 0; i < *n; i++, ix += *incx, iy += *incy)
	dy[iy] = dx[ix];
}

double mf_ddot(const int *n, const double *dx, const int *incx,
	       const double *dy, const int *incy)
{
    int i, ix, iy;
    double dtemp = 0.;
    if (*n <= 0) return 0.;
    ix = START(*n, *incx);
    iy = START(*n, *incy);
    for (i = 0; i < *n; i++, ix += *incx, iy += *incy)
	dtemp += dx[ix] * dy[iy];
    return dtemp;
}

void mf_dscal(const int *n, const double *da, double *dx, const int *incx)
{
    int i;
    if (*n <= 0 || *incx <= 0) return;
    for (i = 0; i < *n * *incx; i += *incx)
	dx[i] = *da * dx[i];
}

void mf_daxpy(const int *n, const double *da, const double *dx,
	      const int *incx, double *dy, const int *incy)
{
    int i, ix, iy;
    if (*n <= 0 || *da == 0.) return;
    ix = START(*n, *incx);
    iy = START(*n, *incy);
    for (i = 0; i < *n; i++, ix += *incx, iy += *incy)
	dy[iy] += *da * dx[ix];
}

/* Column-major element (i, j), 1-based as in the Fortran source. */
#define A(i, j) a[((i) - 1) + ((j) - 1) * (long) *lda]
#define T(i, j) t[((i) - 1) + ((j) - 1) * (long) *ldt]

/* Cholesky factorization of a symmetric positive definite matrix.
   R's version uses a tolerance for positive definiteness. */
void mf_dpofa(double *a, const int *lda, const int *n, int *info)
{
    static const int one = 1;
    const double eps = 1e-14;
    int j, k, km1;
    double s, t;

    for (j = 1; j <= *n; j++) {
	*info = j;
	s = 0.;
	for (k = 1; k <= j - 1; k++) {
	    km1 = k - 1;
	    t = A(k, j) - mf_ddot(&km1, &A(1, k), &one, &A(1, j), &one);
	    t /= A(k, k);
	    A(k, j) = t;
	    s += t * t;
	}
	s = A(j, j) - s;
	if (s <= eps * fabs(A(j, j))) return;
	A(j, j) = sqrt(s);
    }
    *info = 0;
}

/* Solve  T x = b  or  T' x = b  for triangular T; the solution overwrites b.
   job = 00: T lower, 01: T upper, 10: T' with T lower, 11: T' with T upper. */
void mf_dtrsl(double *t, const int *ldt, const int *n, double *b,
	      const int *job, int *info)
{
    static const int one = 1;
    int j, jj, len;
    double temp;

    /* check for zero diagonal elements */
    for (*info = 1; *info <= *n; (*info)++)
	if (T(*info, *info) == 0.) return;
    *info = 0;

    switch ((*job % 10 != 0 ? 2 : 1) + (*job % 100 / 10 != 0 ? 2 : 0)) {
    case 1: /* T x = b, T lower triangular */
	b[0] /= T(1, 1);
	for (j = 2; j <= *n; j++) {
	    temp = -b[j - 2];
	    len = *n - j + 1;
	    mf_daxpy(&len, &temp, &T(j, j - 1), &one, &b[j - 1], &one);
	    b[j - 1] /= T(j, j);
	}
	break;
    case 2: /* T x = b, T upper triangular */
	b[*n - 1] /= T(*n, *n);
	for (jj = 2; jj <= *n; jj++) {
	    j = *n - jj + 1;
	    temp = -b[j];
	    mf_daxpy(&j, &temp, &T(1, j + 1), &one, &b[0], &one);
	    b[j - 1] /= T(j, j);
	}
	break;
    case 3: /* T' x = b, T lower triangular */
	b[*n - 1] /= T(*n, *n);
	for (jj = 2; jj <= *n; jj++) {
	    j = *n - jj + 1;
	    len = jj - 1;
	    b[j - 1] -= mf_ddot(&len, &T(j + 1, j), &one, &b[j], &one);
	    b[j - 1] /= T(j, j);
	}
	break;
    case 4: /* T' x = b, T upper triangular */
	b[0] /= T(1, 1);
	for (j = 2; j <= *n; j++) {
	    len = j - 1;
	    b[j - 1] -= mf_ddot(&len, &T(1, j), &one, &b[0], &one);
	    b[j - 1] /= T(j, j);
	}
	break;
    }
}
