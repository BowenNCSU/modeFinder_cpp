/* Internal declarations shared by the modeFinder C sources. */

#ifndef MF_INTERNAL_H
#define MF_INTERNAL_H

#include <stdlib.h>

#if defined(_MSC_VER)
# define MF_THREAD_LOCAL __declspec(thread)
#else
# define MF_THREAD_LOCAL _Thread_local
#endif

/* linpack.c: Fortran-style BLAS / LINPACK replacements */
void mf_dcopy(const int *n, const double *dx, const int *incx,
	      double *dy, const int *incy);
double mf_ddot(const int *n, const double *dx, const int *incx,
	       const double *dy, const int *incy);
void mf_dscal(const int *n, const double *da, double *dx, const int *incx);
void mf_daxpy(const int *n, const double *da, const double *dx,
	      const int *incx, double *dy, const int *incy);
void mf_dpofa(double *a, const int *lda, const int *n, int *info);
void mf_dtrsl(double *t, const int *ldt, const int *n, double *b,
	      const int *job, int *info);

/* lbfgsb.c: L-BFGS-B, with the same interface as R's lbfgsb() */
typedef double mf_optimfn(int n, double *x, void *ex);
typedef void mf_optimgr(int n, double *x, double *gr, void *ex);

void mf_lbfgsb(int n, int m, double *x, double *l, double *u, int *nbd,
	       double *Fmin, mf_optimfn fminfn, mf_optimgr fmingr, int *fail,
	       void *ex, double factr, double pgtol,
	       int *fncount, int *grcount, int maxit, char *msg,
	       int trace, int nREPORT);

/* kde.c: R's stats::density() with its default arguments (gaussian kernel,
   bw = "nrd0", cut = 3) evaluated at n_out points.  x must be sorted and
   finite with nx >= 2.  Returns 0 on success, -1 if out of memory. */
int mf_density(const double *x, int nx, int n_out,
	       double *out_x, double *out_y);

#endif /* MF_INTERNAL_H */
