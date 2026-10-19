/*
 * modeFinder -- univariate mode estimation via Bernstein polynomials.
 *
 * Plain C99 port of the modeFinder R package (Liu, B. and Ghosh, S. K.
 * (2020) "On empirical estimation of mode based on weakly dependent
 * samples", Computational Statistics & Data Analysis 152, 107046).
 *
 * Copyright (C) 2018  Bowen Liu.  Licensed under the GNU General Public
 * License, version 3 or (at your option) any later version.
 */

#ifndef MODEFINDER_H
#define MODEFINDER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MF_MESSAGE_SIZE 512

/* Return codes of mf_emp_mode() */
enum {
    MF_OK = 0,
    MF_ERR_TOO_SMALL = 1,  /* sample size must be larger than 3^(10/3) */
    MF_ERR_NONFINITE = 2,  /* data contains NaN or infinite values */
    MF_ERR_BADARG = 3,     /* invalid option value */
    MF_ERR_NOMEM = 4,      /* memory allocation failed */
    MF_ERR_NUMERIC = 5     /* no finite Anderson-Darling p-value */
};

typedef enum {
    MF_SMOOTH_BERNSTEIN = 0,  /* Bernstein polynomials density estimate */
    MF_SMOOTH_KERNEL = 1      /* gaussian kernel density, as R's density() */
} mf_smooth_option;

/* Estimator that produced a result */
typedef enum {
    MF_METHOD_BERNSTEIN = 0,
    MF_METHOD_KERNEL = 1,
    MF_METHOD_EMPIRICAL = 2
} mf_method;

/* Arguments of emp_mode(); initialize with mf_default_options(). */
typedef struct {
    int has_fix_lower;   /* 0: lower bound is min(0, min(data) - 1/n) */
    double fix_lower;
    int has_fix_upper;   /* 0: upper bound is max(1, max(data) + 1/n) */
    double fix_upper;
    int smooth;          /* 0: simple empirical mode, no smoothing */
    mf_smooth_option smooth_option;
    int smooth_density;  /* also return the smoothed density curve */
    int density_points;  /* number of points of the density curve */
    int m_degree;        /* Bernstein degree; <= 0 chooses it automatically */
    int sample_size;     /* subsample size for the Anderson-Darling test */
    int m_knots;         /* empirical mode knots; <= 0 uses the sample size */
} mf_options;

typedef struct {
    mf_method method;
    double mode;

    /* MF_METHOD_BERNSTEIN */
    int degree_chosen;   /* 1: degree chosen via alpha, 0: m_degree given */
    int degree;          /* Bernstein degree m used */
    double alpha;        /* optimal alpha, m = ceil(n^alpha) (if chosen) */
    double p_value;      /* Anderson-Darling p-value for the degree used */
    double max_point;    /* maximum of the density, rescaled to [0, 1] */
    double prob_mid;     /* empirical mass within 1/m of max_point */
    double prob_left;    /* empirical mass of [0, 2/m] */
    double prob_right;   /* empirical mass of (1 - 2/m, 1] */

    /* MF_METHOD_EMPIRICAL */
    int knots;

    /* Density curve (smooth && smooth_density), freed by mf_result_free() */
    int n_density;
    double *density_x;
    double *density_y;

    /* Diagnostics the R package printed to the console */
    char message[MF_MESSAGE_SIZE];
} mf_result;

void mf_default_options(mf_options *opt);

/* Estimate the mode of data[0..n-1].  opt may be NULL for the defaults.
   data is not modified.  On success returns MF_OK and fills *res, which
   must later be released with mf_result_free(). */
int mf_emp_mode(const double *data, size_t n, const mf_options *opt,
		mf_result *res);

void mf_result_free(mf_result *res);

const char *mf_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* MODEFINDER_H */
