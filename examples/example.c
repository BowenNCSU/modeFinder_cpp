/* Example of calling the modeFinder C library directly; see Makefile. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "modefinder.h"

int main(void)
{
    enum { N = 100000 };
    static double data[N];
    mf_options opt;
    mf_result res;
    int status;
    static const double probs[] = {0.01, 0.05, 0.10, 0.90, 0.95, 0.99};

    /* Beta(2, 5) sample as the ratio X / (X + Y) of Gamma(2) and Gamma(5)
       variates, each a sum of exponentials; its mode is 0.2. */
    srand(1);
    for (int i = 0; i < N; i++) {
        double x = 0, y = 0;
        for (int k = 0; k < 2; k++) x -= log1p(-rand() / (RAND_MAX + 1.0));
        for (int k = 0; k < 5; k++) y -= log1p(-rand() / (RAND_MAX + 1.0));
        data[i] = x / (x + y);
    }

    mf_default_options(&opt);
    status = mf_emp_mode(data, N, &opt, &res);
    if (status != MF_OK) {
        fprintf(stderr, "mf_emp_mode: %s\n", mf_strerror(status));
        return 1;
    }
    printf("%s", res.message);
    printf("Bernstein: mode %g, degree %d, alpha %g, p-value %g\n",
           res.mode, res.degree, res.alpha, res.p_value);
    /* quantiles of the smoothed distribution, from the cached Fn(i / m) */
    printf("quantiles:");
    for (int i = 0; i < (int) (sizeof(probs) / sizeof(probs[0])); i++)
        printf(" P%02.0f %.4f", probs[i] * 100,
               mf_bern_quantile(res.bern_fn, res.degree, res.lower,
                                res.upper, probs[i]));
    printf("\n");
    mf_result_free(&res);

    opt.smooth_option = MF_SMOOTH_KERNEL;
    if (mf_emp_mode(data, N, &opt, &res) == MF_OK) {
        printf("kernel:    mode %g, bandwidth %g\n", res.mode, res.bandwidth);
        printf("quantiles:");
        for (int i = 0; i < (int) (sizeof(probs) / sizeof(probs[0])); i++)
            printf(" P%02.0f %.4f", probs[i] * 100,
                   mf_kernel_quantile(res.kde_w, res.kde_n, res.kde_lo,
                                      res.kde_up, res.bandwidth, probs[i]));
        printf("\n");
    }
    mf_result_free(&res);

    opt.smooth = 0;
    if (mf_emp_mode(data, N, &opt, &res) == MF_OK)
        printf("empirical: mode %g, knots %d\n", res.mode, res.knots);
    mf_result_free(&res);
    return 0;
}
