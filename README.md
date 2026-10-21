# modeFinder

<!-- badges: start -->

<!-- badges: end -->

The goal of modeFinder is to compute the univariate mode estiamtes via
simple empirical density estimates, kernel density estimates or
Bernstein polynomials.

## Mode estimation via Bernstein polynomials

The mode estimation via Bernstein polynomials is based on the draft of:

Liu, Bowen, and Sujit K. Ghosh. *“On empirical estimation of mode based
on weakly dependent samples.”* Computational Statistics & Data Analysis
152 (2020): 107046.

## Simple empirical mode estimation

It is natural to estimate the population distribution function by the
empirical distribution function, \(F_n\) given by

\[
F_n = \frac{1}{n} \sum^n_{i = 1} I \{X_i \leq x\}
\]

and for any \(X^*\) such that \(X^{(i)} < X^* < X^{(i + 1)}\),
\(i = 1, 2, \dots, n\), the slope of the empirical distribution function
can be approximated by

\[
\frac{ F_n (X^{(i + 1)}) - F_n (X^{(i)})}{X^{(i + 1)} - X^{(i)}}.
\]

Consequently, we can give a simple mode estimator, \(u^{(\hat{k})}\)
based on the empirical distribution function by

\[
\hat{k} = \underset{0 \leq i \leq m^* - 1}{argmax} \frac{ F_n (u^{(i + 1)}) - F_n (u^{(i)})}{u^{(i + 1)} - u^{(i)}}.
\]

where
\(\{u^{(1)}, u^{(2)}, \dots, u^{(m^*)}\} = \{ \frac{1}{m^*}, \frac{2}{m^*}, \dots, \frac{m^*}{m^*}\}\).
However, the simple empirical mode estimation does not collaborate with
the possible smoothness characteristics of the underlying distribution
function.

## Installation

Install the Python package from the repository root (a C compiler is
required; numpy is optional):

``` sh
pip install .
```

## Python and C

I've ported the whole algorithm to plain C99 with no R or Rcpp dependency,
wrapped it as a Python package, and removed the R package. In
a clean virtual environment, `pip install .` builds it and all 25 Python
tests pass. R isn't installed on this machine, so I couldn't compare results
against the R function directly.

### Using it from Python

``` python
import modefinder
modefinder.emp_mode(data)                          # Bernstein polynomials
modefinder.emp_mode(data, smooth_option="kernel")  # kernel density
modefinder.emp_mode(data, smooth=False)            # simple empirical
```

- **Arguments:** the same as the R function, plus `verbose=True` to print
  the diagnostics R prints.
- **Result:** a `dict` with R's names made Python-friendly
  (`optimal_alpha`, `optimal_p_value`, `given_degree`, `p_value`,
  `number_of_knots`, `x`/`y`).
- **Input:** a list, an `array.array`, or any numpy array. numpy is
  optional.
- **Speed:** each method takes about 0.1 s on 10⁶ points.
- **Weights (kernel only):** `weights=` takes one finite, nonnegative
  number per data point. They are divided by their sum and replace the
  equal 1/n mass each point gets in the kernel estimate, as R's
  `density(x, weights = w)`. The bandwidth is still `bw.nrd0(data)`,
  unweighted. `ModeFinder` takes the same argument, and in C it is
  `mf_options.weights` (NULL for equal weights). Passing weights with
  Bernstein or `smooth=False` raises `ValueError`.

  ``` python
  modefinder.emp_mode(prices, smooth_option="kernel", weights=volumes)
  ```

### Quantiles of the smoothed distribution

`ModeFinder` fits a sample once, the same fit as `emp_mode(data)`, and
caches the smoothed distribution. You can then ask it for any number of
quantiles without refitting. It supports both smoothing options:

``` python
fit = modefinder.ModeFinder(data)                         # Bernstein
fit = modefinder.ModeFinder(data, smooth_option="kernel") # kernel density
fit.mode                                # == emp_mode(data, ...)["mode"]
fit.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])   # P01 ... P99
fit.quantile(0.5)                       # scalar in, float out
fit.cdf(x), fit.pdf(x)                  # smoothed CDF and density
fit.result                              # the emp_mode() dict
```

- **Bernstein:** the smoothed CDF on the support `[lower, upper]` is
  F(x) = Σᵢ Fn(i/m) C(m,i) uⁱ(1−u)ᵐ⁻ⁱ, where u = (x − lower)/(upper − lower).
  Its derivative is the density curve `emp_mode()` returns as `y`. The cache
  is the m + 1 values Fn(i/m), and `pdf` is `y / (upper − lower)`.
- **Kernel:** R's `density()` bins the sample onto a regular grid gⱼ (512
  points by default) and convolves it with a Gaussian kernel. So its curve
  is the mixture f(x) = Σⱼ wⱼ φ((x − gⱼ)/h)/h, with h = `bw.nrd0(data)`
  (`fit.bandwidth`), and the CDF is F(x) = Σⱼ wⱼ Φ((x − gⱼ)/h). The cache is
  the grid weights wⱼ. The support is the whole real line
  (`fit.lower = -inf`, `fit.upper = inf`), and `emp_mode()`'s `y` is a
  linear interpolation of `pdf`.
- **Quantile rule:** `quantile(p)` returns inf{x : F(x) ≥ p}, solved by
  Newton's method with a bisection safeguard to about machine precision.
  Above the median the kernel method solves on 1 − F, so P99.99… keeps its
  precision. `quantile(0)` and `quantile(1)` are `lower` and `upper`.
- **Cost:** the sample isn't kept. A quantile costs O(m) or O(grid size).
  P01–P99 together take about 0.1 ms (Bernstein) or 0.6 ms (kernel).
- **Input and output types:** `quantile`, `cdf` and `pdf` accept a scalar, a
  sequence or a numpy array, and return a float, a list or an array of the
  same shape.
- **Tails:** smoothing widens the distribution slightly, so tail quantiles
  sit a little outside the sample quantiles. On 10⁵ standard-normal points,
  P99 is about 2.38 (Bernstein) or 2.34 (kernel), against 2.33. For
  Bernstein, pass a larger `m_degree` for a tighter fit. The kernel spills
  mass past a hard boundary: for Beta(2,5), its P01 is 0.023 against 0.026.

#### Calling the Bernstein or kernel quantiles

The `smooth_option` argument to `ModeFinder` picks the method when you fit.
After that, the same `quantile()` call works for either:

``` python
import modefinder

# Bernstein (the default)
bern = modefinder.ModeFinder(data)
bern.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])

# Kernel
kern = modefinder.ModeFinder(data, smooth_option="kernel")
kern.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
```

- **Checking which you have:** `fit.method` is `"Bernstein"` or
  `"kernel"`. `repr(fit)` shows it as well.
- **Switching methods:** make a second object. Each one fits the data once
  and keeps only its own fitted curve, so the two can be used side by side.
- **Other arguments:** they're the same as `emp_mode()`. `m_degree`,
  `fix_lower`, `fix_upper` and `sample_size` only affect Bernstein;
  `density_points` sets the kernel's grid size.
- **Accepted input:** `quantile`, `cdf` and `pdf` take a single number, a
  list or a numpy array for both methods.

From C, run `mf_emp_mode()` with the option you want, then call the
matching function on the result:

``` c
mf_options opt;
mf_result res;
mf_default_options(&opt);                     /* Bernstein */
mf_emp_mode(data, n, &opt, &res);
double q = mf_bern_quantile(res.bern_fn, res.degree, res.lower, res.upper, 0.95);
mf_result_free(&res);

opt.smooth_option = MF_SMOOTH_KERNEL;         /* kernel */
mf_emp_mode(data, n, &opt, &res);
q = mf_kernel_quantile(res.kde_w, res.kde_n, res.kde_lo, res.kde_up,
                       res.bandwidth, 0.95);
mf_result_free(&res);
```

- **Bernstein functions:** `mf_bern_quantile()`, `mf_bern_cdf()` and
  `mf_bern_pdf()` take `res.bern_fn`, `res.degree`, `res.lower` and
  `res.upper`.
- **Kernel functions:** `mf_kernel_quantile()`, `mf_kernel_cdf()` and
  `mf_kernel_pdf()` take `res.kde_w`, `res.kde_n`, `res.kde_lo`,
  `res.kde_up` and `res.bandwidth`.
- **Example:** `examples/example.c` does both and prints P01–P99 for each.

### Using it from C

- `make` builds `libmodefinder.dylib`/`.so` and `libmodefinder.a`.
- Include `csrc/modefinder.h` and call `mf_emp_mode(data, n, &opt, &res)`;
  `examples/example.c` shows the usage and `make example` runs it.
- The library holds no global state, so it is safe to call from several
  threads.

### What's new

- `csrc/modefinder.c`: the core algorithm, translated line by line from
  the Rcpp source (`src/emp_mode.cpp` in the git history), with the same steps, order and messages.
- `csrc/kde.c`: a C version of R's `density()`, which the kernel option
  used to call. It matches a brute-force density calculation to about
  6e-5.
- `csrc/lbfgsb.c` and `csrc/linpack.c`: R's L-BFGS-B optimizer and the
  small linear-algebra routines it needs. The optimizer code is
  token-for-token identical to the
  copy the R package vendored (`src/lbfgsb.h` in the git history).
- `modefinder/` (the Python wrapper), `pyproject.toml`, `setup.py`,
  `MANIFEST.in`, `Makefile` and `tests/`.
- This README and `.gitignore` updated for the Python and C build.

### Checks

- The C code compiles with full warnings on, and the example runs cleanly
  under AddressSanitizer and UBSan.
- A source package builds and installs in a fresh environment.
- The estimates look right: about 0.2 for Beta(2,5) samples, and 0 for
  samples whose mode sits on the boundary.

### Where it differs from the R version

- **Bugs fixed:**
  - The kernel method read past the end of the density curve when the
    sample was bigger than `density_points`.
  - Three integer overflows that only show up with very large samples or
    degrees.
  - The tie-breaking random pick between the two boundaries could, rarely,
    return a point outside the support.
- **Stricter input checks:** NaN/Inf values, `m_degree < 2` and
  `fix_upper <= fix_lower` now raise errors instead of producing garbage.
- **Kernel option:**
  - It reproduces `density()` from R 4.4 and later.
  - Its FFT isn't R's, so results agree to rounding error rather than bit
    for bit.
  - `smooth_option` is case-insensitive in Python.
- **Random tie-break:** when the boundary pick is random, it uses C's
  `rand()`, so it won't match R's sequence.

Nothing is committed yet. If you have R available, comparing `emp_mode()`
outputs on a few datasets would be the remaining check.

### Note:

macOS requires the Xcode command line tools to compile the C code,
otherwise *xcrun: error* is reported. Run `xcode-select --install`.
