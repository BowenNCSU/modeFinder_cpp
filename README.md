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
a clean virtual environment, `pip install .` builds it and all 8 Python
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
