"""Univariate mode estimation via Bernstein polynomials.

Python interface to the C port of the modeFinder R package.  See
Liu, B. and Ghosh, S. K. (2020) "On empirical estimation of mode based on
weakly dependent samples", Computational Statistics & Data Analysis 152,
107046, https://doi.org/10.1016/j.csda.2020.107046.
"""

import math
import numbers

from ._modefinder import _emp_mode, _smooth

__all__ = ["emp_mode", "ModeFinder"]
__version__ = "0.1.0"


def emp_mode(data, fix_lower=None, fix_upper=None, smooth=True,
             smooth_option="Bernstein", smooth_density=False,
             density_points=512, m_degree=None, sample_size=1000,
             m_knots=None, verbose=False):
    """Empirical mode estimate of a univariate sample.

    Parameters mirror ``modeFinder::emp_mode()`` in R.

    data : sequence of float or float64 array-like
        Sample; more than 38 finite values are required.
    fix_lower, fix_upper : float, optional
        Fixed support bounds.  By default the support is
        ``[min(0, min(data) - 1/n), max(1, max(data) + 1/n)]``.
    smooth : bool
        ``False`` gives the simple empirical mode estimate.
    smooth_option : {"Bernstein", "kernel"}
        Bernstein polynomials (default) or gaussian kernel density estimate.
    smooth_density : bool
        Also return the smoothed density curve as ``x`` and ``y``.
    density_points : int
        Number of points of the density curve.
    m_degree : int, optional
        Degree of the Bernstein polynomials.  By default it is chosen as
        ``ceil(n ** alpha)`` with alpha maximizing the Anderson-Darling
        p-value, stopping once it exceeds 0.95.
    sample_size : int
        Subsample size used by the Anderson-Darling criterion.
    m_knots : int, optional
        Number of knots of the simple empirical estimate; defaults to the
        sample size.
    verbose : bool
        Print the diagnostics the R package prints.

    Returns
    -------
    dict
        Always has ``mode``.  The Bernstein method adds ``optimal_alpha`` and
        ``optimal_p_value`` (degree chosen) or ``given_degree`` and
        ``p_value`` (``m_degree`` given), plus ``degree``, ``max_point``,
        ``prob_mid``, ``prob_left`` and ``prob_right``.  The empirical method
        adds ``number_of_knots``.  ``x`` and ``y`` hold the density curve if
        ``smooth_density`` is set.
    """
    raw = _fit(data, fix_lower, fix_upper, smooth, smooth_option,
               smooth_density, density_points, m_degree, sample_size,
               m_knots, verbose)
    return _public(raw)


def _as_float64(data):
    if hasattr(data, "__array__") and getattr(data, "dtype", None) != "float64":
        # e.g. float32 / int numpy arrays: hand the C code a float64 buffer
        import numpy as np
        data = np.ascontiguousarray(data, dtype=np.float64)
    return data


def _fit(data, fix_lower, fix_upper, smooth, smooth_option, smooth_density,
         density_points, m_degree, sample_size, m_knots, verbose):
    """Run the C estimator; returns its raw result dict."""
    option = str(smooth_option).lower()
    if option == "bernstein":
        option = "Bernstein"
    raw = _emp_mode(_as_float64(data), fix_lower, fix_upper, bool(smooth),
                    option, bool(smooth_density), int(density_points),
                    m_degree, int(sample_size), m_knots)
    if verbose and raw["message"]:
        print(raw["message"], end="")
    return raw


def _public(raw):
    """The emp_mode() result: raw renamed after the R function's output."""
    res = {}
    method = raw["method"]
    if method == "Bernstein":
        if raw["degree_chosen"]:
            res["optimal_alpha"] = raw["alpha"]
            res["optimal_p_value"] = raw["p_value"]
        else:
            res["given_degree"] = raw["degree"]
            res["p_value"] = raw["p_value"]
        res["mode"] = raw["mode"]
        for key in ("degree", "max_point", "prob_mid", "prob_left",
                    "prob_right"):
            res[key] = raw[key]
    elif method == "empirical":
        res["number_of_knots"] = raw["knots"]
        res["mode"] = raw["mode"]
    else:
        res["mode"] = raw["mode"]
    if "x" in raw:
        res["x"] = raw["x"]
        res["y"] = raw["y"]
    return res


class ModeFinder:
    """Smoothed fit of a sample, kept for repeated queries.

    Fits the sample once, exactly as ``emp_mode(data, ...)`` with the
    Bernstein (default) or kernel method, and caches the smoothed
    distribution so that quantiles, the CDF and the density can then be
    evaluated any number of times without refitting::

        fit = ModeFinder(data)                       # or smooth_option="kernel"
        fit.mode
        fit.quantile([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])

    Bernstein: the smoothed distribution on the support ``[lower, upper]``
    has CDF

        F(x) = sum_{i=0}^{m} Fn(i / m) * C(m, i) * u^i * (1 - u)^(m - i),
        u = (x - lower) / (upper - lower),

    where Fn is the empirical CDF of the rescaled sample and m the degree;
    its derivative is the density curve ``emp_mode()`` returns as ``y``.
    The m + 1 values Fn(i / m) are cached.

    Kernel: the density curve of R's ``density()`` that ``emp_mode()``
    returns as ``y`` is the gaussian mixture

        f(x) = sum_j w_j * dnorm(x, g_j, bandwidth)

    of the sample linearly binned onto a regular grid g_j (512 points by
    default), with ``bandwidth`` = ``bw.nrd0(data)``.  Its CDF is
    ``F(x) = sum_j w_j * pnorm(x, g_j, bandwidth)`` on the whole real line.
    The grid weights w_j are cached.

    In both cases the sample itself is not kept.  Parameters are those of
    ``emp_mode()`` that apply to the smoothing methods.

    Attributes
    ----------
    result : dict
        What ``emp_mode()`` returns for the same arguments.
    method : str
        ``"Bernstein"`` or ``"kernel"``.
    mode : float
        The mode estimate, ``result["mode"]``.
    degree : int or None
        Degree m of the Bernstein polynomials (Bernstein only).
    bandwidth : float or None
        Bandwidth of the gaussian kernel (kernel only).
    lower, upper : float
        Support of the smoothed distribution; ``-inf`` and ``inf`` for the
        kernel method.
    """

    def __init__(self, data, fix_lower=None, fix_upper=None,
                 smooth_option="Bernstein", smooth_density=False,
                 density_points=512, m_degree=None, sample_size=1000,
                 verbose=False):
        raw = _fit(data, fix_lower, fix_upper, True, smooth_option,
                   smooth_density, density_points, m_degree, sample_size,
                   None, verbose)
        self.result = _public(raw)
        self.method = raw["method"]
        self.mode = raw["mode"]
        if self.method == "Bernstein":
            self.degree = raw["degree"]
            self.bandwidth = None
            self.lower = raw["lower"]
            self.upper = raw["upper"]
            # Fn(i / m), i = 0..m, as float64 bytes
            self._coef = (raw["bern_fn"], raw["lower"], raw["upper"], 0.0)
        else:
            self.degree = None
            self.bandwidth = raw["bandwidth"]
            self.lower = -math.inf
            self.upper = math.inf
            # grid weights w_j, as float64 bytes, and the grid's end points
            self._coef = (raw["kde_w"], raw["kde_lo"], raw["kde_up"],
                          raw["bandwidth"])

    def __repr__(self):
        if self.method == "Bernstein":
            smooth = f"degree={self.degree}"
        else:
            smooth = f"bandwidth={self.bandwidth!r}"
        return (f"ModeFinder(method={self.method!r}, mode={self.mode!r}, "
                f"{smooth}, support=[{self.lower!r}, {self.upper!r}])")

    def _eval(self, kind, values):
        if isinstance(values, numbers.Real):
            return _smooth(self.method, kind, *self._coef,
                           (float(values),))[0]
        if hasattr(values, "__array__"):
            import numpy as np
            arr = np.asarray(values, dtype=np.float64)
            out = _smooth(self.method, kind, *self._coef,
                          np.ascontiguousarray(arr).ravel())
            return np.array(out, dtype=np.float64).reshape(arr.shape)
        return _smooth(self.method, kind, *self._coef, values)

    def quantile(self, q):
        """Quantiles of the smoothed distribution, ``inf{x: F(x) >= q}``.

        q is a probability in [0, 1] or a sequence / array of them, e.g.
        ``[0.01, 0.05, 0.10, 0.90, 0.95, 0.99]``.  Returns a float for a
        scalar, a list for a sequence and an array of q's shape for an
        array.  ``quantile(0)`` and ``quantile(1)`` are ``lower`` and
        ``upper``.  Raises ValueError for q outside [0, 1].
        """
        return self._eval("quantile", q)

    def cdf(self, x):
        """Smoothed CDF F(x); 0 below ``lower`` and 1 from ``upper`` on."""
        return self._eval("cdf", x)

    def pdf(self, x):
        """Smoothed density in the units of x; 0 outside the support.

        Bernstein: ``emp_mode()``'s ``y`` divided by ``upper - lower``, as
        ``y`` is the density of the sample rescaled to [0, 1].  Kernel: the
        gaussian mixture itself, of which ``y`` is a linear interpolation.
        """
        return self._eval("pdf", x)
