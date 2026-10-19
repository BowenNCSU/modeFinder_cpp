"""Univariate mode estimation via Bernstein polynomials.

Python interface to the C port of the modeFinder R package.  See
Liu, B. and Ghosh, S. K. (2020) "On empirical estimation of mode based on
weakly dependent samples", Computational Statistics & Data Analysis 152,
107046, https://doi.org/10.1016/j.csda.2020.107046.
"""

from ._modefinder import _emp_mode

__all__ = ["emp_mode"]
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
    option = str(smooth_option).lower()
    if option == "bernstein":
        option = "Bernstein"
    if hasattr(data, "__array__") and getattr(data, "dtype", None) != "float64":
        # e.g. float32 / int numpy arrays: hand the C code a float64 buffer
        import numpy as np
        data = np.ascontiguousarray(data, dtype=np.float64)
    raw = _emp_mode(data, fix_lower, fix_upper, bool(smooth), option,
                    bool(smooth_density), int(density_points), m_degree,
                    int(sample_size), m_knots)
    if verbose and raw["message"]:
        print(raw["message"], end="")

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
