import array
import math
import random
import statistics
import unittest

import modefinder


def beta_sample(n, a=2.0, b=5.0, seed=1):
    rng = random.Random(seed)
    return [rng.betavariate(a, b) for _ in range(n)]


class EmpModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Beta(2, 5) has mode (2 - 1) / (2 + 5 - 2) = 0.2
        cls.data = beta_sample(100000)

    def test_bernstein(self):
        res = modefinder.emp_mode(self.data)
        self.assertAlmostEqual(res["mode"], 0.2, delta=0.03)
        self.assertIn("optimal_alpha", res)
        self.assertGreater(res["optimal_p_value"], 0.05)
        self.assertEqual(res["degree"],
                         math.ceil(len(self.data) ** res["optimal_alpha"]))

    def test_bernstein_given_degree(self):
        res = modefinder.emp_mode(self.data, m_degree=50, smooth_density=True,
                                  density_points=100)
        self.assertEqual(res["given_degree"], 50)
        self.assertIn("p_value", res)
        self.assertAlmostEqual(res["mode"], 0.2, delta=0.05)
        self.assertEqual(len(res["x"]), 100)
        self.assertEqual(len(res["y"]), 100)
        self.assertTrue(all(y >= 0 for y in res["y"]))

    def test_kernel(self):
        res = modefinder.emp_mode(self.data, smooth_option="kernel",
                                  smooth_density=True)
        self.assertAlmostEqual(res["mode"], 0.2, delta=0.03)
        self.assertEqual(len(res["x"]), 512)
        # the density integrates to about one
        dx = res["x"][1] - res["x"][0]
        self.assertAlmostEqual(sum(res["y"]) * dx, 1.0, delta=0.01)

    def test_empirical(self):
        res = modefinder.emp_mode(self.data, smooth=False, m_knots=50)
        self.assertEqual(res["number_of_knots"], 50)
        self.assertAlmostEqual(res["mode"], 0.21, delta=0.03)
        res = modefinder.emp_mode(self.data, smooth=False)
        self.assertEqual(res["number_of_knots"], len(self.data))

    def test_input_types(self):
        data = self.data[:2000]
        ref = modefinder.emp_mode(data)["mode"]
        self.assertEqual(modefinder.emp_mode(tuple(data))["mode"], ref)
        self.assertEqual(modefinder.emp_mode(array.array("d", data))["mode"], ref)
        try:
            import numpy as np
        except ImportError:
            return
        self.assertEqual(modefinder.emp_mode(np.asarray(data))["mode"], ref)
        self.assertEqual(modefinder.emp_mode(np.asarray(data)[::-1])["mode"], ref)

    def test_input_not_modified(self):
        data = array.array("d", self.data[:1000])
        copy = array.array("d", data)
        modefinder.emp_mode(data)
        self.assertEqual(data, copy)

    def test_shifted_support(self):
        # the support bounds are widened automatically outside [0, 1]
        data = [10 * x - 3 for x in self.data[:20000]]
        res = modefinder.emp_mode(data)
        self.assertAlmostEqual(res["mode"], -1.0, delta=0.3)
        res = modefinder.emp_mode(data, fix_lower=-3, fix_upper=7)
        self.assertAlmostEqual(res["mode"], -1.0, delta=0.3)

    def test_errors(self):
        with self.assertRaises(ValueError):
            modefinder.emp_mode([0.5] * 38)
        with self.assertRaises(ValueError):
            modefinder.emp_mode(self.data[:100] + [float("nan")])
        with self.assertRaises(ValueError):
            modefinder.emp_mode(self.data, smooth_option="spline")
        with self.assertRaises(ValueError):
            modefinder.emp_mode(self.data, m_degree=1)
        with self.assertRaises(ValueError):
            modefinder.emp_mode(self.data, fix_lower=1, fix_upper=0)
        with self.assertRaises(TypeError):
            modefinder.emp_mode(["a"] * 100)


def beta25_cdf(x):
    return 1 - (1 - x) ** 6 - 6 * x * (1 - x) ** 5


def beta25_quantile(p):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if beta25_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class ModeFinderTest(unittest.TestCase):
    PROBS = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]

    @classmethod
    def setUpClass(cls):
        cls.data = beta_sample(100000)
        cls.fit = modefinder.ModeFinder(cls.data)

    def test_same_fit_as_emp_mode(self):
        self.assertEqual(self.fit.result, modefinder.emp_mode(self.data))
        self.assertEqual(self.fit.method, "Bernstein")
        self.assertIsNone(self.fit.bandwidth)
        self.assertEqual(self.fit.mode, self.fit.result["mode"])
        self.assertEqual(self.fit.degree, self.fit.result["degree"])
        self.assertEqual((self.fit.lower, self.fit.upper), (0.0, 1.0))

    def test_quantiles_invert_cdf(self):
        qs = self.fit.quantile(self.PROBS)
        self.assertEqual(qs, sorted(qs))
        for p, q in zip(self.PROBS, qs):
            self.assertAlmostEqual(self.fit.cdf(q), p, delta=1e-12)
        self.assertEqual(self.fit.quantile(0), self.fit.lower)
        self.assertEqual(self.fit.quantile(1), self.fit.upper)

    def test_quantiles_beta(self):
        for p, q in zip(self.PROBS, self.fit.quantile(self.PROBS)):
            self.assertAlmostEqual(q, beta25_quantile(p), delta=0.02)

    def test_quantiles_normal(self):
        rng = random.Random(2)
        data = [rng.gauss(5, 2) for _ in range(100000)]
        fit = modefinder.ModeFinder(data)
        self.assertLess(fit.lower, min(data))
        self.assertGreater(fit.upper, max(data))
        dist = statistics.NormalDist(5, 2)
        for p, q in zip(self.PROBS, fit.quantile(self.PROBS)):
            self.assertAlmostEqual(q, dist.inv_cdf(p), delta=0.15)

    def test_cdf_pdf(self):
        fit = self.fit
        self.assertEqual(fit.cdf(-0.5), 0.0)
        self.assertEqual(fit.cdf(1.0), 1.0)
        self.assertEqual(fit.pdf(-0.5), 0.0)
        self.assertEqual(fit.pdf(1.5), 0.0)
        # the density integrates to one (trapezoidal rule)
        k = 4000
        ys = fit.pdf([i / k for i in range(k + 1)])
        self.assertAlmostEqual((sum(ys) - (ys[0] + ys[-1]) / 2) / k, 1.0,
                               delta=1e-4)
        # and is emp_mode()'s density curve on the data scale
        res = modefinder.emp_mode(self.data, smooth_density=True,
                                  density_points=100)
        pdf = fit.pdf(res["x"])
        scale = fit.upper - fit.lower
        for y, f in zip(res["y"], pdf):
            self.assertAlmostEqual(f * scale, y, delta=1e-9 * max(1.0, y))

    def test_shifted_support(self):
        data = [10 * x - 3 for x in self.data[:20000]]
        fit = modefinder.ModeFinder(data, fix_lower=-3, fix_upper=7)
        base = modefinder.ModeFinder(self.data[:20000], fix_lower=0,
                                     fix_upper=1)
        for q, b in zip(fit.quantile(self.PROBS), base.quantile(self.PROBS)):
            self.assertAlmostEqual(q, 10 * b - 3, delta=1e-6)

    def test_return_types(self):
        self.assertIsInstance(self.fit.quantile(0.5), float)
        self.assertIsInstance(self.fit.quantile((0.1, 0.9)), list)
        self.assertEqual(self.fit.quantile(array.array("d", [0.1, 0.9])),
                         self.fit.quantile([0.1, 0.9]))
        try:
            import numpy as np
        except ImportError:
            return
        p = np.array([[0.01, 0.05, 0.1], [0.9, 0.95, 0.99]])
        q = self.fit.quantile(p)
        self.assertIsInstance(q, np.ndarray)
        self.assertEqual(q.shape, p.shape)
        self.assertEqual(q.ravel().tolist(), self.fit.quantile(p.ravel().tolist()))
        self.assertEqual(self.fit.quantile(np.float32(0.5)),
                         self.fit.quantile(0.5))

    def test_errors(self):
        for bad in (-0.1, 1.1, float("nan"), [0.5, 2]):
            with self.assertRaises(ValueError):
                self.fit.quantile(bad)
        with self.assertRaises(TypeError):
            self.fit.quantile(["a"])



class KernelModeFinderTest(unittest.TestCase):
    PROBS = ModeFinderTest.PROBS

    @classmethod
    def setUpClass(cls):
        cls.data = beta_sample(100000)
        cls.fit = modefinder.ModeFinder(cls.data, smooth_option="kernel")

    def test_same_fit_as_emp_mode(self):
        fit = self.fit
        self.assertEqual(fit.result,
                         modefinder.emp_mode(self.data, smooth_option="kernel"))
        self.assertEqual(fit.method, "kernel")
        self.assertIsNone(fit.degree)
        self.assertGreater(fit.bandwidth, 0)
        self.assertEqual((fit.lower, fit.upper), (-math.inf, math.inf))
        self.assertEqual(modefinder.ModeFinder(self.data[:5000],
                                               smooth_option="KERNEL").method,
                         "kernel")

    def test_quantiles_invert_cdf(self):
        qs = self.fit.quantile(self.PROBS)
        self.assertEqual(qs, sorted(qs))
        for p, q in zip(self.PROBS, qs):
            self.assertAlmostEqual(self.fit.cdf(q), p, delta=1e-12)
        self.assertEqual(self.fit.quantile([0, 1]), [-math.inf, math.inf])
        # tail probabilities keep their precision on both sides
        for p in (1e-10, 1 - 1e-10):
            q = self.fit.quantile(p)
            self.assertAlmostEqual(self.fit.cdf(q), p, delta=1e-15)

    def test_quantiles_beta(self):
        for p, q in zip(self.PROBS, self.fit.quantile(self.PROBS)):
            self.assertAlmostEqual(q, beta25_quantile(p), delta=0.01)

    def test_quantiles_normal(self):
        rng = random.Random(2)
        data = [rng.gauss(5, 2) for _ in range(100000)]
        fit = modefinder.ModeFinder(data, smooth_option="kernel")
        dist = statistics.NormalDist(5, 2)
        for p, q in zip(self.PROBS, fit.quantile(self.PROBS)):
            self.assertAlmostEqual(q, dist.inv_cdf(p), delta=0.05)

    def test_cdf_pdf(self):
        fit = self.fit
        self.assertEqual(fit.cdf(-math.inf), 0.0)
        self.assertEqual(fit.cdf(math.inf), 1.0)
        self.assertEqual(fit.pdf(-2.0), 0.0)
        # the density integrates to one (trapezoidal rule)
        a, b, k = -0.5, 1.5, 20000
        ys = fit.pdf([a + (b - a) * i / k for i in range(k + 1)])
        self.assertAlmostEqual((sum(ys) - (ys[0] + ys[-1]) / 2) * (b - a) / k,
                               1.0, delta=1e-6)
        # and emp_mode()'s density curve linearly interpolates it
        res = modefinder.emp_mode(self.data, smooth_option="kernel",
                                  smooth_density=True)
        top = max(res["y"])
        for y, f in zip(res["y"], fit.pdf(res["x"])):
            self.assertAlmostEqual(f, y, delta=1e-3 * top)

    def test_weights(self):
        data = self.data[:20000]
        base = modefinder.ModeFinder(data, smooth_option="kernel")
        # equal weights of any scale give the unweighted fit
        same = modefinder.ModeFinder(data, smooth_option="kernel",
                                     weights=[3.0] * len(data))
        self.assertEqual(same.result, base.result)
        self.assertEqual(same.quantile(self.PROBS), base.quantile(self.PROBS))
        # weight 0 on the points above 0.3: almost no mass is left there
        fit = modefinder.ModeFinder(data, smooth_option="kernel",
                                    weights=[float(x <= 0.3) for x in data])
        self.assertGreater(fit.cdf(0.3 + 3 * fit.bandwidth), 0.99)
        self.assertEqual(fit.bandwidth, base.bandwidth)
        # weight 2 is close to repeating the point (only the bandwidth differs)
        w = [2.0] * 2000 + [1.0] * (len(data) - 2000)
        fit = modefinder.ModeFinder(data, smooth_option="kernel", weights=w)
        rep = modefinder.ModeFinder(data + data[:2000], smooth_option="kernel")
        for a, b in zip(fit.quantile(self.PROBS), rep.quantile(self.PROBS)):
            self.assertAlmostEqual(a, b, delta=1e-3)
        # the weights move with their points when the data are sorted
        order = sorted(range(len(data)), key=lambda i: -data[i])
        back = modefinder.ModeFinder([data[i] for i in order], smooth_option="kernel",
                                     weights=[w[i] for i in order])
        self.assertEqual(back.quantile(self.PROBS), fit.quantile(self.PROBS))

    def test_weights_errors(self):
        data = self.data[:1000]
        for bad in ([1.0] * 999, [-1.0] + [1.0] * 999, [0.0] * 1000,
                    [float("nan")] + [1.0] * 999):
            with self.assertRaises(ValueError):
                modefinder.ModeFinder(data, smooth_option="kernel", weights=bad)
        with self.assertRaises(ValueError):
            modefinder.ModeFinder(data, weights=[1.0] * 1000)        # Bernstein
        with self.assertRaises(ValueError):
            modefinder.emp_mode(data, smooth=False, weights=[1.0] * 1000)

    def test_shifted_data(self):
        # the kernel estimate is equivariant under affine maps of the data
        data = [10 * x - 3 for x in self.data[:20000]]
        fit = modefinder.ModeFinder(data, smooth_option="kernel")
        base = modefinder.ModeFinder(self.data[:20000], smooth_option="kernel")
        self.assertAlmostEqual(fit.bandwidth, 10 * base.bandwidth, delta=1e-9)
        for q, b in zip(fit.quantile(self.PROBS), base.quantile(self.PROBS)):
            self.assertAlmostEqual(q, 10 * b - 3, delta=1e-9)

    def test_errors(self):
        for bad in (-0.1, 1.1, float("nan"), [0.5, 2]):
            with self.assertRaises(ValueError):
                self.fit.quantile(bad)
        with self.assertRaises(ValueError):
            modefinder.ModeFinder(self.data, smooth_option="spline")

if __name__ == "__main__":
    unittest.main()
