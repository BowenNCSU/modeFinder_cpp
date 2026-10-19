import array
import math
import random
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


if __name__ == "__main__":
    unittest.main()
