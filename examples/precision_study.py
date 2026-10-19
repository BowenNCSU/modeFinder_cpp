"""Monte Carlo precision study of modefinder.emp_mode().

Draws repeated samples from normal and beta distributions with known modes
and reports, for each estimator, the bias, standard deviation, RMSE and the
mean absolute error of the mode estimate.  Requires numpy.

    python examples/precision_study.py [--reps 200] [--sizes 100 1000 10000]
"""

import argparse
import time

import numpy as np

import modefinder

# name, sampler(rng, n), true mode, scale used to make errors comparable
DISTRIBUTIONS = [
    ("Normal(0, 1)", lambda rng, n: rng.normal(0, 1, n), 0.0, 1.0),
    ("Normal(5, 2)", lambda rng, n: rng.normal(5, 2, n), 5.0, 2.0),
    ("Beta(2, 5)", lambda rng, n: rng.beta(2, 5, n), 0.2, 1.0),
    ("Beta(5, 2)", lambda rng, n: rng.beta(5, 2, n), 0.8, 1.0),
    ("Beta(2, 2)", lambda rng, n: rng.beta(2, 2, n), 0.5, 1.0),
    ("Beta(3, 10)", lambda rng, n: rng.beta(3, 10, n), 2 / 11, 1.0),
    ("Beta(1, 3)", lambda rng, n: rng.beta(1, 3, n), 0.0, 1.0),  # boundary mode
]

METHODS = [
    ("Bernstein", {}),
    ("kernel", {"smooth_option": "kernel"}),
    ("empirical", {"smooth": False}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100, 1000, 10000, 100000])
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print(f"{args.reps} replications per cell; errors are divided by the "
          "distribution's scale (sd for normals)\n")
    header = (f"{'distribution':<13} {'n':>7} {'method':<10} {'bias':>9} "
              f"{'sd':>8} {'RMSE':>8} {'MAE':>8} {'ms/call':>8}")
    print(header)
    print("-" * len(header))
    for name, sampler, mode, scale in DISTRIBUTIONS:
        for n in args.sizes:
            rng = np.random.default_rng([args.seed, n])
            samples = [sampler(rng, n) for _ in range(args.reps)]
            for method, kwargs in METHODS:
                t0 = time.perf_counter()
                est = np.array([modefinder.emp_mode(x, **kwargs)["mode"]
                                for x in samples])
                ms = (time.perf_counter() - t0) / args.reps * 1e3
                err = (est - mode) / scale
                print(f"{name:<13} {n:>7} {method:<10} {err.mean():>9.4f} "
                      f"{err.std(ddof=1):>8.4f} "
                      f"{np.sqrt(np.mean(err ** 2)):>8.4f} "
                      f"{np.mean(np.abs(err)):>8.4f} {ms:>8.2f}")
        print()


if __name__ == "__main__":
    main()
