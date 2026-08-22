"""
Benchmark for pystochastic.random (crandom + drandom).

Measures generation time for each distribution as a function of the
number of samples requested. Errors on individual distributions are
reported but do not stop the benchmark.
"""

import time
import numpy as np

from pystochastic.random import crandom
from pystochastic.random import drandom


# ---------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------

def benchmark_call(fn, n_runs=5):
    """Times a zero-argument callable n_runs times."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)


def run_group(label, functions, sample_sizes, n_runs=5):
    """
    functions : dict {name: callable(n) -> samples}
    """
    print("=" * 70)
    print(f"random benchmark -- {label}")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n in sample_sizes:
        print(f"n = {n:,}")
        for name, fn in functions.items():
            try:
                mean_t, min_t, max_t = benchmark_call(lambda: fn(n), n_runs=n_runs)
                print(f"  {name:<18s} : {mean_t*1000:8.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  {name:<18s} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main():

    sample_sizes = [1_000, 10_000, 100_000, 1_000_000]
    n_runs = 5

    continuous_laws = {
        "uniform": lambda n: crandom.uniform(0, 1, n),
        "exponential": lambda n: crandom.exponential(1, n),
        "normal": lambda n: crandom.normal(0, 1, n),
        "gamma (int shape)": lambda n: crandom.gamma(3, 1, n),
        "gamma (frac shape)": lambda n: crandom.gamma(2.5, 1, n),
        "beta": lambda n: crandom.beta(2, 3, n),
        "weibull": lambda n: crandom.weibull(1.5, 1, n),
        "frechet": lambda n: crandom.frechet(2, 1, 0, n),
        "cauchy": lambda n: crandom.cauchy(0, 1, n),
        "gumbel": lambda n: crandom.gumbel(0, 1, n),
        "kumaraswamy": lambda n: crandom.kumaraswamy(2, 3, n),
        "fisher": lambda n: crandom.fisher(4, 10, n),
        "pareto": lambda n: crandom.pareto(1, 3, n),
        "rayleigh": lambda n: crandom.rayleigh(1, n),
    }

    discrete_laws = {
        "duniform": lambda n: drandom.duniform(10, n),
        "bernoulli": lambda n: drandom.bernoulli(0.3, n),
        "rademacher": lambda n: drandom.rademacher(0.5, n),
        "binomial": lambda n: drandom.binomial(0.3, 20, n),
        "poisson": lambda n: drandom.poisson(3, n),
        "hypergeometric": lambda n: drandom.hypergeometric(50, 20, 10, n),
        "geometric": lambda n: drandom.geometric(0.3, n),
        "negative_binomial": lambda n: drandom.negative_binomial(0.3, 5, n),
        "yule_simon": lambda n: drandom.yule_simon(2, n),
    }

    run_group("continuous laws (crandom)", continuous_laws, sample_sizes, n_runs=n_runs)
    run_group("discrete laws (drandom)", discrete_laws, sample_sizes, n_runs=n_runs)


if __name__ == "__main__":
    main()
