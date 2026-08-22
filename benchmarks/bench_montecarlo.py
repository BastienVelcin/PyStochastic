"""
Benchmark for pystochastic.montecarlo (MonteCarloEstimator, MonteCarloProcess).
"""

import time
import numpy as np

from pystochastic.random import crandom
from pystochastic.processes.diffusion.vasicek import Vasicek
from pystochastic.montecarlo.montecarlo import MonteCarlo, MonteCarloProcess


# ---------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------

def time_call(fn, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)


# ---------------------------------------------------------------------
# MonteCarloEstimator benchmark
# ---------------------------------------------------------------------

def run_estimator_benchmark(sample_sizes, n_runs=5):
    print("=" * 70)
    print("MonteCarloEstimator benchmark")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n in sample_sizes:
        print(f"n_simulations = {n:,}")
        samples = crandom.gamma(3, 2, n)
        mc = MonteCarloEstimator(samples)

        for name, fn in {
            "estimate": lambda: mc.estimate(),
            "mean_estimator": lambda: mc.mean_estimator(),
            "confidence_interval": lambda: mc.confidence_interval(),
            "confidence_curve": lambda: mc.confidence_curve(),
        }.items():
            try:
                mean_t, min_t, max_t = time_call(fn, n_runs=n_runs)
                print(f"  {name:<20s} : {mean_t*1000:8.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  {name:<20s} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


# ---------------------------------------------------------------------
# MonteCarloProcess benchmark
# ---------------------------------------------------------------------

def run_process_benchmark(sample_sizes, n_runs=3):
    print("=" * 70)
    print("MonteCarloProcess benchmark (Vasicek, method='exact')")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n in sample_sizes:
        print(f"n_simulations = {n:,}")

        def build():
            process = Vasicek(reversion_speed=2, mu=1.5, volatility=0.3, r_0=0, n_steps=200)
            return MonteCarloProcess(process, n_simulations=n)

        try:
            mean_t, min_t, max_t = time_call(build, n_runs=n_runs)
            print(f"  construction (incl. simulate) : {mean_t:8.4f} s (min={min_t:.4f}, max={max_t:.4f})")
        except Exception as e:
            print(f"  construction (incl. simulate) : ERROR ({type(e).__name__}: {e})")
            print("-" * 70)
            continue

        mc = build()

        for name, fn in {
            "estimate": lambda: mc.estimate(),
            "mean_path": lambda: mc.mean_path(plot_sim=False),
        }.items():
            try:
                mean_t, min_t, max_t = time_call(fn, n_runs=n_runs)
                print(f"  {name:<28s} : {mean_t:8.4f} s (min={min_t:.4f}, max={max_t:.4f})")
            except Exception as e:
                print(f"  {name:<28s} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main():
    run_estimator_benchmark(sample_sizes=[1_000, 10_000, 100_000, 1_000_000], n_runs=5)
    run_process_benchmark(sample_sizes=[100, 1_000, 5_000], n_runs=3)


if __name__ == "__main__":
    main()
