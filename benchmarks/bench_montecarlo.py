"""
Benchmark for pystochastic.montecarlo.MonteCarlo, and for the process-level
methods (values_at, mean_path) that feed it, as a function of the pool size
(number of samples) and, for process-level benchmarks, the number of
discretization steps.
"""

import time
import numpy as np

from pystochastic.random import continuous
from pystochastic.montecarlo import MonteCarlo
from pystochastic.processes.diffusion.vasicek import Vasicek
from pystochastic.random.setseed import seed


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
# MonteCarlo -- benchmark every method as a function of pool size
# ---------------------------------------------------------------------

def run_montecarlo_benchmark(sample_sizes, n_runs=5):
    print("=" * 70)
    print("MonteCarlo benchmark (pooled samples)")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n in sample_sizes:
        print(f"n_pool_values = {n:,}")
        seed(0)
        samples = continuous.normal(0, 1, n)
        mc = MonteCarlo(samples)

        methods = {
            "estimate": lambda: mc.estimate(),
            "half_width": lambda: mc.half_width(),
            "moment (order=2)": lambda: mc.moment(order=2),
            "variance": lambda: mc.variance(),
            "std": lambda: mc.std(),
            "standard_error": lambda: mc.standard_error(),
            "confidence_interval": lambda: mc.confidence_interval(),
            "confidence_curve": lambda: mc.confidence_curve(),
            "bias_estimator": lambda: mc.bias_estimator(reference=0),
            "mse_estimator": lambda: mc.mse_estimator(reference=0),
            "rmse_estimator": lambda: mc.rmse_estimator(reference=0),
            "quantile(0.9)": lambda: mc.quantile(0.9),
            "min": lambda: mc.min(),
            "max": lambda: mc.max(),
            "median": lambda: mc.median(),
            "skewness": lambda: mc.skewness(),
            "kurtosis": lambda: mc.kurtosis(),
            "histogram": lambda: mc.histogram(plot=False),
            "ecdf": lambda: mc.ecdf(plot=False),
        }

        for name, fn in methods.items():
            try:
                mean_t, min_t, max_t = time_call(fn, n_runs=n_runs)
                print(f"  {name:<24s} : {mean_t*1000:9.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  {name:<24s} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


# ---------------------------------------------------------------------
# Process -- values_at / mean_path, as a function of n_simulations and steps
# ---------------------------------------------------------------------

def run_process_pipeline_benchmark(simulations, time_steps, n_runs=3):
    print("=" * 70)
    print("Process pipeline benchmark (simulate -> values_at / mean_path -> MonteCarlo)")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)

        for n_simulations in simulations:
            print(f"Simulations : {n_simulations:,}")

            def build():
                return Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, T=5, steps=n_steps)

            def simulate_only():
                p = build()
                p.simulate(n_simulations=n_simulations)
                return p

            try:
                mean_t, min_t, max_t = time_call(simulate_only, n_runs=n_runs)
                print(f"  simulate()               : {mean_t:8.4f} s (min={min_t:.4f}, max={max_t:.4f})")
            except Exception as e:
                print(f"  simulate()               : ERROR ({type(e).__name__}: {e})")
                print("-" * 70)
                continue

            p = simulate_only()

            try:
                mean_t, min_t, max_t = time_call(lambda: p.values_at(t=5), n_runs=n_runs)
                print(f"  values_at(t=5)           : {mean_t*1000:8.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  values_at(t=5)           : ERROR ({type(e).__name__}: {e})")

            try:
                mean_t, min_t, max_t = time_call(lambda: p.mean_path(plot_sim=False), n_runs=n_runs)
                print(f"  mean_path()              : {mean_t*1000:8.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  mean_path()              : ERROR ({type(e).__name__}: {e})")

            try:
                def full_pipeline():
                    samples = p.values_at(t=5)
                    return MonteCarlo(samples).confidence_interval()
                mean_t, min_t, max_t = time_call(full_pipeline, n_runs=n_runs)
                print(f"  values_at + confidence_interval : {mean_t*1000:8.3f} ms (min={min_t*1000:.3f}, max={max_t*1000:.3f})")
            except Exception as e:
                print(f"  values_at + confidence_interval : ERROR ({type(e).__name__}: {e})")

        print("-" * 70)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main():
    run_montecarlo_benchmark(sample_sizes=[1_000, 10_000, 100_000, 1_000_000], n_runs=5)
    run_process_pipeline_benchmark(simulations=[100, 1_000, 5_000], time_steps=[100, 500], n_runs=3)


if __name__ == "__main__":
    main()