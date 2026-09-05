"""
Benchmark for pystochastic.sde (EulerMaruyama, Milstein).

Compares the vectorized path (diagonal diffusion) against the sequential
path (full matrix diffusion) for EulerMaruyama, with and without
parallelization (multiprocess) on the sequential path, and benchmarks
Milstein (always vectorized, restricted to autonomous 1D SDEs).
"""

import time
import numpy as np

from pystochastic.sde.eulermaruyama import EulerMaruyama
from pystochastic.sde.milstein import Milstein
from pystochastic.sde.rungekutta import RungeKutta


# ---------------------------------------------------------------------
# Test functions -- matrix form (sequential path)
# ---------------------------------------------------------------------

def cheap_drift(x, t):
    return x


def cheap_diffusion_matrix(x, t):
    return 0.1 * np.eye(len(x))


def normal_drift(x, t):
    return np.sin(x) + 0.5 * x


def normal_diffusion_matrix(x, t):
    return np.diag(0.1 + 0.05 * np.abs(x))


# ---------------------------------------------------------------------
# Test functions -- diagonal form (vectorized path), same underlying SDE
# ---------------------------------------------------------------------

def cheap_diffusion_diag(x, t):
    return 0.1 * np.ones_like(x)


def normal_diffusion_diag(x, t):
    return 0.1 + 0.05 * np.abs(x)


# ---------------------------------------------------------------------
# Milstein test functions (autonomous, 1D only)
# ---------------------------------------------------------------------

def milstein_drift(x):
    return -x


def milstein_diffusion(x):
    return 0.3 * np.ones_like(x)


# ---------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------

def benchmark_euler(n_simulations, n_steps, mu, sigma, n_runs=3, dim=1, parallel=False, n_workers=None):
    x_0 = np.ones(dim)
    times = []
    for _ in range(n_runs):
        solver = EulerMaruyama(drift=mu, diffusion=sigma, initial=1.0, T=1,
                           steps=n_steps)
        start = time.perf_counter()
        solver.solve(plot=False, parallel=parallel, n_workers=n_workers,n_simulations=n_simulations)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)


def benchmark_milstein(n_simulations, n_steps, mu, sigma, n_runs=3):
    times = []
    for _ in range(n_runs):
        solver = Milstein(drift=mu, diffusion=sigma, initial=1.0, T=1,
                           steps=n_steps)
        start = time.perf_counter()
        solver.solve(plot=False, n_simulations=n_simulations)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)

def benchmark_rungekutta(n_simulations, n_steps, mu, sigma, n_runs=3):
    times = []
    for _ in range(n_runs):
        solver = RungeKutta(drift=mu, diffusion=sigma, initial=1.0, T=1,
                           steps=n_steps)
        start = time.perf_counter()
        solver.solve(plot=False, n_simulations=n_simulations)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)


# ---------------------------------------------------------------------
# Comparison runner (Euler-Maruyama : matrix vs diagonal vs matrix+parallel)
# ---------------------------------------------------------------------

def run_euler_comparison(label, mu, sigma_matrix, sigma_diag, simulations, time_steps, n_runs=3, n_workers=None):

    print("=" * 70)
    print(f"EulerMaruyama benchmark -- {label}")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)

        for n_simulations in simulations:

            mat_mean, mat_min, mat_max = benchmark_euler(
                n_simulations, n_steps, mu, sigma_matrix, n_runs=n_runs, parallel=False,
            )
            par_mean, par_min, par_max = benchmark_euler(
                n_simulations, n_steps, mu, sigma_matrix, n_runs=n_runs, parallel=True, n_workers=n_workers,
            )
            diag_mean, diag_min, diag_max = benchmark_euler(
                n_simulations, n_steps, mu, sigma_diag, n_runs=n_runs, parallel=False,
            )

            speedup_par = mat_mean / par_mean if par_mean > 0 else float("inf")
            speedup_vec = mat_mean / diag_mean if diag_mean > 0 else float("inf")

            print(
                f"Simulations : {n_simulations:,}\n"
                f"  Matrix (sequential)        : {mat_mean:.4f} s (min={mat_min:.4f}, max={mat_max:.4f})\n"
                f"  Matrix (parallel)          : {par_mean:.4f} s (min={par_min:.4f}, max={par_max:.4f})"
                f"   -> speedup x{speedup_par:.1f}\n"
                f"  Diagonal (vectorized)      : {diag_mean:.4f} s (min={diag_min:.4f}, max={diag_max:.4f})"
                f"   -> speedup x{speedup_vec:.1f}"
            )

        print("-" * 70)


def run_milstein_benchmark(simulations, time_steps, n_runs=3):
    print("=" * 70)
    print("Milstein benchmark (always vectorized)")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)
        for n_simulations in simulations:
            mean_t, min_t, max_t = benchmark_milstein(
                n_simulations, n_steps, milstein_drift, milstein_diffusion, n_runs=n_runs,
            )
            print(f"Simulations : {n_simulations:<8,} : {mean_t:.4f} s (min={min_t:.4f}, max={max_t:.4f})")
        print("-" * 70)

def run_rungekutta_benchmark(simulations, time_steps, n_runs=3):
    print("=" * 70)
    print("Runge-Kutta benchmark (always vectorized)")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)
        for n_simulations in simulations:
            mean_t, min_t, max_t = benchmark_rungekutta(
                n_simulations, n_steps, milstein_drift, milstein_diffusion, n_runs=n_runs,
            )
            print(f"Simulations : {n_simulations:<8,} : {mean_t:.4f} s (min={min_t:.4f}, max={max_t:.4f})")
        print("-" * 70)
# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main():

    simulations = [100, 1_000, 5_000, 10_000]
    time_steps = [100, 500]
    n_runs = 3

    run_euler_comparison(
        "cheap drift/diffusion",
        mu=cheap_drift,
        sigma_matrix=cheap_diffusion_matrix,
        sigma_diag=cheap_diffusion_diag,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_euler_comparison(
        "normal drift/diffusion",
        mu=normal_drift,
        sigma_matrix=normal_diffusion_matrix,
        sigma_diag=normal_diffusion_diag,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_milstein_benchmark(simulations, time_steps, n_runs=n_runs)
    run_rungekutta_benchmark(simulations, time_steps, n_runs=n_runs)


if __name__ == "__main__":
    main()
