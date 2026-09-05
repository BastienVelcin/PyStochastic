"""
Benchmark for pystochastic.processes (Vasicek, CIR, OrnsteinUhlenbeck,
GeometricBrownianMotion, Poisson, Brownian).

Measures simulation time as a function of the number of simulations and
time steps, for each available method ("exact", "euler-maruyama",
"milstein", "runge-kutta" when applicable).
"""

import time
import numpy as np

from pystochastic.processes.diffusion.vasicek import Vasicek
from pystochastic.processes.diffusion.cox_ingersoll_ross import CIR
from pystochastic.processes.diffusion.ornstein_uhlenbeck import OrnsteinUhlenbeck
from pystochastic.processes.diffusion.geometric_brownian_motion import GeometricBrownianMotion
from pystochastic.processes.diffusion.constant_elasticity_variance import CEV
from pystochastic.processes.diffusion.heston import Heston
from pystochastic.processes.diffusion.hull_white import HullWhite
from pystochastic.processes.jump.compound_poisson import CompoundPoisson
from pystochastic.processes.jump.poisson import PoissonProcess
from pystochastic.processes.elementary.brownian import Brownian
from pystochastic.dist import Normal

# ---------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------

def benchmark_simulate(build_process, n_simulations, method=None, n_runs=3, **simulate_kwargs):
    """
    build_process : callable() -> a fresh process instance (rebuilt each run,
        so that internal caches -- if any -- don't distort the timing).
    """
    times = []
    for _ in range(n_runs):
        process = build_process()
        start = time.perf_counter()
        if method is not None:
            process.simulate(n_simulations=n_simulations, method=method, plot=False,**simulate_kwargs)
        else:
            process.simulate(n_simulations=n_simulations, plot=False,**simulate_kwargs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.min(times), np.max(times)


def run_process_group(label, build_process, methods, simulations, time_steps, n_runs=3):
    print("=" * 70)
    print(f"processes benchmark -- {label}")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)

        for n_simulations in simulations:
            print(f"Simulations : {n_simulations:,}")
            for method in methods:
                try:
                    mean_t, min_t, max_t = benchmark_simulate(
                        lambda: build_process(n_steps),
                        n_simulations=n_simulations,
                        method=method,
                        n_runs=n_runs,
                    )
                    print(f"  {method:<16s} : {mean_t:8.4f} s (min={min_t:.4f}, max={max_t:.4f})")
                except Exception as e:
                    print(f"  {method:<16s} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


def run_no_method_group(label, build_process, simulations, time_steps, n_runs=3):
    """For processes without a `method` argument (Poisson, Brownian)."""
    print("=" * 70)
    print(f"processes benchmark -- {label}")
    print("=" * 70)
    print(f"Runs : {n_runs}")
    print("-" * 70)

    for n_steps in time_steps:
        print(f"Number of time steps : {n_steps}")
        print("-" * 70)

        for n_simulations in simulations:
            try:
                mean_t, min_t, max_t = benchmark_simulate(
                    lambda: build_process(n_steps),
                    n_simulations=n_simulations,
                    method=None,
                    n_runs=n_runs,
                )
                print(f"Simulations : {n_simulations:<8,} : {mean_t:8.4f} s (min={min_t:.4f}, max={max_t:.4f})")
            except Exception as e:
                print(f"Simulations : {n_simulations:<8,} : ERROR ({type(e).__name__}: {e})")
        print("-" * 70)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------

def main():

    simulations = [100, 1_000, 5_000, 10_000]
    time_steps = [100, 500]
    methods = ["exact", "euler-maruyama", "milstein", "runge-kutta"]
    n_runs = 3

    run_process_group(
        "Vasicek",
        build_process=lambda n_steps: Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_process_group(
        "CIR",
        build_process=lambda n_steps: CIR(mean=2, speed=0.05, volatility=0.1, initial=0.03, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_process_group(
        "OrnsteinUhlenbeck",
        build_process=lambda n_steps: OrnsteinUhlenbeck(speed=2, volatility=0.5, initial=0, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_process_group(
        "GeometricBrownianMotion",
        build_process=lambda n_steps: GeometricBrownianMotion(mu=0.05, volatility=0.2, initial=100, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_process_group(
        "CEV",
        build_process=lambda n_steps: CEV(speed=0.05, volatility=0.2, elasticity=0.4, initial=100, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )
    run_process_group(
        "Heston",
        build_process=lambda n_steps: Heston(mu=0.05, long_variance=0.2, reverting_rate=0.05, variance_volatility=0.3, correlation = 0.6, initial_price = 100, initial_variance= 0.9, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )
    run_process_group(
        "Hull-White",
        build_process=lambda n_steps: HullWhite(reversion_speed=0.05, calibration = lambda t : 0.05 + 0.01 * np.sin(2 * np.pi * t) , volatility= lambda t : 0.02 + 0.005 * t, initial=100, steps=n_steps),
        methods=methods,
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )
    run_no_method_group(
        "Poisson",
        build_process=lambda n_steps: PoissonProcess(intensity=3, steps=n_steps),
        simulations=[100, 1_000, 5_000],
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_no_method_group(
        "Compound Poisson",
        build_process=lambda n_steps: CompoundPoisson(intensity = 1, distribution = Normal(0,1),steps = n_steps),
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )

    run_no_method_group(
        "Brownian",
        build_process=lambda n_steps: Brownian(1, steps=n_steps),
        simulations=simulations,
        time_steps=time_steps,
        n_runs=n_runs,
    )


if __name__ == "__main__":
    main()
