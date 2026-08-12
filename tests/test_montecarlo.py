"""Tests for Monte Carlo utilities."""

import numpy as np
import pytest

from pystochastic.montecarlo import MonteCarloEstimator, MonteCarloProcess
from pystochastic.processes import Brownian


def test_monte_carlo_estimator_initialization():
    samples = np.arange(10, dtype=float)
    mc = MonteCarloEstimator(samples, n_simulations=5)

    assert mc.samples.shape == (10,)
    assert mc.n_simulations == 5


def test_monte_carlo_estimate():
    samples = np.arange(1, 11, dtype=float)
    mc = MonteCarloEstimator(samples, n_simulations=10)

    assert mc.estimate() == pytest.approx(5.5)
    assert mc.estimate(function=lambda x: x**2) == pytest.approx(38.5)


def test_monte_carlo_estimate_with_n():
    samples = np.arange(1, 11, dtype=float)
    mc = MonteCarloEstimator(samples, n_simulations=10)

    assert mc.estimate(n=4) == pytest.approx(2.5)


def test_monte_carlo_mean_estimator():
    samples = np.arange(1, 11, dtype=float)
    mc = MonteCarloEstimator(samples, n_simulations=10)

    mean, half_width = mc.mean_estimator(confidence=0.95)

    assert mean == pytest.approx(5.5)
    assert half_width > 0


def test_monte_carlo_confidence_interval():
    samples = np.arange(1, 11, dtype=float)
    mc = MonteCarloEstimator(samples, n_simulations=10)

    lower, upper = mc.confidence_interval()

    assert lower < 5.5 < upper
    assert upper - lower > 0


@pytest.mark.parametrize("n_simulations", [0, 1, 11])
def test_monte_carlo_invalid_simulation_count(n_simulations):
    with pytest.raises(ValueError):
        MonteCarloEstimator(np.arange(10), n_simulations=n_simulations)


def test_monte_carlo_process():
    process = Brownian(t_n=1, n_steps=20)
    mc = MonteCarloProcess(process, n_simulations=20)

    assert mc.ech.shape == (20, 21, 1)

    values = mc.values_at(t_0=0.5)
    assert values.shape == (20,)

    estimate = mc.estimate(t_0=0.5)
    assert np.asarray(estimate).shape == ()

    mean_path = mc.mean_path(plot_sim=False)
    assert mean_path.shape == (21, 1)


def test_monte_carlo_process_uses_closest_time():
    process = Brownian(t_n=1, n_steps=10)
    mc = MonteCarloProcess(process, n_simulations=5)

    values = mc.values_at(t_0=0.47)

    # 0.47 is closest to 0.5 on a grid with dt = 0.1.
    expected_index = 5
    assert np.allclose(values, mc.ech[:, expected_index, 0])
