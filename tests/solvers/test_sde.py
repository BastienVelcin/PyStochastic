"""Tests for stochastic differential equation solvers."""

import numpy as np
import pytest

from pystochastic.sde import EulerMaruyama, Milstein


def test_euler_maruyama_deterministic_equation():
    """With zero diffusion, EM should reproduce the deterministic Euler scheme."""
    solver = EulerMaruyama(
        drift=lambda x, t: np.ones_like(x),
        diffusion=lambda x, t: np.zeros((1, 1)),
        initial=0,
        t_0=0,
        t_n=1,
        n_steps=10)

    paths = solver.solve(n_simulations=4)

    assert paths.shape == (4, 11, 1)
    assert np.allclose(paths[:, 0, 0], 0)
    assert np.allclose(paths[:, -1, 0], 1)


def test_euler_maruyama_zero_drift_zero_diffusion():
    solver = EulerMaruyama(
        drift=lambda x, t: np.zeros_like(x),
        diffusion=lambda x, t: np.zeros((1, 1)),
        initial=3,
        t_0=0,
        t_n=1,
        n_steps=20)

    paths = solver.solve(n_simulations=3)

    assert np.allclose(paths, 3)


def test_euler_maruyama_multidimensional_deterministic():
    solver = EulerMaruyama(
        drift=lambda x, t: np.array([1.0, -2.0]),
        diffusion=lambda x, t: np.zeros((2, 2)),
        initial=[0, 1],
        t_0=0,
        t_n=1,
        n_steps=10)

    paths = solver.solve(n_simulations=2)

    assert paths.shape == (2, 11, 2)
    assert np.allclose(paths[:, -1, :], [1, -1])


def test_euler_maruyama_invalid_time_interval():
    with pytest.raises(ValueError):
        EulerMaruyama(t_0=1, t_n=0)


def test_euler_maruyama_plot_rejects_dimension_above_three():
    solver = EulerMaruyama(
        drift=lambda x, t: np.ones(4),
        diffusion=lambda x, t: np.zeros((4, 4)),
        initial=[0, 0, 0, 0],
        t_n=1,
        n_steps=10)

    with pytest.raises(ValueError):
        solver.solve(n_simulations=1,plot=True)


def test_milstein_derivative():
    solver = Milstein(
        drift=lambda x: x,
        diffusion=lambda x: 2 * x,
        initial=1,
        t_n=1,
        n_steps=10)

    derivative = solver.approx_derivative_diffusion(np.array([3.0]))

    assert derivative == pytest.approx(2)


def test_milstein_deterministic_equation():
    solver = Milstein(
        drift=lambda x: np.ones_like(x),
        diffusion=lambda x: np.zeros_like(x),
        initial=0,
        t_0=0,
        t_n=1,
        n_steps=10)

    paths = solver.solve(n_simulations=2,plot=False)

    assert paths.shape == (2, 11, 1)
    assert np.allclose(paths[:, 0, 0], 0)


def test_milstein_rejects_multidimensional_initial_condition():
    with pytest.raises(NotImplementedError):
        Milstein(
            drift=lambda x: x,
            diffusion=lambda x: x,
            initial=[0, 0],
        )


def test_milstein_invalid_time_interval():
    with pytest.raises(ValueError):
        Milstein(
            drift=lambda x: x,
            diffusion=lambda x: x,
            t_0=1,
            t_n=0,
        )
