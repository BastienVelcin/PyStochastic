"""Tests for stochastic differential equation solvers."""

import numpy as np
import pytest

from pystochastic.sde import EulerMaruyama, Milstein


def test_euler_maruyama_deterministic_equation():
    """With zero diffusion, EM should reproduce the deterministic Euler scheme."""
    solver = EulerMaruyama(
        mu=lambda x, t: np.ones_like(x),
        sigma=lambda x, t: np.zeros((1, 1)),
        x_0=0,
        t_0=0,
        t_n=1,
        n_steps=10,
        n_simulations=4,
    )

    paths = solver.solve()

    assert paths.shape == (4, 11, 1)
    assert np.allclose(paths[:, 0, 0], 0)
    assert np.allclose(paths[:, -1, 0], 1)


def test_euler_maruyama_zero_drift_zero_diffusion():
    solver = EulerMaruyama(
        mu=lambda x, t: np.zeros_like(x),
        sigma=lambda x, t: np.zeros((1, 1)),
        x_0=3,
        t_0=0,
        t_n=1,
        n_steps=20,
        n_simulations=3,
    )

    paths = solver.solve()

    assert np.allclose(paths, 3)


def test_euler_maruyama_multidimensional_deterministic():
    solver = EulerMaruyama(
        mu=lambda x, t: np.array([1.0, -2.0]),
        sigma=lambda x, t: np.zeros((2, 2)),
        x_0=[0, 1],
        t_0=0,
        t_n=1,
        n_steps=10,
        n_simulations=2,
    )

    paths = solver.solve()

    assert paths.shape == (2, 11, 2)
    assert np.allclose(paths[:, -1, :], [1, -1])


def test_euler_maruyama_invalid_time_interval():
    with pytest.raises(ValueError):
        EulerMaruyama(t_0=1, t_n=0)


def test_euler_maruyama_plot_rejects_dimension_above_three():
    solver = EulerMaruyama(
        mu=lambda x, t: np.ones(4),
        sigma=lambda x, t: np.zeros((4, 4)),
        x_0=[0, 0, 0, 0],
        t_n=1,
        n_steps=5,
        n_simulations=1,
    )

    with pytest.raises(ValueError):
        solver.solve(plot=True)


def test_milstein_derivative():
    solver = Milstein(
        mu=lambda x: x,
        sigma=lambda x: 2 * x,
        x_0=1,
        t_n=1,
        n_steps=10,
        n_simulations=1,
    )

    derivative = solver.approx_derivative_diffusion(np.array([3.0]))

    assert derivative == pytest.approx(2)


def test_milstein_deterministic_equation():
    solver = Milstein(
        mu=lambda x: np.ones_like(x),
        sigma=lambda x: np.zeros_like(x),
        x_0=0,
        t_0=0,
        t_n=1,
        n_steps=10,
        n_simulations=2,
    )

    paths = solver.solve(plot=False)

    assert paths.shape == (2, 11, 1)
    assert np.allclose(paths[:, 0, 0], 0)


def test_milstein_rejects_multidimensional_initial_condition():
    with pytest.raises(NotImplementedError):
        Milstein(
            mu=lambda x: x,
            sigma=lambda x: x,
            x_0=[0, 0],
        )


def test_milstein_invalid_time_interval():
    with pytest.raises(ValueError):
        Milstein(
            mu=lambda x: x,
            sigma=lambda x: x,
            t_0=1,
            t_n=0,
        )
