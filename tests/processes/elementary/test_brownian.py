import numpy as np
import pytest
from pystochastic.processes import Brownian


def test_scalar_and_multidimensional_construction():
    w = Brownian(cov=2.0, T=2, steps=20)
    assert w.dim == 1
    assert w.cov.shape == (1, 1)
    q = np.array([[1.0, .4], [.4, 2.0]])
    w = Brownian(cov=q, T=2, steps=20)
    assert w.dim == 2
    assert np.allclose(w.cov, q)


@pytest.mark.parametrize("T", [0, -1, "1"])
def test_invalid_T(T):
    with pytest.raises(ValueError):
        Brownian(T=T)


def test_invalid_covariance():
    with pytest.raises(ValueError):
        Brownian(cov=np.array([[1., 2.], [2., 1.]]))


@pytest.mark.parametrize("n", [1, 3, 20])
def test_shapes_and_initial_condition(n):
    w = Brownian(T=1, steps=30)
    path = w.simulate(n)
    assert path.shape == (n, 31, 1)
    assert w.increments.shape == (n, 30, 1)
    assert np.allclose(path[:, 0], 0)
    assert np.allclose(np.diff(path, axis=1), w.increments)


def test_final_position_and_path_statistics_api():
    w = Brownian(steps=30)
    path = w.simulate(10)
    assert np.allclose(w.final_position(), path[:, -1])
    maximum, index, time = w.max()
    minimum, index_min, time_min = w.min()
    assert maximum.shape == minimum.shape == (10,)
    assert index.shape == index_min.shape == (10,)
    assert time.shape == time_min.shape == (10,)


def test_empirical_terminal_moments():
    w = Brownian(cov=2, T=1.5, steps=40)
    terminal = w.simulate(20000)[:, -1, 0]
    assert np.mean(terminal) == pytest.approx(0, abs=.06)
    assert np.var(terminal) == pytest.approx(3, rel=.05)


def test_theoretical_moments_and_density():
    w = Brownian(cov=2, T=1.5, steps=20)
    assert w.expectation(.7) == 0
    assert np.allclose(w.covariance_matrix(.7), [[1.4]])
    assert np.allclose(w.variance(.7), [1.4])
    assert w.covariance(.7, 0, 0) == pytest.approx(1.4)
    assert w.density(1, 0) == pytest.approx(1 / np.sqrt(4*np.pi))
    assert w.density(0, 0) == 0


def test_correlated_terminal_covariance():
    q = np.array([[1., .6], [.6, 2.]])
    w = Brownian(cov=q, T=1, steps=30)
    terminal = w.simulate(15000)[:, -1, :]
    assert np.allclose(np.cov(terminal, rowvar=False), q, rtol=.06, atol=.06)


def test_invalid_covariance_indices():
    w = Brownian()
    for i, j in [(-1, 0), (0, 1), (1, 0)]:
        with pytest.raises(ValueError):
            w.covariance(1, i, j)
