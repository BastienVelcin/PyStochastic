import numpy as np
import pytest
from pystochastic.processes import BrownianBridge


def test_construction():
    b = BrownianBridge(T=2, steps=20)
    assert b.dim == 1
    assert b.path is None


def test_shapes_and_endpoints():
    b = BrownianBridge(T=1, steps=50)
    path = b.simulate(20)
    assert path.shape == (20, 51, 1)
    assert np.allclose(path[:, 0], 0)
    assert np.allclose(path[:, -1], 0)


def test_multidimensional_shape():
    b = BrownianBridge(dim=3, T=1, steps=20)
    assert b.simulate(5).shape == (5, 21, 3)


def test_moments():
    b = BrownianBridge(T=2, steps=100)
    t = .7
    v = t * (2-t) / 2
    assert b.expectation(t) == 0
    assert np.allclose(b.variance(t), [v])
    assert np.allclose(b.covariance_matrix(t), [[v]])
    assert b.covariance(t, 0, 0) == pytest.approx(v)


def test_empirical_variance():
    b = BrownianBridge(T=1, steps=100)
    values = b.simulate(20000)[:, 50, 0]
    assert np.var(values) == pytest.approx(.25, rel=.05)


def test_density_and_boundary():
    b = BrownianBridge(T=1)
    assert b.density(.5, 0) == pytest.approx(1/np.sqrt(2*np.pi*.25))
    assert b.density(0, 0) == 0
    assert b.density(1, 0) == 0


def test_invalid_indices():
    b = BrownianBridge(dim=2)
    with pytest.raises(ValueError):
        b.covariance(.5, 0, 2)
