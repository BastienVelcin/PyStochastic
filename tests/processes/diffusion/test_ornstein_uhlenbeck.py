import numpy as np
import pytest
from pystochastic.processes import OrnsteinUhlenbeck


def test_representations():
    r = OrnsteinUhlenbeck(speed=2, volatility=.5, initial=0)
    assert r.dim == 1 and r._diagonal
    r = OrnsteinUhlenbeck(speed=[2,3], volatility=[.5,.7], initial=[0,1])
    assert r.dim == 2 and r._diagonal
    r = OrnsteinUhlenbeck(speed=np.eye(2), volatility=np.eye(2), initial=[0,1])
    assert r._diagonal


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        OrnsteinUhlenbeck(speed=np.eye(2), volatility=np.eye(2), initial=[0])


def test_methods_and_shapes():
    r = OrnsteinUhlenbeck(speed=2, volatility=.5, initial=0, steps=30)
    for method in ["euler-maruyama", "milstein", "runge-kutta", "exact"]:
        assert r.simulate(10, method).shape == (10,31,1)


def test_multidimensional_em():
    r = OrnsteinUhlenbeck(speed=[2,3], volatility=[.5,.7], initial=[0,1], steps=30)
    assert r.simulate(10).shape == (10,31,2)


def test_drift_and_diffusion():
    r = OrnsteinUhlenbeck(speed=[2,3], volatility=[.5,.7], initial=[0,1])
    x = np.array([.5,1.5])
    assert np.allclose(r.drift(x), [1,4.5])
    assert np.allclose(r.diffusion(x), [.5,.7])


def test_exact_moments():
    r = OrnsteinUhlenbeck(speed=2, volatility=.5, initial=.5, T=1, steps=100)
    x = r.simulate(20000, "exact")[:, -1, 0]
    assert np.mean(x) == pytest.approx(r.expectation(1), rel=.025, abs=.02)
    assert np.var(x) == pytest.approx(r.variance(1), rel=.08)


def test_density():
    r = OrnsteinUhlenbeck(speed=2, volatility=.5, initial=0)
    assert r.density(1,0) > 0
    assert r.density(0,0) == 0
