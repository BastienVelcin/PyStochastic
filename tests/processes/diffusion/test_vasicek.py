import numpy as np
import pytest
from pystochastic.processes import Vasicek


def test_representations():
    r = Vasicek(speed=2, mean=.05, volatility=.1, initial=.03)
    assert r.dim == 1 and r._diagonal
    r = Vasicek(speed=[1,2], mean=[.05,.1], volatility=[.1,.2], initial=[.03,.04])
    assert r.dim == 2 and r._diagonal
    r = Vasicek(speed=np.eye(2), mean=[.05,.1], volatility=np.eye(2), initial=[.03,.04])
    assert r._diagonal


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        Vasicek(speed=np.eye(2), mean=[.05,.1], volatility=np.eye(2), initial=[.03])


def test_methods_and_shapes():
    r = Vasicek(speed=2, mean=.05, volatility=.1, initial=.03, steps=30)
    for method in ["euler-maruyama", "milstein", "runge-kutta", "exact"]:
        assert r.simulate(10, method).shape == (10,31,1)


def test_multidimensional_em():
    r = Vasicek(speed=[1,2], mean=[.05,.1], volatility=[.1,.2], initial=[.03,.04], steps=30)
    assert r.simulate(10).shape == (10,31,2)


def test_drift():
    r = Vasicek(speed=[1,2], mean=[.05,.1], volatility=[.1,.2], initial=[.03,.04])
    assert np.allclose(r.drift(np.array([.03,.08])), [.02,.04])


def test_exact_moments():
    r = Vasicek(speed=2, mean=.05, volatility=.1, initial=.03, T=1, steps=100)
    x = r.simulate(20000, "exact")[:, -1, 0]
    m = .05 + (.03-.05)*np.exp(-2)
    v = .1**2/4*(1-np.exp(-4))
    assert np.mean(x) == pytest.approx(m, rel=.03, abs=.002)
    assert np.var(x) == pytest.approx(v, rel=.08)


def test_density_and_covariance():
    r = Vasicek(speed=2, mean=.05, volatility=.1, initial=.03)
    assert r.density(1,.03) > 0
    assert r.density(0,.03) == 0
    assert r.covariance(0.5,0,0) == pytest.approx(r.variance(.5)[0])
