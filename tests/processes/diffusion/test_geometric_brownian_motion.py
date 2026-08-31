import numpy as np
import pytest
from pystochastic.processes import GeometricBrownianMotion


def test_scalar_vector_and_matrix_representations():
    s = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1)
    assert s.dim == 1 and s._diagonal
    s = GeometricBrownianMotion(mu=[.1,.2], volatility=[.2,.3], initial=[1,2])
    assert s.dim == 2 and s._diagonal
    s = GeometricBrownianMotion(mu=[.1,.2], volatility=np.diag([.2,.3]), initial=[1,2])
    assert s._diagonal
    s = GeometricBrownianMotion(mu=[.1,.2], volatility=np.array([[.2,.1],[.05,.3]]), initial=[1,2])
    assert not s._diagonal


def test_invalid_initial_and_dimensions():
    with pytest.raises(ValueError): GeometricBrownianMotion(initial=0)
    with pytest.raises(ValueError): GeometricBrownianMotion(initial=-1)
    with pytest.raises(ValueError): GeometricBrownianMotion(mu=[.1,.2], volatility=np.eye(2), initial=[1])


@pytest.mark.parametrize("method", ["euler-maruyama", "milstein", "runge-kutta", "exact"])
def test_scalar_methods(method):
    s = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, steps=30)
    path = s.simulate(10, method)
    assert path.shape == (10, 31, 1)


def test_exact_positivity():
    s = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, steps=50)
    assert np.all(s.simulate(500, "exact") > 0)


def test_multidimensional_em():
    s = GeometricBrownianMotion(mu=[.1,.2], volatility=[.2,.3], initial=[1,2], steps=30)
    assert s.simulate(10).shape == (10,31,2)


def test_drift_and_diffusion():
    s = GeometricBrownianMotion(mu=[.1,.2], volatility=[.2,.3], initial=[1,2])
    x = np.array([2.,3.])
    assert np.allclose(s.drift(x), [.2,.6])
    assert np.allclose(s.diffusion(x), [.4,.9])


def test_exact_moments():
    s = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, T=1, steps=100)
    x = s.simulate(20000, "exact")[:, -1, 0]
    assert np.mean(x) == pytest.approx(np.exp(.1), rel=.025)
    assert np.var(x) == pytest.approx(np.exp(.2)*(np.exp(.04)-1), rel=.08)


def test_moments_and_density():
    s = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1)
    assert s.expectation(.7) == pytest.approx(np.exp(.07))
    assert s.variance(.7)[0] > 0
    assert s.covariance(.7,0,0) == pytest.approx(s.variance(.7)[0])
    assert s.density(1,1) > 0
    assert s.density(0,1) == 0
    assert s.density(1,-1) == 0


def test_invalid_method():
    with pytest.raises(ValueError):
        GeometricBrownianMotion().simulate(method="invalid")
