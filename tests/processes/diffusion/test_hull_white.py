import numpy as np
import pytest
from pystochastic.processes import HullWhite


def make_hw(**kw):
    p = dict(reversion_speed=.5, calibration=.1, volatility=.02, initial=0, T=1, steps=100)
    p.update(kw)
    return HullWhite(**p)


def test_constant_and_callable_coefficients():
    h = make_hw()
    assert h.dim == 1
    h = make_hw(calibration=lambda t:.1+.01*t, volatility=lambda t:.02)
    assert callable(h.calibration) and callable(h.volatility)


@pytest.mark.parametrize("kw", [{"reversion_speed":0},{"reversion_speed":-1},{"volatility":0},{"volatility":-1}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError): make_hw(**kw)


def test_drift_and_diffusion():
    h = make_hw(reversion_speed=.5, calibration=.1, volatility=.02)
    assert h.drift(np.array([.2]), .3) == pytest.approx(-.05)
    assert h.diffusion(np.array([.2]), .3) == pytest.approx(.02)


def test_constant_coefficient_moments():
    h = make_hw(reversion_speed=.5, calibration=.1, volatility=.02, initial=0)
    t = 1
    assert h.expectation(t) == pytest.approx(.2*(1-np.exp(-.5)))
    assert h.variance(t) == pytest.approx(.02**2*(1-np.exp(-1)))


def test_time_dependent_expectation():
    import scipy.integrate
    cal = lambda s:.1+.02*s
    h = make_hw(calibration=cal, initial=0)
    t=.8
    expected=np.exp(-.5*t)*scipy.integrate.quad(lambda s:np.exp(.5*s)*cal(s),0,t)[0]
    assert h.expectation(t) == pytest.approx(expected)


def test_time_dependent_variance():
    import scipy.integrate
    sig=lambda s:.02+.005*s
    h=make_hw(volatility=sig)
    t=.8
    expected=np.exp(-t)*scipy.integrate.quad(lambda s:np.exp(s)*sig(s)**2,0,t)[0]
    assert h.variance(t) == pytest.approx(expected)


def test_exact_simulation():
    h=make_hw(steps=50)
    path=h.simulate(100, "exact")
    assert path.shape == (100,51,1)
    assert np.all(np.isfinite(path))


def test_exact_terminal_moments():
    h=make_hw(steps=100)
    x=h.simulate(30000,"exact")[:,-1,0]
    m=.2*(1-np.exp(-.5))
    v=.02**2*(1-np.exp(-1))
    assert np.mean(x)==pytest.approx(m,rel=.02,abs=.0015)
    assert np.var(x)==pytest.approx(v,rel=.06)


def test_density():
    h=make_hw()
    assert h.density(1,.1)>0
    assert h.density(0,0)==0
