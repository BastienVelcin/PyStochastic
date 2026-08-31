import numpy as np
import pytest
from pystochastic.processes import CEV


def make_cev(**kw):
    p = dict(speed=.5, volatility=.2, elasticity=.7, initial=1, T=1, steps=200)
    p.update(kw)
    return CEV(**p)


def test_construction():
    c = make_cev()
    assert c.dim == 1
    assert c.is_autonomous


@pytest.mark.parametrize("kw", [{"volatility":-1},{"elasticity":-1},{"volatility":"x"},{"elasticity":"x"},{"speed":"x"}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError): make_cev(**kw)


def test_drift_and_diffusion():
    c = make_cev(speed=.5, volatility=.2, elasticity=.5)
    assert c.drift(np.array([2.])) == pytest.approx(1.)
    assert c.diffusion(np.array([4.])) == pytest.approx(.4)
    assert c.diffusion(np.array([-1.])) == pytest.approx(0.)


def test_expectation_formula_current_api():
    c = make_cev(speed=.5, initial=2)
    assert c.expectation(.7) == pytest.approx(2*np.exp(.35))


def test_simulation_shape_and_finiteness():
    c = make_cev(speed=.2, volatility=.1, elasticity=.5, steps=100)
    path = c.simulate(500)
    assert path.shape == (500,101,1)
    assert np.all(np.isfinite(path))


def test_invalid_method():
    with pytest.raises(ValueError): make_cev().simulate(method="invalid")
