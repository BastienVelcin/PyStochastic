import numpy as np
import pytest
from pystochastic.processes import CIR


def make_cir(**kw):
    p = dict(speed=2, mean=.05, volatility=.1, initial=.03, T=1, steps=200)
    p.update(kw)
    return CIR(**p)


def test_valid_construction():
    c = make_cir()
    assert c.dim == 1


@pytest.mark.parametrize("kw", [{"speed":0},{"speed":-1},{"mean":0},{"mean":-1},{"volatility":0},{"volatility":-1},{"initial":-1}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError): make_cir(**kw)


def test_drift_and_diffusion():
    c = make_cir()
    assert c.drift(np.array([0.]))[0] > 0
    assert c.diffusion(np.array([0.]))[0] == 0
    assert c.diffusion(np.array([.04]))[0] == pytest.approx(.02)


def test_shape_and_nonnegativity_reasonable_case():
    c = make_cir()
    path = c.simulate(1000)
    assert path.shape == (1000,201,1)
    assert np.all(np.isfinite(path))
    assert np.all(path >= 0)


def test_expectation():
    c = make_cir()
    t = .7
    expected = .05 + (.03-.05)*np.exp(-2*t)
    assert c.expectation(t) == pytest.approx(expected)


def test_density():
    c = make_cir()
    assert c.density(1,.03) >= 0
    assert c.density(0,.03).item() == 0


def test_terminal_mean_against_theory():
    c = make_cir()
    x = c.simulate(20000)[:, -1, 0]
    expected = .05 + (.03-.05)*np.exp(-2)
    assert np.mean(x) == pytest.approx(expected, rel=.08, abs=.003)
