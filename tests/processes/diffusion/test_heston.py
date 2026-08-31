import numpy as np
import pytest
from pystochastic.processes import Heston


def make_heston(**kw):
    p = dict(mu=.05, long_variance=.04, reverting_rate=2, variance_volatility=.2,
             correlation=-.5, initial_price=100, initial_variance=.04, T=1, steps=100)
    p.update(kw)
    return Heston(**p)


def test_construction():
    h = make_heston()
    assert h.dim == 1


@pytest.mark.parametrize("kw", [{"initial_price":0},{"initial_price":-1},{"initial_variance":0},{"long_variance":0},{"reverting_rate":0},{"variance_volatility":0},{"correlation":1.1},{"correlation":-1.1}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError): make_heston(**kw)


def test_feller_condition():
    assert make_heston(reverting_rate=2,long_variance=.04,variance_volatility=.2).feller_condition
    assert not make_heston(reverting_rate=1,long_variance=.01,variance_volatility=.3).feller_condition


def test_drift_and_diffusion():
    h = make_heston(mu=.1)
    x = np.array([100., .01])
    assert np.allclose(h.drift(x), [10., .06])
    d = h.diffusion(x)
    assert d.shape == (2,2)
    assert d[0,1] == 0 and d[1,0] == 0


def test_simulation_and_component_paths():
    h = make_heston(steps=50)
    path = h.simulate(1000)
    assert path.shape == (1000,51,2)
    assert h.price.shape == (1000,51,1)
    assert h.var.shape == (1000,51,1)
    assert h.path.shape == (1000,51,1)
    assert np.all(np.isfinite(path))
    assert np.all(h.price > 0)
    assert np.all(h.var >= 0)


def test_correlation_boundary_values():
    for rho in [-1,0,1]:
        assert make_heston(correlation=rho).correlation == rho
