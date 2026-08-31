import numpy as np
import pytest
from pystochastic.processes import PoissonProcess


def test_construction():
    p=PoissonProcess(intensity=2,T=1,steps=50)
    assert p.dim==1 and p.path is None


@pytest.mark.parametrize("lam", [0,-1,"x",None])
def test_invalid_intensity(lam):
    with pytest.raises(ValueError): PoissonProcess(intensity=lam)


def test_shape_and_integer_monotonic_paths():
    p=PoissonProcess(intensity=2,T=1,steps=50)
    path=p.simulate(500)
    assert path.shape==(500,51,1)
    assert np.all(path[:,0,0]==0)
    assert np.all(np.diff(path[:,:,0],axis=1)>=0)
    assert np.all(path[:,:,0]==np.floor(path[:,:,0]))


def test_moments():
    p=PoissonProcess(intensity=2,T=1)
    assert p.expectation(.7)==pytest.approx(1.4)
    assert p.variance(.7)==pytest.approx(1.4)
    x=p.simulate(30000)[:,-1,0]
    assert np.mean(x)==pytest.approx(2,rel=.02)
    assert np.var(x)==pytest.approx(2,rel=.04)


def test_density_is_pmf_on_integer_values():
    p=PoissonProcess(intensity=2)
    assert p.density(1,3)==pytest.approx(np.exp(-2)*2**3/6)
    assert p.density(0,0)==0
