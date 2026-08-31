import numpy as np
import pytest
from pystochastic.processes import CompoundPoisson
from pystochastic.dist import Normal


def test_construction_and_distribution_validation():
    p=CompoundPoisson(intensity=2,distribution=Normal(1,.25),steps=20)
    assert p.dim==1
    with pytest.raises(ValueError): CompoundPoisson(intensity=0)
    with pytest.raises(ValueError): CompoundPoisson(intensity=-1)
    with pytest.raises(ValueError): CompoundPoisson(distribution="normal")


def test_shape_and_initial_value():
    p=CompoundPoisson(intensity=2,distribution=Normal(1,.25),steps=30)
    path=p.simulate(100)
    assert path.shape==(100,31,1)
    assert np.allclose(path[:,0,0],0)


def test_expectation_and_variance():
    p=CompoundPoisson(intensity=3,distribution=Normal(2,.25))
    assert p.expectation(1.5)==pytest.approx(9)
    assert p.variance(1.5)==pytest.approx(3*1.5*(.25+4))


def test_zero_mean_jumps_can_change_sign():
    p=CompoundPoisson(intensity=5,distribution=Normal(0,1),steps=100)
    x=p.simulate(500)[:,-1,0]
    assert np.any(x>0) and np.any(x<0)


def test_positive_jumps_are_non_decreasing():
    from pystochastic.dist import Poisson
    p=CompoundPoisson(intensity=2,distribution=Poisson(2),steps=100)
    path=p.simulate(500)
    assert np.all(np.diff(path[:,:,0],axis=1)>=0)


def test_terminal_mean():
    p=CompoundPoisson(intensity=2,distribution=Normal(1.5,.25),steps=100)
    x=p.simulate(20000)[:,-1,0]
    assert np.mean(x)==pytest.approx(3,rel=.04,abs=.05)
