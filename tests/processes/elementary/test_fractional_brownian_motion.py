import numpy as np
import pytest
from pystochastic.processes import FractionalBrownianMotion


@pytest.mark.parametrize("H", [.1, .5, .9])
def test_valid_hurst(H):
    f = FractionalBrownianMotion(hurst=H, T=2, steps=30)
    assert f.hurst == H
    assert f.dim == 1


@pytest.mark.parametrize("H", [0, 1, -0.1, 1.1])
def test_invalid_hurst(H):
    with pytest.raises(ValueError):
        FractionalBrownianMotion(hurst=H)


def test_shapes_and_initial_condition():
    f = FractionalBrownianMotion(hurst=.7, T=1, steps=30)
    path = f.simulate(10)
    assert path.shape == (10, 31, 1)
    assert np.allclose(path[:, 0], 0)


def test_variance_formula():
    f = FractionalBrownianMotion(hurst=.7, T=2, steps=30)
    assert np.allclose(f.variance(1), [1])
    assert np.allclose(f.variance(2), [2**1.4])
    assert f.expectation(.5) == 0


def test_terminal_variance():
    f = FractionalBrownianMotion(hurst=.75, T=1, steps=40)
    x = f.simulate(10000)[:, -1, 0]
    assert np.mean(x) == pytest.approx(0, abs=.05)
    assert np.var(x) == pytest.approx(1, rel=.07)


def test_covariance_matrix_grid_is_symmetric():
    f = FractionalBrownianMotion(hurst=.7, T=1, steps=20)
    assert np.allclose(f.time_covar_matrix, f.time_covar_matrix.T)
    assert np.all(np.linalg.eigvalsh(f.time_covar_matrix) >= -1e-10)


def test_density():
    f = FractionalBrownianMotion(hurst=.5, T=2, steps=20)
    assert f.density(2, 0) == pytest.approx(1/np.sqrt(4*np.pi))
    assert f.density(0, 0) == 0
