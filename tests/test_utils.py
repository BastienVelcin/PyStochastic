"""Tests for utility functions."""

import numpy as np

from pystochastic.utils import is_pos_def, default_drift, default_diffusion


def test_is_pos_def():
    assert is_pos_def(np.eye(2))
    assert is_pos_def(np.array([[2.0, 0.5], [0.5, 1.0]]))
    assert not is_pos_def(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_default_drift():
    x = np.array([1.0, 2.0])
    result = default_drift(x, 0.5)
    print(result)
    assert np.allclose(result, x)


def test_default_diffusion():
    x = np.array([1.0, 2.0])
    result = default_diffusion(x, 0.5)

    assert np.shape(result) == (2, 2)
    assert np.allclose(result, np.diag(x))
