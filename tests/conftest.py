"""Shared pytest configuration for PyStochastic tests."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Make stochastic tests reproducible."""
    np.random.seed(12345)
