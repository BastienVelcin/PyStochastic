"""
Tests for the distribution interfaces.
"""

import pytest

from pystochastic.dist import Distribution, DiscreteDistribution
from pystochastic.dist.dist import ContinuousDistribution


# ======================================================================
# Abstract interfaces
# ======================================================================

def test_distribution_is_abstract():
    with pytest.raises(TypeError):
        ContinuousDistribution()


def test_discrete_distribution_is_abstract():
    with pytest.raises(TypeError):
        DiscreteDistribution()


def test_continuous_interface_methods():
    methods = [
        "cdf",
        "sample",
        "mean",
        "variance",
        "entropy",
        "support",
    ]

    for method in methods:
        assert hasattr(Distribution, method)


def test_discrete_interface_methods():
    methods = [
        "cdf",
        "sample",
        "mean",
        "variance",
        "entropy",
        "support",
    ]

    for method in methods:
        assert hasattr(DiscreteDistribution, method)