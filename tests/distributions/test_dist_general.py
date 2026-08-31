"""Tests for probability distributions."""

import numpy as np
import pytest

from pystochastic.dist import (
    Uniform,
    Exponential,
    Normal,
    Gamma,
    Beta,
    Weibull,
    Frechet,
    Cauchy,
    Gumbel,
    Kumaraswamy,
    Fisher,
    Pareto,
    Rayleigh,
)


DISTRIBUTIONS = [
    Uniform,
    Exponential,
    Normal,
    Gamma,
    Beta,
    Weibull,
    Frechet,
    Cauchy,
    Gumbel,
    Kumaraswamy,
    Fisher,
    Pareto,
    Rayleigh,
]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_distribution_can_be_instantiated(distribution):
    """Every public distribution should be constructible with defaults."""
    X = distribution()
    assert X is not None


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_distribution_support_is_a_pair(distribution):
    """Every distribution should expose a two-sided support."""
    X = distribution()
    support = X.support()

    assert isinstance(support, tuple)
    assert len(support) == 2
    assert support[0] <= support[1]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_distribution_sampling_shape(distribution):
    """Sampling should return the requested number of observations."""
    X = distribution()
    samples = np.asarray(X.sample(32))

    assert samples.shape == (32,)
    assert np.all(np.isfinite(samples))


def test_uniform_properties():
    X = Uniform(2, 6)

    assert X.support() == (2, 6)
    assert X.mean() == pytest.approx(4)
    assert X.variance() == pytest.approx(4 / 3)
    assert X.pdf(4) == pytest.approx(1 / 4)
    assert X.pdf(1) == pytest.approx(0)
    assert X.cdf(1) == pytest.approx(0)
    assert X.cdf(4) == pytest.approx(0.5)
    assert X.cdf(7) == pytest.approx(1)


def test_exponential_properties():
    X = Exponential(alpha=2)

    assert X.support() == (0, np.inf)
    assert X.mean() == pytest.approx(0.5)
    assert X.variance() == pytest.approx(0.25)
    assert X.pdf(0) == pytest.approx(2)
    assert X.cdf(0.5) == pytest.approx(1 - np.exp(-1))


def test_gamma_properties():
    X = Gamma(k=2, theta=3)

    assert X.mean() == pytest.approx(2 / 3)
    assert X.variance() == pytest.approx(2 / 9)
    assert X.pdf(1) == pytest.approx(9 * np.exp(-3))
    assert X.cdf(0) == pytest.approx(0)
    assert X.support() == (0, np.inf)


def test_beta_properties():
    X = Beta(a=2, b=3)

    assert X.support() == (0, 1)
    assert X.mean() == pytest.approx(2 / 5)
    assert X.variance() == pytest.approx(6 / 150)
    assert X.pdf(0.5) == pytest.approx(1.5)
    assert X.cdf(0) == pytest.approx(0)
    assert X.cdf(1) == pytest.approx(1)


def test_weibull_properties():
    X = Weibull(k=2, l=3)

    assert X.support() == (0, np.inf)
    assert X.mean() == pytest.approx(3 * np.sqrt(np.pi) / 2)
    assert X.cdf(0) == pytest.approx(0)
    assert X.pdf(0) == pytest.approx(0)


def test_distribution_invalid_parameters():
    with pytest.raises(ValueError):
        Uniform(1, 1)

    with pytest.raises(ValueError):
        Exponential(0)

    with pytest.raises(ValueError):
        Normal(var=0)

    with pytest.raises(ValueError):
        Gamma(k=0)

    with pytest.raises(ValueError):
        Gamma(theta=0)

    with pytest.raises(ValueError):
        Beta(a=0)

    with pytest.raises(ValueError):
        Beta(b=0)

    with pytest.raises(ValueError):
        Weibull(k=0)

    with pytest.raises(ValueError):
        Weibull(l=0)


def test_normal_properties():
    """Check the Normal distribution API and its theoretical moments."""
    X = Normal(mu=2, var=3)

    assert X.pdf(2) == pytest.approx(1 / ( np.sqrt(2 * 3 * np.pi)))
    assert X.cdf(2) == pytest.approx(0.5)
    assert X.support() == (-np.inf, np.inf)

    # These two assertions are deliberately useful regression tests:
    # the current implementation stores ``mean`` as an attribute and
    # later uses ``self.mu`` in sample()/mean(). They should pass once
    # that naming inconsistency is fixed.
    assert X.mean() == pytest.approx(2)
    assert X.variance() == pytest.approx(3)
    samples = X.sample(100)
    assert samples.shape == (100,)
