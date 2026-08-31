"""
Tests for elementary probability distributions.
"""

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


# ======================================================================
# Helpers
# ======================================================================

@pytest.mark.parametrize(
    "distribution",
    [
        Uniform(0, 1),
        Exponential(2),
        Normal(0, 1),
        Gamma(3, 2),
        Beta(2, 3),
        Weibull(2, 1),
        Frechet(3, 1),
        Cauchy(0, 1),
        Gumbel(0, 1),
        Kumaraswamy(2, 3),
        Fisher(5, 10),
        Pareto(3, 1),
        Rayleigh(1),
    ],
)
def test_support_is_valid(distribution):
    support = distribution.support()

    assert len(support) == 2
    assert support[0] <= support[1]


@pytest.mark.parametrize(
    "distribution",
    [
        Uniform(0, 1),
        Exponential(2),
        Normal(0, 1),
        Gamma(3, 2),
        Beta(2, 3),
        Weibull(2, 1),
        Frechet(3, 1),
        Cauchy(0, 1),
        Gumbel(0, 1),
        Kumaraswamy(2, 3),
        Fisher(5, 10),
        Pareto(3, 1),
        Rayleigh(1),
    ],
)
def test_cdf_is_between_zero_and_one(distribution):
    support = distribution.support()

    lo, hi = support

    if not callable(distribution.mean()):
        mean = 1
    else:
        mean = distribution.mean()

    if not callable(distribution.variance()):
        variance = 1
    else:
        variance = distribution.variance()
    if np.isneginf(lo):
        lo = mean - 10 * np.sqrt(
            variance
        )

    if np.isposinf(hi):
        hi = mean + 10 * np.sqrt(
            variance
        )

    x = np.linspace(lo, hi, 200)

    cdf = np.asarray(
        [distribution.cdf(xi) for xi in x]
    )

    assert np.all(cdf >= -1e-12)
    assert np.all(cdf <= 1 + 1e-12)


@pytest.mark.parametrize(
    "distribution",
    [
        Uniform(0, 1),
        Exponential(2),
        Normal(0, 1),
        Gamma(3, 2),
        Beta(2, 3),
        Weibull(2, 1),
        Frechet(3, 1),
        Cauchy(0, 1),
        Gumbel(0, 1),
        Kumaraswamy(2, 3),
        Fisher(5, 10),
        Pareto(3, 1),
        Rayleigh(1),
    ],
)
def test_sample_has_correct_size(distribution):
    n = 1000

    samples = distribution.sample(n)

    assert np.asarray(samples).size == n


# ======================================================================
# Specific mathematical tests
# ======================================================================

def test_uniform_moments():
    distribution = Uniform(2, 6)

    assert distribution.mean() == pytest.approx(4)
    assert distribution.variance() == pytest.approx(16 / 12)


def test_exponential_moments():
    distribution = Exponential(2)

    assert distribution.mean() == pytest.approx(0.5)
    assert distribution.variance() == pytest.approx(0.25)


def test_normal_moments():
    distribution = Normal(2, 3)

    assert distribution.mean() == pytest.approx(2)
    assert distribution.variance() == pytest.approx(3)


def test_gamma_moments():
    distribution = Gamma(3, 2)

    assert distribution.mean() == pytest.approx(1.5)
    assert distribution.variance() == pytest.approx(0.75)


def test_beta_moments():
    distribution = Beta(2, 3)

    assert distribution.mean() == pytest.approx(2 / 5)


# ======================================================================
# Parameter validation
# ======================================================================

def test_uniform_rejects_equal_bounds():
    with pytest.raises(ValueError):
        Uniform(1, 1)


def test_exponential_rejects_non_positive_parameter():
    with pytest.raises(ValueError):
        Exponential(0)

    with pytest.raises(ValueError):
        Exponential(-1)


def test_normal_rejects_non_positive_standard_deviation():
    with pytest.raises(ValueError):
        Normal(0, 0)

    with pytest.raises(ValueError):
        Normal(0, -1)


def test_gamma_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        Gamma(0, 1)

    with pytest.raises(ValueError):
        Gamma(1, 0)