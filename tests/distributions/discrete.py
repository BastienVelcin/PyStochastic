"""
Tests for discrete probability distributions.
"""

import numpy as np
import pytest

from pystochastic.dist import (
    DUniform,
    Bernoulli,
    Rademacher,
    Binomial,
    Poisson,
    Hypergeometric,
    Geometric,
    NegativeBinomial,
    YuleSimon,
)


# ======================================================================
# Helpers
# ======================================================================

@pytest.mark.parametrize(
    "distribution",
    [
        DUniform(10),
        Bernoulli(0.3),
        Rademacher(),
        Binomial(0.4,10),
        Poisson(3),
        Hypergeometric(20, 8, 5),
        Geometric(0.4),
        NegativeBinomial(0.4,5 ),
        YuleSimon(3),
    ],
)
def test_sample_size(distribution):
    n = 1000

    samples = np.asarray(
        distribution.sample(n)
    )

    assert samples.size == n


@pytest.mark.parametrize(
    "distribution",
    [
        DUniform(10),
        Bernoulli(0.3),
        Rademacher(),
        Binomial(0.4,10),
        Poisson(3),
        Hypergeometric(20, 8, 5),
        Geometric(0.4),
        NegativeBinomial(0.4,5 ),
        YuleSimon(3),
    ],
)
def test_cdf_is_between_zero_and_one(distribution):
    samples = np.asarray(
        distribution.sample(500)
    )

    lo = int(np.min(samples))
    hi = int(np.max(samples))

    x = np.arange(lo, hi + 1)

    cdf = np.asarray(
        [distribution.cdf(xi) for xi in x]
    )

    assert np.all(cdf >= -1e-12)
    assert np.all(cdf <= 1 + 1e-12)
    assert np.all(np.diff(cdf) >= -1e-12)


# ======================================================================
# PMF
# ======================================================================

def test_bernoulli_pmf():
    distribution = Bernoulli(0.3)

    assert distribution.pmf(0) == pytest.approx(0.7)
    assert distribution.pmf(1) == pytest.approx(0.3)


def test_rademacher_pmf():
    distribution = Rademacher()

    assert distribution.pmf(-1) == pytest.approx(0.5)
    assert distribution.pmf(1) == pytest.approx(0.5)


def test_binomial_pmf():
    distribution = Binomial(0.5,10 )

    assert distribution.pmf(0) == pytest.approx(1 / 1024)
    assert distribution.pmf(10) == pytest.approx(1 / 1024)


def test_poisson_pmf():
    distribution = Poisson(3)

    assert distribution.pmf(0) == pytest.approx(
        np.exp(-3)
    )


# ======================================================================
# Moments
# ======================================================================

def test_bernoulli_moments():
    distribution = Bernoulli(0.3)

    assert distribution.mean() == pytest.approx(0.3)
    assert distribution.variance() == pytest.approx(0.21)


def test_binomial_moments():
    distribution = Binomial(0.4,10 )

    assert distribution.mean() == pytest.approx(4)
    assert distribution.variance() == pytest.approx(2.4)


def test_poisson_moments():
    distribution = Poisson(3)

    assert distribution.mean() == pytest.approx(3)
    assert distribution.variance() == pytest.approx(3)


def test_rademacher_moments():
    distribution = Rademacher()

    assert distribution.mean() == pytest.approx(0)
    assert distribution.variance() == pytest.approx(1)

# ======================================================================
# PMF normalization
# ======================================================================

def test_bernoulli_pmf_sums_to_one():
    distribution = Bernoulli(0.3)

    probabilities = [
        distribution.pmf(0),
        distribution.pmf(1),
    ]

    assert sum(probabilities) == pytest.approx(1)


def test_binomial_pmf_sums_to_one():
    distribution = Binomial(0.4,10 )

    probabilities = [
        distribution.pmf(k)
        for k in range(11)
    ]

    assert sum(probabilities) == pytest.approx(1)


def test_poisson_pmf_is_approximately_normalized():
    distribution = Poisson(3)

    probabilities = [
        distribution.pmf(k)
        for k in range(30)
    ]

    assert sum(probabilities) == pytest.approx(
        1,
        abs=1e-8,
    )


# ======================================================================
# Parameter validation
# ======================================================================

def test_bernoulli_rejects_invalid_probability():
    with pytest.raises(ValueError):
        Bernoulli(-0.1)

    with pytest.raises(ValueError):
        Bernoulli(1.1)


def test_binomial_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        Binomial(0.5, -1 )

    with pytest.raises(ValueError):
        Binomial(-0.1, 10 )

    with pytest.raises(ValueError):
        Binomial(1.1, 10 )


def test_poisson_rejects_invalid_parameter():
    with pytest.raises(ValueError):
        Poisson(-1)


def test_geometric_rejects_invalid_probability():
    with pytest.raises(ValueError):
        Geometric(0)

    with pytest.raises(ValueError):
        Geometric(1.1)