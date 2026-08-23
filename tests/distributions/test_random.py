"""Tests for the random-number generation layer."""

import numpy as np
import pytest

from pystochastic.random import continuous


@pytest.mark.parametrize(
    "generator,args",
    [
        (continuous.uniform, (0, 1)),
        (continuous.exponential, (2,)),
        (continuous.normal, (0, 1)),
        (continuous.gamma, (2, 3)),
        (continuous.beta, (2, 3)),
    ],
)
def test_continuous_generator_shape(generator, args):
    samples = np.asarray(generator(*args, n=64))

    assert samples.shape == (64,)
    assert np.all(np.isfinite(samples))


def test_uniform_generator_bounds():
    samples = continuous.uniform(-2, 5, n=1000)
    assert np.all(samples >= -2)
    assert np.all(samples <= 5)


def test_exponential_generator_is_positive():
    samples = continuous.exponential(alpha=2, n=1000)
    assert np.all(samples >= 0)


def test_normal_generator_moments():
    samples = continuous.normal(mean=2, sd=3, n=100_000)

    assert np.mean(samples) == pytest.approx(2, abs=0.05)
    assert np.var(samples) == pytest.approx(9, abs=0.15)


def test_gamma_generator_moments():
    samples = continuous.gamma(p=2, theta=3, n=100_000)

    assert np.mean(samples) == pytest.approx(2 / 3, abs=0.02)
    assert np.var(samples) == pytest.approx(2 / 9, abs=0.02)


def test_beta_generator_bounds():
    samples = continuous.beta(a=2, b=3, n=1000)

    assert np.all(samples >= 0)
    assert np.all(samples <= 1)


@pytest.mark.parametrize(
    "call",
    [
        lambda: continuous.uniform(0, 1, n=0),
        lambda: continuous.exponential(1, n=0),
        lambda: continuous.normal(0, 1, n=0),
        lambda: continuous.gamma(1, 1, n=0),
        lambda: continuous.beta(1, 1, n=0),
    ],
)
def test_generators_reject_invalid_sample_count(call):
    with pytest.raises(ValueError):
        call()


def test_generators_reject_invalid_parameters():
    with pytest.raises(ValueError):
        continuous.exponential(0)

    with pytest.raises(ValueError):
        continuous.normal(0, 0)

    with pytest.raises(ValueError):
        continuous.gamma(0, 1)

    with pytest.raises(ValueError):
        continuous.gamma(1, 0)

    with pytest.raises(ValueError):
        continuous.beta(0, 1)

    with pytest.raises(ValueError):
        continuous.beta(1, 0)

    with pytest.raises(ValueError):
        continuous.uniform(2, 1)
