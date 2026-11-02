
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pystochastic.finance.pricing.analytical.heston import HestonPricer


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model():
    """Standard Heston model satisfying the Feller condition."""
    return HestonPricer(
        rf_ir=0.05,
        long_variance=0.04,
        reverting_rate=2.0,
        variance_volatility=0.30,
        correlation=-0.70,
        initial_variance=0.04,
        dividend=0.02
    )


@pytest.fixture
def option():
    return SimpleNamespace(
        strike=100.0,
        maturity=1.0,
        option_type="call"
    )


# ============================================================
# 1. CONSTRUCTOR AND PARAMETER VALIDATION
# ============================================================

@pytest.mark.parametrize("value", [-0.1, 0, np.nan, np.inf, "invalid"])
def test_invalid_long_variance(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, value, 2, 0.3, -0.7, 0.04)


@pytest.mark.parametrize("value", [-1, 0, np.nan, np.inf, "invalid"])
def test_invalid_reverting_rate(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, 0.04, value, 0.3, -0.7, 0.04)


@pytest.mark.parametrize("value", [-1, 0, np.nan, np.inf, "invalid"])
def test_invalid_variance_volatility(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, 0.04, 2, value, -0.7, 0.04)


@pytest.mark.parametrize("value", [-1.01, 1.01, np.nan, np.inf])
def test_invalid_correlation(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, 0.04, 2, 0.3, value, 0.04)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, "invalid"])
def test_invalid_initial_variance(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, 0.04, 2, 0.3, -0.7, value)


@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf, "invalid"])
def test_invalid_dividend(value):
    with pytest.raises(ValueError):
        HestonPricer(0.05, 0.04, 2, 0.3, -0.7, 0.04, value)


def test_invalid_risk_free_rate():
    with pytest.raises(ValueError):
        HestonPricer(0.05 + 1j, 0.04, 2, 0.3, -0.7, 0.04)


def test_feller_condition_warning():
    """A warning should be raised when Feller condition is violated."""
    with pytest.warns(UserWarning, match="Feller"):
        HestonPricer(
            0.05, 0.04, 0.5, 0.5, -0.7, 0.04
        )


def test_feller_condition_satisfied():
    """No warning should be raised when Feller condition holds."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HestonPricer(
            0.05, 0.04, 2, 0.3, -0.7, 0.04
        )


# ============================================================
# 2. PRICING OUTPUT
# ============================================================

def test_call_price_is_finite(model, option):
    price = model.price(option, 100)

    assert np.isfinite(price)


def test_put_price_is_finite(model):
    put = SimpleNamespace(
        strike=100.0,
        maturity=1.0,
        option_type="put"
    )

    price = model.price(put, 100)

    assert np.isfinite(price)


def test_call_price_is_positive(model, option):
    assert model.price(option, 100) > 0


def test_put_price_is_positive(model):
    put = SimpleNamespace(
        strike=100.0,
        maturity=1.0,
        option_type="put"
    )

    assert model.price(put, 100) > 0


# ============================================================
# 3. FINANCIAL PROPERTIES
# ============================================================

def test_put_call_parity(model):
    """Check the European put-call parity."""

    spot = 100
    strike = 100
    maturity = 1

    call = SimpleNamespace(
        strike=strike,
        maturity=maturity,
        option_type="call"
    )

    put = SimpleNamespace(
        strike=strike,
        maturity=maturity,
        option_type="put"
    )

    call_price = model.price(call, spot)
    put_price = model.price(put, spot)

    expected = (
        spot * np.exp(-model.dividend * maturity)
        - strike * np.exp(-model.rf_ir * maturity)
    )

    assert call_price - put_price == pytest.approx(
        expected, abs=1e-6
    )


def test_call_price_bounds(model, option):
    """Check European call no-arbitrage bounds."""

    spot = 100
    strike = option.strike
    maturity = option.maturity

    price = model.price(option, spot)

    lower_bound = max(
        spot * np.exp(-model.dividend * maturity)
        - strike * np.exp(-model.rf_ir * maturity),
        0
    )

    upper_bound = spot * np.exp(-model.dividend * maturity)

    assert price >= lower_bound - 1e-6
    assert price <= upper_bound + 1e-6


def test_put_price_bounds(model):
    """Check European put no-arbitrage bounds."""

    spot = 100
    strike = 100
    maturity = 1

    put = SimpleNamespace(
        strike=strike,
        maturity=maturity,
        option_type="put"
    )

    price = model.price(put, spot)

    lower_bound = max(
        strike * np.exp(-model.rf_ir * maturity)
        - spot * np.exp(-model.dividend * maturity),
        0
    )

    upper_bound = strike * np.exp(-model.rf_ir * maturity)

    assert price >= lower_bound - 1e-6
    assert price <= upper_bound + 1e-6


# ============================================================
# 4. PARAMETER SENSITIVITY
# ============================================================

def test_call_price_increases_with_spot(model, option):
    price_1 = model.price(option, 90)
    price_2 = model.price(option, 100)
    price_3 = model.price(option, 110)

    assert price_1 < price_2 < price_3


def test_call_price_increases_with_strike_decrease(model):
    """A call becomes more valuable when its strike decreases."""

    option_1 = SimpleNamespace(
        strike=90, maturity=1, option_type="call"
    )

    option_2 = SimpleNamespace(
        strike=100, maturity=1, option_type="call"
    )

    assert model.price(option_1, 100) > model.price(option_2, 100)


def test_put_price_increases_with_strike(model):
    """A put becomes more valuable when its strike increases."""

    put_1 = SimpleNamespace(
        strike=90, maturity=1, option_type="put"
    )

    put_2 = SimpleNamespace(
        strike=100, maturity=1, option_type="put"
    )

    assert model.price(put_1, 100) < model.price(put_2, 100)


# ============================================================
# 5. EDGE CASES
# ============================================================

def test_zero_initial_variance():
    """Heston pricing should accept a zero initial variance."""

    model = HestonPricer(
        0.05, 0.04, 2, 0.3, -0.7, 0
    )

    call = SimpleNamespace(
        strike=100, maturity=1, option_type="call"
    )

    price = model.price(call, 100)

    assert np.isfinite(price)
    assert price > 0


def test_zero_correlation():
    """The model should support independent Brownian motions."""

    model = HestonPricer(
        0.05, 0.04, 2, 0.3, 0, 0.04
    )

    call = SimpleNamespace(
        strike=100, maturity=1, option_type="call"
    )

    assert np.isfinite(model.price(call, 100))
