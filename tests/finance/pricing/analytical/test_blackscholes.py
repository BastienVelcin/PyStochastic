
import numpy as np
import pytest

from pystochastic.finance.pricing.analytical.blackscholes import BlackScholesPricer


# ============================================================
# FIXTURES
# ============================================================

class EuropeanOption:
    """Minimal European option implementing the pricer interface."""

    def __init__(self, strike, maturity, option_type):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type

    def payoff(self, underlying):
        if self.option_type == "call":
            return np.maximum(underlying - self.strike, 0)

        return np.maximum(self.strike - underlying, 0)


@pytest.fixture
def market():
    return {
        "initial": 100.0,
        "strike": 100.0,
        "rf_ir": 0.05,
        "volatility": 0.20,
        "dividend": 0.0,
        "T": 1.0,
    }


@pytest.fixture
def call_option(market):
    return EuropeanOption(
        strike=market["strike"],
        maturity=market["T"],
        option_type="call",
    )


@pytest.fixture
def put_option(market):
    return EuropeanOption(
        strike=market["strike"],
        maturity=market["T"],
        option_type="put",
    )


@pytest.fixture
def pricer(market):
    return BlackScholesPricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=market["dividend"],
    )


# ============================================================
# 1. BASIC PRICING
# ============================================================

def test_call_price_reference(pricer, call_option, market):
    price = pricer.price(call_option, market["initial"])

    assert price == pytest.approx(10.45058357, abs=1e-6)


def test_put_price_reference(pricer, put_option, market):
    price = pricer.price(put_option, market["initial"])

    assert price == pytest.approx(5.57352602, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_is_positive(pricer, market, option_type):
    option = EuropeanOption(
        market["strike"],
        market["T"],
        option_type,
    )

    price = pricer.price(option, market["initial"])

    assert price >= 0
    assert np.isfinite(price)


# ============================================================
# 2. PUT-CALL PARITY
# ============================================================

def test_put_call_parity(pricer, call_option, put_option, market):
    call = pricer.price(call_option, market["initial"])
    put = pricer.price(put_option, market["initial"])

    expected = (
        market["initial"] * np.exp(-market["dividend"] * market["T"])
        - market["strike"] * np.exp(-market["rf_ir"] * market["T"])
    )

    assert call - put == pytest.approx(expected, abs=1e-10)


def test_put_call_parity_with_dividend(market):
    dividend = 0.02

    call = EuropeanOption(100, 1, "call")
    put = EuropeanOption(100, 1, "put")

    pricer = BlackScholesPricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=dividend,
    )

    call_price = pricer.price(call, market["initial"])
    put_price = pricer.price(put, market["initial"])

    expected = (
        market["initial"] * np.exp(-dividend * market["T"])
        - market["strike"] * np.exp(-market["rf_ir"] * market["T"])
    )

    assert call_price - put_price == pytest.approx(expected, abs=1e-10)


# ============================================================
# 3. PRICE BOUNDS
# ============================================================

def test_call_price_upper_bound(pricer, call_option, market):
    price = pricer.price(call_option, market["initial"])

    upper_bound = market["initial"] * np.exp(
        -market["dividend"] * market["T"]
    )

    assert price <= upper_bound


def test_put_price_upper_bound(pricer, put_option, market):
    price = pricer.price(put_option, market["initial"])

    upper_bound = market["strike"] * np.exp(
        -market["rf_ir"] * market["T"]
    )

    assert price <= upper_bound


def test_call_price_lower_bound(pricer, call_option, market):
    price = pricer.price(call_option, market["initial"])

    lower_bound = max(
        market["initial"] * np.exp(-market["dividend"] * market["T"])
        - market["strike"] * np.exp(-market["rf_ir"] * market["T"]),
        0,
    )

    assert price >= lower_bound - 1e-10


def test_put_price_lower_bound(pricer, put_option, market):
    price = pricer.price(put_option, market["initial"])

    lower_bound = max(
        market["strike"] * np.exp(-market["rf_ir"] * market["T"])
        - market["initial"] * np.exp(-market["dividend"] * market["T"]),
        0,
    )

    assert price >= lower_bound - 1e-10


# ============================================================
# 4. SENSITIVITY TO PARAMETERS
# ============================================================

def test_call_price_increases_with_volatility(market, call_option):
    low_vol = BlackScholesPricer(
        market["rf_ir"], 0.10, market["dividend"]
    )

    high_vol = BlackScholesPricer(
        market["rf_ir"], 0.40, market["dividend"]
    )

    price_low = low_vol.price(call_option, market["initial"])
    price_high = high_vol.price(call_option, market["initial"])

    assert price_high > price_low


def test_put_price_increases_with_volatility(market, put_option):
    low_vol = BlackScholesPricer(
        market["rf_ir"], 0.10, market["dividend"]
    )

    high_vol = BlackScholesPricer(
        market["rf_ir"], 0.40, market["dividend"]
    )

    price_low = low_vol.price(put_option, market["initial"])
    price_high = high_vol.price(put_option, market["initial"])

    assert price_high > price_low


def test_call_price_increases_with_initial_value(market, pricer):
    option = EuropeanOption(100, 1, "call")

    low_price = pricer.price(option, 90)
    high_price = pricer.price(option, 110)

    assert high_price > low_price


def test_put_price_decreases_with_initial_value(market, pricer):
    option = EuropeanOption(100, 1, "put")

    low_price = pricer.price(option, 90)
    high_price = pricer.price(option, 110)

    assert low_price > high_price


def test_call_price_decreases_with_strike(pricer, market):
    low_strike = EuropeanOption(90, 1, "call")
    high_strike = EuropeanOption(110, 1, "call")

    price_low = pricer.price(low_strike, market["initial"])
    price_high = pricer.price(high_strike, market["initial"])

    assert price_low > price_high


def test_put_price_increases_with_strike(pricer, market):
    low_strike = EuropeanOption(90, 1, "put")
    high_strike = EuropeanOption(110, 1, "put")

    price_low = pricer.price(low_strike, market["initial"])
    price_high = pricer.price(high_strike, market["initial"])

    assert price_high > price_low


# ============================================================
# 5. DIVIDEND YIELD
# ============================================================

def test_call_price_decreases_with_dividend(market, call_option):
    no_dividend = BlackScholesPricer(
        market["rf_ir"],
        market["volatility"],
        dividend=0,
    )

    with_dividend = BlackScholesPricer(
        market["rf_ir"],
        market["volatility"],
        dividend=0.05,
    )

    price_without = no_dividend.price(call_option, market["initial"])
    price_with = with_dividend.price(call_option, market["initial"])

    assert price_with < price_without


def test_put_price_increases_with_dividend(market, put_option):
    no_dividend = BlackScholesPricer(
        market["rf_ir"],
        market["volatility"],
        dividend=0,
    )

    with_dividend = BlackScholesPricer(
        market["rf_ir"],
        market["volatility"],
        dividend=0.05,
    )

    price_without = no_dividend.price(put_option, market["initial"])
    price_with = with_dividend.price(put_option, market["initial"])

    assert price_with > price_without


# ============================================================
# 6. INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "rf_ir",
    [np.nan, np.inf, -np.inf, "0.05"],
)
def test_invalid_risk_free_rate(rf_ir):
    with pytest.raises(ValueError):
        BlackScholesPricer(
            rf_ir=rf_ir,
            volatility=0.2,
        )


@pytest.mark.parametrize(
    "volatility",
    [0, -0.1, np.nan, np.inf, -np.inf],
)
def test_invalid_volatility(volatility):
    with pytest.raises(ValueError):
        BlackScholesPricer(
            rf_ir=0.05,
            volatility=volatility,
        )


@pytest.mark.parametrize(
    "dividend",
    [-0.01, np.nan, np.inf, -np.inf],
)
def test_invalid_dividend(dividend):
    with pytest.raises(ValueError):
        BlackScholesPricer(
            rf_ir=0.05,
            volatility=0.2,
            dividend=dividend,
        )


@pytest.mark.parametrize(
    "initial",
    [0, -100, np.nan, np.inf, -np.inf],
)
def test_invalid_initial_value(pricer, call_option, initial):
    with pytest.raises(ValueError):
        pricer.price(call_option, initial)


def test_invalid_option_type(pricer, market):
    option = EuropeanOption(100, 1, "invalid")

    with pytest.raises(ValueError):
        pricer.price(option, market["initial"])


# ============================================================
# 7. NEGATIVE INTEREST RATES
# ============================================================

def test_negative_interest_rate_is_accepted():
    pricer = BlackScholesPricer(
        rf_ir=-0.01,
        volatility=0.2,
    )

    option = EuropeanOption(100, 1, "call")

    price = pricer.price(option, 100)

    assert price > 0
    assert np.isfinite(price)
