
import numpy as np
import pytest

from pystochastic.finance.models.blackscholes import BlackScholes


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model():
    """Standard Black-Scholes model."""
    return BlackScholes(
        rf_ir=0.05,
        volatility=0.20,
        maturity=1.0,
        strike=100.0
    )


@pytest.fixture
def market_parameters():
    return {
        "initial": 100.0,
        "dividend": 0.0
    }


# ============================================================
# 1. ANALYTICAL PRICING
# ============================================================

def test_call_exact_price(model, market_parameters):
    """Check the analytical price of a European call."""

    price = model.price(
        **market_parameters,
        option_type="call",
        method="exact"
    )

    assert price == pytest.approx(10.45058357, abs=1e-6)


def test_put_exact_price(model, market_parameters):
    """Check the analytical price of a European put."""

    price = model.price(
        **market_parameters,
        option_type="put",
        method="exact"
    )

    assert price == pytest.approx(5.57352602, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_exact_price_is_positive(model, market_parameters, option_type):
    """Option prices must be non-negative."""

    price = model.price(
        **market_parameters,
        option_type=option_type,
        method="exact"
    )

    assert price >= 0


# ============================================================
# 2. CALL-PUT PARITY
# ============================================================

def test_call_put_parity_exact(model, market_parameters):
    """Check the theoretical call-put parity."""

    results = model.call_put_parity(**market_parameters)

    theoretical_value = (
        market_parameters["initial"]
        * np.exp(-market_parameters["dividend"] * model.maturity)
        - model.strike * np.exp(-model.rf_ir * model.maturity)
    )

    assert results["theoretical_value"] == pytest.approx(
        theoretical_value
    )

    assert results["bs_cp"] == pytest.approx(
        theoretical_value,
        abs=1e-10
    )


def test_call_put_parity_montecarlo(model, market_parameters):
    """Check that the Monte-Carlo parity estimate is finite."""

    results = model.call_put_parity(
        **market_parameters,
        n_simulations=100_000
    )

    assert np.isfinite(results["mc_cp"])


# ============================================================
# 3. PRICE BOUNDS
# ============================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_option_price_bounds(model, market_parameters, option_type):
    """Check the theoretical lower and upper bounds."""

    S0 = market_parameters["initial"]
    q = market_parameters["dividend"]
    K = model.strike
    r = model.rf_ir
    T = model.maturity

    discount_S = S0 * np.exp(-q * T)
    discount_K = K * np.exp(-r * T)

    price = model.price(
        **market_parameters,
        option_type=option_type,
        method="exact"
    )

    if option_type == "call":
        lower_bound = max(discount_S - discount_K, 0)
        upper_bound = discount_S
    else:
        lower_bound = max(discount_K - discount_S, 0)
        upper_bound = discount_K

    assert price >= lower_bound - 1e-10
    assert price <= upper_bound + 1e-10


# ============================================================
# 4. MONTE-CARLO PRICING
# ============================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_montecarlo_output_shape(model, market_parameters, option_type):
    """Check the shape of the Monte-Carlo output."""

    n_runs = 10

    prices = model.price(
        **market_parameters,
        option_type=option_type,
        method="montecarlo",
        n_simulations=10_000,
        n_runs=n_runs
    )

    assert isinstance(prices, np.ndarray)
    assert prices.shape == (n_runs,)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_montecarlo_convergence(model, market_parameters, option_type):
    """Check that Monte-Carlo estimates are close to exact prices."""

    exact_price = model.price(
        **market_parameters,
        option_type=option_type,
        method="exact"
    )

    mc_prices = model.price(
        **market_parameters,
        option_type=option_type,
        method="montecarlo",
        n_simulations=100_000,
        n_runs=1
    )

    # A deliberately conservative tolerance for a stochastic test.
    assert mc_prices[0] == pytest.approx(exact_price, abs=0.50)


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

def test_summary_keys(model, market_parameters):
    """Check that summary returns all expected statistics."""

    results = model.summary(
        **market_parameters,
        n_simulations=2_000,
        n_runs=20
    )

    expected_keys = {
        "exact_price",
        "mean_mc_price",
        "std_mc_price",
        "se_mean_mc_price",
        "bias_mc_price",
        "rmse_mc_price"
    }

    assert set(results.keys()) == expected_keys


def test_summary_statistics_are_finite(model, market_parameters):
    """Check that summary statistics are finite."""

    results = model.summary(
        **market_parameters,
        n_simulations=2_000,
        n_runs=20
    )

    for value in results.values():
        assert np.isfinite(value)


def test_summary_standard_error(model, market_parameters):
    """Check the standard error formula."""

    results = model.summary(
        **market_parameters,
        n_simulations=2_000,
        n_runs=50
    )

    expected_se = results["std_mc_price"] / np.sqrt(50)

    assert results["se_mean_mc_price"] == pytest.approx(expected_se)


def test_summary_bias(model, market_parameters):
    """Check the Monte-Carlo bias calculation."""

    results = model.summary(
        **market_parameters,
        n_simulations=2_000,
        n_runs=20
    )

    expected_bias = (
        results["mean_mc_price"] - results["exact_price"]
    )

    assert results["bias_mc_price"] == pytest.approx(expected_bias)


# ============================================================
# 6. INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize("volatility", [0, -0.1])
def test_invalid_volatility(volatility):
    with pytest.raises(ValueError):
        BlackScholes(0.05, volatility, 1.0, 100.0)


@pytest.mark.parametrize("maturity", [0, -1.0])
def test_invalid_maturity(maturity):
    with pytest.raises(ValueError):
        BlackScholes(0.05, 0.2, maturity, 100.0)


@pytest.mark.parametrize("strike", [0, -100.0])
def test_invalid_strike(strike):
    with pytest.raises(ValueError):
        BlackScholes(0.05, 0.2, 1.0, strike)


@pytest.mark.parametrize("initial", [0, -10.0])
def test_invalid_initial(model, initial):
    with pytest.raises(ValueError):
        model.price(initial=initial)


@pytest.mark.parametrize("dividend", [-0.01, -1.0])
def test_invalid_dividend(model, dividend):
    with pytest.raises(ValueError):
        model.price(dividend=dividend)


def test_invalid_option_type(model):
    with pytest.raises(ValueError):
        model.price(option_type="digital")


def test_invalid_method(model):
    with pytest.raises(ValueError):
        model.price(method="binomial")


@pytest.mark.parametrize("n_simulations", [0, -10, 1.5])
def test_invalid_number_of_simulations(model, n_simulations):
    with pytest.raises(ValueError):
        model.price(
            method="montecarlo",
            n_simulations=n_simulations
        )


@pytest.mark.parametrize("n_runs", [0, -10, 1.5])
def test_invalid_number_of_runs(model, n_runs):
    with pytest.raises(ValueError):
        model.price(
            method="montecarlo",
            n_runs=n_runs
        )
