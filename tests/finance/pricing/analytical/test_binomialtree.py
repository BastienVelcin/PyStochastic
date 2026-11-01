
import numpy as np
import pytest

from pystochastic.finance.models.binomialtree import BinomialTree
from pystochastic.finance.pricing.analytical.binomialtree import BinomialTreePricer
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
        self.exercise_style = "european"

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
    return BinomialTreePricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=market["dividend"],
        T=market["T"],
        steps=1000,
    )


# ============================================================
# 1. BINOMIAL TREE SIMULATION
# ============================================================

def test_tree_shape():
    tree = BinomialTree(100, 1.1, 0.9, 1, 10).simulate()

    assert tree.shape == (11, 11)


def test_tree_initial_value():
    tree = BinomialTree(100, 1.1, 0.9, 1, 10).simulate()

    assert tree[0, 0] == pytest.approx(100)


def test_tree_first_step():
    tree = BinomialTree(100, 1.1, 0.9, 1, 10).simulate()

    assert tree[1, 0] == pytest.approx(110)
    assert tree[1, 1] == pytest.approx(90)


def test_tree_terminal_values():
    S0 = 100
    up = 1.1
    down = 0.9
    steps = 4

    tree = BinomialTree(S0, up, down, 1, steps).simulate()

    expected = np.array([
        S0 * up**4,
        S0 * up**3 * down,
        S0 * up**2 * down**2,
        S0 * up * down**3,
        S0 * down**4,
    ])

    np.testing.assert_allclose(
        tree[-1, :steps + 1],
        expected,
    )


def test_tree_recombining_property():
    S0 = 100
    up = 1.1
    down = 0.9

    tree = BinomialTree(S0, up, down, 1, 2).simulate()

    assert tree[2, 1] == pytest.approx(S0 * up * down)


def test_tree_unpopulated_nodes_are_zero():
    steps = 5
    tree = BinomialTree(100, 1.1, 0.9, 1, steps).simulate()

    for i in range(steps + 1):
        assert np.all(tree[i, i + 1:] == 0)


# ============================================================
# 2. CRR PARAMETERS
# ============================================================

def test_up_down_factors(pricer):
    expected_up = np.exp(
        pricer.volatility * np.sqrt(pricer.dt)
    )

    expected_down = np.exp(
        -pricer.volatility * np.sqrt(pricer.dt)
    )

    assert pricer.up == pytest.approx(expected_up)
    assert pricer.down == pytest.approx(expected_down)


def test_up_down_product(pricer):
    assert pricer.up * pricer.down == pytest.approx(1.0)


def test_risk_neutral_probability(pricer):
    p = (
        np.exp((pricer.rf_ir - pricer.dividend) * pricer.dt)
        - pricer.down
    ) / (pricer.up - pricer.down)

    assert 0 < p < 1


def test_risk_neutral_expectation(pricer):
    p = (
        np.exp((pricer.rf_ir - pricer.dividend) * pricer.dt)
        - pricer.down
    ) / (pricer.up - pricer.down)

    expected_growth = np.exp(
        (pricer.rf_ir - pricer.dividend) * pricer.dt
    )

    assert (
        p * pricer.up + (1 - p) * pricer.down
    ) == pytest.approx(expected_growth)


# ============================================================
# 3. EUROPEAN OPTION PRICING
# ============================================================

@pytest.mark.parametrize(
    "option_type",
    ["call", "put"],
)
def test_option_price_is_positive(
    pricer,
    market,
    option_type,
):
    option = EuropeanOption(
        market["strike"],
        market["T"],
        option_type,
    )

    price = pricer.price(option, market["initial"])

    assert price >= 0
    assert np.isfinite(price)


def test_call_price_reference(pricer, call_option, market):
    price = pricer.price(call_option, market["initial"])

    assert price == pytest.approx(10.44858410, abs=1e-6)


def test_put_call_parity(pricer, call_option, put_option, market):
    call = pricer.price(call_option, market["initial"])
    put = pricer.price(put_option, market["initial"])

    expected = (
        market["initial"] * np.exp(
            -market["dividend"] * market["T"]
        )
        - market["strike"] * np.exp(
            -market["rf_ir"] * market["T"]
        )
    )

    assert call - put == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "option_type",
    ["call", "put"],
)
def test_option_price_bounds(
    pricer,
    market,
    option_type,
):
    option = EuropeanOption(
        market["strike"],
        market["T"],
        option_type,
    )

    price = pricer.price(option, market["initial"])

    if option_type == "call":
        upper_bound = market["initial"] * np.exp(
            -market["dividend"] * market["T"]
        )
    else:
        upper_bound = market["strike"] * np.exp(
            -market["rf_ir"] * market["T"]
        )

    assert price <= upper_bound + 1e-10


# ============================================================
# 4. DIVIDEND YIELD
# ============================================================

def test_pricing_with_dividend(market):
    dividend = 0.02

    option = EuropeanOption(
        market["strike"],
        market["T"],
        "call",
    )

    pricer = BinomialTreePricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=dividend,
        T=market["T"],
        steps=1000,
    )

    price = pricer.price(option, market["initial"])

    bs = BlackScholesPricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=dividend,
    )

    reference = bs.price(option, market["initial"])

    assert price == pytest.approx(reference, abs=0.01)


# ============================================================
# 5. CONVERGENCE TO BLACK-SCHOLES
# ============================================================

@pytest.mark.parametrize(
    "option_type",
    ["call", "put"],
)
def test_binomial_convergence_to_black_scholes(
    market,
    option_type,
):
    option = EuropeanOption(
        market["strike"],
        market["T"],
        option_type,
    )

    pricer = BinomialTreePricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=market["dividend"],
        T=market["T"],
        steps=5000,
    )

    binomial_price = pricer.price(
        option,
        market["initial"],
    )

    bs = BlackScholesPricer(
        rf_ir=market["rf_ir"],
        volatility=market["volatility"],
        dividend=market["dividend"],
    )

    bs_price = bs.price(option, market["initial"])

    assert binomial_price == pytest.approx(
        bs_price,
        abs=5e-4,
    )


# ============================================================
# 6. NO-ARBITRAGE CONDITION
# ============================================================

def test_no_arbitrage_violation(call_option, market):
    pricer = BinomialTreePricer(
        rf_ir=1.0,
        volatility=0.01,
        dividend=0.0,
        T=1.0,
        steps=1,
    )

    with pytest.raises(ValueError, match="No-arbitrage"):
        pricer.price(call_option, market["initial"])


# ============================================================
# 7. INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "volatility",
    [0, -0.1, np.nan, np.inf],
)
def test_invalid_volatility(volatility):
    with pytest.raises(ValueError):
        BinomialTreePricer(
            rf_ir=0.05,
            volatility=volatility,
            dividend=0,
            T=1,
            steps=100,
        )


@pytest.mark.parametrize(
    "dividend",
    [-0.01, np.nan, np.inf],
)
def test_invalid_dividend(dividend):
    with pytest.raises(ValueError):
        BinomialTreePricer(
            rf_ir=0.05,
            volatility=0.2,
            dividend=dividend,
            T=1,
            steps=100,
        )


@pytest.mark.parametrize(
    "T",
    [0, -1, np.nan, np.inf],
)
def test_invalid_maturity(T):
    with pytest.raises(ValueError):
        BinomialTreePricer(
            rf_ir=0.05,
            volatility=0.2,
            dividend=0,
            T=T,
            steps=100,
        )


@pytest.mark.parametrize(
    "steps",
    [0, -1, 1.5],
)
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        BinomialTreePricer(
            rf_ir=0.05,
            volatility=0.2,
            dividend=0,
            T=1,
            steps=steps,
        )


def test_invalid_risk_free_rate():
    with pytest.raises(ValueError):
        BinomialTreePricer(
            rf_ir=np.nan,
            volatility=0.2,
            dividend=0,
            T=1,
            steps=100,
        )
