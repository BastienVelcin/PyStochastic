import numpy as np
from pystochastic.finance.models import BinomialTree

def _validate_parameters(initial , dividend, option_type):

    if not isinstance(initial, (int, np.integer, float, np.floating)) or initial <= 0 or not np.isfinite(initial) or np.isnan(initial):
        raise ValueError("The initial value must be a strictly positive real number.")

    if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
        raise ValueError("The continuous dividend rate must be a positive or zero real number.")

    if not option_type in ["call", "put"]:
        raise ValueError("The option type must be either 'call' or 'put'.")


class BinomialTreePricer:

    def __init__(self, rf_ir, volatility, dividend, T, steps):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(volatility, (int, np.integer, float, np.floating)) or volatility <= 0 or not np.isfinite(volatility) or np.isnan(volatility):
            raise ValueError("The volatility parameter must be a strictly positive float.")

        if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
            raise ValueError("The maturity date must be a positive float.")

        if not isinstance(T, (int, np.integer, float, np.floating)) or T <= 0 or not np.isfinite(T) or np.isnan(T):
            raise ValueError("The time horizon parameter must be a strictly positive float.")

        if not isinstance(steps, (int, np.integer)) or steps <= 0 or not np.isfinite(steps) or np.isnan(steps):
            raise ValueError("The number of steps must be a strictly positive integer.")

        self.rf_ir = rf_ir
        self.volatility = volatility
        self.dividend = dividend
        self.T = T
        self.steps = steps
        self.dt = self.T / self.steps

        self.up = np.exp(self.volatility * np.sqrt(self.dt))
        self.down = np.exp(-self.volatility * np.sqrt(self.dt))


    def price(self, option, initial):

        tree = BinomialTree(initial, self.up, self.down, self.T, self.steps).simulate()
        payoff = option.payoff(tree[-1])

        p = (np.exp((self.rf_ir - self.dividend) * self.dt) - self.down) / (self.up - self.down)

        if not 0 < p < 1:
            raise ValueError("No-arbitrage condition is not satisfied.")

        for i in range(self.steps - 1, -1, -1):
            payoff = np.exp(-self.rf_ir * self.dt) * (p * payoff[:-1] + (1 - p) * payoff[1:])

        return payoff[0]