import numpy as np
from pystochastic.dist import Normal
from rich.console import Console
from rich.table import Table
from rich import box
from pystochastic.random.setseed import seed as _setseed
"""
============================================================
Module BLACK-SCHOLES
============================================================

Description
-----------
This module provides a way to price an european option using the Black-Scholes formula.

Examples
--------
>> model = BlackScholes(rf_ir = 0.02, volatility = 0.25, maturity = 1, strike = 10)
"""

def _validate_parameters(initial , dividend, option_type):

    if not isinstance(initial, (int, np.integer, float, np.floating)) or initial <= 0 or not np.isfinite(initial) or np.isnan(initial):
        raise ValueError("The initial value must be a strictly positive real number.")

    if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
        raise ValueError("The continuous dividend rate must be a positive or zero real number.")

    if not option_type in ["call", "put"]:
        raise ValueError("The option type must be either 'call' or 'put'.")

class BlackScholesPricer:

    """
    Black-Scholes class

    The Black-Scholes formula prices a european option (put or call) depending on an underlying asset (with a constant
    mean interest rate and volatility), a risk-free interest rate, a maturity date (in year) and a strike value.
    In the Black-Scholes model, we assume that the underlying asset follows a Geometric Brownian Motion (GBM) :
                            dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Parameters
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    volatility : float
        Volatility of the underlying asset.
    maturity : float
        Maturity date of the option. The maturity date is given in year.
    strike : float
        Strike price of the option.

    Attributes
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    volatility : float
        Volatility of the underlying asset.
    maturity : float
        Maturity date of the option. The maturity date is given in year.
    strike : float
        Strike of the option.
    Examples
    --------

    """

    def __init__(self, rf_ir, volatility, dividend=0):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(volatility, (int, np.integer, float, np.floating)) or volatility <= 0 or not np.isfinite(volatility) or np.isnan(volatility):
            raise ValueError("The volatility parameter must be a strictly positive float.")

        if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
            raise ValueError("The maturity date must be a positive float.")

        self.rf_ir = rf_ir
        self.volatility = volatility
        self.dividend = dividend

        self.name = "Black-Scholes"

    def price(self, option, initial):

        option_type = option.option_type
        strike = option.strike
        maturity = option.maturity

        _validate_parameters(initial, self.dividend, option_type)

        d1 = (np.log(initial / strike) + (self.rf_ir - self.dividend + 0.5 * self.volatility ** 2)*maturity) / (self.volatility * np.sqrt(maturity))
        d2 = d1 - self.volatility * np.sqrt(maturity)
        phi = Normal(0,1).cdf

        if option_type == "call":
            return initial * np.exp(-self.dividend * maturity) * phi(d1) - strike * np.exp(-self.rf_ir * maturity) * phi(d2)
        return strike * np.exp(-self.rf_ir * maturity) * phi(-d2) - initial * np.exp(-self.dividend * maturity) * phi(-d1)
