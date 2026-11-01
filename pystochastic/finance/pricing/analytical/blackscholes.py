import numpy as np
from pystochastic.dist import Normal
from pystochastic.processes.diffusion import GeometricBrownianMotion
from pystochastic.finance.pricing.validation import _validate_parameters

"""
============================================================
Module BLACK-SCHOLES PRICER
============================================================

Description
-----------
This module provides a way to price an european option using the Black-Scholes formula. The pricing is done under the risk-neutral measure.
This modules provides  a class for a pricer with the Black-Scholes model, with the following methods:
    - .underlying() : Returns the underlying asset of the option, considered under the risk-neutral measure. For the Black-Scholes model, 
                      the underlying asset is a Geometric Brownian Motion. The return object is the instance of the GeometricBrownianMotion class
                      from pystochastic.processes.diffusion.geometricbrownianmotion.
    - .price() : Price an option under the risk-neutral measure, with a specified spot price.

Examples
--------
>> model = BlackScholesPricer(rf_ir = 0.02, volatility = 0.25, dividend = 0.01)
"""

class BlackScholesPricer:

    """
    Black-Scholes Pricer class

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
    dividend : float
        Continuous dividend rate of the option.

    Attributes
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    volatility : float
        Volatility of the underlying asset.
    dividend : float
        Continuous dividend rate of the option.

    Examples
    --------
    >> call = EuropeanOption(100, 1, 'call')
    >> model = BlackScholesPricer(rf_ir = 0.02, volatility = 0.25, dividend = 0.01)
    >> call.price(model, 90)

    Notes
    -----
    The computation of the Geometrical Brownian Motion is done under the risk-neutral measure.
    """

    def __init__(self, rf_ir, volatility, dividend=0):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(volatility, (int, np.integer, float, np.floating)) or volatility <= 0 or not np.isfinite(volatility) or np.isnan(volatility):
            raise ValueError("The volatility parameter must be a strictly positive float.")

        if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
            raise ValueError("The continuous dividend rate must be a positive float.")

        self.rf_ir = rf_ir
        self.volatility = volatility
        self.dividend = dividend

        self.name = "Black-Scholes Pricer"

    def underlying(self, option, initial, steps = 365):

        # Under the risk-neutral measure, the underlying asset is a Geometric Brownian Motion of drift x-> rf_ir * x and diffusion x -> volatility * x.

        return GeometricBrownianMotion(mu = self.rf_ir - self.dividend, volatility = self.volatility, initial = initial, T=option.maturity, steps = steps)

    def price(self, option, initial):

        # BLACK - SCHOLES FORMULA FOR AN EUROPEAN CALL
        #
        # The call price is given by the following formula:
        # C = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
        #
        # The put price is given by the following formula:
        # P = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)
        #
        # Where S is the spot price, r is the risk-free interest rate, q is the continuous dividend rate, T is the maturity date, N(x) is the standard normal
        # cumulative distribution function, and d1 = (ln(S/K) + (r - q + 0.5 * v^2)T) / (v * sqrt(T)) and d2 = d1 - v * sqrt(T).


        option_type = option.option_type
        strike = option.strike
        maturity = option.maturity

        _validate_parameters(initial, option_type)

        d1 = (np.log(initial / strike) + (self.rf_ir - self.dividend + 0.5 * self.volatility ** 2)*maturity) / (self.volatility * np.sqrt(maturity))
        d2 = d1 - self.volatility * np.sqrt(maturity)

        phi = Normal(0,1).cdf

        if option_type == "call":
            return initial * np.exp(-self.dividend * maturity) * phi(d1) - strike * np.exp(-self.rf_ir * maturity) * phi(d2)
        return strike * np.exp(-self.rf_ir * maturity) * phi(-d2) - initial * np.exp(-self.dividend * maturity) * phi(-d1)
