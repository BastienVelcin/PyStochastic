import numpy as np
from pystochastic.processes.diffusion import GeometricBrownianMotion
from pystochastic.finance.models import BinomialTree
from pystochastic.finance.pricing.validation import _validate_parameters

"""
============================================================
Module BINOMIAL TREE PRICER / COX-ROSS-RUBINSTEIN PRICER
============================================================

Description
-----------
This module provides a way to price an option using the Cox-Ross-Rubinstein . The pricing is done under the risk-neutral measure.
This modules provides a class for a pricer with the Cox-Ross-Rubinstein model, with the following methods:
    - .underlying() : Returns the underlying asset of the option, considered under the risk-neutral measure. For the Cox-Ross-Rubinstein model, 
                      the underlying asset is a Geometric Brownian Motion. The return object is the instance of the GeometricBrownianMotion class
                      from pystochastic.processes.diffusion.geometricbrownianmotion.
    - .tree() : Returns the binomial tree of the option, considered under the risk-neutral measure. The return object is the instance of the BinomialTree class
                from pystochastic.finance.models.binomialtree.
    - .price() : Price an option under the risk-neutral measure, with a specified spot price.

Examples
--------
>> put = AmericanOption(100, 1, 'put')
>> model = BinomialTreePricer(rf_ir = 0.02, volatility = 0.25, dividend = 0.01, T = 1, steps = 365)
>> put.price(model, 90)
"""

class BinomialTreePricer:

    """
    Binomial Tree Pricer class

    The Binomial Tree pricing method (or Cox-Ross-Rubinstein models) prices an option (european or american) using
    a tree built from two constant returns rates : a growth rate (u) and a decline rate (d). Given S_i the price of an asset
    at the time i, it is possible to extend the tree from the S_i with 2 branches :
        - Branch 1 : S_i * u
        - Branch 2 : S_i * d

    The optimal price can be obtained by an induction on the payoff tree.

    Parameters
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    volatility : float
        Volatility of the underlying asset.
    dividend : float
        Continuous dividend rate of the option.
    steps : int
        Number of steps of the binomial tree.

    Attributes
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    volatility : float
        Volatility of the underlying asset.
    dividend : float
        Continuous dividend rate of the option.
    steps : int
        Number of steps of the binomial tree.

    Examples
    --------
    >> call = AmericanOption(100, 1, 'call')
    >> model = BinomialTreePricer(rf_ir = 0.02, volatility = 0.25, dividend = 0.01, steps = 365)
    >> call.price(model, 90)
    """

    def __init__(self, rf_ir, volatility, dividend, steps):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(volatility, (int, np.integer, float, np.floating)) or volatility <= 0 or not np.isfinite(volatility) or np.isnan(volatility):
            raise ValueError("The volatility parameter must be a strictly positive float.")

        if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
            raise ValueError("The continuous dividend rate must be a positive float.")

        if not isinstance(steps, (int, np.integer)) or steps <= 0 or not np.isfinite(steps) or np.isnan(steps):
            raise ValueError("The number of steps must be a strictly positive integer.")

        self.rf_ir = rf_ir
        self.volatility = volatility
        self.dividend = dividend
        self.steps = steps

        self.name = "Binomial Tree Pricer"

    def underlying(self, option, initial):
        return GeometricBrownianMotion(mu = self.rf_ir - self.dividend, volatility = self.volatility, initial = initial, T=option.maturity, steps = self.steps)

    def tree(self,option, initial):

        up = np.exp(self.volatility * np.sqrt(dt))
        down = np.exp(-self.volatility * np.sqrt(dt))

        return BinomialTree(initial, up, down, option.maturity, self.steps).simulate()
    
    def price(self, option, initial):

        up = np.exp(self.volatility * np.sqrt(dt))
        down = np.exp(-self.volatility * np.sqrt(dt))
        
        dt = option.maturity / self.steps
        _validate_parameters(initial, option.option_type)

        tree = self.tree(option, initial)

        # We compute the payoff at the last step
        payoff = option.payoff(tree[-1])

        # The risk-neutral probability of the option is given by the following formula:
        p = (np.exp((self.rf_ir - self.dividend) * dt) - down) / (up - down)

        # We check if the no-arbitrage condition is satisfied, i.e p is in [0,1].
        if not 0 < p < 1:
            raise ValueError("No-arbitrage condition is not satisfied.")


        if option.exercise_style == 'european':
            # The price for a european option is given by the final value of this induction formula:
            # P_{i,j} = exp(-rT) (P_{i-1,j+1} * p + P_{i+1,j} * (1-p))

            for i in range(self.steps - 1, -1, -1):
                payoff = np.exp(-self.rf_ir * dt) * (p * payoff[:-1] + (1 - p) * payoff[1:])

        elif option.exercise_style == 'american':
            # The price for an american option is given by the final value of this induction formula:
            # P_{i,j} = max[Payoff(S_{i,j}),(exp(-rT) (P_{i-1,j+1} * p + P_{i+1,j} * (1-p))]

            for i in range(self.steps - 1, -1, -1):
                payoff_at_i = option.payoff(tree[i, :i+1])

                induction_value = np.exp(-self.rf_ir * dt) * (p * payoff[:-1] + (1 - p) * payoff[1:])
                payoff = np.maximum(payoff_at_i, induction_value)

        return payoff[0]