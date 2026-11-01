import numpy as np
import warnings
from scipy import integrate as integrate
from pystochastic.processes.diffusion import Heston
from pystochastic.finance.pricing.validation import _validate_parameters

"""
============================================================
Module HESTON PRICER
============================================================

Description
-----------
This module provides a way to price an option using the Heston Model. The pricing is done under the risk-neutral measure.
This modules provides a class for a pricer with the Heston model, with the following methods:
    - .underlying() : Returns the underlying asset of the option, considered under the risk-neutral measure. For the Heston model, 
                      the underlying asset is a Heston Model Process. The return object is the instance of the Hestob class
                      from pystochastic.processes.diffusion.heston.
    - .price() : Price an option under the risk-neutral measure, with a specified spot price. We compute the price with the Fourier Transform method.

Examples
--------
>> put = EuropeanOption(100, 1, 'put')
>> model = BinomialTreePricer(rf_ir = 0.02, volatility = 0.25, dividend = 0.01, T = 1, steps = 365)
>> put.price(model, 90)
"""

# (S1) http://www0.cs.ucl.ac.uk/staff/g.germano/papers/EurJOperRes_2017.pdf

class HestonPricer:

    """
    Heston Pricer class

    The Heston model prices a european option (put or call) depending on an underlying asset with a stochastic volatility,
    a risk-free interest rate, a maturity date (in year) and a strike value.
    In the Heston model, we assume that the underlying asset follows a Geometric Brownian Motion (GBM) :
                            dS_t = (rf_ir - q)* S_t * dt + sqrt(nu_t) * S_t * dW_t^(1)
                            dnu_t = kappa * (theta - nu_t) * dt + sigma * sqrt(nu_t) * dW_t^(2)

    Parameters
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    long_variance : float
        Mean long variance of the variance process (theta).
    reverting_rate : float
        Mean reverting rate of the variance process (kappa).
    variance_volatility : float
        Volatility of the variance process (sigma).
    correlation : float
        Correlation parameter between W^(1) and W^(2).
    initial_variance : float
        Initial value of the variance process.
    dividend : float
        Continuous dividend rate of the option (q).

    Attributes
    ----------
    rf_ir : float
        Risk-free interest rate of the market.
    long_variance : float
        Mean long variance of the variance process (theta).
    reverting_rate : float
        Mean reverting rate of the variance process (kappa).
    variance_volatility : float
        Volatility of the variance process (sigma).
    correlation : float
        Correlation parameter between W^(1) and W^(2).
    initial_variance : float
        Initial value of the variance process.
    dividend : float
        Continuous dividend rate of the option (q).

    Examples
    --------
    >> call = EuropeanOption(100, 1, 'call')
    >> model = HestonPricer(0.05, 0.04, 2, 10, -0.70,0,0)
    >> call.price(model, 90)

    Notes
    -----
    The computation of the Geometrical Brownian Motion is done under the risk-neutral measure.
    """

    def __init__(self, rf_ir , long_variance, reverting_rate, variance_volatility, correlation, initial_variance, dividend=0):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(long_variance, (int, np.integer, float, np.floating)) or long_variance <= 0 or not np.isfinite(long_variance) or np.isnan(long_variance):
            raise ValueError("The long variance parameter must be a strictly positive float.")

        if not isinstance(reverting_rate, (int, np.integer, float, np.floating)) or reverting_rate <= 0 or not np.isfinite(reverting_rate) or np.isnan(reverting_rate):
            raise ValueError("The mean reverting rate parameter must be a strictly positive float.")

        if not isinstance(variance_volatility, (int, np.integer, float, np.floating)) or variance_volatility <= 0 or not np.isfinite(variance_volatility) or np.isnan(variance_volatility):
            raise ValueError("The volatility parameter of the variance must be a strictly positive float.")

        if not isinstance(correlation, (int, np.integer, float, np.floating)) or not -1 <= correlation <= 1 or not np.isfinite(correlation) or np.isnan(correlation):
            raise ValueError("The correlation parameter must be between -1 and 1.")

        if not isinstance(initial_variance, (int, np.integer, float, np.floating)) or initial_variance < 0 or not np.isfinite(initial_variance) or np.isnan(initial_variance):
            raise ValueError("The initial variance parameter must be a positive float.")

        if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
            raise ValueError("The continuous dividend rate must be a positive float.")

        self.rf_ir = rf_ir
        self.dividend = dividend
        self.long_variance = long_variance
        self.reverting_rate = reverting_rate
        self.variance_volatility = variance_volatility
        self.correlation = correlation
        self.initial_variance = initial_variance

        self.name = "Heston Pricer"

        if 2 * self.reverting_rate * self.long_variance < self.variance_volatility ** 2:
            warnings.warn(
                "The Feller condition is not satisfied. The variance process may reach zero."
            )

    def underlying(self, option, initial, steps = 365):
        return Heston(mu = self.rf_ir - self.dividend,
                      long_variance = self.long_variance,
                      reverting_rate = self.reverting_rate,
                      variance_volatility = self.variance_volatility,
                      correlation = self.correlation,
                      initial_variance = self.initial_variance,
                      initial = initial,
                      T = option.maturity,
                      steps = steps)


    def price(self, option, initial):

        # We price the option using the Fourier Transform method.

        _validate_parameters(initial, option.option_type)

        F = initial * np.exp((self.rf_ir - self.dividend)* option.maturity)

        kappa = self.reverting_rate
        sigma = self.variance_volatility
        theta = self.long_variance
        rho = self.correlation
        v0 = self.initial_variance
        T = option.maturity
        K = option.strike
        r = self.rf_ir
        q = self.dividend

        # We refer to (S1) for the definition of the characteristic function. Note that some operations are simplified
        # to optimize the computation.
        def characteristic(u):
            chi = kappa - rho * sigma * 1j * u
            d = np.sqrt(chi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))

            H = (d+chi) + (d-chi) * np.exp(-d*T)
            A = v0* (u**2 + 1j*u) * (1- np.exp(-d*T)) / H

            D = np.log(2*d / H) + (kappa - d)*T / 2

            return np.exp(1j * u * np.log(F / initial) - kappa * theta * rho * T * 1j * u / sigma - A + 2*kappa*theta/(sigma**2)*D)

        phi_minus_i = F/initial

        def first_integrand(u):
            return np.real(
                np.exp(-1j * u * np.log(K / initial))
                * characteristic(u - 1j)
                / (1j * u * phi_minus_i)
            )

        def second_integrand(u):
            return np.real(
                np.exp(-1j * u * np.log(K / initial))
                * characteristic(u)
                / (1j * u)
            )

        # We use the quadrature method to compute the integrals.

        I1 = integrate.quad(func = first_integrand,
                            a = 0,
                            b = np.inf,
                            limit=200,
                            epsabs=1e-8,
                            epsrel=1e-8)[0]

        I2 = integrate.quad(func = second_integrand,
                            a = 0,
                            b = np.inf,
                            limit=200,
                            epsabs=1e-8,
                            epsrel=1e-8
                            )[0]

        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi

        if option.option_type == "call":
            return initial * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

        elif option.option_type == "put":
            return  K * np.exp(-r * T) * (1 - P2) - initial * np.exp(-q * T) *(1 - P1)