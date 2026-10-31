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

def _validate_parameters(initial , dividend, option_type, method, n_simulations, n_runs, antithetic):

    if not isinstance(initial, (int, np.integer, float, np.floating)) or initial <= 0 or not np.isfinite(initial) or np.isnan(initial):
        raise ValueError("The initial value must be a strictly positive real number.")

    if not isinstance(dividend, (int, np.integer, float, np.floating)) or dividend < 0 or not np.isfinite(dividend) or np.isnan(dividend):
        raise ValueError("The continuous dividend rate must be a positive or zero real number.")

    if not option_type in ["call", "put"]:
        raise ValueError("The option type must be either 'call' or 'put'.")

    if not method in ["exact", "montecarlo"]:
        raise ValueError("The method must be either 'exact' or 'montecarlo'.")

    if not isinstance(n_simulations, (int, np.integer)) or n_simulations <= 0 or not np.isfinite(n_simulations) or np.isnan(n_simulations):
        raise ValueError("The number of simulations must be a strictly positive integer.")

    if not isinstance(antithetic, bool):
        raise ValueError("The antithetic variable must be a boolean.")

    if not isinstance(n_runs, (int, np.integer)) or n_runs < 1 or not np.isfinite(n_runs) or np.isnan(n_runs):
        raise ValueError("The number of Monte-Carlo runs must be an integer greater than 1.")


class BlackScholes:

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

    def __init__(self, rf_ir, volatility, maturity, strike):

        if not isinstance(rf_ir, (int, np.integer, float, np.floating)) or not np.isfinite(rf_ir) or np.isnan(rf_ir):
            raise ValueError("The risk-free interest rate parameter must be a float.")

        if not isinstance(volatility, (int, np.integer, float, np.floating)) or volatility <= 0 or not np.isfinite(volatility) or np.isnan(volatility):
            raise ValueError("The volatility parameter must be a strictly positive float.")

        if not isinstance(maturity, (int, np.integer, float, np.floating)) or maturity <= 0 or not np.isfinite(maturity) or np.isnan(maturity):
            raise ValueError("The maturity date must be a strictly positive float.")

        if not isinstance(strike, (int, np.integer, float, np.floating)) or strike <= 0 or not np.isfinite(strike) or np.isnan(strike):
            raise ValueError("The strike parameter must be a strictly positive float.")

        self.rf_ir = rf_ir
        self.volatility = volatility
        self.maturity = maturity
        self.strike = strike

        self.name = "Black-Scholes"

    def price(self, initial = 1, dividend = 0, option_type="call", method = "exact", n_simulations = 10000, n_runs = 1, antithetic = False, seed = None):

        _validate_parameters(initial, dividend, option_type, method, n_simulations, n_runs, antithetic)

        if method == "exact":

            d1 = (np.log(initial / self.strike) + (self.rf_ir - dividend + 0.5 * self.volatility ** 2)*self.maturity) / (self.volatility * np.sqrt(self.maturity))
            d2 = d1 - self.volatility * np.sqrt(self.maturity)
            phi = Normal(0,1).cdf

            if option_type == "call":
                return initial * np.exp(-dividend * self.maturity) * phi(d1) - self.strike * np.exp(-self.rf_ir * self.maturity) * phi(d2)
            else:
                return self.strike * np.exp(-self.rf_ir * self.maturity) * phi(-d2) - initial * np.exp(-dividend * self.maturity) * phi(-d1)

        elif method == "montecarlo":

            # We simulate the GBM under the risk-neutral measure:
            #               dS_t = (rf_ir - dividend) * S_t * dt + volatility * S_t * dW_t^Q
            # where W_t^Q is a brownian motion under the risk-neutral measure Q.

            if antithetic:
                if n_simulations % 2 != 0:
                    raise ValueError(
                        "The number of simulations must be even when using antithetic variates."
                    )

                if seed is not None:
                    _setseed(seed)
                N = Normal(0, 1).sample((n_simulations // 2) * n_runs).reshape(n_simulations // 2, n_runs)
                Z = np.concatenate((N, -N), axis=0)

            else:
                if seed is not None:
                    _setseed(seed)
                Z = Normal(0, 1).sample(n_simulations * n_runs).reshape(n_simulations, n_runs)

            maturity_values = initial * np.exp(
                (self.rf_ir - dividend - self.volatility ** 2 / 2) * self.maturity + self.volatility * np.sqrt(
                    self.maturity) * Z)

            if option_type == "call":
                payoff = np.maximum(maturity_values - self.strike,0)
            else:
                payoff = np.maximum(self.strike - maturity_values,0)

            return np.exp(-self.rf_ir * self.maturity) * np.mean(payoff, axis = 0)


    def summary(self, initial = 1, dividend = 0, option_type = "call", n_simulations = 10000, n_runs = 1000, antithetic = False, seed = None):

        _validate_parameters(initial, dividend, option_type, "exact", n_simulations, n_runs, antithetic)

        exact_price = self.price(initial = initial,
                                 dividend = dividend,
                                 option_type = option_type,
                                 method = "exact")

        montecarlo_prices = self.price(initial = initial,
                                       dividend = dividend,
                                       option_type = option_type,
                                       method = "montecarlo",
                                       n_simulations = n_simulations,
                                       n_runs = n_runs,
                                       antithetic = antithetic,
                                       seed = seed)

        mean_mc_price = np.mean(montecarlo_prices)

        if n_runs == 1:
            sd_mc_price = np.std(montecarlo_prices)
        else:
            sd_mc_price = np.std(montecarlo_prices, ddof=1)
        results = {"exact_price": exact_price,
                   "mean_mc_price": mean_mc_price,
                   "std_mc_price": sd_mc_price,
                   "se_mean_mc_price": sd_mc_price / np.sqrt(n_runs),
                    "bias_mc_price": mean_mc_price - exact_price,
                    "rmse_mc_price": np.sqrt(np.mean((montecarlo_prices - exact_price) ** 2))
                   }
        return results

    def display_summary(self, initial = 1, dividend = 0, option_type = "call", n_simulations = 10000, n_runs = 1000, antithetic = False, seed = None):
        results = self.summary(initial = initial,
                               dividend=dividend,
                               option_type = option_type,
                               n_simulations = n_simulations,
                               n_runs = n_runs,
                               antithetic = antithetic,
                               seed = seed)

        table = Table(
            title="[bold light_coral]PyStochastic[/] [dim]|[/] "
                  "[bold white]Black-Scholes Pricing[/]",
            box=box.ROUNDED,
            border_style="light_coral",
            header_style="bold indian_red1",
            title_justify="left",
            padding=(0, 2),
            show_lines=False,
        )

        table.add_column("Statistic", style="white", min_width=30)
        table.add_column("Value", justify="right", style="light_salmon1")

        names = {
            "exact_price": "Exact price",
            "mean_mc_price": "Mean Monte Carlo price",
            "std_mc_price": "MC standard deviation",
            "se_mean_mc_price": "Standard error of the mean",
            "bias_mc_price": "Monte Carlo bias",
            "rmse_mc_price": "Root mean squared error",
        }

        for key, value in results.items():
            label = names.get(key, key.replace("_", " ").title())

            if key == "exact_price":
                table.add_row(
                    f"[bold]{label}[/]",
                    f"[bold deep_pink2]{value:.6f}[/]",
                )
            else:
                table.add_row(label, f"{value:.6f}")

        Console().print()
        Console().print(table)

    def call_put_parity(self , initial = 1, dividend = 0, n_simulations = 10000, n_runs = 100, antithetic = False, seed = None):

        _validate_parameters(initial, dividend, "call", "exact", n_simulations, n_runs, antithetic)

        # The call-put parity is defined by
        # C - P = S * exp(-dividend * maturity) - K * exp(-rf_ir * maturity)

        parity = initial * np.exp(-dividend * self.maturity) - self.strike * np.exp(-self.rf_ir * self.maturity)
        bs_call = self.price(initial = initial,
                             dividend = dividend,
                             option_type="call",
                             method = "exact")

        bs_put = self.price(initial=initial,
                             dividend=dividend,
                             option_type="put",
                             method="exact")

        # Since (S_T - K)^+ - (K-S_T)^+ = S_T - K, we can compute the Monte-Carlo approximation of the call-put parity.
        # To minimize the variance, we compute the call and the put with the same normal samples.

        if antithetic:
            if n_simulations % 2 != 0:
                raise ValueError(
                    "The number of simulations must be even when using antithetic variates."
                )
            if seed is not None:
                _setseed(seed)
            N = Normal(0, 1).sample((n_simulations // 2) * n_runs).reshape((n_simulations // 2) , n_runs)
            Z = np.concatenate((N, -N), axis = 0)

        else:
            if seed is not None:
                _setseed(seed)
            Z = Normal(0, 1).sample(n_simulations * n_runs).reshape(n_simulations, n_runs)
        maturity_values = initial * np.exp(
            (self.rf_ir - dividend - self.volatility ** 2 / 2) * self.maturity + self.volatility * np.sqrt(
                self.maturity) * Z)


        mc_approx =  np.exp(-self.rf_ir * self.maturity) * np.mean(maturity_values-self.strike, axis = 0)


        results = {"theoretical_value": parity,
                   "bs_cp" : bs_call - bs_put,
                   "mc_cp" : np.mean(mc_approx),
                   "mc_cp_var" : np.var(mc_approx, ddof=1) if n_runs > 1 else np.var(mc_approx, ddof = 0),
                   "mc_cp_std" : np.std(mc_approx, ddof = 1) if n_runs > 1 else np.std(mc_approx, ddof = 0)}

        return results

    def display_call_put_parity(self, initial = 1, dividend = 0, n_simulations = 10000, n_runs = 100, antithetic = False, seed = None):

        results = self.call_put_parity(initial = initial,
                                       dividend = dividend,
                                       n_simulations = n_simulations,
                                       n_runs = n_runs,
                                       antithetic =antithetic,
                                       seed = seed)
        table = Table(
            title="[bold light_coral]PyStochastic[/] [dim]|[/] "
                  "[bold white]Black-Scholes Pricing[/]",
            box=box.ROUNDED,
            border_style="light_coral",
            header_style="bold indian_red1",
            title_justify="left",
            padding=(0, 2),
            show_lines=False,
        )

        names = {
            "theoretical_value": "Theoretical value for Call-Put parity",
            "bs_cp": "Call-Put (Black-Scholes formula)",
            "mc_cp": "Call-Put (Monte-Carlo method)",
            "mc_cp_var": "Variance of the Monte-Carlo approximation",
            "mc_cp_std": "Standard deviation of the Monte-Carlo approximation",
        }

        table.add_column("Statistic", style="white", min_width=30)
        table.add_column("Value", justify="right", style="light_salmon1")

        for key, value in results.items():
            label = names.get(key, key.replace("_", " ").title())
            if key == "theoretical_value":
                table.add_row(
                    f"[bold]{label}[/]",
                    f"[bold deep_pink2]{value:.6f}[/]",
                )
            else:
                table.add_row(label, f"{value:.6f}")

        Console().print()
        Console().print(table)
