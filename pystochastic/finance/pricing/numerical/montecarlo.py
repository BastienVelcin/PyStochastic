import numpy as np
from pystochastic.random.setseed import seed as _setseed
from pystochastic.montecarlo import MonteCarlo
from pystochastic.dist import Normal
from pystochastic.processes.diffusion import GeometricBrownianMotion

class MonteCarloPricer:

    def __init__(self, underlying, n_simulations = 10000, antithetic = False, seed = None):

        self.underlying = underlying
        self.n_simulations = n_simulations
        self.antithetic = antithetic
        self.seed = seed

    def price(self, option, rf_ir, dividend):

        localunderlying = self.underlying(rf_ir-dividend,  )
        self.underlying.simulate(n_simulations = self.n_simulations, plot = False)
        final_values = self.underlying.final_position()
        payoff = option.payoff(final_values)
        mean_payoff = np.mean(payoff)

        return mean_payoff * np.exp(-rf_ir * self.underlying.T)