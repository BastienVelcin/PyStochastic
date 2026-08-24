"""
============================================================
Module HESTON
============================================================

Description
-----------
This module provides a way to simulate a Heston model with given parameters. Heston model is used to determine an asset price where
the volatility is given by a Cox-Ingersoll-Ross process.

This module provides a general class "Heston", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> H = Heston(mean=[2,1],speed=np.ones((2,2)),speed=np.ones((2,2)),initial=[1,1],t_0=0,t_n=1,steps=1000) #Ornstein-Uhlenbeck process with mean [2,1] and diffusion and form parameter np.ones((2,2)) and starting point [1,1]
>>
>> H.simulate() #Simulate the Ornstein-Uhlenbeck process path
>>
>> H.plot() #Plot the Ornstein-Uhlenbeck process path
"""
import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian
from pystochastic.processes.diffusion import DiffusionProcess
from pystochastic.sde import EulerMaruyama


class Heston(DiffusionProcess):
    def __init__(
            self,
            mu = 1,
            initial_price=1,
            initial_variance = 1,
            long_variance = 1,
            correlation = 0,
            reverting_rate = 1,
            volatility_volatility = 1,
            t_0=0,
            t_n=1,
            steps=1000):

        super().__init__(t_0 = t_0,
                         t_n = t_n,
                         steps = steps)

        self.name = "Heston process"
        self.mu = mu
        self.initial_price = initial_price
        self.initial_variance = initial_variance
        self.long_variance = long_variance
        self.correlation = correlation
        self.reverting_rate = reverting_rate
        self.volatility_volatility = volatility_volatility
        self.price = None
        self.variance = None
        self.vol = None #The volatility will be sqrt(self.variance)
        self.path = (self.price, self.variance)
        self.dim = 1 #Plot dimension

    @property
    def feller_condition(self):
        return 2 * self.reverting_rate * self.long_variance >= self.volatility_volatility ** 2

    def drift(self,x,t=None):
        price = x[0]
        variance = np.maximum(x[1], 0)

        drift = np.array([self.mu * price, self.reverting_rate*(self.long_variance- variance)])
        return drift

    def diffusion(self, x, t=None):

        # In this implementation, the form of x is x = (price, variance)
        price = x[0]
        variance = np.maximum(x[1], 0)

        diffusion = np.array([[price * np.sqrt(variance),0],
                              [0, self.volatility_volatility * np.sqrt(variance)]])

        return diffusion


    def simulate(self,n_simulations=1,method=None,plot=False,parallel=False,n_workers=None):

        self.n_simulations = n_simulations

        covariance_matrix = np.array([[1, self.correlation], [self.correlation, 1]])

        W = Brownian(variance = covariance_matrix, t_0 = self.t_0, t_n = self.t_n, steps = self.steps)
        W.simulate(n_simulations = n_simulations)

        self.path = EulerMaruyama(drift=self.drift,
                                  diffusion=self.diffusion,
                                  initial=np.array([self.initial_price, self.initial_variance]),
                                  t_0=self.t_0,
                                  t_n=self.t_n,
                                  n_steps=self.steps).solve(n_simulations=n_simulations,
                                                           plot=False,
                                                           parallel=parallel,
                                                           n_workers=n_workers,
                                                           brownian_sequence=W.increments)
        self.price = self.path[:, :, 0]
        self.variance = self.path[:, :, 1]
        self.vol = np.sqrt(np.maximum(self.variance, 0))
        if plot:
            fig_price = go.Figure()
            fig_vol = go.Figure()
            for sim in range(self.n_simulations):
                fig_price.add_trace(go.Scatter(x=self.t,
                                               y=self.price[sim,:],
                                               mode="lines",
                                               line=dict(width=2),
                                               name=f"Path {sim+1}"))
                fig_vol.add_trace(go.Scatter(x=self.t,
                                             y=self.vol[sim, :],
                                             mode="lines",
                                             line=dict(width=2),
                                             name=f"Path {sim+1}"))

            fig_price.update_layout(title=f"Price evolution of the asset price",
                                    xaxis_title="t",
                                    yaxis_title="Price $S_t$",
                                    template="plotly_white")
            fig_vol.update_layout(title=f"Volatility evolution",
                                    xaxis_title="t",
                                    yaxis_title="Volatility $\nu_t$",
                                    template="plotly_white")

            fig_vol.show()
            fig_price.show()

        return self.path


    def covariance_matrix(self,t):
        pass
    def covariance(self,t,i,j):
        pass
    def expectation(self,t):
        pass
    def variance(self,t):
        pass