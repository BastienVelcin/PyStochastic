"""
============================================================
Module POISSON
============================================================

Description
-----------
This module provides a way to simulate a Poisson process with a given intensity parameter.

This module provides a general class "Poisson", with the following built-in methods:
    - .simulate() : Simulate a Poisson process path in 1D.
    - .plot() : Plot the Poisson process path.
    - .mean() : Mean of the Poisson process at a given time.
    - .variance() : Variance of the Poisson process at a given time.

Examples
--------
>> P = Poisson(intensity=2,T=1, steps=1000) #Poisson process with intensity 2
>>
>> P.simulate() #Simulate the Poisson process path
>>
>> P.plot() #Plot the Poisson process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.random import discrete
from pystochastic.dist import Poisson
from pystochastic.processes.jump.jump_process import JumpProcess

class PoissonProcess(JumpProcess):

    """
    Poisson class

    A Poisson process (N_t)_{t>=0} is a unidimensional stochastic process that satisfies the following assertions:
        - For all 0<= t_1 <= ... < t_k, the random variables (N_t_k - N_t_{k-1}) , ... , (N_t_1 - N_0) are independent.
        - P(N_{t+h}-N_t = 1) = intensity*h + o(h) when h ---> 0+
        - P(N_{t+h}-N_t > 1) = o(h) when h ---> 0+

    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Poisson_point_process

    Parameters
    ----------
    intensity : float
        Intensity parameter. Must be strictly positive.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    intensity : float
        Intensity parameter.
    T : float
        Final time.
    steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the process. Here, the dimension is 1.
    t : np.ndarray
        Time interval on which we want to simulate the process.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated process.
    name : str
        Name of the process.

    Examples
    --------
    >> P = Poisson(intensity=2,T=1, steps=1000)
    >> P.simulate()
    >> P.plot()
    """

    def __init__(self,
                 intensity=1,
                 T = 1,
                 steps=1000):

        super().__init__(T = T,
                         steps = steps,
                         dim = 1,
                         name = "Poisson process")

        if intensity <= 0:
            raise ValueError(
                "The intensity must be strictly positive."
            )

        self.intensity = intensity


    def simulate(self,n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a Poisson process path using Poisson random variables

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.

        Returns
        -------
        np.ndarray
            Path of the simulated Poisson process of the form ``(n_simulations, steps + 1)``.
        """

        self.path = np.zeros((n_simulations, self.steps + 1, 1))

        # The increments of a Poisson process follows : N_t_{i+1} - N_t_i ~ Poisson(intensity*dt)
        increments = discrete.poisson(self.intensity * self.dt, n_simulations * self.steps).reshape((n_simulations, self.steps, 1))

        # We compute N_t with cumsum

        self.path[:, 1:] = np.cumsum(increments, axis=1)

        self.n_simulations = n_simulations

        self.path = self.path.astype(int)
        if plot:
            self.plot()
        return self.path


    def expectation(self,t):
        return self.intensity*t

    def variance(self,t):
        return self.intensity * t

    def density(self,t,x):
        if t == 0:
            return np.array([0])
        P = Poisson(lam  = t*self.intensity)
        return P.pmf(np.floor(x))