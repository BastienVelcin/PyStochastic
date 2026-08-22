"""
============================================================
Module POISSON
============================================================

Description
-----------
This module provides a way to simulate a Poisson process with a given intensity parameter.

This module provides a general class "CIR", with the following built-in methods:
    - .simulate() : Simulate a Poisson process path in 1D.
    - .plot() : Plot the Poisson process path.
    - .mean() : Mean of the Poisson process at a given time.
    - .variance() : Variance of the Poisson process at a given time.

Examples
--------
>> P = Poisson(intensity=2,t_0=0, t_n=1, steps=1000) #Poisson process with intensity 2
>>
>> P.simulate() #Simulate the Poisson process path
>>
>> P.plot() #Plot the Poisson process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.random import crandom, drandom

class Poisson:

    """
    Poisson class

    A Poisson process (N_t)_{t>=0} is an unidimensional stochastic process that satisfies the following assertions:
        - For all 0<= t_1 <= ... < t_k, the random variables (N_t_k - N_t_{k-1}) , ... , (N_t_1 - N_0) are independent.
        - P(N_{t+h}-N_t = 1) = intensity*h + o(h) when h ---> 0+
        - P(N_{t+h}-N_t > 1) = o(h) when h ---> 0+

    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Poisson_point_process

    Parameters
    ----------
    intensity : float
        Intensity parameter. Must be strictly positive.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    intensity : float
        Intensity parameter.
    t_0 : float
        Initial time.
    t_n : float
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

    Examples
    --------
    >> P = Poisson(intensity=2,t_0=0, t_n=1, steps=1000)
    >> P.simulate()
    >> P.plot()
    """

    def __init__(self,
                 intensity=1,
                 t_0=0,
                 t_n=10,
                 steps=1000):

        self.intensity = intensity
        self.t_0 = t_0
        self.t_n = t_n
        self.steps = steps
        self.n_simulations = None
        self.dim = 1
        self.t = np.linspace(t_0,t_n,steps+1)
        self.dt = (t_n-t_0)/steps
        self.path = None


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

        self.path = np.zeros((n_simulations, self.steps + 1))

        # The increments of a Poisson process follows : N_t_{i+1} - N_t_i ~ Poisson(intensity*dt)
        increments = drandom.poisson(self.intensity * self.dt, n_simulations* self.steps).reshape((n_simulations, self.steps))

        # We compute N_t with cumsum

        self.path[:, 1:] = np.cumsum(increments, axis=1)

        self.n_simulations = n_simulations

        if plot:
            self.plot()

        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Poisson process.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        fig = go.Figure()

        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t,
                                     y=self.path[sim,:],
                                     mode="lines",
                                     line=dict(width=2,shape="hv")))
        fig.show()

    def expectation(self,t):

        """
        Expectation method.

        Return the expectation of the Poisson process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Expectation of the Poisson process at a time t

        Notes
        -----
        The expectation of the Poisson process at every time t is equal to the intensity times t, since the Poisson process at a time t
        follows a Poisson distribution of parameter intensity times t.
        """

        return self.intensity*t

    def variance(self,t):

        """
        Variance method.

        Return the variance of the Poisson process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Variance of the Poisson process at a time t

        Notes
        -----
        The variance of the Poisson process at every time t is equal to the intensity times t, since the Poisson process at a time t
        follows a Poisson distribution of parameter intensity times t.
        """

        return self.intensity * t