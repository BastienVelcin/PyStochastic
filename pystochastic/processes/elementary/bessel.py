"""
============================================================
Module BESSEL
============================================================

Description
-----------
This module provides a way to simulate a Bessel process with a given order.

This module provides a general class "Bessel", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> BSL = Bessel(order=2,t_0=0, t_n=1, steps=1000)
>>
>> BSL.simulate() #Simulate the Bessel process path
>>
>> BSL.plot() #Plot the Bessel process path
"""

import numpy as np
import scipy
import plotly.graph_objects as go

from pystochastic.processes import Brownian
from pystochastic.random import continuous
from pystochastic.utils import _decompose
from pystochastic.processes import Process
from pystochastic.random.setseed import *

class Bessel(Process):

    """
    Bessel class

    A Bessel of order n process is a stochastic process that satisfies the following equation:
                                        B_t = ||W_t||
    where W_t denotes an n-dimensional standard Brownian motion.

    Parameters
    ----------
    order : int
        Order of the Bessel process.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    order : int
        Order of the Bessel process.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the process.
    t : np.ndarray
        Time interval on which we want to simulate the process.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated process.
    name : str
        Name of the process

    Examples
    --------
    >> BSL = Bessel(order=2,t_0=0, t_n=1, steps=1000)
    >> BSL.simulate() #Simulate the Bessel process path
    >> BSL.plot() #Plot the Bessel process path
    """

    def __init__(self,
                 order=1,
                 t_0=1e-5,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0=t_0,
                         t_n=t_n,
                         steps=steps)
        self.initial = 0
        self.order = order
        self.name = f"Bessel process of order {self.order}"
        self.dim = 1



    def simulate(self, n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a Bessel process path using the norm of a Brownian motion.

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Path of the simulated Bessel process of the form ``(n_simulations, steps + 1, dim)``.
        """

        if not isinstance(self.order, (int, np.integer)):
            raise ValueError(
                "The order of the Bessel process must be an integer to compute the path with the exact method."
            )

        self.n_simulations = n_simulations

        W = Brownian(np.eye(self.order),t_0 = self.t_0, t_n = self.t_n, steps = self.steps)
        W.simulate(n_simulations=n_simulations)

        self.path = np.linalg.norm(W.path,axis=2)[:,:,None]

        if plot:
            self.plot()

        return self.path

    def expectation(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the expectation of the Bessel process."
        )

    def covariance_matrix(self, t):
        pass

    def covariance(self, t,i,j):
        pass

    def variance(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the variance of the Bessel process."
        )