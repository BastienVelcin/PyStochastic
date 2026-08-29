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
>> BSL = Bessel(order=2, T=1, steps=1000)
>>
>> BSL.simulate() #Simulate the Bessel process path
>>
>> BSL.plot() #Plot the Bessel process path
"""

import numpy as np
import scipy
from pystochastic.processes.elementary.brownian import Brownian
from pystochastic.processes.process import Process
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
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    order : int
        Order of the Bessel process.
    T : float
        Final time.
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
    >> BSL = Bessel(order=2, T=1, steps=1000)
    >> BSL.simulate() #Simulate the Bessel process path
    >> BSL.plot() #Plot the Bessel process path
    """

    def __init__(self,
                 order=1,
                 T=1,
                 steps=1000):

        self.order = order

        super().__init__(T=T,
                         steps=steps,
                         dim = 1,
                         name = f"Bessel process of order {self.order}")

        self.initial = 0
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

        W = Brownian(np.eye(self.order),T = self.T, steps = self.steps)
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

    def density(self,t,x):

        if self.dim > 1:
            raise ValueError(
                "The density is only implemented for 1D processes yet."
            )

        if t == 0:
            return np.array([0])
        return 1/(2**((self.order/2)-1)*scipy.special.gamma(self.order/2) * np.sqrt(t)) * (x/np.sqrt(t))**(self.order-1) * np.exp(-x**2 /(2*t))
