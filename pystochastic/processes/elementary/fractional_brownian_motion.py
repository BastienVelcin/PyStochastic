"""
============================================================
Module FRACTIONAL BROWNIAN MOTION
============================================================

Description
-----------
This module provides a way to simulate a Fractional Brownian Motion with a given hurst index.

This module provides a general class "FractionalBrownianMotion", which inherits from the methods of Process abstract class.

Examples
--------
>> F = FractionalBrownianMotion(hurst=0.7,T=1,steps=1000) #Fractional Brownian motion with hurst index 0.7
>>
>> F.simulate() #Simulate the Fractional Brownian motion path
>>
>> F.plot() #Plot the Fractional Brownian motion path
"""

import numpy as np
from pystochastic.random import continuous
from pystochastic.processes.process import Process
from pystochastic.processes.process import _validate_t
from pystochastic.dist import Normal

class FractionalBrownianMotion(Process):
    """
    Fractional Brownian motion class

    The Fractional Brownian Motion is an extension of the Brownian motion. It is a zero-mean Gaussian process which covariance function is
    given by the following equation:
                        Cov(B(t),B(s)) = (|t|^2H + |s|^2H - |t-s|^2H)/2,
    where, H is the Hurst index, which is a float between 0 and 1.

    Parameters
    ----------
    hurst : float
        Hurst index of the Fractional Brownian motion. Must be a float between 0 and 1.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    hurst : float
        Hurst index of the Fractional Brownian motion
    T : float
        Final time.
    steps : int
        Number of time steps.
    path : np.ndarray
        Path of the simulated Fractional Brownian motion.
    t : np.ndarray
        Time interval on which we want to simulate the Fractional Brownian motion.
    dim : int
        Dimension of the Fractional Brownian motion.
    name : str
        Name of the process

    Examples
    --------
    >> F = FractionalBrownianMotion(hurst=0.7,T=1,steps=1000)
    >> F.simulate()
    >> F.plot()
    """

    def __init__(self,
                 hurst= 0.5,
                 T= 1,
                 steps= 1000):

        self.hurst = hurst

        super().__init__(T=T,
                         steps=steps,
                         dim = 1,
                         name = f"{self.hurst}-Fractional Brownian Motion")

        if not isinstance(hurst, (float, int, np.floating, np.integer)):
            raise ValueError(
                "The Hurst index must be a float strictly between 0 and 1."
            )

        if not 0 < hurst < 1:
            raise ValueError(
                "The Hurst index must be a float strictly between 0 and 1."
            )

    @property
    def time_matrix(self):
        return np.repeat(np.atleast_2d(self.t), self.t.size, axis=0)

    @property
    def time_covar_matrix(self):
        epsi = 1e-10
        return np.eye(self.t.size) * epsi + (
                    np.abs(self.time_matrix) ** (2 * self.hurst) + (np.abs(self.time_matrix.T)) ** (
                        2 * self.hurst) - np.abs(self.time_matrix - self.time_matrix.T) ** (2 * self.hurst)) / 2

    @property
    def cholesky_time_covar_matrix(self):
        return np.linalg.cholesky(self.time_covar_matrix).T

    def simulate(self, n_simulations=1, plot=False):

        """
        Simulate method.

        Simulate a Fractional Brownian motion using the Cholesky decomposition of the time covariance matrix.

        Returns
        -------
        np.ndarray
            Path of the simulated Fractional Brownian motion.

        """
        self.n_simulations = n_simulations

        Z = continuous.normal(0, 1, n_simulations * (self.steps + 1)).reshape(self.steps + 1, n_simulations)

        self.path = (Z.T @ self.cholesky_time_covar_matrix)[:, :, None]
        self.path[:,0,0] = np.zeros(self.dim)
        if plot:
            self.plot()

        return self.path

    def expectation(self, t):

        """
        Expectation method.

        Return the expectation of the Fractional Brownian motion at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between 0 and T.

        Returns
        -------
        float
            0 : Expectation of the Fractional Brownian motion at a time t

        Notes
        -----
        The expectation of the Fractional Brownian motion at every time t is always 0, since W_t ~ N(0,t*Q), where Q is the covariance matrix.
        """
        t = _validate_t(t)

        return 0


    def variance(self, t):

        """
        Variance method.

        Return the variance of the Fractional Brownian Motion coordinates at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated.

        Returns
        -------
        np.ndarray
            Variance of the Fractional Brownian Motion path coordinates at a given time t.
        """

        return t**(2*self.hurst) * np.ones(self.steps)

    def density(self,t,x):

        if self.dim > 1:
            raise ValueError(
                "The density is only implemented for 1D processes yet."
            )

        t = _validate_t(t)

        if t == 0:
            return np.array([0])
        N = Normal(mu=0, var=t ** (2 * self.hurst))
        return N.pdf(x)
