"""
============================================================
Module BROWNIAN BRIDGE
============================================================

Description
-----------
This module provides a way to simulate a Brownian bridge.

This module provides a general class "BrownianBridge", which inherits from the methods of Process abstract class.

Examples
--------
>> B = BrownianBridge(dim=2,T=1,steps=1000) #Brownian bridge with covariance matrix np.eye(2)
>>
>> B.simulate() #Simulate the Brownian bridge path
>>
>> B.plot() #Plot the Brownian bridge path
"""

import numpy as np
from pystochastic.processes.process import Process
from pystochastic.processes.elementary.brownian import Brownian
from pystochastic.processes.process import _validate_t
from pystochastic.dist import Normal
class BrownianBridge(Process):

    """
    Brownian motion class

    The Brownian bridge is a standard Brownian motion which terminal value is equal to 0.

    Parameters
    ----------
    dim : int
        Dimension of the brownian motion. The dimension must coincide with the dimension of the covariance matrix. Must be a strictly positive integer.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    dim : int
        Dimension of the brownian motion.
    T : float
        Final time.
    steps : int
        Number of time steps.
    path : np.ndarray
        Path of the simulated Brownian motion.
    t : np.ndarray
        Time interval on which we want to simulate the Brownian motion.
    name : str
        Name of the process

    Examples
    --------
    >> B = BrownianBridge(dim=2,T=1,steps=1000)
    >> B.simulate()
    >> B.plot()
    """
    def __init__(self,
                 dim=1,
                 T = 1,
                 steps=1000):

        super().__init__(T = T,
                         steps = steps,
                         dim = dim,
                         name = "Brownian Bridge" )

    def simulate(self,n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a Brownian Bridge path using the following formula:
            B_t = W_t - (t/T * W_{T})

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Path of the simulated Brownian Bridge process of the form ``(n_simulations, steps + 1, dim)``.
        """
        self.n_simulations = n_simulations
        W = Brownian(variance = np.eye(self.dim), T = self.T, steps = self.steps)
        W.simulate(n_simulations = n_simulations)
        self.path = np.zeros((n_simulations,self.steps+1, self.dim))

        self.path = W.path - np.einsum("t,sd->std",self.t / self.T,W.path[:, -1, :])
        if plot:
            self.plot()

        return self.path

    def expectation(self,t):

        t = _validate_t(t)

        return 0

    def variance(self,t):

        t = _validate_t(t)

        if self.dim == 1:
            return t*(self.T - t)/self.T
        return np.diag(t*(self.T - t)/self.T*np.eye(self.dim))

    def covariance_matrix(self,t):

        t = _validate_t(t)

        if self.dim == 1:
            return t*(self.T - t)/self.T
        return t*(self.T - t)/self.T*np.eye(self.dim)

    def covariance(self,t,i,j):

        t = _validate_t(t)

        if not 0 <= i < self.dim or not isinstance(i, (int, np.integer)):
            raise ValueError(
                "The first coordinate must be an integer between 0 and the dimension (excluded)."
            )

        if not 0 <= j < self.dim or not isinstance(j, (int, np.integer)):
            raise ValueError(
                "The second coordinate must be an integer between 0 and the dimension (excluded)."
            )

        return t*(self.T - t)/self.T if i == j else 0

    def density(self,t,x):

        if self.dim > 1:
            raise ValueError(
                "The density is only implemented for 1D processes yet."
            )

        t = _validate_t(t)

        if t == 0 or t == self.T:
            return np.array([0])
        N = Normal(mu = 0, var = t*(self.T - t)/self.T)
        return N.pdf(x)