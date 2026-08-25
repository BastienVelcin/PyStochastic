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
>> B = BrownianBridge(dim=2,t_0=0,t_n=1,steps=1000) #Brownian bridge with covariance matrix np.eye(2)
>>
>> B.simulate() #Simulate the Brownian bridge path
>>
>> B.plot() #Plot the Brownian bridge path
"""

import numpy as np
from pystochastic.processes.process import Process
from pystochastic.processes.elementary.brownian import Brownian

class BrownianBridge(Process):

    """
    Brownian motion class

    The Brownian bridge is a standard Brownian motion which terminal value is equal to 0.

    Parameters
    ----------
    dim : int
        Dimension of the brownian motion. The dimension must coincide with the dimension of the covariance matrix. Must be a strictly positive integer.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    dim : int
        Dimension of the brownian motion.
    t_0 : float
        Initial time.
    t_n : float
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
    >> B = BrownianBridge(dim=2,t_0=0,t_n=1,steps=1000)
    >> B.simulate()
    >> B.plot()
    """
    def __init__(self,dim=1,t_0=0,t_n=1,steps=1000):

        super().__init__(t_0 = t_0,
                         t_n = t_n,
                         steps = steps)
        self.name = "Brownian Bridge"
        self.dim = dim

    def simulate(self,n_simulations=1,plot=False):

        W = Brownian(variance = np.eye(self.dim), t_0 = self.t_0, t_n = self.t_n, steps = self.steps)
        W.simulate(n_simulations = n_simulations)
        self.path = np.zeros((n_simulations,self.steps+1, self.dim))
        self.path = W.path - (self.t/self.t_n * W.path[:, -1, :])[:,:,None]
        if plot:
            self.plot()

        self.n_simulations = n_simulations
        return self.path

    def expectation(self,t):
        return 0

    def variance(self,t):
        return np.diag(t*(self.t_n - t)/self.t_n*np.eye(self.dim))

    def covariance_matrix(self,t):
        return t*(self.t_n - t)/self.t_n*np.eye(self.dim)

    def covariance(self,t,i,j):
        return t*(self.t_n - t)/self.t_n if i == j else 0