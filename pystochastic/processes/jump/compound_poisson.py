"""
============================================================
Module COMPOUND POISSON
============================================================

Description
-----------
This module provides a way to simulate a Compound Poisson process with a given intensity parameter.

This module provides a general class "CompoundPoisson", with the following built-in methods:
    - .simulate() : Simulate a Poisson process path in 1D.
    - .plot() : Plot the Poisson process path.
    - .mean() : Mean of the Poisson process at a given time.
    - .variance() : Variance of the Poisson process at a given time.

Examples
--------
>> P = CompoundPoisson(intensity=2,t_0=0, t_n=1, steps=1000) #Poisson process with intensity 2
>>
>> P.simulate() #Simulate the Poisson process path
>>
>> P.plot() #Plot the Poisson process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.dist import *
from pystochastic.processes.jump.jump_process import JumpProcess
from pystochastic.processes.jump.poisson import PoissonProcess

class CompoundPoisson(JumpProcess):
    """
    Compound Poisson class

    A Compound Poisson process is a stochastic process defined as a sum of random variables X_i, where the number of random variables
    summed at a specified time is defined by a Poisson process.

    Parameters
    ----------
    intensity : float
        Intensity parameter of the poisson process. Must be strictly positive.
    distribution : pystochastic.dist.ContinuousDistribution or pystochastic.dist.DiscreteDistribution
        Distribution of the random variables X_i.
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
    name : str
        Name of the process.

    Examples
    --------
    >> P = Poisson(intensity=2,t_0=0, t_n=1, steps=1000)
    >> P.simulate()
    >> P.plot()
    """
    def __init__(self,
                 intensity = 1,
                 distribution=Normal(0,1),
                 t_0=0,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0 = t_0,
                         t_n = t_n,
                         steps = steps)

        self.name = f"Compound Poisson process with {type(distribution).__name__} distribution of parameters {distribution.__dict__}"

        if intensity <= 0:
            raise ValueError(
                "The intensity must be strictly positive."
            )

        self.intensity = intensity
        self.distribution = distribution

    def simulate(self,n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a Compound Poisson process path using Poisson random variables

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Path of the simulated Compound Poisson process of the form ``(n_simulations, steps + 1)``.
        """
        self.n_simulations = n_simulations
        self.path = np.zeros((n_simulations,self.steps+1))

        N = PoissonProcess(intensity=self.intensity,t_0=self.t_0, t_n=self.t_n, steps=self.steps)
        N.simulate(n_simulations=n_simulations)

        max_val_sim = np.max(N.path[:,-1]).astype(int)

        #If the Poisson process is a constant zero process, we return the path which is constant zero.
        if max_val_sim == 0:
            if plot:
                self.plot()
            return self.path

        # We simulate the specified distribution. We need to simulate at most the maximum value of all the Poisson processes times
        # the number of simulations.

        X = self.distribution.sample(max_val_sim * n_simulations).reshape((n_simulations,max_val_sim))

        # We compute the cumulative sum of the simulated distribution
        jump_sums = np.cumsum(X, axis=1)

        # We initialize the process to 0
        jump_sums = np.concatenate([np.zeros((n_simulations, 1)),jump_sums,],axis=1,)

        # For each time, we take the cumulative sum of the simulated distribution at the corresponding time of the Poisson process
        self.path = np.take_along_axis(jump_sums,N.path,axis=1)

        if plot:
            self.plot()

        return self.path

    def expectation(self,t):
        if self.distribution.mean() is not None or self.distribution.mean() != np.inf:
            return self.intensity * t * self.distribution.mean()
        return None

    def variance(self,t):
        if self.distribution.variance() is not None or self.distribution.variance() != np.inf:
            moment2 = self.distribution.variance() + self.distribution.mean()**2
            return self.intensity * t * moment2
        return None
