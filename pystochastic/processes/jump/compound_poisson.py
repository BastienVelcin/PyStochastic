"""
============================================================
Module COMPOUND POISSON
============================================================

Description
-----------
This module provides a way to simulate a Compound Poisson process with a given intensity parameter.

This module provides a general class "CompoundPoisson", which inherits from the methods of Process and JumpProcess abstract classes.

Examples
--------
>> P = CompoundPoisson(intensity=2,T=1, steps=1000) #Poisson process with intensity 2
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
from pystochastic.processes.process import _validate_t

class CompoundPoisson(JumpProcess):
    """
    Compound Poisson class

    A Compound Poisson process is a stochastic process defined as a sum of random variables X_i, where the number of random variables
    summed at a specified time is defined by a Poisson process.

    Parameters
    ----------
    intensity : float
        Intensity parameter of the poisson process. Must be strictly positive.
    distribution : pystochastic.dist.Distribution
        Distribution of the jump sizes.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    intensity : float
        Intensity parameter.
    distribution : pystochastic.dist.Distribution
        Distribution of the jump sizes.
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
    >> P = CompoundPoisson(intensity=2, distribution=Normal(-1,0.5)T=1, steps=1000)
    >> P.simulate()
    >> P.plot()
    """
    def __init__(self,
                 intensity = 1,
                 distribution=Normal(0,1),
                 T = 1,
                 steps=1000):

        super().__init__(T = T,
                         steps = steps,
                         dim = 1,
                         name = "Compound Poisson process")

        if not isinstance(intensity, (int, float, np.integer, np.floating)):
            raise ValueError(
                "The intensity must be a strictly positive real number."
            )
        if intensity <= 0:
            raise ValueError(
                "The intensity must be strictly positive."
            )

        if not isinstance(distribution, Distribution):
            raise ValueError(
                "The distribution must be a Distribution object (pystochastic.dist.Distribution)."
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

        if not n_simulations >= 1 or not isinstance(n_simulations, (int, np.integer)):
            raise ValueError(
                "The number of simulations must be a strictly positive integer."
            )

        self.n_simulations = n_simulations
        self.path = np.zeros((n_simulations,self.steps+1))

        N = PoissonProcess(intensity=self.intensity,T=self.T, steps=self.steps)
        N.simulate(n_simulations=n_simulations)

        max_val_sim = np.max(N.path[:,-1,0]).astype(int)

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
        self.path = np.atleast_3d(np.take_along_axis(jump_sums,N.path[:,:,0],axis=1))
        if plot:
            self.plot()

        return self.path

    def expectation(self,t):

        t = _validate_t(t)

        if self.distribution.mean() is not None and self.distribution.mean() != np.inf:
            return self.intensity * t * self.distribution.mean()
        return None

    def variance(self,t):

        t = _validate_t(t)

        if self.distribution.variance() is not None and self.distribution.variance() != np.inf:
            moment2 = self.distribution.variance() + self.distribution.mean()**2
            return self.intensity * t * moment2
        return None
