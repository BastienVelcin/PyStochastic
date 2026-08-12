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

Examples
--------
>> P = Poisson(intensity=2,t_0=0, t_n=1, n_steps=1000) #Poisson process with intensity 2
>>
>> P.simulate() #Simulate the Poisson process path
>>
>> P.plot() #Plot the Poisson process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom

class Poisson():

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
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    intensity : float
        Intensity parameter.
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    n_steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the process. Here, the dimension is 1.
    t : np.ndarray
        Time interval on which we want to simulate the GBM.
    path : np.ndarray
        Path of the simulated GBM.

    Examples
    --------
    >> P = Poisson(intensity=2,t_0=0, t_n=1, n_steps=1000)
    >> P.simulate()
    >> P.plot()
    """

    def __init__(self,intensity=1,t_0=0, t_n=10, n_steps=1000):
        self.intensity = intensity
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.dim = 1
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.path = None


    def simulate(self,n_simulations=1):

        """
        Simulate method.

        Simulate a Poisson process path using exponential random variables.

        Returns
        -------
        np.ndarray
            Path of the simulated Poisson process.
        """

        self.path = np.zeros((n_simulations,self.n_steps+1))

        for sim in range(n_simulations):
            T = [0]

            # We simulate exponential random variables until the total sum of them exceeds the final time.
            while T[-1] < self.t_n:
                E = crandom.exponential(self.intensity).item()
                T.append(T[-1] + E)

            for i in range(1,self.n_steps+1):
                # The Poisson process value is given by the number of exponential sums that are smaller than the current time.
                self.path[sim,i]= sum(T<= self.t[i])

        # When the first simulation is launched, we define the global number of simulations
        self.n_simulations = n_simulations

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