"""
============================================================
Module GEOMETRIC BROWNIAN MOTION
============================================================

Description
-----------
This module provides a way to simulate a Geometric Brownian Motion (GBM) with a given mean and covariance matrix.

This module provides a general class "GeometricBrownianMotion", with the following built-in methods:
    - .simulate() : Simulate a Geometric Brownian Motion path, with both exact and Euler-Maruyama methods.
    - .plot() : Plot the Geometric Brownian Motion motion path.

Examples
--------
>> S = GeometricBrownianMotion(mu=[2,1],sigma=np.eye(2),S_0=[1,1],t_0=0,t_n=1,n_steps=1000) #Geometric Brownian Motion with mean [2,1] and covariance matrix np.eye(2) and starting point [1,1]
>>
>> S.simulate() #Simulate the Brownian motion path
>>
>> S.plot() #Plot the Brownian motion path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes.brownian import Brownian

class GeometricBrownianMotion:

    """
    Geometric Brownian Motion class

    The Geometric Brownian Motion is a stochastic process that satisfies the following equation:
                                 dS_t = mu*S_tdt + sigma*S_t dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Geometric_Brownian_motion

    Parameters
    ----------
    mu : float, or list, or np.ndarray
        Constant vector drift of the model.
    sigma : float, or np.ndarray
        Constant matrix drift of the model. The dimension of the matrix must coincide with the dimension of the starting point and the vector drift.
    S_0 : float, or list, or np.ndarray
        Initial condition of the model. The dimension of the starting point must coincide with the dimension of the covariance matrix and the vector drift.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mu : float, or list, or np.ndarray
        Constant vector drift of the model.
    sigma : float, or np.ndarray
        Factor diffusion matrix of the model.
    S_0 : float, or list, or np.ndarray
        Initial condition of the model.
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    n_steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the GBM.
    t : np.ndarray
        Time interval on which we want to simulate the GBM.
    path : np.ndarray
        Path of the simulated GBM.

    Examples
    --------
    >> S = GeometricBrownianMotion(mu=[2,1],sigma=np.eye(2),S_0=[1,1],t_0=0,t_n=1,n_steps=1000)
    >> S.simulate()
    >> S.plot()
    """

    def __init__(self, mu=1, sigma=1, S_0=1,t_0=0, t_n=1, n_steps=1000):

        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_2d(sigma)
        self.S_0 = np.atleast_1d(S_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.dim = np.size(S_0)
        self.t = np.linspace(t_0, t_n, n_steps+1)
        self.path = None

        if not(np.shape(self.sigma)[0] == np.shape(self.sigma)[1] == self.dim == np.size(self.mu)):
            raise ValueError(
                "The dimension of the volatility matrix, of the drift and of the starting point must coincide."
            )

    def simulate(self,n_simulations=1, method="exact"):

        """
        Simulate method.

        Simulate a Geometric Brownian Motion path using both the Euler-Maruyama method and the explicit solution.

        Returns
        -------
        np.ndarray
            Path of the simulated Geometric Brownian Motion.
        """
        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(lambda x,t : self.mu*x,
                                      lambda x,t : self.sigma*x,
                                      self.S_0,
                                      self.t_0,
                                      self.t_n,
                                      self.n_steps,
                                      self.n_simulations).solve()

        elif method == "exact":
            self.path = np.zeros((n_simulations,self.n_steps+1, self.dim))

            for sim in range(n_simulations):
                # For every simulation, we compute a different Brownian increment array.
                W = Brownian(np.eye(self.dim), self.dim, self.t_0, self.t_n, self.n_steps)

                for i in range(0, self.n_steps+1):
                    # The explicit solution is given by S_t = S_0 * exp((mu - 1/2 * sigma^2) * t + sigma * W_t)
                    self.path[sim,i, :] = self.S_0 * np.exp(
                        (self.mu - np.sum(self.sigma ** 2, axis=1) / 2) * self.t[i] + self.sigma   @ W.path[i,:])
        else:
            raise ValueError(
                "The method must be either 'euler-maruyama' or 'exact'."
            )

        self.n_simulations = n_simulations
        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Geometric Brownian Motion. The path can be plotted only in 1D, 2D or 3D.
        """

        if self.dim > 3:
            raise ValueError(
                "The path can be plotted only for 1D, 2D and 3D."
            )

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        fig = go.Figure()

        if self.dim == 1:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=self.path[sim,:, 0],
                                         mode="lines",
                                         line=dict(width=2)))

        elif self.dim == 2:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.path[sim,:, 0],
                                         y=self.path[sim, :, 1],
                                         mode="lines",
                                         line=dict(width=2)))

        else:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter3d(x=self.path[sim,:, 0],
                                           y=self.path[sim,:, 1],
                                           z=self.path[sim,:, 2],
                                           mode="lines",
                                           line=dict(width=2)))

        fig.show()
