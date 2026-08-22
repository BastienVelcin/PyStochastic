"""
============================================================
Module GEOMETRIC BROWNIAN MOTION
============================================================

Description
-----------
This module provides a way to simulate a Geometric Brownian Motion (GBM) with a given mean and covariance matrix.

This module provides a general class "GeometricBrownianMotion", which inherits from the methods of Process and
DiffusionProcess abstract classes.

Examples
--------
>> S = GeometricBrownianMotion(mu=[2,1],volatility=np.eye(2),initial=[1,1],t_0=0,t_n=1,steps=1000) #Geometric Brownian Motion with mean [2,1] and covariance matrix np.eye(2) and starting point [1,1]
>>
>> S.simulate() #Simulate the Brownian motion path
>>
>> S.plot() #Plot the Brownian motion path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes.continuous.brownian import Brownian
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.utils import _decompose
from pystochastic.random.setseed import *

class GeometricBrownianMotion(DiffusionProcess):

    """
    Geometric Brownian Motion class

    The Geometric Brownian Motion is a stochastic process that satisfies the following equation:
                                 dS_t = mu*S_tdt + volatility*S_t dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Geometric_Brownian_motion

    Parameters
    ----------
    mu : float, or list, or np.ndarray
        Constant vector drift of the model.
    volatility : float, or np.ndarray
        Constant matrix drift of the model. The dimension of the matrix must coincide with the dimension of the starting point and the vector mu.
    initial : None, float, or list, or np.ndarray
        Initial condition of the model. The dimension of the starting point must coincide with the dimension of the covariance matrix and the vector mu.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mu : float, or list, or np.ndarray
        Constant vector drift of the model.
    volatility : float, or np.ndarray
        Factor diffusion matrix of the model.
    initial : None, float, or list, or np.ndarray
        Initial condition of the model.
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the GBM.
    t : np.ndarray
        Time interval on which we want to simulate the GBM.
    path : np.ndarray
        Path of the simulated GBM.
    _diagonal : bool
        Specify if volatility is an array that works well with vectorization.

    Examples
    --------
    >> S = GeometricBrownianMotion(mu=[2,1],volatility=np.eye(2),initial=[1,1],t_0=0,t_n=1,steps=1000)
    >> S.simulate()
    >> S.plot()
    """

    def __init__(self,
                 mu=1,
                 volatility=1,
                 initial=None,
                 t_0=0,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0=t_0,
                         t_n=t_n,
                         steps=steps)

        self.mu = np.atleast_1d(mu)

        self.volatility = np.atleast_1d(volatility)
        self.m_volatility = np.atleast_2d(volatility)

        if self.volatility.ndim == 1:
            self.volatility = np.diag(self.volatility)

        if initial is None:
            initial = np.ones(np.size(mu))

        self.initial = np.atleast_1d(initial)
        self.dim = np.size(self.initial)

        if self.volatility.ndim == 1 and not(np.shape(self.volatility) == self.dim == np.size(self.mu)):
            raise ValueError(
                "The dimension of the volatility matrix, of the drift and of the starting point must coincide."
            )

        elif self.volatility.ndim != 1 and not(np.shape(self.volatility)[0] == np.shape(self.volatility)[1] == self.dim == np.size(self.mu)):
            raise ValueError(
                "The dimension of the volatility matrix, of the drift and of the starting point must coincide."
            )

        # We check if the diffusion is a scalar, a vector or a diagonal matrix.
        _volatility, _volatility_diag = _decompose(volatility) # Form : (Scalar/Vec/Matrix, Bool)

        self._diagonal = _volatility_diag # True or False depending on the type of volatility.

        # If volatility is a scalar, a vector or a diagonal matrix, we force it to be vector to use vectorization
        # Otherwise, we assign to the diffusion the matrix, and we will use the sequential method.
        if self._diagonal:
            self.volatility = _volatility


    def drift(self, x, t=None):

        return self.mu * x

    def diffusion(self, x, t=None):

        # If volatility is a scalar, a vector or a diagonal matrix, then self.volatility is a vector, and the diffusion is given
        # by volatility * x. Otherwise, self.volatility is a matrix, and the diffusion is given by volatility @ x.

        if self._diagonal:
            return self.volatility * x

        return  x @ self.volatility.T

    def simulate(self,n_simulations=1, method="exact",plot=False,parallel=False,n_workers=None):

        """
        Simulate method.

        Simulate a Geometric Brownian Motion path using both the Euler-Maruyama method and the explicit solution.

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        method : {"exact", "euler-maruyama", "milstein"}
            Simulation method to use.
        plot : bool
            Specify if the path should be plotted.
        parallel: bool
            In the case the vectorization doesn't work, the user can specify the usage of parallel computing.
        n_workers: int
            Number of workers to use in parallel computing.

        Returns
        -------
        np.ndarray
            Path of the simulated Geometric Brownian Motion of the form ``(n_simulations, steps + 1, dim)``.
        """

        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(self.drift,
                                      self.diffusion,
                                      self.initial,
                                      self.t_0,
                                      self.t_n,
                                      self.steps,
                                      n_simulations).solve(plot=plot,
                                                           parallel=parallel,
                                                           n_workers=n_workers)
        elif method == "milstein":
            if self.dim > 1:
                raise ValueError(
                    "The Milstein method is only implemented for 1D processes."
                )

            from pystochastic.sde import Milstein
            self.path = Milstein(self.drift,
                                      self.diffusion,
                                      self.initial,
                                      self.t_0,
                                      self.t_n,
                                      self.steps,
                                      n_simulations).solve(plot=plot)

        elif method == "exact":
            self.path = np.zeros((n_simulations,self.steps+1, self.dim))
            W = Brownian(np.eye(self.dim), self.t_0, self.t_n, self.steps)
            W.simulate(n_simulations=n_simulations)
            #If the volatility is a scalar, a vector or a diagonal matrix, then volatility is reshaped as a vector,
            # so we need to sum on the only available axis. Otherwise, volatility is a matrix, and we need to sum on the axis 1.
            if self._diagonal:
                for i in range(0, self.steps+1):
                    # The explicit solution is given by S_t = initial * exp((mu - 1/2 * volatility^2) * t + volatility * W_t)

                    self.path[:,i, :] = self.initial * np.exp(
                        (self.mu - np.sum(self.volatility ** 2, axis=0) / 2) * self.t[i] + self.volatility * W.path[:,i,:])
            else:
                for i in range(0, self.steps + 1):
                    # The explicit solution is given by S_t = initial * exp((mu - 1/2 * volatility^2) * t + volatility * W_t)

                    self.path[:, i, :] = self.initial * np.exp(
                        (self.mu - np.sum(self.volatility * 2, axis=1) / 2) * self.t[i] + self.volatility  @ W.path[:,i,:])

            if plot:
                self.n_simulations = n_simulations
                self.plot()
        else:
            raise ValueError(
                "The method must be either 'euler-maruyama', 'milstein' or 'exact'."
            )

        self.n_simulations = n_simulations
        return self.path

    def expectation(self, t):

        """
        Expectation method.

        Return the expectation of the GBM path at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Expectation of the GBM path at a time t

        Notes
        -----
        The expectation of the GBM path at every time t with a fixed initial is given by initial * exp(mu*t)
        """

        return self.initial * np.exp(self.mu * t)

    def covariance(self,t, i,j):

        """
        Covariance method.

        Return the covariance between the i-th and j-th coordinates
        of the GBM at a given time t.

        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated. Must be between t_0 and t_n.
        i : int
            Index of the first coordinate. It must verify 0 <= i < dim.
        j : int
            Index of the second coordinate. It must verify 0 <= j < dim.

        Returns
        -------
        float
            Covariance between the i-th and j-th coordinates.

        Notes
        -----
        The covariance of the i-th and j-th coordinate of the GBM path at a time t is given by
            S_{0,i} * S_{0,j} * exp((mu_{i} + mu_{j})*t) * (exp((volatility*volatility^T)_{ij}*t) - 1).
        """

        return self.initial[i]*self.initial[j]*np.exp((self.mu[i]+self.mu[j])*t)*(np.exp((self.m_volatility*self.m_volatility.T)[i,j] * t)-1)

    def covariance_matrix(self,t):

        """
        Covariance Matrix method.

        Return the covariance of the GBM at a given time t.

        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.

        Returns
        -------
        np.ndarray
            Covariance matrix of the GBM at a time t.
        """


        covar = np.zeros((self.dim,self.dim))

        for i in range(self.dim):
            for j in range(i,self.dim):
                covar[i,j] = self.covariance(t,i,j)
                covar[j,i] = covar[i,j]
        return covar

    def variance(self, t):

        """
        Variance method.

        Return the variance of the GBM coordinates at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated.

        Returns
        -------
        np.ndarray
            Variance of the GBM path coordinates at a given time t.
        """

        return np.array([self.covariance(t,i,i) for i in range(self.dim)])