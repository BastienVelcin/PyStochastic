"""
============================================================
Module GEOMETRIC BROWNIAN MOTION
============================================================

Description
-----------
This module provides a way to simulate a Geometric Brownian Motion (GBM) with a given mean and covariance matrix.

This module provides a general class "GeometricBrownianMotion", with the following built-in methods:
    - .simulate() : Simulate a Geometric Brownian Motion path, with both exact, Milstein (only in 1D) and Euler-Maruyama methods.
    - .plot() : Plot the Geometric Brownian Motion path.
    - .mean() : Mean of the Geometric Brownian Motion process at a given time.
    - .covariance() : Covariance between two coordinates of the Geometric Brownian Motion process at a given time.
    - .covariance_matrix() : Covariance matrix of the Geometric Brownian Motion process at a given time.
    - .variance() : Variance of the Geometric Brownian Motion process at a given time.

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
from pystochastic.utils import _decompose

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
    S_0 : None, float, or list, or np.ndarray
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
    S_0 : None, float, or list, or np.ndarray
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
    _diagonal : bool
        Specify if sigma is an array that works well with vectorization.

    Examples
    --------
    >> S = GeometricBrownianMotion(mu=[2,1],sigma=np.eye(2),S_0=[1,1],t_0=0,t_n=1,n_steps=1000)
    >> S.simulate()
    >> S.plot()
    """

    def __init__(self,
                 mu=1,
                 sigma=1,
                 S_0=None,
                 t_0=0,
                 t_n=1,
                 n_steps=1000):

        self.mu = np.atleast_1d(mu)

        self.sigma = np.atleast_1d(sigma)

        if self.sigma.ndim == 1:
            self.sigma = np.diag(self.sigma)

        if S_0 == None:
            S_0 = np.ones(np.size(mu))

        self.S_0 = np.atleast_1d(S_0)
        self.dim = np.size(self.S_0)

        if self.sigma.ndim == 1 and not(np.shape(self.sigma) == self.dim == np.size(self.mu)):
            raise ValueError(
                "The dimension of the volatility matrix, of the drift and of the starting point must coincide."
            )

        elif self.sigma.ndim != 1 and not(np.shape(self.sigma)[0] == np.shape(self.sigma)[1] == self.dim == np.size(self.mu)):
            raise ValueError(
                "The dimension of the volatility matrix, of the drift and of the starting point must coincide."
            )

        # We check if the diffusion is a scalar, a vector or a diagonal matrix.
        _sigma, _sigma_diag = _decompose(sigma) # Form : (Scalar/Vec/Matrix, Bool)

        self._diagonal = _sigma_diag # True or False depending on the type of sigma.

        # If sigma is a scalar, a vector or a diagonal matrix, we force it to be vector to use vectorization
        # Otherwise, we assign to the diffusion the matrix, and we will use the sequential method.
        if self._diagonal:
            self.sigma = _sigma

        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.t = np.linspace(t_0, t_n, n_steps+1)
        self.path = None



    def drift(self, x, t=None):

        """
        Drift function

        Evaluate the drift of the GBM at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the drift is evaluated.
        t : float
            Time at which the drift is evaluated.

        Returns
        -------
        float :
            Drift evaluated at x and t.
        """

        return self.mu * x

    def diffusion(self, x, t=None):

        """
        Diffusion function

        Evaluate the diffusion of the GBM at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the diffusion is evaluated.
        t : float
            Time at which the diffusion is evaluated.

        Returns
        -------
        float :
            Diffusion evaluated at x and t.
        """

        # If sigma is a scalar, a vector or a diagonal matrix, then self.sigma is a vector, and the diffusion is given
        # by sigma * x. Otherwise, self.sigma is a matrix, and the diffusion is given by sigma @ x.

        if self._diagonal:
            return self.sigma * x

        return  x @ self.sigma.T

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
            Path of the simulated Geometric Brownian Motion of the form ``(n_simulations, n_steps + 1, dim)``.
        """

        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(self.drift,
                                      self.diffusion,
                                      self.S_0,
                                      self.t_0,
                                      self.t_n,
                                      self.n_steps,
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
                                      self.S_0,
                                      self.t_0,
                                      self.t_n,
                                      self.n_steps,
                                      n_simulations).solve(plot=plot)

        elif method == "exact":
            self.path = np.zeros((n_simulations,self.n_steps+1, self.dim))
            W = Brownian(np.eye(self.dim), self.t_0, self.t_n, self.n_steps)
            W.simulate(n_simulations=n_simulations)
            #If the volatility is a scalar, a vector or a diagonal matrix, then sigma is reshaped as a vector,
            # so we need to sum on the only available axis. Otherwise, sigma is a matrix, and we need to sum on the axis 1.
            if self._diagonal:
                for i in range(0, self.n_steps+1):
                    # The explicit solution is given by S_t = S_0 * exp((mu - 1/2 * sigma^2) * t + sigma * W_t)

                    self.path[:,i, :] = self.S_0 * np.exp(
                        (self.mu - np.sum(self.sigma ** 2, axis=0) / 2) * self.t[i] + self.sigma * W.path[:,i,:])
            else:
                for i in range(0, self.n_steps + 1):
                    # The explicit solution is given by S_t = S_0 * exp((mu - 1/2 * sigma^2) * t + sigma * W_t)

                    self.path[:, i, :] = self.S_0 * np.exp(
                        (self.mu - np.sum(self.sigma * 2, axis=1) / 2) * self.t[i] + self.sigma  @ W.path[:,i,:])

            if plot:
                self.n_simulations = n_simulations
                self.plot()
        else:
            raise ValueError(
                "The method must be either 'euler-maruyama', 'milstein' or 'exact'."
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

    def mean(self, t):

        """
        Mean method.

        Return the mean of the GBM path at a given time t.

        Parameters
        ----------
        t : float
            Time at which the mean is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Mean of the GBM path at a time t

        Notes
        -----
        The mean of the GBM path at every time t with a fixed S_0 is given by S_0 * exp(mu*t)
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.S_0 * np.exp(self.mu * t)

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
            S_{0,i} * S_{0,j} * exp((mu_{i} + mu_{j})*t) * (exp((sigma*sigma^T)_{ij}*t) - 1).
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        if not 0 <= i < self.dim:
            raise ValueError(
                "The first coordinate must be between 0 and the dimension (excluded)."
            )

        if not 0 <= j < self.dim:
            raise ValueError(
                "The second coordinate must be between 0 and the dimension (excluded)."
            )
        return self.S_0[i]*self.S_0[j]*np.exp((self.mu[i]+self.mu[j])*t)*(np.exp((self.sigma*self.sigma.T)[i,j] * t)-1)

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

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

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

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return np.array([self.covariance(t,i,i) for i in range(self.dim)])