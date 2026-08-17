"""
============================================================
Module VASIECK
============================================================

Description
-----------
This module provides a way to simulate a Vasieck process with a given long-term mean, diffusion and form parameter.

This module provides a general class "Vasieck", with the following built-in methods:
    - .drift() : Drift function of the Vasieck model.
    - .diffusion() : Diffusion function of the Vasieck model.
    - .simulate() : Simulate an Vasieck process path, with both exact (only in 1D), Milstein (only in 1D) and Euler-Maruyama methods.
    - .plot() : Plot the Vasieck process path.
    - .mean() : Mean of the Vasieck process at a given time.
    - .covariance_matrix() : Covariance matrix of the Vasieck process at a given time.
    - .covariance() : Covariance between two coordinates of the Vasieck process at a given time.
    - .variance() : Variance of the Vasieck process at a given time.

Examples
--------
>> R = Vasieck(reversion_speed=2,mean=3,volatility=1,r_0=0,t_0=0,t_n=1,n_steps=1000) #Vasieck process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the Vasieck process path
>>
>> R.plot() #Plot the Vasieck process path
"""

import numpy as np
import scipy
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom
from pystochastic.utils import _decompose

class Vasicek():

    """
    Vasicek class

    A Vasicek process is a stochastic process that satisfies the following equation:
                                 dR_t = a*(b - R_t)dt+sigma*dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Parameters
    ----------
    mu : float, or np.ndarray
        Long term mean vector of the model. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    reversion_speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    volatility : float, or np.ndarray
        Volatility matrix. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    r_0 : float, or list, or np.ndarray
        Initial condition of the model. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mu : float, or np.ndarray
        Long term mean vector of the model.
    reversion_speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    volatility : float, or np.ndarray
        Volatility matrix.
    r_0 : float, or list, or np.ndarray
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
        Dimension of the process.
    t : np.ndarray
        Time interval on which we want to simulate the process.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated process.
    _diagonal : bool
        Specify if sigma is an array that works well with vectorization.

    Examples
    --------
    >> R = Vasieck(reversion_speed=2,mean=3,volatility=1,r_0=0,t_0=0,t_n=1,n_steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,
                 mu=1,
                 reversion_speed=1,
                 volatility=1,
                 r_0=0,
                 t_0=0,
                 t_n=1,
                 n_steps=1000,
                 n_simulations=1):

        self.mu = np.atleast_1d(mu)

        self.reversion_speed = np.atleast_1d(reversion_speed)
        self.volatility = np.atleast_1d(volatility)

        self.m_reversion_speed = np.atleast_2d(reversion_speed)
        self.m_volatility = np.atleast_2d(volatility)


        if self.reversion_speed.ndim == 1:
            self.reversion_speed = np.diag(self.reversion_speed)

        if self.volatility.ndim == 1:
            self.volatility = np.diag(self.volatility)

        self.dim = self.mu.size
        self.r_0 = np.atleast_1d(r_0)

        if not(np.shape(self.reversion_speed)[0] == np.shape(self.reversion_speed)[1] == self.dim == np.shape(self.volatility)[0] == np.shape(self.volatility)[1]  == self.r_0.size):
            raise ValueError(
                "The dimension of the the mean, signa, theta, and of the starting point must coincide."
            )

        # We check if the reversion_speed and volatility are a scalar, a vector or a diagonal matrix.
        _reversion_speed, _reversion_speed_diag = _decompose(reversion_speed)  # Form : (Scalar/Vec/Matrix, Bool)
        _volatility, _volatility_diag = _decompose(volatility)  # Form : (Scalar/Vec/Matrix, Bool)

        self._diagonal = _reversion_speed_diag and _volatility_diag # True if the reversion speed and the volatility supports vectorization, False otherwise.

        # If the reversion speed and the volatility are a scalar, a vector or a diagonal matrix, we force it to be vector to use vectorization
        # Otherwise, we assign them their matrix form, and we will use the sequential method.
        if self._diagonal:
            self.reversion_speed = _reversion_speed
            self.volatility = _volatility

        if np.all(self.reversion_speed <= 0) or np.all(self.volatility < 0):
            raise ValueError(
                "The sigma and theta parameters should be greater than 0."
            )

        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps
        self.path = None

    def drift(self,x,t=None):

        """
        Drift function

        Evaluate the drift of the Vasicek process at a given point x and time t.

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

        if self._diagonal:
            return self.reversion_speed * (self.mu - x)
        return self.reversion_speed @ (self.mu - x)

    def diffusion(self,x,t=None):

        """
        Diffusion function

        Evaluate the diffusion of the Vasicek process at a given point x and time t.

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

        return self.volatility

    def simulate(self, n_simulations=1, method="euler-maruyama",plot=False, parallel=False,n_workers=None):

        """
        Simulate method.

        Simulate a Vasicek process path using both the Euler-Maruyama method and the induction formula.

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
            Path of the simulated Vasicek process of the form ``(n_simulations, n_steps + 1, dim)``.
        """

        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(self.drift,
                                      self.diffusion,
                                      self.r_0,
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
                                 self.r_0,
                                 self.t_0,
                                 self.t_n,
                                 self.n_steps,
                                 n_simulations).solve(plot=plot)
        elif method == "exact":
            if self.dim > 1:
                raise ValueError(
                    "The exact method is only implemented for 1D processes."
                )

            self.path = np.zeros((n_simulations, self.n_steps+1, 1))
            self.path[:,0] = self.r_0

            Z = crandom.normal(0, 1, self.n_steps * n_simulations).reshape((n_simulations, self.n_steps))

            for i in range(1,self.n_steps+1):
                #  The induction formula is given by R_t = (mean + R_{t-1} - mean) * exp(-reversion_speed * dt) + volatility * sqrt(1 - exp(-2 * reversion_speed * dt)) / (2 * theta)) * Z[i-1])
                self.path[:,i,0] = (self.mu+ (self.path[:,i-1,0] - self.mu) * np.exp(-self.reversion_speed * self.dt) + self.volatility * np.sqrt((1 - np.exp(-2 * self.reversion_speed * self.dt)) / (2 * self.reversion_speed)) * Z[:,i-1])

            if plot:
                self.n_simulations = n_simulations
                self.plot()
        else:
            raise ValueError(
                "The method must be either 'euler-maruyama', 'milstein' or 'exact'."
            )

        # When the first simulation is launched, we define the global number of simulations
        self.n_simulations = n_simulations

        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Vasieck process. The path can be plotted only in 1D, 2D or 3D.
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
                                         y=self.path[sim,:, 1],
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

    def mean(self,t):

        """
        Mean method.

        Return the mean of the Ornstein-Uhlenbeck process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the mean is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Mean of the Ornstein-Uhlenbeck process at a time t

        Notes
        -----
        The mean of the Ornstein-Uhlenbeck process  at every time t with a fixed R_0 is given by
                            R_0 * exp(-theta*t) + mean * (Id - exp(-theta*t))
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.mu + scipy.linalg.expm(-self.m_reversion_speed * (t - self.t_0)) @ (self.r_0 - self.mu)

    def covariance_matrix(self, t):

        """
        Covariance Matrix method.

        Return the covariance of the Ornstein-Uhlenbeck process at a given time t.
        The covariance matrix satisfies the following Lyapunov equation :
            P'(t) = -theta*P(t) - P(t)*theta^T + (Sigma*Sigma^T)

        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.

        Returns
        -------
        np.ndarray
            Covariance matrix of the Ornstein-Uhlenbeck process at a time t.
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        Q = self.m_volatility @ self.m_volatility.T

        # We define the Lyapunov equation, where s is the time at which we want to evaluate the solution of the
        # Lyapunov equation, and p is the state of the process at time t (as a vector).
        def ode(s, p):

            #Because p was flatten (inputted as a vector), we need to reshape it as a matrix like the covariance one.
            P = p.reshape(self.dim, self.dim)

            # We define the right-hand side of the Lyapunov equation
            dP = -self.m_reversion_speed @ P- P @ self.m_reversion_speed.T + Q

            # We return the flattened version of the right-hand side of the Lyapunov equation. We use the method ravel
            # to flatten the array column by column instead of row by row.
            return dP.ravel()

        # We solve the Lyapunov equation using the scipy.integrate.solve_ivp function, with the initial condition p=0, since
        # R_0, the initial condition of the process, is a deterministic vector.

        solution = scipy.integrate.solve_ivp(ode, (0, t),np.zeros(self.dim ** 2))

        # solution.y is a 2D array, with the first dimension corresponding to the time, and the second dimension to the state.
        # Because we want the solution at the time t, we need to select the last column of the array, and to reshape it as a matrix,
        # instead of a vector.

        return solution.y[: , -1].reshape(self.dim, self.dim)

    def covariance(self, t,i,j):

        """
        Covariance Matrix method.

        Return the covariance between the i-th and j-th coordinates
        of the Ornstein-Uhlenbeck process at a given time t.


        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.
        i : int
            Index of the first coordinate. It must verify 0 <= i < dim.
        j : int
            Index of the second coordinate. It must verify 0 <= j < dim.

        Returns
        -------
        np.ndarray
            Covariance between the i-th and j-th coordinates.

        Notes
        -----
        This method is using the covariance matrix method, which solves the Lyapunov equation.
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.covariance_matrix(t)[i,j]

    def variance(self,t):

        """
        Variance method.

        Return the variance of the Ornstein-Uhlenbeck process at a given time t.


        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.

        Returns
        -------
        np.ndarray
            Variance of the Ornstein-Uhlenbeck process at the time t.

        Notes
        -----
        This method is using the covariance matrix method, which solves the Lyapunov equation.
        """

        return np.diag(self.covariance_matrix(t))