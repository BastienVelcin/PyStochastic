"""
============================================================
Module Vasicek
============================================================

Description
-----------
This module provides a way to simulate a Vasicek process with a given long-term mean, diffusion and form parameter.

This module provides a general class "Vasicek", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> R = Vasicek(speed=2,mean=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000) #Vasicek process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the Vasicek process path
>>
>> R.plot() #Plot the Vasicek process path
"""

import numpy as np
import scipy
import plotly.graph_objects as go
from pystochastic.random import continuous
from pystochastic.utils import _decompose
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.random.setseed import *

class Vasicek(DiffusionProcess):

    """
    Vasicek class

    A Vasicek process is a stochastic process that satisfies the following equation:
                                 dR_t = - speed*(R_t - mean)dt+volatility*dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Parameters
    ----------
    mean : float, or np.ndarray
        Long term mean vector of the model. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    volatility : float, or np.ndarray
        Volatility matrix. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    initial : float, or list, or np.ndarray
        Initial condition of the model. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mean : float, or np.ndarray
        Long term mean vector of the model.
    speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    volatility : float, or np.ndarray
        Volatility matrix.
    initial : float, or list, or np.ndarray
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
        Dimension of the process.
    t : np.ndarray
        Time interval on which we want to simulate the process.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated process.
    _diagonal : bool
        Specify if sigma is an array that works well with vectorization.
    name : str
        Name of the process
    is_autonomous : bool
        Specify if the process SDE is autonomous.

    Examples
    --------
    >> R = Vasicek(speed=2,mean=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,
                 mean=1,
                 speed=1,
                 volatility=1,
                 initial=0,
                 t_0=0,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0=t_0,
                         t_n=t_n,
                         steps=steps)

        self.name = "Vasicek process"
        self.is_autonomous = True
        self.mean = np.atleast_1d(mean)

        self.speed = np.atleast_1d(speed)
        self.volatility = np.atleast_1d(volatility)

        self.m_speed = np.atleast_2d(speed)
        self.m_volatility = np.atleast_2d(volatility)


        if self.speed.ndim == 1:
            self.speed = np.diag(self.speed)

        if self.volatility.ndim == 1:
            self.volatility = np.diag(self.volatility)

        self.dim = self.mean.size
        self.initial = np.atleast_1d(initial)

        if not(np.shape(self.speed)[0] == np.shape(self.speed)[1] == self.dim == np.shape(self.volatility)[0] == np.shape(self.volatility)[1]  == self.initial.size):
            raise ValueError(
                "The dimension of the the mean, signa, theta, and of the starting point must coincide."
            )

        # We check if the speed and volatility are a scalar, a vector or a diagonal matrix.
        _speed, _speed_diag = _decompose(speed)  # Form : (Scalar/Vec/Matrix, Bool)
        _volatility, _volatility_diag = _decompose(volatility)  # Form : (Scalar/Vec/Matrix, Bool)

        self._diagonal = _speed_diag and _volatility_diag # True if the reversion speed and the volatility supports vectorization, False otherwise.

        # If the reversion speed and the volatility are a scalar, a vector or a diagonal matrix, we force it to be vector to use vectorization
        # Otherwise, we assign them their matrix form, and we will use the sequential method.
        if self._diagonal:
            self.speed = _speed
            self.volatility = _volatility

        if np.all(self.speed <= 0) or np.all(self.volatility < 0):
            raise ValueError(
                "The sigma and theta parameters should be greater than 0."
            )


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
            return self.speed * (self.mean - x)
        return self.speed @ (self.mean - x)

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

    def _simulate_exact(self, n_simulations=1,plot=False):

        """
        Exact simulation method.

        Simulate a Vasicek process path using the induction formula.

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Path of the simulated Vasicek process of the form ``(n_simulations, steps + 1, dim)``.
        """

        if self.dim > 1:
            raise ValueError(
                "The exact method is only implemented for 1D processes."
            )

        self.path = np.zeros((n_simulations, self.steps+1, 1))
        self.path[:,0] = self.initial

        Z = continuous.normal(0, 1, self.steps * n_simulations).reshape((n_simulations, self.steps))

        for i in range(1,self.steps+1):
            #  The induction formula is given by R_t = (mean + R_{t-1} - mean) * exp(-speed * dt) + volatility * sqrt(1 - exp(-2 * speed * dt)) / (2 * theta)) * Z[i-1])
            self.path[:,i,0] = (self.mean+ (self.path[:,i-1,0] - self.mean) * np.exp(-self.speed * self.dt) + self.volatility * np.sqrt((1 - np.exp(-2 * self.speed * self.dt)) / (2 * self.speed)) * Z[:,i-1])

        if plot:
            self.n_simulations = n_simulations
            self.plot()

        return self.path

    def expectation(self,t):

        """
        Expectation method.

        Return the expectation of the Ornstein-Uhlenbeck process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Expectation of the Ornstein-Uhlenbeck process at a time t

        Notes
        -----
        The expectation of the Ornstein-Uhlenbeck process at every time t with a fixed initial is given by
                            initial * exp(-speed*t) + mean * (Id - exp(-volatility*t))
        """

        return self.mean + scipy.linalg.expm(-self.m_speed * (t - self.t_0)) @ (self.initial - self.mean)

    def covariance_matrix(self, t):

        """
        Covariance Matrix method.

        Return the covariance of the Ornstein-Uhlenbeck process at a given time t.
        The covariance matrix satisfies the following Lyapunov equation :
            P'(t) = -volatility*P(t) - P(t)*volatility^T + (speed*speed^T)

        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.

        Returns
        -------
        np.ndarray
            Covariance matrix of the Ornstein-Uhlenbeck process at a time t.
        """

        Q = self.m_volatility @ self.m_volatility.T

        # We define the Lyapunov equation, where s is the time at which we want to evaluate the solution of the
        # Lyapunov equation, and p is the state of the process at time t (as a vector).
        def ode(s, p):

            #Because p was flatten (inputted as a vector), we need to reshape it as a matrix like the covariance one.
            P = p.reshape(self.dim, self.dim)

            # We define the right-hand side of the Lyapunov equation
            dP = -self.m_speed @ P- P @ self.m_speed.T + Q

            # We return the flattened version of the right-hand side of the Lyapunov equation. We use the method ravel
            # to flatten the array column by column instead of row by row.
            return dP.ravel()

        # We solve the Lyapunov equation using the scipy.integrate.solve_ivp function, with the initial condition p=0, since
        # initial, the initial condition of the process, is a deterministic vector.

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