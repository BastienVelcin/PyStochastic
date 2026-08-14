"""
============================================================
Module ORNSTEIN UHLENBECK
============================================================

Description
-----------
This module provides a way to simulate an Ornstein-Uhlenbeck process with a given long-term mean, diffusion and form parameter.

This module provides a general class "OrnsteinUhlenbeck", with the following built-in methods:
    - .simulate() : Simulate an Ornstein-Uhlenbeck process path, with both exact (only in 1D) and Euler-Maruyama methods.
    - .plot() : Plot the Ornstein-Uhlenbeck process path.
    - .mean() : Mean of the Ornstein-Uhlenbeck process at a given time.
    - .covariance_matrix() : Covariance matrix of the Ornstein-Uhlenbeck process at a given time.
    - .covariance() : Covariance between two coordinates of the Ornstein-Uhlenbeck process at a given time.
    - .variance() : Variance of the Ornstein-Uhlenbeck process at a given time.

Examples
--------
>> R = OrnsteinUhlenbeck(mean=[2,1],sigma=np.ones((2,2)),theta=np.ones((2,2)),r_0=[1,1],t_0=0,t_n=1,n_steps=1000) #Ornstein-Uhlenbeck process with mean [2,1] and diffusion and form parameter np.ones((2,2)) and starting point [1,1]
>>
>> R.simulate() #Simulate the  Ornstein-Uhlenbeck process path
>>
>> R.plot() #Plot the Ornstein-Uhlenbeck process path
"""

import numpy as np
import scipy
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom

class OrnsteinUhlenbeck:

    """
    Ornstein Uhlenbeck class

    An Ornstein Uhlenbeck process is a stochastic process that satisfies the following equation:
                                 dR_t = - theta*(R_t - mean)dt+sigma*dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Parameters
    ----------
    mu : float, or list, or np.ndarray
        Long term mean vector of the model.
    sigma : float, or np.ndarray
        Constant matrix diffusion of the model. The dimension of the matrix must coincide with the dimension of the starting point and the vector drift.
    theta : float, or np.ndarray
        Constant matrix shape parameter. The dimension of the matrix must coincide with the dimension of the starting point and the vector drift.
    r_0 : float, or list, or np.ndarray
        Initial condition of the model. The dimension of the starting point must coincide with the dimension of sigma and theta.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mu : float, or list, or np.ndarray
        Long term mean vector of the model.
    sigma : float, or np.ndarray
        Constant matrix diffusion of the model.
    theta : float, or np.ndarray
        Constant matrix shape parameter.
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

    Examples
    --------
    >> R = OrnsteinUhlenbeck(mean=[2,1],sigma=np.ones((2,2)),theta=np.ones((2,2)),r_0=[1,1],t_0=0,t_n=1,n_steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,mu=0,sigma=1,theta=1,r_0=0,t_0=0, t_n=1, n_steps=1000):
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_2d(sigma)
        self.theta = np.atleast_2d(theta)

        if np.all(self.sigma < 0) or np.all(self.theta <=0):
            raise ValueError(
                "The sigma and theta parameters should be greater than 0."
            )

        self.r_0 = np.atleast_1d(r_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.dim = np.size(self.mu)
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps
        self.path = None

        if not(np.shape(self.sigma)[0] == np.shape(self.sigma)[1] == self.dim == np.shape(self.theta)[0] == np.shape(self.theta)[1] ==  self.r_0.size):
            raise ValueError(
                "The dimension of the mean, sigma, theta, and of the starting point must coincide."
            )

    def simulate(self, n_simulations=1, method="euler-maruyama"):

        """
        Simulate method.

        Simulate an Ornstein Uhlenbeck process path using both the Euler-Maruyama method and the induction formula.

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.
        method : {"exact", "euler-maruyama"}, default="euler-maruyama"
            Simulation method to use.

        Returns
        -------
        np.ndarray
            Path of the simulated Ornstein Uhlenbeck process of the form ``(n_simulations, n_steps + 1, dim)``.
        """

        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(lambda x,t : (self.mu-x) @ self.theta.T ,
                                      lambda x,t : self.sigma,
                                      self.r_0,
                                      self.t_0,
                                      self.t_n,
                                      self.n_steps,
                                      n_simulations).solve()

        elif method == "exact":
            if self.dim > 1:
                raise ValueError(
                    "The exact method is only implemented for 1D processes."
                )

            self.path = np.zeros((n_simulations,self.n_steps+1, 1))
            self.path[:,0] = self.r_0

            for sim in range(n_simulations):
                # For every simulation, we compute different normal samples
                Z = crandom.normal(0, 1, self.n_steps)

                for i in range(1,self.n_steps+1):
                    # The induction formula is given by R_t = (mean + R_{t-1} - mean) * exp(-theta * dt) + sigma * sqrt(1 - exp(-2 * theta * dt)) / (2 * theta)) * Z[i-1])
                    self.path[sim,i] = (self.mu+ (self.path[sim,i-1] - self.mu) * np.exp(-self.theta * self.dt) + self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * self.dt)) / (2 * self.theta)) * Z[i-1])

        else:
            raise ValueError(
                "The method must be either 'euler-maruyama' or 'exact'."
            )

        # When the first simulation is launched, we define the global number of simulations
        self.n_simulations = n_simulations

        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Ornstein Uhlenbeck process. The path can be plotted only in 1D, 2D or 3D.
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

        return self.r_0 @ scipy.linalg.expm(- self.theta * t) + self.mu @ (np.eye(self.dim) - scipy.linalg.expm(- self.theta * t))

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

        Q = self.sigma @ self.sigma.T

        # We define the Lyapunov equation, where s is the time at which we want to evaluate the solution of the
        # Lyapunov equation, and p is the state of the process at time t (as a vector).
        def ode(s, p):

            #Because p was flatten (inputted as a vector), we need to reshape it as a matrix like the covariance one.
            P = p.reshape(self.dim, self.dim)

            # We define the right-hand side of the Lyapunov equation
            dP = -self.theta @ P- P @ self.theta.T + Q

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