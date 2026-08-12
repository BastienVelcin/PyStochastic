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
    - .simulate() : Simulate an Vasieck process path, with both exact (only in 1D) and Euler-Maruyama methods.
    - .plot() : Plot the Vasieck process path.

Examples
--------
>> R = Vasieck(reversion_speed=2,mean=3,volatility=1,r_0=0,t_0=0,t_n=1,n_steps=1000) #Vasieck process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the Vasieck process path
>>
>> R.plot() #Plot the Vasieck process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom

class Vasicek():

    """
    Vasicek class

    A Vasicek process is a stochastic process that satisfies the following equation:
                                 dR_t = a*(b - R_t)dt+sigma*dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    Parameters
    ----------

    reversion_speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    mean : float, or np.ndarray
        Long term mean vector of the model. The dimension of the vector must coincide with the dimension of the reversion speed matrix.
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
    reversion_speed : float, or list, or np.ndarray
        Speed of reversion matrix of the model.
    mean : float, or np.ndarray
        Long term mean vector of the model.
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

    Examples
    --------
    >> R = Vasieck(reversion_speed=2,mean=3,volatility=1,r_0=0,t_0=0,t_n=1,n_steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,reversion_speed=1,mean=1,volatility=1,r_0=0,t_0=0, t_n=1, n_steps=1000, n_simulations=1):
        self.reversion_speed = np.atleast_2d(reversion_speed)
        self.mean = np.atleast_1d(mean)
        self.volatility = np.atleast_2d(volatility)

        if np.any(self.reversion_speed <= 0) or np.any(self.volatility < 0):
            raise ValueError(
                "The sigma and theta parameters should be greater than 0."
            )

        self.r_0 = np.atleast_1d(r_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.dim = np.size(self.mean)
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps
        self.path = None

        if not(np.shape(self.reversion_speed)[0] == np.shape(self.reversion_speed)[1] == self.dim == np.shape(self.volatility)[0] == np.shape(self.volatility)[1]  == self.r_0.size):
            raise ValueError(
                "The dimension of the the mean, signa, theta, and of the starting point must coincide."
            )

    def drift(self,x,t):

        """
        Drift method

        Drift function of the Vasicek process.

        Returns
        -------
        np.ndarray
            Drift of the Vasicek process evaluated at a time t and a point x.
        """

        return self.reversion_speed @ (self.mean-x)

    def diffusion(self,x,t):

        """
        Diffusion method

        Diffusion function of the Vasicek process.

        Returns
        -------
        np.ndarray
            Diffusion of the Vasicek process evaluated at a time t and a point x.
        """

        return self.volatility

    def simulate(self, n_simulations=1, method="euler-maruyama"):

        """
        Simulate method.

        Simulate a Vasicek process path using both the Euler-Maruyama method and the induction formula.

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.
        method : {"exact", "euler-maruyama"}, default="euler-maruyama"
            Simulation method to use.

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
                                      n_simulations).solve()

        elif method == "exact":
            if self.dim > 1:
                raise ValueError(
                    "The exact method is only implemented for 1D processes."
                )

            self.path = np.zeros((n_simulations, self.n_steps+1, 1))
            self.path[:,0] = self.r_0

            for sim in range(n_simulations):
                # For every simulation, we compute different normal samples
                Z = crandom.normal(0, 1, self.n_steps)

                for i in range(1,self.n_steps+1):
                    #  The induction formula is given by R_t = (mean + R_{t-1} - mean) * exp(-reversion_speed * dt) + volatility * sqrt(1 - exp(-2 * reversion_speed * dt)) / (2 * theta)) * Z[i-1])
                    self.path[sim,i] = (self.mean+ (self.path[sim,i-1] - self.mean) * np.exp(-self.reversion_speed * self.dt) + self.volatility * np.sqrt((1 - np.exp(-2 * self.reversion_speed * self.dt)) / (2 * self.reversion_speed)) * Z[i-1])

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