"""
============================================================
Module COX-INGERSOLL-ROSS
============================================================

Description
-----------
This module provides a way to simulate a Cox-Ingersoll-Ross process with a given long-term mean, diffusion and form parameter.

This module provides a general class "CIR", with the following built-in methods:
    - .drift() : Drift function of the CIR model.
    - .diffusion() : Diffusion function of the CIR model.
    - .simulate() : Simulate a Cox-Ingersoll-Ross model path in 1D, with both exact and Euler-Maruyama methods.
    - .plot() : Plot the Cox-Ingersoll-Ross model path.
    - .mean() : Mean of the CIR process at a given time.
    - .variance() : Variance of the CIR process at a given time.

Examples
--------
>> R = CIR(a=2,b=3,sigma=1,r_0=0,t_0=0,t_n=1,n_steps=1000) #Cox-Ingersoll-Ross process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the CIR process path
>>
>> R.plot() #Plot the CIR process path
"""

import numpy as np
import plotly.graph_objects as go
import scipy

class CIR:

    """
    CIR class

    A Cox-Ingersoll-Ross process is an unidimensional stochastic process that satisfies the following equation:
                                 dR_t = a*(b - R_t)dt + sigma*sqrt(R_t)dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model

    Parameters
    ----------
    a : float
        Speed of adjustment to the mean and volatility.
    b : float
        Mean parameter.
    sigma : float
        Volatility parameter.
    r_0 : float, or list, or np.ndarray
        Initial condition of the model.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    a : float
        Speed of adjustment to the mean and volatility.
    b : float
        Mean parameter.
    sigma : float
        Volatility parameter.
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
        Dimension of the process. Here, the dimension is 1.
    t : np.ndarray
        Time interval on which we want to simulate the CIR.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated CIR.
    nu : float
        First parameter of the Noncentral chi-squared distribution for the exact simulation equation
    factor : float
        Factor for the second parameter of the Noncentral chi-squared distribution for the exact simulation equation.
    c : float
        Standardization coefficient for the Noncentral chi-squared distribution for the exact simulation equation.

    Examples
    --------
    >>> R = CIR(a=2,b=3,sigma=1,r_0=0,t_0=0,t_n=1,n_steps=1000)
    >>> R.simulate()
    >>> R.plot()
    """

    def __init__(self, a=1, b=1, sigma=1, r_0=0, t_0=0, t_n=1, n_steps=1000):

        self.a = a
        self.b = b
        self.sigma = sigma
        self.r_0 = r_0

        if (self.a <= 0) or (self.b < 0) or (self.sigma <= 0) or (self.r_0 < 0):
            raise ValueError(
                "The model parameters must satisfy a > 0, b >= 0, sigma > 0 and r_0 >= 0."
            )

        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.t = np.linspace(t_0, t_n, n_steps + 1)
        self.dt = (t_n - t_0) / n_steps
        self.path = None
        self.dim = 1

        self.nu = (4*self.a*self.b)/(self.sigma**2)
        self.factor = (4*self.a*np.exp(-self.a * self.dt))/((self.sigma**2)*(1-np.exp(-self.a * self.dt)))
        self.c = ((self.sigma**2)*(1-np.exp(-self.a * self.dt)))/(4*self.a)

    def drift(self,x,t):
        return self.a * (self.b-x)

    def diffusion(self,x,t):
        return self.sigma * np.sqrt(np.maximum(x,0))

    def simulate(self, n_simulations=1,method="exact"):

        """
        Simulate method.

        Simulate a Cox-Ingersoll-Ross process path using both the Euler-Maruyama method and exact solution.

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.
        method : {"exact", "euler-maruyama"}, default="exact"
            Simulation method to use.

        Returns
        -------
        np.ndarray
            Path of the simulated Cox-Ingersoll-Ross process of the form ``(n_simulations, n_steps + 1, dim)``.
        """

        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            if (2 * self.a * self.b < self.sigma ** 2):
                raise ValueError(
                    "The model parameters are inconsistent with the model. Please choose a, b and sigma such that 2*a*b >= sigma^2"
                )

            self.path = EulerMaruyama(self.drift,
                                      self.diffusion,
                                      self.r_0,
                                      self.t_0,
                                      self.t_n,
                                      self.n_steps,
                                      n_simulations).solve()

        elif method == "exact":

            self.path = np.zeros((n_simulations,self.n_steps+1, 1))
            self.path[:,0] = self.r_0
            for sim in range(n_simulations):
                for i in range(1,self.n_steps+1):

                    # The induction formula is given by R_{t+1} = c*Z, where Z ~ NCX2(df=nu, nc=R_t * factor)
                    Y = scipy.stats.ncx2.rvs(df=self.nu, nc=self.path[sim,i-1] * self.factor)
                    self.path[sim,i] = self.c*Y

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

        Plot the simulated path of the Cox-Ingersoll-Ross process.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        fig = go.Figure()

        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t,
                                     y=self.path[sim,:,0],
                                     mode="lines",
                                     line=dict(width=2)))
        fig.show()

    def mean(self,t):

        """
        Mean method.

        Return the mean of the CIR path at a given time t.

        Returns
        -------
        float
            Mean of the CIR path at a time t

        Parameters
        ----------
        t : float
            Time at which the mean is evaluated. Must be between t_0 and t_n.

        Notes
        -----
        The mean of the CIR path at every time t with a fixed r_0 is given by r_0 * exp(-a*t) + b*(1-exp(-a*t))
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.r_0 * np.exp(-self.a * t) + self.b*(1-np.exp(-self.a * t))

    def variance(self,t):

        """
        Variance method.

        Return the variance of the CIR path at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            Variance of the CIR path at a time t

        Notes
        -----
        The variance of the CIR path at every time t with a fixed r_0 is given by
        r_0 * sigma^2/a * (exp(-a*t)-exp(-2*a*t)) + (b*sigma^2)/(2*a) * (1-exp(-a*t))^2
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.r_0 * (self.sigma**2 / self.a) * (np.exp(-self.a*t)-np.exp(-2 * self.a * t)) + (self.b * self.sigma**2)/(2*self.a) * (1-np.exp(-self.a*t))**2