"""
============================================================
Module COX-INGERSOLL-ROSS
============================================================

Description
-----------
This module provides a way to simulate a Cox-Ingersoll-Ross process with a given long-term mean, diffusion and form parameter.

This module provides a general class "CIR", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> R = CIR(a=2,b=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000) #Cox-Ingersoll-Ross process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the CIR process path
>>
>> R.plot() #Plot the CIR process path
"""

import numpy as np
from pystochastic.random.setseed import *
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess

class CIR(DiffusionProcess):

    """
    CIR class

    A Cox-Ingersoll-Ross process is a unidimensional stochastic process that satisfies the following equation:
                                 dR_t = a*(b - R_t)dt + volatility*sqrt(R_t)dW_t,
    For more information, please refer to :
        - https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model

    Parameters
    ----------
    speed : float
        Speed of adjustment to the mean and volatility.
    mean : float
        Mean parameter.
    volatility : float
        Volatility parameter.
    initial : float, or list, or np.ndarray
        Initial condition of the model.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    speed : float
        Speed of adjustment to the mean and volatility.
    mean : float
        Mean parameter.
    volatility : float
        Volatility parameter.
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
    name : str
        Name of the process
    is_autonomous : bool
        Specify if the process SDE is autonomous.

    Examples
    --------
    >> R = CIR(a=2,b=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,
                 speed=1,
                 mean=1,
                 volatility=1,
                 initial=0,
                 t_0=0,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0=t_0,
                         t_n=t_n,
                         steps=steps)

        self.speed = speed
        self.mean = mean
        self.volatility = volatility
        self.initial = initial
        self.name = "Cox-Ingersoll-Ross process"

        if (self.speed <= 0) or (self.mean < 0) or (self.volatility <= 0) or (self.initial < 0):
            raise ValueError(
                "The model parameters must satisfy a > 0, b >= 0, volatility > 0 and initial >= 0."
            )

        self.dim = 1
        self.is_autonomous = True

    @property
    def nu(self):
        return (4*self.speed*self.mean)/(self.volatility**2)

    @property
    def factor(self):
        return (4*self.speed*np.exp(-self.speed * self.dt))/((self.volatility**2)*(1-np.exp(-self.speed * self.dt)))

    @property
    def c(self):
        return ((self.volatility ** 2) * (1 - np.exp(-self.speed * self.dt))) / (4 * self.speed)

    @property
    def feller_condition(self):
        return 2 * self.speed * self.mean >= self.volatility ** 2

    def drift(self,x,t=None):
        return self.speed * (self.mean-x)

    def diffusion(self,x,t=None):
        return self.volatility * np.sqrt(np.maximum(x,0))

    def _simulate_exact(self, n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a Cox-Ingersoll-Ross process path using both the exact solution.

        Parameters
        ----------
        n_simulations : int, default=1
            Number of trajectories to simulate.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Path of the simulated Cox-Ingersoll-Ross process of the form ``(n_simulations, steps + 1, dim)``.
        """

        self.path = np.zeros((n_simulations,self.steps+1, 1))
        self.path[:,0] = self.initial
        for i in range(1,self.steps+1):
            rng = get_rng()
            # The induction formula is given by R_{t+1} = c*Z, where Z ~ NCX2(df=nu, nc=R_t * factor)
            Y = rng.noncentral_chisquare(df=self.nu, nonc=self.path[:,i-1] * self.factor)
            self.path[:,i] = self.c*Y

        if plot:
            self.n_simulations = n_simulations
            self.plot()
        return self.path

    def expectation(self,t):

        """
        Expectation method.

        Return the expectation of the CIR path at a given time t.

        Returns
        -------
        float
            Expectation of the CIR path at a time t

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between t_0 and t_n.

        Notes
        -----
        The expectation of the CIR path at every time t with a fixed initial is given by initial * exp(-speed*t) + b*(1-exp(-speed*t))
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.initial * np.exp(-self.speed * t) + self.mean*(1-np.exp(-self.speed * t))

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
        The variance of the CIR path at every time t with a fixed initial is given by
        initial * volatility^2/a * (exp(-a*t)-exp(-2*a*t)) + (b*volatility^2)/(2*a) * (1-exp(-a*t))^2
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.initial * (self.volatility**2 / self.speed) * (np.exp(-self.speed*t)-np.exp(-2 * self.speed * t)) + (self.mean * self.volatility**2)/(2*self.speed) * (1-np.exp(-self.speed*t))**2

    def covariance_matrix(self,t):
        return self.variance(t)

    def covariance(self,t,i,j):
        return self.variance(t)