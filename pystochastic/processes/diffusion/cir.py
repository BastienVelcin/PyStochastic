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
>> R = CIR(speed=2,mean=3,volatility=1,initial=0,T=1,steps=1000) #Cox-Ingersoll-Ross process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the CIR process path
>>
>> R.plot() #Plot the CIR process path
"""

import numpy as np
import scipy

from pystochastic.random.setseed import *
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.processes.process import _validate_t

class CIR(DiffusionProcess):

    """
    CIR class

    A Cox-Ingersoll-Ross process is a unidimensional stochastic process that satisfies the following equation:
                                 dR_t = speed*(mean - R_t)dt + volatility*sqrt(R_t)dW_t,

    Parameters
    ----------
    speed : float
        Speed of mean reversion. Must be strictly positive.
    mean : float
        Mean parameter. Must be positive.
    volatility : float
        Volatility parameter. Must be strictly positive.
    initial : float, or list, or np.ndarray
        Initial condition of the model. Must be positive.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    speed : float
        Speed of mean reversion.
    mean : float
        Mean parameter.
    volatility : float
        Volatility parameter.
    initial : float, or list, or np.ndarray
        Initial condition of the model.
    T : float
        Final time.
    steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the process. Here, the dimension is 1 since we consider only the unidimensional case.
    t : np.ndarray
        Time interval on which we want to simulate the CIR.
    dt : float
        Time step length.
    path : np.ndarray
        Path of the simulated CIR.
    nu : float
        First parameter of the Noncentral chi-squared distribution for the exact simulation equation
    nc_factor : float
        Factor for the second parameter of the Noncentral chi-squared distribution for the exact simulation equation.
    c : float
        Standardization coefficient for the Noncentral chi-squared distribution for the exact simulation equation.
    name : str
        Name of the process
    is_autonomous : bool
        Specify if the process SDE is autonomous.

    Examples
    --------
    >> R = CIR(speed=2,mean=3,volatility=1,initial=0,T=1,steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,
                 mean=1,
                 speed=1,
                 volatility=1,
                 initial=0,
                 T=1,
                 steps=1000):

        super().__init__(T = T,
                         steps=steps,
                         dim = 1,
                         name = "Cox-Ingersoll-Ross process")

        self.speed = speed
        self.mean = mean
        self.volatility = volatility
        self.initial = initial

        if (self.speed <= 0) or (self.mean <= 0) or (self.volatility <= 0) or (self.initial < 0):
            raise ValueError(
                "The model parameters must satisfy speed > 0, mean > 0, volatility > 0 and initial >= 0."
            )

        self.is_autonomous = True

    @property
    def nu(self):
        return (4*self.speed*self.mean)/(self.volatility**2)

    @property
    def nc_factor(self):
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
        rng = get_rng()

        for i in range(1,self.steps+1):

            # The induction formula is given by R_{t+1} = c*Z, where Z ~ NCX2(df=nu, nc=R_t * nc_factor)
            Y = rng.noncentral_chisquare(df=self.nu, nonc=self.path[:,i-1] * self.nc_factor)
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
            Time at which the expectation is evaluated. Must be between 0 and T.

        Notes
        -----
        The expectation of the CIR path at every time t with a fixed initial is given by initial * exp(-speed*t) + mean*(1-exp(-speed*t))
        """

        t = _validate_t(t)

        return self.initial * np.exp(-self.speed * t) + self.mean*(1-np.exp(-self.speed * t))

    def variance(self,t):

        """
        Variance method.

        Return the variance of the CIR path at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated. Must be between 0 and T.

        Returns
        -------
        float
            Variance of the CIR path at a time t

        Notes
        -----
        The variance of the CIR path at every time t with a fixed initial is given by
        initial * volatility^2/speed * (exp(-speed*t)-exp(-2*speed*t)) + (mean*volatility^2)/(2*speed) * (1-exp(-speed*t))^2
        """

        t = _validate_t(t)

        return self.initial * (self.volatility**2 / self.speed) * (np.exp(-self.speed*t)-np.exp(-2 * self.speed * t)) + (self.mean * self.volatility**2)/(2*self.speed) * (1-np.exp(-self.speed*t))**2

    def density(self, t, x):

        """
        Density method.

        Return the density of the CIR process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the density is evaluated.
        x :
            Point at which the density is evaluated.

        Returns
        -------
        np.ndarray
            Dist of the CIR process path coordinates at a given time t.

        Notes
        -----
        When the density is evaluated at t=0, the function returns 0 instead of returning the Dirac distribution.
        """

        t = _validate_t(t)


        if not x > 0 or t == 0:
            return np.array([0])

        x = np.asarray(x)

        nu = (4 * self.speed * self.mean) / (self.volatility ** 2)

        c = ((self.volatility ** 2) * (1 - np.exp(-self.speed * t)) / (4 * self.speed))

        nc_factor = (4 * self.speed * np.exp(-self.speed * t) / ((self.volatility ** 2) * (1 - np.exp(-self.speed * t))))

        noncentrality = nc_factor * self.initial

        return scipy.stats.ncx2.pdf(x / c,df=nu,nc=noncentrality) / c