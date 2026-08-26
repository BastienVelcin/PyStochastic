"""
============================================================
Module HullWhite
============================================================

Description
-----------
This module provides a way to simulate a HullWhite process with a given long-term mean, diffusion and form parameter.

This module provides a general class "HullWhite", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> R = HullWhite(speed=2,mean=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000) #HullWhite process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the HullWhite process path
>>
>> R.plot() #Plot the HullWhite process path
"""

import numpy as np
import scipy
from pystochastic.random import continuous
from pystochastic.utils import _decompose
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.random.setseed import *

class HullWhite(DiffusionProcess):

    """
    HullWhite class

    A HullWhite process is a stochastic process that satisfies the following equation:
                                 dR_t = [calibration(t) + mean * R_t]dt + volatility(t)*dW_t,

    Parameters
    ----------
    mean : float, or np.ndarray
        Long term mean value of the model.
    calibration : float or function
        Calibration function of the model
    volatility : float or function
        Volatility function of the model
    initial : float
        Initial condition of the model.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mean : float, or np.ndarray
        Long term mean value of the model.
    calibration : float or function
        Calibration function of the model
    volatility : float or function
        Volatility function of the model
    initial : float
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
    name : str
        Name of the process
    is_autonomous : bool
        Specify if the process SDE is autonomous.

    Examples
    --------
    >> R = HullWhite(speed=2,mean=3,volatility=1,initial=0,t_0=0,t_n=1,steps=1000)
    >> R.simulate()
    >> R.plot()
    """

    def __init__(self,
                 mean= 1,
                 calibration=lambda t : t,
                 volatility=lambda t : 1,
                 initial=0,
                 t_0=0,
                 t_n=1,
                 steps=1000):

        super().__init__(t_0=t_0,
                         t_n=t_n,
                         steps=steps)

        self.name = "HullWhite process"

        if not isinstance(mean, (int, float)):
            raise ValueError(
                "The mean parameter should be a real number."
            )

        if isinstance(calibration, (int, float)):
            calibration = lambda t, c = calibration : c
        elif not callable(calibration):
            raise ValueError(
                "The calibration parameter should be a real number or a function."
            )
        if isinstance(volatility, (int, float)):
            volatility = lambda t, v=volatility: v
        elif not callable(volatility):
            raise ValueError(
                "The volatility parameter should be a real number or a function."
            )
        if not isinstance(initial, (int, float)):
            raise ValueError(
                "The initial value should be a real number"
            )

        self.mean = mean
        self.calibration = calibration
        self.volatility = volatility
        self.initial = initial
        self.dim = 1
        self.is_autonomous = False

    def drift(self,x,t=None):

        """
        Drift function

        Evaluate the drift of the HullWhite process at a given point x and time t.

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

        return self.calibration(t) - (self.mean * x)

    def diffusion(self,x,t=None):

        """
        Diffusion function

        Evaluate the diffusion of the HullWhite process at a given point x and time t.

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

        return self.volatility(t)

    def _simulate_exact(self, n_simulations=1,plot=False):
        raise NotImplementedError(
            "The exact method is not implemented for this process yet."
        )

    def expectation(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the expectation of the Hull-White process."
        )

    def covariance_matrix(self, t):
        pass

    def covariance(self, t,i,j):
        pass

    def variance(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the variance of the Hull-White process."
        )