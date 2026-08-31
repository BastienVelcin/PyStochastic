"""
============================================================
Module HullWhite
============================================================

Description
-----------
This module provides a way to simulate a HullWhite process with a given reverting speed, calibration and volatility parameters.

This module provides a general class "HullWhite", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> H = HullWhite(reversion_speed=2 , calibration = lambda t : t, volatility = lambda t : 1, initial = 1, T = 1, steps = 1000) #HullWhite process with reverting speed 2, calibration function and volatility function and starting point 1.
>>
>> H.simulate() #Simulate the HullWhite process path
>>
>> H.plot() #Plot the HullWhite process path
"""

import numpy as np
import scipy
import plotly.graph_objects as go
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.processes.process import _validate_t
from pystochastic.dist import Normal
from pystochastic.random import get_rng


class HullWhite(DiffusionProcess):

    """
    HullWhite class

    A HullWhite process is a stochastic process that satisfies the following equation:
                                 dR_t = [calibration(t) - reversion_speed * R_t]dt + volatility(t)*dW_t

    Parameters
    ----------
    reversion_speed : float, or np.ndarray
        Reverting speed value of the model.
    calibration : float or function
        Calibration function of the model
    volatility : float or function
        Volatility function of the model
    initial : float
        Initial condition of the model.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    reversion_speed : float, or np.ndarray
        Reverting speed value of the model.
    calibration : float or function
        Calibration function of the model
    volatility : float or function
        Volatility function of the model
    initial : float
        Initial condition of the model.
    T : float
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
    >> H = HullWhite(reversion_speed=2 , calibration = lambda t : t, volatility = lambda t : 1, initial = 1, T = 1, steps = 1000)
    >> H.simulate()
    >> H.plot()
    """

    def __init__(self,
                 reversion_speed= 1,
                 calibration=lambda t : t,
                 volatility=lambda t : 1,
                 initial=0,
                 T=1,
                 steps=1000):

        super().__init__(T=T,
                         steps=steps,
                         dim = 1,
                         name="Hull-White model")

        if not isinstance(reversion_speed, (int, float)):
            raise ValueError(
                "The reverting speed parameter should be a real number."
            )
        if not reversion_speed > 0:
            raise ValueError(
                "The reverting speed parameter should be strictly positive."
            )

        if isinstance(calibration, (int, float)):
            calibration = lambda t, c = calibration : c
        elif not callable(calibration):
            raise ValueError(
                "The calibration parameter should be a real number or a function."
            )
        if isinstance(volatility, (int, float)):
            if volatility <= 0:
                raise ValueError(
                    "The volatility parameter should be strictly positive."
                )
            volatility = lambda t, v=volatility: v
        elif not callable(volatility):
            raise ValueError(
                "The volatility parameter should be a real number or a function."
            )
        if not isinstance(initial, (int, float)):
            raise ValueError(
                "The initial value should be a real number"
            )

        self.reversion_speed = reversion_speed
        self.calibration = calibration
        self.volatility = volatility
        self.initial = initial
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

        return self.calibration(t) - (self.reversion_speed * x)

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

    def _simulate_exact(self,n_simulations=1,plot=False):
        self.path = np.zeros((n_simulations,self.steps+1,1))
        self.path[:,0,0] = self.initial

        #For a Hull White model, the distribution of X_{i+1}|X_i is a normal distribution with the mean and variance being given by the next equations:
        N = Normal(0, 1)
        for i in range(self.steps):
            mean = self.path[:, i, 0] * np.exp(-self.reversion_speed * self.dt)  + np.exp(-self.reversion_speed * self.t[i + 1]) * scipy.integrate.quad(lambda s: np.exp(self.reversion_speed * s) * self.calibration(s), self.t[i], self.t[i + 1])[0]

            variance = np.exp(-2 * self.reversion_speed * self.t[i + 1]) * scipy.integrate.quad(lambda s: np.exp(2 * self.reversion_speed * s) * self.volatility(s) ** 2, self.t[i], self.t[i + 1])[0]

            self.path [:,i+1,0] = N.sample(n_simulations)*np.sqrt(variance) + mean

        if plot:
            self.plot()
        return self.path

    def expectation(self,t):
        t = _validate_t(t)

        integral = scipy.integrate.quad(lambda s: np.exp(self.reversion_speed * s) * self.calibration(s), 0, t)[0]

        return np.exp(-self.reversion_speed * t) * (self.initial + integral)

    def variance(self,t):
        t = _validate_t(t)

        integral = scipy.integrate.quad(lambda s: np.exp(2 * self.reversion_speed * s) * self.volatility(s) ** 2, 0, t)[0]

        return np.exp(-2*self.reversion_speed*t)*integral
    def density(self,t,x):

        t = _validate_t(t)

        if t == 0:
            return np.array([0])
        x = np.asarray(x)
        int1 = scipy.integrate.quad(lambda s : np.exp(self.reversion_speed * s)*self.calibration(s), 0, t)[0]
        int2 = scipy.integrate.quad(lambda s : np.exp(2 * self.reversion_speed * s)*self.volatility(s)**2, 0, t)[0]
        N = Normal(mu = np.exp(-self.reversion_speed * t) * (self.initial + int1), var = np.exp(-2*self.reversion_speed*t)*int2)
        return N.pdf(x)
