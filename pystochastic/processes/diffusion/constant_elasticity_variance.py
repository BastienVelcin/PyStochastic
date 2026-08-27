"""
============================================================
Module CONSTANT ELASTICITY OF VARIANCE
============================================================

Description
-----------
This module provides a way to simulate a Constant elasticity of variance process with a given speed, volatility and elasticity parameters.

This module provides a general class "CEV", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> C = CEV(speed=3,volatility=2.2,elasticity=0.5,initial=12,T=1,steps=1000) #CEV process with speed 3, volatility 2.2, elasticity 0.5 and starting point 12.
>>
>> C.simulate() #Simulate the CEV process path
>>
>> C.plot() #Plot the CEV process path
"""

import numpy as np
from pystochastic.processes.diffusion.diffusion_process import DiffusionProcess
from pystochastic.random.setseed import *

class CEV(DiffusionProcess):

    """
    CEV class

    An CEV process is a stochastic process that satisfies the following equation:
            dR_t = - speed*dt+volatility * R_t^elasticity *dW_t,

    Parameters
    ----------
    speed : float
        Constant diffusion of the model.
    variance : float
        Constant variance parameter.
    elasticity : float
        Constant elasticity parameter between the price and the volatility.
    initial : float
        Initial condition of the model.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    speed : float
        Constant diffusion of the model.
    variance : float
        Constant variance parameter.
    elasticity : float
        Constant elasticity parameter between the price and the volatility.
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
    >> C = CEV(speed=3,volatility=2.2,elasticity=0.5,initial=12,T=1,steps=1000)
    >> C.simulate()
    >> C.plot()
    """

    def __init__(self,
                 speed=1,
                 volatility=1,
                 elasticity=1,
                 initial=1,
                 T = 1,
                 steps=1000):

        super().__init__(T = T,
                         steps=steps)

        self.name = "CEV process"

        if not isinstance(speed, (int, float)):
            raise ValueError(
                "The speed parameter should be a real number."
            )
        if not isinstance(volatility, (int, float)) or elasticity < 0:
            raise ValueError(
                "The speed parameter should be a positive real number."
            )
        if not isinstance(elasticity, (int, float)) or elasticity < 0:
            raise ValueError(
                "The elasticity parameter should be a positive real number."
            )

        self.speed = speed
        self.volatility = volatility
        self.elasticity = elasticity
        self.initial = initial
        self.dim = 1
        self.is_autonomous = True

    def drift(self, x, t=None):

        """
        Drift function

        Evaluate the drift of the CEV process at a given point x and time t.

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

        return self.speed * x


    def diffusion(self, x, t=None):

        """
        Diffusion function

        Evaluate the diffusion of the CEV process at a given point x and time t.

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

        return self.volatility * np.power(x,self.elasticity)

    def _simulate_exact(self, n_simulations=1, plot=False):

        raise NotImplementedError(
            "The exact method is not implemented for this process yet."
        )


    def expectation(self,t):

        """
        Expectation method.

        Return the expectation of the CEV process at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated. Must be between 0 and T.

        Returns
        -------
        float
            Expectation of the CEV process at a time t

        Notes
        -----
        The expectation of the CEV process at every time t with a fixed initial is given by
                            initial * exp(-speed*t) + mean * (Id - exp(-volatility*t))
        """

        return self.initial * np.exp(self.speed * t)

    def covariance_matrix(self, t):
        pass

    def covariance(self, t,i,j):
        pass

    def variance(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the variance of the CEV process."
        )