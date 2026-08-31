"""
============================================================
Module RUNGE-KUTTA
============================================================

Description
-----------
This module provides a way to approximatively solve unidimensional and autonomous SDEs using the Runge-Kutta method.

This module provides a general class "RungeKutta", which is an SDEs solver, with the following built-in methods:
    - .solve() : SDEs solving function, with the Milstein induction formula.

The Runge-Kutta class also handles the solution plotting.

Examples
--------
>> solver = RungeKutta(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, T=1, steps=1000, n_simulations=10) #Runge-Kutta SDE with drift x(t) = x and diffusion 0.1*x(t)
>>
>> solver.solve(plot=True) #Plot the evolution of the SDE
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian
from pystochastic.utils import default_drift, default_diffusion

class RungeKutta:

    """
    Runge-Kutta method for SDEs.

    The Runge-Kutta method is a numerical method for solving autonomous stochastic differential equations (ASDEs) of the form
                                        dX_t = drift(X_t)dt + diffusion(X_t)dW_t,
    where X_t is a random variable, W_t is a Standard Wiener process, drift(X_t) is the drift function, and diffusion(X_t)
    is the diffusion function.
    Note that the differentiation is taken in the sense of Ito.

    Parameters
    ----------
    drift : function of one argument
        Drift function of the SDE. The calculations need to be batch-compatible with numpy.
    diffusion : function of one argument
        Diffusion function of the SDE. The calculations need to be batch-compatible with numpy.
    initial : float, or list, or np.ndarray
        Initial condition.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.
    n_simulations : int
        Number of solution simulations. Must be a strictly positive integer.

    Attributes
    ----------
    drift : function
        Drift function of the SDE.
    diffusion : function
        Diffusion function of the SDE.
    initial : float, list, or np.ndarray
        Initial condition.
    T : float
        Final time.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    steps : int
        Number of time steps.
    dt : float
        Time step

    Examples
    --------
    >> solver = RungeKutta(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, T=1, steps=1000, n_simulations=10)
    >> solver.solve(plot=True)

    """

    def __init__(self,
                 drift=lambda x : 1,diffusion=lambda x : 1,
                 initial=0,
                 T = 1,
                 steps=1000):

        if not isinstance(initial, (int, float, np.integer, np.floating, np.ndarray)):
            raise NotImplementedError(
                "Milstein is currently implemented only for autonomous one-dimensional SDEs. Please specify the initial condition as a number."
            )

        if not 0 < T or not isinstance(steps, (int, np.integer, float, np.floating)):
            raise ValueError(
                "The final time must be a strictly positive number."
            )

        if steps <= 0 or not isinstance(steps, (int, np.integer)):
            raise ValueError(
                "The number of steps must be a strictly positive integer."
            )

        initial = np.atleast_1d(initial)
        self.drift = drift
        self.diffusion = diffusion
        self.initial = initial
        self.T = T
        self.steps = steps
        self.n_simulations = None
        self.t = np.linspace(0,T,steps+1)
        self.dt = T/steps
        self.dim = 1

    def solve(self, n_simulations = 1, plot=True):

        """
        Solve method.

        Solve the SDE using the Runge-Kutta method.

        Parameters
        ----------
        n_simulations : int
            Number of solution simulations. Must be a strictly positive integer.
        plot : bool
            Specifies whether to plot the evolution of the SDE.

        Returns
        -------
        np.ndarray
            Array of every simulated path of the approximation of the SDE solution.
        """

        if n_simulations < 1 or not isinstance(n_simulations, (int, np.integer)):
            raise ValueError(
                "The number of simulations must be a strictly positive integer."
            )

        self.n_simulations = n_simulations

        # Initialization of the 'n_simulations' simulations.
        Y = np.zeros((self.n_simulations,self.steps+1,1))

        # Fixing the initial condition on every simulation.
        Y[:,0,:] = self.initial

        fig = go.Figure()

        W = Brownian(1, self.T, self.steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        for i in range(1,self.steps+1):
            # Runge-Kutta induction formula.
            Y[:,i,:] = Y[:,i-1,:] + self.drift(Y[:,i-1,:])*self.dt + self.diffusion(Y[:,i-1,:]) * dW[:,i-1,:] + (1/2)*(self.diffusion(Y[:,i-1,:] + self.drift(Y[:,i-1,:])*self.dt+self.diffusion(Y[:,i-1,:])*np.power(self.dt,1/2)) - self.diffusion(Y[:,i-1,:]))*(dW[:,i-1,:]**2-self.dt)*np.power(self.dt,-1/2)

        # Plotting is allowed only if the user has specified the plot parameter to True.

        if plot:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=Y[sim,:,0],
                                         mode="lines",
                                         line=dict(width=2),
                                         name=f"Path {sim+1}"))
            fig.update_layout(
                title=f"Simulation with Runge-Kutta method.",
                xaxis_title="t",
                template="plotly_white",
            )
            fig.show()

        return Y