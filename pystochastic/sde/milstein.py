"""
============================================================
Module MILSTEIN
============================================================

Description
-----------
This module provides a way to approximatively solve unidimensional and autonomous SDEs using the Milstein method.

This module provides a general class "Milstein", which is an SDEs solver, with the following built-in methods:
    - .approx_derivative_diffusion() : Approximation of the derivative of the diffusion function at a given point.
    - .solve() : SDEs solving function, with the Milstein induction formula.

The Milstein class also handles the solution plotting.

Examples
--------
>> solver = Milstein(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, T=1, steps=1000, n_simulations=10) #Milstein SDE with drift x(t) = x and diffusion 0.1*x(t)
>>
>> solver.solve(plot=True) #Plot the evolution of the SDE
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian

class Milstein:

    """
    Milstein method for SDEs.

    The Milstein method is a numerical method for solving autonomous stochastic differential equations (ASDEs) of the form
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
    >> solver = Milstein(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, T=1, steps=1000, n_simulations=10)
    >> solver.solve(plot=True)

    """

    def __init__(self,
                 drift=lambda x : 1,diffusion=lambda x : 1,
                 initial=0,
                 T=1,
                 steps=1000):

        if not isinstance(initial, (int, float, np.integer, np.floating)):
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

    def approx_derivative_diffusion(self,x, eps=1e-6):

        """
        Approx derivative diffusion method.

        Compute an approximation of the derivative of the diffusion function at a given point numerically to avoid
        symbolic computations.

        Parameters
        ----------
        x : float or np.ndarray
            Point at which we want to approximate the derivative of the diffusion function.
        eps : float
            Specifies the gap between diffusion(x) and diffusion(x+eps).
        plot : bool
            Specifies whether to plot the evolution of the SDE.
        brownian_increments : np.ndarray or None
            Brownian increments used in the Milstein computation. If None, Brownian increments are computed.
        diffusion_derivative : function or float
            Derivative of the diffusion function. If None, the derivative is approximated numerically.

        Returns
        -------
        float or np.ndarray
            Approximation of the derivative of the diffusion function at x.
        """

        if eps <= 0 and not isinstance(eps, (int, np.integer, float, np.floating)):
            raise ValueError(
                "The gap between diffusion(x) and diffusion(x+eps) must be strictly positive."
            )
        # Rate of change of the diffusion function at x, with a small gap eps.
        return (self.diffusion(x + eps)- self.diffusion(x - eps)) / (2 * eps)

    def solve(self, n_simulations=1, plot=True, brownian_increments=None, diffusion_derivative=None):

        """
        Solve method.

        Solve the SDE using the Milstein method.

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

        if brownian_increments is None:
            W = Brownian(1, self.T, self.steps)
            W.simulate(self.n_simulations)
            dW = W.increments
        else:
            if not brownian_increments.shape == (self.n_simulations, self.steps, 1) or not brownian_increments == (self.n_simulations, self.steps):
                raise ValueError(
                    "The brownian sequence must have the same shape as the number of simulations."
                )
            dW = np.atleast_3d(brownian_increments)

        W = Brownian(1, self.T, self.steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        if diffusion_derivative is None:
            for i in range(1,self.steps+1):
                # Milstein induction formula.
                Y[:,i,:] = Y[:,i-1,:] + self.drift(Y[:,i-1,:])*self.dt + self.diffusion(Y[:,i-1,:]) * dW[:,i-1,:] + (1/2)*self.diffusion(Y[:,i-1,:])*self.approx_derivative_diffusion(Y[:,i-1,:])*(dW[:,i-1,:]**2-self.dt)
        else:
            for i in range(1, self.steps + 1):
                # Milstein induction formula.
                Y[:, i, :] = Y[:, i - 1, :] + self.drift(Y[:, i - 1, :]) * self.dt + self.diffusion(Y[:, i - 1, :]) * dW[:, i - 1, :] + (1 / 2) * self.diffusion(Y[:, i - 1, :]) * diffusion_derivative(Y[:, i - 1, :]) * (dW[:, i - 1, :] ** 2 - self.dt)
        # Plotting is allowed only if the user has specified the plot parameter to True.

        if plot:
            fig = go.Figure()
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=Y[sim,:,0],
                                         mode="lines",
                                         line=dict(width=2),
                                         name=f"Path {sim+1}"))
            fig.update_layout(
                title=f"Simulation with Milstein method.",
                xaxis_title="t",
                template="plotly_white",
            )
            fig.show()
        return Y