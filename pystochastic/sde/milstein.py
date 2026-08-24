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
>> solver = Milstein(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10) #Milstein SDE with drift x(t) = x and diffusion 0.1*x(t)
>>
>> solver.solve(plot=True) #Plot the evolution of the SDE
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian
from pystochastic.utils import default_drift, default_diffusion

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
        Drift function of the SDE.
    diffusion : function of one argument
        Diffusion function of the SDE.
    initial : float, or list, or np.ndarray
        Initial condition.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
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
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    n_steps : int
        Number of time steps.
    dt : float
        Time step

    Examples
    --------
    >> solver = Milstein(drift=lambda x : x, diffusion=lambda x : 0.1*x, initial=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10)
    >> solver.solve(plot=True)

    """

    def __init__(self,
                 drift=lambda x : 1,diffusion=lambda x : 1,
                 initial=0,
                 t_0=0,
                 t_n=1,
                 n_steps=1000):

        if np.size(initial) != 1:
            raise NotImplementedError(
                "Milstein is currently implemented only for autonomous one-dimensional SDEs."
            )
        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        initial = np.atleast_1d(initial)
        self.drift = drift
        self.diffusion = diffusion
        self.initial = initial
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps

    def approx_derivative_diffusion(self,x, eps=1e-6):

        """
        Approx derivative diffusion method.

        Compute an approximation of the derivative of the diffusion function at a given point numerically, to avoid
        symbolic computations.

        Parameters
        ----------
        eps : float
            Specifies the gap between diffusion(x) and diffusion(x+eps).

        Returns
        -------
        float or np.ndarray
            Approximation of the derivative of the diffusion function at x.
        """

        # Rate of change of the diffusion function at x, with a small gap eps.
        return (self.diffusion(x + eps)- self.diffusion(x - eps)) / (2 * eps)

    def solve(self, n_simulations=1, plot=True):

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
            :param n_simulations:
        """

        if n_simulations < 1 or not isinstance(n_simulations, int):
            raise ValueError(
                "The number of simulations must be a strictly positive integer."
            )

        self.n_simulations = n_simulations

        # Initialization of the 'n_simulations' simulations.
        Y = np.zeros((self.n_simulations,self.n_steps+1,1))

        # Fixing the initial condition on every simulation.
        Y[:,0,:] = self.initial

        fig = go.Figure()

        W = Brownian(1, self.t_0, self.t_n, self.n_steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        for i in range(1,self.n_steps+1):
            # Milstein induction formula.
            Y[:,i,:] = Y[:,i-1,:] + self.drift(Y[:,i-1,:])*self.dt + self.diffusion(Y[:,i-1,:]) * dW[:,i-1,:] + (1/2)*self.diffusion(Y[:,i-1,:])*self.approx_derivative_diffusion(Y[:,i-1,:])*(dW[:,i-1,:]**2-self.dt)

        # Plotting is allowed only if the user has specified the plot parameter to True.

        if plot == True:
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