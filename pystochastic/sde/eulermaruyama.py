"""
============================================================
Module EULER MARUYAMA
============================================================

Description
-----------
This module provides a way to approximatively solve multidimensional SDEs using the Euler-Maruyama method.

This module provides a general class "EulerMaruyama", which is an SDEs solver, built with the following methods:
    - .solve() : SDEs solving function, with the Euler-Maruyama induction formula.

The EulerMaruyama class also handles the solution plotting.

Examples
--------
>> solver = EulerMaruyama(mu=lambda x,t : x, sigma=lambda x,t : 0.1*x, x_0=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10) #Euler-Maruyama SDE with drift x(t) = x and diffusion 0.1*x(t)
>>
>> solver.solve(plot=True) #Plot the evolution of the SDE
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import *
from pystochastic.utils import default_drift, default_diffusion
import multiprocessing
from multiprocessing import Pool

class EulerMaruyama:

    """
    Euler-Maruyama method for SDEs.

    The Euler-Maruyama method is a numerical method for solving stochastic differential equations (SDEs) of the form
                                        dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t,
    where X_t is a random variable, W_t is a Standard Wiener process, mu(X_t,t) is the drift function, and sigma(X_t,t)
    is the diffusion function.
    Note that the differentiation is taken in the sense of Ito.

    Parameters
    ----------
    mu : function of two arguments
        Drift function of the SDE.
    sigma : function of two arguments
        Diffusion function of the SDE.
    x_0 : float, or list, or np.ndarray
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
    mu : function
        Drift function of the SDE.
    sigma : function
        Diffusion function of the SDE.
    x_0 : float, or list, or np.ndarray
        Initial condition.
    dim : int
        Dimension of the SDE coefficients.
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    n_steps : int
        Number of time steps.
    dt : float
        Time step.
    n_simulations : int
        Number of solution simulations.

    Examples
    --------
    >> solver = EulerMaruyama(mu=lambda x,t : x, sigma=lambda x,t : 0.1*x, x_0=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10)
    >> solver.solve(plot=True)
    """

    def __init__(self,
                 mu=default_drift,
                 sigma=default_diffusion,
                 x_0=0,
                 t_0=0,
                 t_n=1,
                 n_steps=1000,
                 n_simulations=1):

        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        self.mu = mu
        self.sigma = sigma
        self.x_0 = np.atleast_1d(x_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.dim = np.size(x_0)
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps

    def solve(self, plot=True):

        """
        Solve method.

        Solve the SDE using the Euler-Maruyama method.

        Parameters
        ----------
        plot : bool
            Specifies whether to plot the evolution of the SDE.

        Returns
        -------
        np.ndarray
            Array of every simulated path of the approximation of the SDE solution.
        """

        # Initialization of the 'n_simulations' simulations.
        Y = np.zeros((self.n_simulations,self.n_steps+1,self.dim))

        # Fixing the initial condition on every simulation.
        Y[:,0,:] = self.x_0

        fig = go.Figure()

        W = Brownian(np.eye(self.dim), self.t_0, self.t_n, self.n_steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        for sim in range(self.n_simulations):
            # For every simulation, we compute a different Brownian increment array.
            for i in range(1,self.n_steps+1):
                # Euler-Maruyama induction formula.
                Y[sim,i,:] = Y[sim,i-1,:] + self.mu(Y[sim,i-1,:],self.t[i-1])*self.dt + self.sigma(Y[sim,i-1,:],self.t[i-1]) @ dW[sim,i-1,:]
        # Plotting is allowed only for 1D, 2D and 3D and if the user has specified the plot parameter to True.

        if plot == True and self.dim <= 3:
            for sim in range(self.n_simulations):
                if self.dim == 1:
                    fig.add_trace(go.Scatter(x=self.t,
                                             y=Y[sim,:,0],
                                             mode="lines",
                                             line=dict(width=2)))

                elif self.dim == 2:
                    fig.add_trace(go.Scatter(x=Y[sim,:, 0],
                                             y=Y[sim,:, 1],
                                             mode="lines",
                                             line=dict(width=2)))
                else:
                    fig.add_trace(go.Scatter3d(x=Y[sim,:, 0],
                                               y=Y[sim,:, 1],
                                               z=Y[sim,:, 2],
                                               mode="lines",
                                               line=dict(width=2)))

            fig.show()

        elif plot == True and self.dim > 3:
                raise ValueError(
                    "The path can be plotted only for 1D, 2D and 3D."
                )

        return Y
