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
>> solver = Milstein(mu=lambda x : x, sigma=lambda x : 0.1*x, x_0=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10) #Milstein SDE with drift x(t) = x and diffusion 0.1*x(t)
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
                                        dX_t = mu(X_t)dt + sigma(X_t)dW_t,
    where X_t is a random variable, W_t is a Standard Wiener process, mu(X_t,t) is the drift function, and sigma(X_t,t)
    is the diffusion function.
    Note that the differentiation is taken in the sense of Ito.

    Parameters
    ----------
    mu : function of one argument
        Drift function of the SDE.
    sigma : function of one argument
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
    x_0 : float, list, or np.ndarray
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
    n_simulations : int
        Number of solution simulations.

    Examples
    --------
    >> solver = Milstein(mu=lambda x : x, sigma=lambda x : 0.1*x, x_0=0, t_0=0, t_n=1, n_steps=1000, n_simulations=10)
    >> solver.solve(plot=True)

    """

    def __init__(self,
                 mu=default_drift,sigma=default_diffusion,
                 x_0=0,
                 t_0=0,
                 t_n=1,
                 n_steps=1000,
                 n_simulations=100):

        if np.size(x_0) != 1:
            raise NotImplementedError(
                "Milstein is currently implemented only for autonomous one-dimensional SDEs."
            )
        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        x_0 = np.atleast_1d(x_0)
        self.mu = mu
        self.sigma = sigma
        self.x_0 = x_0
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
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
            Specifies the gap between sigma(x) and sigma(x+eps).

        Returns
        -------
        float or np.ndarray
            Approximation of the derivative of the diffusion function at x.
        """

        # Rate of change of the diffusion function at x, with a small gap eps.
        return (self.sigma(x + eps)- self.sigma(x - eps)) / (2 * eps)

    def solve(self, plot=True):

        """
        Solve method.

        Solve the SDE using the Milstein method.

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
        Y = np.zeros((self.n_simulations,self.n_steps+1,1))

        # Fixing the initial condition on every simulation.
        Y[:,0,:] = self.x_0

        fig = go.Figure()

        W = Brownian(1, self.t_0, self.t_n, self.n_steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        for i in range(1,self.n_steps+1):
            # Milstein induction formula.
            Y[:,i,:] = Y[:,i-1,:] + self.mu(Y[:,i-1,:])*self.dt + self.sigma(Y[:,i-1,:]) * dW[:,i-1,:] + (1/2)*self.sigma(Y[:,i-1,:])*self.approx_derivative_diffusion(Y[:,i-1,:])*(dW[:,i-1,:]**2-self.dt)

        # Plotting is allowed only if the user has specified the plot parameter to True.

        if plot == True:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=Y[sim,:,0],
                                         mode="lines",
                                         line=dict(width=2)))
            fig.show()
        return Y