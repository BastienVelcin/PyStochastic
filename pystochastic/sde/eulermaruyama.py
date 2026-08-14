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

Notes
-----
On the euler-maruyama method, the first argument of the drift and diffusion functions is a matrix, which contains a certain point at each simulation.
Then
    - drift(x,t) --> shape (d,) #Vector of length d, where d is the dimension of the SDE
    - diffusion(x,t) --> shape (d,d)  #Matrix of size d*d, where d is the dimension of the SDE

where x.shape = (n_simulations,dim)

"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import *
from pystochastic.utils import default_drift, default_diffusion


def batchify(fct, x, t, output_dim):

    """
    Batchify function.

    Returns a version of ``fct`` that accepts a batch of states. The function to bash takes two arguments :

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional array of shape (n_simulations, dim), which contains the list of all
        points to apply the function to.
    t : float
        Time at which the function is evaluated.
    output_dim : int
        Dimension of the output of the function.

    Returns
    -------
        Batched function
    ""


    """

    # Check if the output of a function has a meaning
    if output_dim < 1:
        raise ValueError("The state dimension must be positive.")


    x = np.asarray(x)

    if x.ndim != 1 or x.size != output_dim:
        raise ValueError("x must be a one-dimensional state of output_dim elements.")


    # Use three rows: using two rows can hide functions that accidentally
    # index the two batch rows as if they were state coordinates.
    x_batch = np.stack([x, x, x])

    try:
        result = np.asarray(fct(x_batch, t))
    except (ValueError, TypeError, IndexError):
        result = None

    if result is not None:

        # Drift: one state vector per batch element.
        if result.shape == (x_batch.shape[0], output_dim):
            return fct

        # Diffusion: one (state_dimension, noise_dimension) matrix per
        # batch element.  The noise dimension need not equal state_dim.
        if (result.ndim == 3 and result.shape[0] == x_batch.shape[0]
                and result.shape[1] == output_dim):
            return fct

    # The function is single-state.  Probe one state to distinguish a drift
    # vector from a diffusion matrix when the batched call was rejected.
    try:
        single_result = np.asarray(fct(x, t))
    except (ValueError, TypeError, IndexError):
        single_result = None

    # np.vectorize applies the single-state function independently to each
    # simulation while retaining its vector/matrix output shape.
    if (single_result is not None and single_result.ndim == 2
            and single_result.shape[0] == output_dim):
        return np.vectorize(fct, signature='(d),()->(d,m)')
    return np.vectorize(fct, signature='(d),()->(d)')

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

        self._mu = batchify(mu, self.x_0, self.t, self.dim)
        self._sigma = batchify(sigma, self.x_0, self.t, self.dim)

    def solve(self, plot=True, parallel=False , n_threads=None):

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

        for i in range(1,self.n_steps+1):

            y_prev = Y[:, i - 1, :]
            t_prev = self.t[i - 1]

            drift = self._mu(y_prev, t_prev) * self.dt
            diffusion = (self._sigma(y_prev, t_prev) @ dW[:, i - 1, :, None]).squeeze(-1)

            Y[:, i, :] = y_prev + drift + diffusion

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

