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

def batchify(fct, x, output_dim):

    """
    Batchify function.

    Returns a version of ``fct`` that accepts a batch of states. The function to bash takes two arguments :

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional array of shape (n_simulations, dim), which contains the list of all
        points to apply the function to.
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
        result = np.asarray(fct(x_batch))
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
        single_result = np.asarray(fct(x))
    except (ValueError, TypeError, IndexError):
        single_result = None

    # np.vectorize applies the single-state function independently to each
    # simulation while retaining its vector/matrix output shape.
    if (single_result is not None and single_result.ndim == 2
            and single_result.shape[0] == output_dim):
        return np.vectorize(fct, signature='(d)->(d,m)')
    return np.vectorize(fct, signature='(d)->(d)')

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
                 n_simulations=1):

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
        self.dim = np.size(x_0)

        # Milstein is currently one-dimensional. Normalize both scalar and
        # 1x1-array coefficient functions to a vector of shape ``(1,)``.
        def as_one_dimensional(coefficient):
            def normalized(x):
                value = np.asarray(coefficient(x))
                if value.size != 1:
                    raise ValueError(
                        "Milstein coefficients must return one value in 1D."
                    )
                return value.reshape(1)

            return np.vectorize(normalized, signature='(d)->(d)')

        self._mu = as_one_dimensional(mu)
        self._sigma = as_one_dimensional(sigma)

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
        x = np.asarray(x)
        return (
            self._sigma(x + eps) - self._sigma(x - eps)
        ) / (2 * eps)

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
            y_prev = Y[:, i - 1, :]
            dW_prev = dW[:, i - 1, :]

            sigma = self._sigma(y_prev)
            sigma_derivative = self.approx_derivative_diffusion(y_prev)

            drift = self._mu(y_prev) * self.dt
            diffusion = sigma * dW_prev
            correction = 0.5 * sigma * sigma_derivative * (
                dW_prev ** 2 - self.dt
            )

            Y[:, i, :] = y_prev + drift + diffusion + correction

        # Plotting is allowed only if the user has specified the plot parameter to True.
        if plot == True:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=Y[sim,:,0],
                                         mode="lines",
                                         line=dict(width=2)))
            fig.show()
        return Y
