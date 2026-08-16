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
The modules provides some other functions, that are used within the .solve() method, to either solve with vectorization
or parallelization the SDE.

Examples
--------
>> solver = EulerMaruyama(mu=lambda x,t : x, sigma=lambda x,t : 0.1*x, x_0=1, t_0=0, t_n=1, n_steps=1000, n_simulations=10)
>> solver.solve(plot=True)
>>
>> solver = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: 0.3, n_simulations=1000)
>> Y = solver.solve(plot=False)
>>
>> solver = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: np.array([[0.3]]))
>> Y = solver.solve(plot=False, parallel=True)
"""

import numpy as np
import plotly.graph_objects as go
from multiprocess import Pool, cpu_count
from pystochastic.processes import Brownian
from pystochastic.utils import default_drift, default_diffusion, _decompose


def _is_vectorizable_diffusion(f, x_0, t_0):

    """
    Is Diagonal Form function

    Tests if f(x_0,t_0) returns a scalar or a vector (diagonal diffusion,
    vectorisable on all simulations at once), or a full (dim,dim) matrix, which requires
    a separate simulation for each step.

    Parameters
    ----------
    f : function
        Function on which we want to test the vectorization capacity.
    x_0 : float, list, or np.ndarray
        Point at which we want to test the vectorization capacity.
    t : float
        Time instant at which we want to test the vectorization capacity.

    Returns
    -------
    bool :
        True if f(x_0,t_0) returns a scalar or a vector, False otherwise.
    """

    test = np.asarray(f(x_0, t_0))
    return test.ndim <= 1


def _simulate_vectorized(mu, sigma, x_0, t, dt, n_steps, dim, dW):

    """
    Simulate Vectorized function.

    If the diffusion is diagonal (sigma returns a scalar or a vector), all simulations are
    treated together at each time step, via numpy vectorization.

    Parameters
    ----------
    mu : function
        Drift function of the SDE.
    sigma : function
        Diffusion function of the SDE.
    x_0 : float, list, or np.ndarray
        Initial condition of the SDE.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    dt : float
        Time step length.
    n_steps : int
        Number of time steps.
    dim : int
        Dimension of the SDE.
    dW : np.ndarray
        Brownian increments

    Returns
    -------
    np.ndarray
        Simulated path with vectorization method.
    """

    n_simulations = dW.shape[0]
    Y = np.zeros((n_simulations, n_steps + 1, dim))

    # Fixing the initial condition on every simulation.
    Y[:, 0, :] = x_0

    for i in range(1, n_steps + 1):

        x_prev = Y[:, i - 1, :]
        t_prev = t[i - 1]

        Y[:, i, :] = x_prev + mu(x_prev, t_prev) * dt + sigma(x_prev, t_prev) * dW[:, i - 1, :]

    return Y


def _simulate_sequential(mu, sigma, x_0, t, dt, n_steps, dim, dW):

    """
    Simulate Sequential function.

    If the diffusion is not diagonal, then we need to simulate each step separately with a for loop
    on simulations.

    Parameters
    ----------
    mu : function
        Drift function of the SDE.
    sigma : function
        Diffusion function of the SDE.
    x_0 : float, list, or np.ndarray
        Initial condition of the SDE.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    dt : float
        Time step length.
    n_steps : int
        Number of time steps.
    dim : int
        Dimension of the SDE.
    dW : np.ndarray
        Brownian increments

    Returns
    -------
    np.ndarray
        Simulated path with sequential method.
    """

    n_simulations = dW.shape[0]
    Y = np.zeros((n_simulations, n_steps + 1, dim))

    # Fixing the initial condition on every simulation.
    Y[:, 0, :] = x_0

    for sim in range(n_simulations):

        Y_sim = Y[sim]
        dW_sim = dW[sim]

        for i in range(1, n_steps + 1):

            x_prev = Y_sim[i - 1]
            t_prev = t[i - 1]

            # The Euler-Maruyama induction formula is given by : Y_{t_{i+1}}  = Y_{t_i} + mu(Y_{t_i},t_i)*dt + sigma(Y_{t_i},t_i)*dW_{t_i}
            Y_sim[i] = x_prev + mu(x_prev, t_prev) * dt + sigma(x_prev, t_prev) @ dW_sim[i - 1]
    return Y


def _simulate_sequential_args(args):
    """Wrapper used to parallelize the simulation of the Euler-Maruyama method."""
    return _simulate_sequential(*args)


class EulerMaruyama:

    """
    Euler-Maruyama method for SDEs.

    The Euler-Maruyama method is a numerical method for solving stochastic differential equations (SDEs) of the form
                                        dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t,
    where X_t is a random variable, W_t is a Standard Wiener process, mu(X_t,t) is the drift function, and sigma(X_t,t)
    is the diffusion function.
    Note that the differentiation is taken in the sense of Ito.

    Two internal simulation strategies, chosen automatically at solve() time:
      - "vectorized" : used when sigma(x,t) returns a scalar or a vector (diagonal
        diffusion, independent noise per dimension). All simulations are advanced
        together at each time step -- by far the fastest option (no dependency on
        the number of CPU cores).

      - "sequential" (optionally parallelized via multiprocess across simulations):
        used when sigma(x,t) returns a full (dim,dim) matrix (correlated noise
        across dimensions), which cannot be vectorized across simulations without
        changing the calling convention of sigma.

    Notice that the parallelization is usefull only for large values of number of simulations.

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
    >> solver = EulerMaruyama(mu=lambda x,t : x, sigma=lambda x,t : 0.1*x, x_0=1, t_0=0, t_n=1, n_steps=1000, n_simulations=10)
    >> solver.solve(plot=True)
    >>
    >> solver = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: 0.3, n_simulations=1000)
    >> Y = solver.solve(plot=False)
    >>
    >> solver = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: np.array([[0.3]]))
    >> Y = solver.solve(plot=False, parallel=True)
    """

    def __init__(self,
                 mu=default_drift,
                 sigma=default_diffusion,
                 x_0=0,
                 t_0=0,
                 t_n=1,
                 n_steps=1000,
                 n_simulations=100):

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
        self.t = np.linspace(t_0, t_n, n_steps + 1)
        self.dt = (t_n - t_0) / n_steps

    def solve(self, plot=True, parallel=False, n_workers=None):

        """
        Solve method.

        Solve the SDE using the Euler-Maruyama method.

        Parameters
        ----------
        plot : bool
            Specifies whether to plot the evolution of the SDE (only in 1D, 2D and 3D).
        parallel : bool
            Only relevant when sigma(x,t) returns a full (dim,dim) matrix (non-diagonal
            diffusion). Ignored when the diffusion is diagonal: the vectorized path is already
            faster than any parallelization could offer.
        n_workers : int
            Number of worker processes. Defaults to the number of available cores.

        Returns
        -------
        np.ndarray
            Simulated path of shape (n_simulations, n_steps+1, dim).
        """

        # We compute the different brownian path and increments that will be used in the simulation.
        W = Brownian(np.eye(self.dim), self.t_0, self.t_n, self.n_steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        # We define the list of arguments that will be passed to the simulation functions.
        args = (self.mu, self.sigma, self.x_0, self.t, self.dt, self.n_steps, self.dim, dW)

        #If the diffusion is diagonal, we use the vectorized method. Otherwise, we use the sequential method.
        if _is_vectorizable_diffusion(self.sigma, self.x_0, self.t_0):
            Y = _simulate_vectorized(*args)

        # SEQUENTIAL METHOD :

        # If the user wants to parallelize the simulation, we use the Pool class from the multiprocessing module.
        # Otherwise, we use a for loop on simulations.
        elif not parallel:
            Y = _simulate_sequential(*args)

        else:
            if n_workers is None:
                #By default, the number of workers is equal to the number of available CPU cores.
                n_workers = cpu_count()

            # We split the increments into chunks, and pass each chunk to a worker process.
            chunks = np.array_split(dW, n_workers, axis=0)

            worker_args = [(self.mu, self.sigma, self.x_0, self.t, self.dt,
                             self.n_steps, self.dim, chunk) for chunk in chunks]

            # We launch the simulations in parallel, and wait for the results.
            with Pool(n_workers) as pool:
                results = pool.map(_simulate_sequential_args, worker_args)

            # We concatenate the results of the simulations of every pools.
            Y = np.concatenate(results, axis=0)

        if plot:
            if self.dim > 3:
                raise ValueError(
                    "The path can be plotted only for 1D, 2D and 3D."
                )

            fig = go.Figure()

            for sim in range(self.n_simulations):

                if self.dim == 1:
                    fig.add_trace(go.Scatter(x=self.t,
                                             y=Y[sim, :, 0],
                                             mode="lines",
                                             line=dict(width=2)))

                elif self.dim == 2:
                    fig.add_trace(go.Scatter(x=Y[sim, :, 0],
                                             y=Y[sim, :, 1],
                                             mode="lines",
                                             line=dict(width=2)))

                else:
                    fig.add_trace(go.Scatter3d(x=Y[sim, :, 0],
                                               y=Y[sim, :, 1],
                                               z=Y[sim, :, 2],
                                               mode="lines",
                                               line=dict(width=2)))

            fig.show()

        return Y