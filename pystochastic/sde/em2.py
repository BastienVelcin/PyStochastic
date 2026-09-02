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
>> solver = EulerMaruyama(drift=lambda x,t : x, diffusion=lambda x,t : 0.1*x, initial=1, T=1, steps=1000)
>> solver.solve(plot=True)
>>
>> solver = EulerMaruyama(drift=lambda x,t: -x, diffusion=lambda x,t: 0.3)
>> Y = solver.solve(plot=False)
>>
>> solver = EulerMaruyama(drift=lambda x,t: -x, diffusion=lambda x,t: np.array([[0.3]]))
>> Y = solver.solve(plot=False, parallel=True)
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian
import jax.numpy as jnp
import jax

class EulerMaruyamaJAX:

    """
    Euler-Maruyama method for SDEs.

    The Euler-Maruyama method is a numerical method for solving stochastic differential equations (SDEs) of the form
                                        dX_t = drift(X_t,t)dt + diffusion(X_t,t)dW_t,
    where X_t is a random variable, W_t is a Standard Wiener process, drift(X_t,t) is the drift function, and diffusion(X_t,t)
    is the diffusion function.
    Note that the differentiation is taken in the sense of Ito.

    Two internal simulation strategies, chosen automatically at solve() time:
      - "vectorized" : used when diffusion(x,t) returns a scalar or a vector (diagonal
        diffusion, independent noise per dimension). All simulations are advanced
        together at each time step -- by far the fastest option (no dependency on
        the number of CPU cores).

      - "sequential" (optionally parallelized via multiprocess across simulations):
        used when diffusion(x,t) returns a full (dim,dim) matrix (correlated noise
        across dimensions), which cannot be vectorized across simulations without
        changing the calling convention of diffusion.

    Notice that the parallelization is usefull only for large values of number of simulations.

    Parameters
    ----------
    drift : function of two arguments
        Drift function of the SDE.
    diffusion : function of two arguments
        Diffusion function of the SDE.
    initial : float, or list, or np.ndarray
        Initial condition.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    drift : function
        Drift function of the SDE.
    diffusion : function
        Diffusion function of the SDE.
    initial : float, or list, or np.ndarray
        Initial condition.
    dim : int
        Dimension of the SDE coefficients.
    T : float
        Final time.
    t : np.ndarray
        Time interval on which we want to solve the SDE.
    steps : int
        Number of time steps.
    dt : float
        Time step.
    n_simulations : int
        Number of solution simulations.

    Examples
    --------
    >> solver = EulerMaruyamaJAX(drift=lambda x,t : x, diffusion=lambda x,t : 0.1*x, initial=1, T=1, steps=1000)
    >> solver.solve(plot=True)
    >>
    >> solver = EulerMaruyamaJAX(drift=lambda x,t: -x, diffusion=lambda x,t: 0.3)
    >> Y = solver.solve(plot=False)
    >>
    >> solver = EulerMaruyamaJAX(drift=lambda x,t: -x, diffusion=lambda x,t: np.array([[0.3]]))
    >> Y = solver.solve(plot=False, parallel=True)
    """

    def __init__(self, drift, diffusion, initial, T=1, steps=1000):
        self.drift = drift
        self.diffusion = diffusion
        self.initial = np.atleast_1d(initial)
        self.T = T
        self.steps = steps
        self.t = jnp.linspace(0, T, steps + 1)
        self.dt = self.T / self.steps
        self.dim = self.initial.shape[0]
        self.n_simulations = None

    def solve(self, n_simulations, plot = True, brownian_increments=None):

        self.n_simulations = n_simulations

        # We compute the brownian increments.
        if brownian_increments is None:
            W = Brownian(np.eye(self.dim), self.T, self.steps)
            W.simulate(self.n_simulations)
            dW = W.increments

        else:
            if not brownian_increments.shape == (self.n_simulations, self.steps, self.dim):
                raise ValueError(
                    "The brownian sequence must have the same shape as the number of simulations."
                )
            dW = brownian_increments

        # We convert it into a JaxNumPy Array

        dW = jnp.array(dW)
        sim, steps, dim = dW.shape

        # To facilitate the computations with jax, we need to swap the steps and the dimension
        dW_steps = jnp.swapaxes(dW, 0, 1)  # (steps, sim, dim)

        # We initialize the vectors of the initial step.
        x0 = self.initial
        x0_batch = jnp.repeat(x0[None, :], sim, axis=0)  # (sim, dim)
        x0_batch = jnp.array(x0_batch, dtype=jnp.float32)

        # We convert the drift and diffusion to
        drift_batch = jax.vmap(self.drift, in_axes=(0, None))
        diffusion_batch = jax.vmap(self.diffusion, in_axes=(0, None))

        # We create the loop core for 1D SDE into a function step_1d :
        def step_1d(x_prev, inputs):

            dW_i, t_prev = inputs

            # We compute the drift and diffusion at the previous time step with batched functions.
            dr = drift_batch(x_prev, t_prev)
            diff = diffusion_batch(x_prev, t_prev)

            # We add a dimension to the diffusion to be able to compute the increment.
            diff = diff.reshape(sim, 1)
            # We compute the Euler-Maruyama induction formula.
            x_next = x_prev + dr * self.dt + diff * dW_i

            #We return the next state 2 times to use the jax.lax.scan
            return x_next, x_next

        # We create the loop core for multidimensional SDE into a function step_md :
        def step_md(x_prev, inputs):
            dW_i, t_prev = inputs
            # We compute the drift and diffusion at
            # the previous time step with batched functions.

            dr = drift_batch(x_prev, t_prev)
            diff = diffusion_batch(x_prev, t_prev)

            # We compute the Euler-Maruyama induction formula.
            incr = jnp.einsum("sij,sj->si", diff, dW_i)
            x_next = x_prev + dr * self.dt + incr

            # We return the next state 2 times to use the jax.lax.scan
            return x_next, x_next

        # We choose the right loop depending on the dimension of the SDE
        dim_step = step_1d if dim == 1 else step_md


        def run(dW_steps):

            # We select the increments used for the EM computation and the time values at which the
            # increments were computed.
            inputs = (dW_steps, self.t[:-1])

            # We compute the chosen step for each time values (and so, each brownian increments)
            _, X = jax.lax.scan(dim_step, x0_batch, inputs)

            # We concatenate the initial condition and the computed path.
            X_full = jnp.concatenate([x0_batch[None, :, :], X], axis=0)

            # We retranspose the path to have the shape (simulations, steps, dim)
            return jnp.swapaxes(X_full, 0, 1)

        # We run the computation with jax.jit for better performance.
        run_jit = jax.jit(run)

        Y = np.array(run_jit(dW_steps))

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
                                                 line=dict(width=2),
                                                 name=f"Path {sim+1}"))

                elif self.dim == 2:
                    fig.add_trace(go.Scatter(x=Y[sim, :, 0],
                                                 y=Y[sim, :, 1],
                                                 mode="lines",
                                                 line=dict(width=2),
                                                 name=f"Path {sim+1}"))

                else:
                    fig.add_trace(go.Scatter3d(x=Y[sim, :, 0],
                                                   y=Y[sim, :, 1],
                                                   z=Y[sim, :, 2],
                                                   mode="lines",
                                                   line=dict(width=2),
                                                   name=f"Path {sim+1}"))

                fig.update_layout(
                    title=f"Simulation with Euler-Maruyama method.",
                    xaxis_title="t",
                    template="plotly_white",
                )

            fig.show()
        return Y
