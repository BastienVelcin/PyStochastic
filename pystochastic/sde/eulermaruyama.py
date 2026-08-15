"""
============================================================
Module EULER MARUYAMA
============================================================
"""

import numpy as np
import plotly.graph_objects as go
from multiprocess import Pool, cpu_count
from pystochastic.processes import Brownian
from pystochastic.utils import default_drift, default_diffusion


def _is_diagonal_form(sigma, x_0, t_0):
    """
    Test si sigma(x_0,t_0) renvoie un scalaire/vecteur (diffusion diagonale,
    vectorisable sur toutes les simulations d'un coup) ou une vraie matrice 2D
    (diffusion pleine/correlee, qui necessite un traitement par simulation).
    """
    test = np.asarray(sigma(x_0, t_0))
    return test.ndim <= 1


def _simulate_vectorized(mu, sigma, x_0, t, dt, n_steps, dim, dW):
    """
    Chemin rapide : diffusion diagonale (sigma renvoie un scalaire/vecteur).
    Une seule boucle, sur le temps -- toutes les simulations sont traitees
    d'un coup a chaque pas, via des operations numpy vectorisees.
    """
    n_simulations = dW.shape[0]
    Y = np.zeros((n_simulations, n_steps + 1, dim))
    Y[:, 0, :] = x_0
    for i in range(1, n_steps + 1):
        x_prev = Y[:, i - 1, :]
        t_prev = t[i - 1]
        Y[:, i, :] = x_prev + mu(x_prev, t_prev) * dt + sigma(x_prev, t_prev) * dW[:, i - 1, :]
    return Y


def _simulate_sequential(mu, sigma, x_0, t, dt, n_steps, dim, dW):
    """
    Chemin general : diffusion matricielle pleine (bruit correle entre dimensions).
    Boucle par simulation, necessaire car sigma(x,t) attend un etat (dim,) et
    renvoie une matrice (dim,dim) combinee via '@'.
    Fonction module-level (et non une methode) pour rester picklable par multiprocess.
    """
    n_simulations = dW.shape[0]
    Y = np.zeros((n_simulations, n_steps + 1, dim))
    Y[:, 0, :] = x_0
    for sim in range(n_simulations):
        Y_sim = Y[sim]
        dW_sim = dW[sim]
        for i in range(1, n_steps + 1):
            x_prev = Y_sim[i - 1]
            t_prev = t[i - 1]
            Y_sim[i] = x_prev + mu(x_prev, t_prev) * dt + sigma(x_prev, t_prev) @ dW_sim[i - 1]
    return Y


def _simulate_sequential_args(args):
    """Wrapper qui deballe un tuple d'arguments -- pratique pour Pool.map."""
    return _simulate_sequential(*args)


class EulerMaruyama:
    """
    Euler-Maruyama method for SDEs.

    Solves dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t.

    Two internal simulation strategies, chosen automatically at solve() time:
      - "vectorized" : used when sigma(x,t) returns a scalar or a vector (diagonal
        diffusion, independent noise per dimension). All simulations are advanced
        together at each time step -- by far the fastest option (no dependency on
        the number of CPU cores).
      - "sequential" (optionally parallelized via multiprocess across simulations)
        : used when sigma(x,t) returns a full (dim,dim) matrix (correlated noise
        across dimensions), which cannot be vectorized across simulations without
        changing the calling convention of sigma.

    In both cases, mu and sigma keep their original signature: mu(x,t) -> (dim,)
    (or broadcastable), sigma(x,t) -> scalar/(dim,) for the diagonal case, or
    (dim,dim) for the full case.

    Examples
    --------
    >> solver = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: 0.3, n_simulations=1000)
    >> Y = solver.solve(plot=False)                    # chemin vectorise automatique
    >>
    >> solver2 = EulerMaruyama(mu=lambda x,t: -x, sigma=lambda x,t: np.array([[0.3]]))
    >> Y2 = solver2.solve(plot=False, parallel=True)    # chemin sequentiel + multiprocess
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
            raise ValueError("The final time must be strictly greater than the initial time.")

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

        Parameters
        ----------
        plot : bool
            Whether to plot the simulated paths (1D, 2D or 3D only).
        parallel : bool
            Only relevant when sigma(x,t) returns a full (dim,dim) matrix (non-diagonal
            diffusion). If True, distributes the n_simulations trajectories across
            several worker processes (multiprocess, dill-based -- supports lambdas).
            Ignored when the diffusion is diagonal: the vectorized path is already
            faster than any parallelization could offer.
        n_workers : int, optional
            Number of worker processes. Defaults to the number of available cores.

        Returns
        -------
        np.ndarray
            Array of shape (n_simulations, n_steps+1, dim).
        """
        W = Brownian(np.eye(self.dim), self.t_0, self.t_n, self.n_steps)
        W.simulate(self.n_simulations)
        dW = W.increments

        args = (self.mu, self.sigma, self.x_0, self.t, self.dt, self.n_steps, self.dim, dW)

        if _is_diagonal_form(self.sigma, self.x_0, self.t_0):
            Y = _simulate_vectorized(*args)
        elif not parallel:
            Y = _simulate_sequential(*args)
        else:
            if n_workers is None:
                n_workers = cpu_count()
            chunks = np.array_split(dW, n_workers, axis=0)
            worker_args = [(self.mu, self.sigma, self.x_0, self.t, self.dt,
                             self.n_steps, self.dim, chunk) for chunk in chunks]
            with Pool(n_workers) as pool:
                results = pool.map(_simulate_sequential_args, worker_args)
            Y = np.concatenate(results, axis=0)

        if plot:
            if self.dim > 3:
                raise ValueError("The path can be plotted only for 1D, 2D and 3D.")
            fig = go.Figure()
            for sim in range(self.n_simulations):
                if self.dim == 1:
                    fig.add_trace(go.Scatter(x=self.t, y=Y[sim, :, 0], mode="lines", line=dict(width=2)))
                elif self.dim == 2:
                    fig.add_trace(go.Scatter(x=Y[sim, :, 0], y=Y[sim, :, 1], mode="lines", line=dict(width=2)))
                else:
                    fig.add_trace(go.Scatter3d(x=Y[sim, :, 0], y=Y[sim, :, 1], z=Y[sim, :, 2],
                                                mode="lines", line=dict(width=2)))
            fig.show()

        return Y