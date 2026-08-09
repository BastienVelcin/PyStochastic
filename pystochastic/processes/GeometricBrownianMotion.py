import numpy as np
import plotly.graph_objects as go
from pystochastic.processes.brownian import Brownian

class GeometricBrownianMotion:
    '''
    This class provide a way to simulate a Geometric Brownian Motion of the following form :
    dX_t = diag(X_t)(mu*dt+sigma*dW_t)

    Parameters :
    - mu : scalar or vectorial drift function of the form mu(x,t) = f(x,t)
    - sigma : scalar or vectorial diffusion function of the form sigma(x,t) = g(x,t)
    - x_0 : initial value (line vector value)
    - t_0 : initial time
    - t_n : final time
    - n_steps : number of time steps
    - n_simulations : number of simulations
    '''
    def __init__(self, mu=1, sigma=1, S_0=1,t_0=0, t_n=1, n_steps=1000):

        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_2d(sigma)
        self.S_0 = np.atleast_1d(S_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.dim = np.size(S_0)
        self.t = np.linspace(t_0, t_n, n_steps+1)
        self.path = None
        if not(np.shape(self.sigma)[0] == np.shape(self.sigma)[1] == self.dim == np.size(self.mu)):
            raise ValueError("The dimension of the volatility matrix, of the drift and of the starting point must coincide.")

    def drift(self, x, t):
        return self.mu

    def diffusion(self, x, t):
        return self.sigma

    def simulate(self,n_simulations=1, method="exact"):
        if method == "euler-maruyama":
            from pystochastic.sde import EulerMaruyama
            self.path = EulerMaruyama(self.drift, self.diffusion, self.S_0, self.t_0, self.t_n, self.n_steps, self.n_simulations).solve()
        elif method == "exact":
            self.path = np.zeros((n_simulations,self.n_steps+1, self.dim))
            for sim in range(n_simulations):
                W = Brownian(np.eye(self.dim), self.dim, self.t_0, self.t_n, self.n_steps)
                for i in range(0, self.n_steps+1):
                    self.path[sim,i, :] = self.S_0 * np.exp(
                        (self.mu - np.sum(self.sigma ** 2, axis=1) / 2) * self.t[i] + self.sigma   @ W.path[i,:])
        else:
            raise ValueError("The method must be either 'euler-maruyama' or 'exact'.")
        self.n_simulations = n_simulations
        return self.path

    def plot(self):
        if self.dim > 3:
            raise ValueError("The path can be plotted only for 1D, 2D and 3D.")
        if self.path is None:
            raise ValueError("The path has not been simulated yet. Please run the simulate method first.")
        fig = go.Figure()

        if self.dim == 1:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t, y=self.path[sim,:, 0], mode="lines", line=dict(width=2)))

        elif self.dim == 2:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.path[sim,:, 0], y=self.path[sim, :, 1], mode="lines", line=dict(width=2)))

        else:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter3d(x=self.path[sim,:, 0], y=self.path[sim,:, 1], z=self.path[sim,:, 2], mode="lines", line=dict(width=2)))

        fig.show()
