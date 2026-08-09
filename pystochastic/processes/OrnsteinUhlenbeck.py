import numpy as np
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom
from pystochastic.sde import EulerMaruyama

class OrnsteinUhlenbeck:
    def __init__(self,mean=0,sigma=1,theta=1,r_0=0,t_0=0, t_n=1, n_steps=1000,n_simulations=1):
        self.mean = np.atleast_1d(mean)
        self.sigma = np.atleast_2d(sigma)
        self.theta = np.atleast_2d(theta)

        if np.any(self.sigma < 0) or np.any(self.theta <=0):
            raise ValueError("The sigma and theta parameters should be greater than 0.")

        self.r_0 = np.atleast_1d(r_0)
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.dim = np.size(self.mean)
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.dt = (t_n-t_0)/n_steps
        self.path = None

        if not(np.shape(self.sigma)[0] == np.shape(self.sigma)[1] == self.dim == np.shape(self.theta)[0] == np.shape(self.theta)[1] ==  self.r_0.size):
            raise ValueError("The dimension of the the mean, signa, theta, and of the starting point must coincide.")

    def drift(self,x,t):
        return self.theta @ (self.mean-x)

    def diffusion(self,x,t):
        return self.sigma

    def simulate(self, method="euler-maruyama"):
        if method == "euler-maruyama":
            self.path = EulerMaruyama(self.drift,self.diffusion,self.r_0,self.t_0,self.t_n,self.n_steps,self.n_simulations).solve()
        elif method == "exact":
            if self.dim > 1:
                raise ValueError("The exact method is only implemented for 1D processes.")
            self.path = np.zeros((self.n_simulations,self.n_steps+1, 1))
            self.path[:,0] = self.r_0
            for sim in range(self.n_simulations):
                Z = crandom.normal(0, 1, self.n_steps)
                for i in range(1,self.n_steps+1):
                    self.path[sim,i] = (self.mean+ (self.path[sim,i-1] - self.mean) * np.exp(-self.theta * self.dt) + self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * self.dt)) / (2 * self.theta)) * Z[i-1])
        else:
            raise ValueError("The method must be either 'euler-maruyama' or 'exact'.")
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
                fig.add_trace(go.Scatter(x=self.path[sim,:, 0], y=self.path[sim,:, 1], mode="lines", line=dict(width=2)))

        else:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter3d(x=self.path[sim,:, 0], y=self.path[sim,:, 1], z=self.path[sim,:, 2], mode="lines", line=dict(width=2)))
        fig.show()