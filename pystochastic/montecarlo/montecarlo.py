import numpy as np
import plotly.graph_objects as go
import pystochastic.processes as processes
from pystochastic.processes import *

class MonteCarlo:
    def __init__(self,process,n_simulations=10000):
        self.process = process
        self.n_simulations = n_simulations
        self.t = self.process.t
        self.ech = None

    def simulate(self):
        self.ech = self.process.simulate(self.n_simulations)
        return self.ech

    def estimate(self, t_0=None, function = lambda x: x):
        if self.ech is None:
            self.simulate()
        if t_0 is None:
            t_0 = self.process.t_n
        if t_0 not in self.t:
            t_index = np.argmin(np.abs(t_0 - self.t))
        else:
            t_index = np.where(self.t == t_0)[0][0]
        return np.mean(self.ech[:,t_index], axis=0)


    def mean_path(self, plot_sim=True):
        if self.ech is None:
            self.simulate()

        meanpath = np.mean(self.ech,axis=0)
        fig = go.Figure()

        if self.process.dim == 1:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter(x=self.t, y=self.ech[sim, :, 0], mode="lines", line=dict(width=1,color="#D1D9ED")))
            fig.add_trace(go.Scatter(x=self.t, y=meanpath[:, 0], mode="lines", line=dict(width=2)))

        elif self.process.dim == 2:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(
                        go.Scatter(x=self.ech[sim, :, 0], y=self.ech[sim, :, 1], mode="lines", line=dict(width=1,color="#D1D9ED")))
            fig.add_trace(go.Scatter(x=meanpath[:, 0], y=meanpath[:, 1], mode="lines", line=dict(width=2)))
        elif self.process.dim ==3:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter3d(x=self.ech[sim, :, 0], y=self.ech[sim, :, 1], z=self.ech[sim, :, 2], mode="lines", line=dict(width=1,color="#D1D9ED")))
            fig.add_trace(go.Scatter3d(x=meanpath[:, 0], y=meanpath[:, 1], z=meanpath[:, 2], mode="lines", line=dict(width=2)))
        fig.show()

        return meanpath