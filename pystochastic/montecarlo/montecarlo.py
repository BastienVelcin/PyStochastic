import numpy as np
import plotly.graph_objects as go
import pystochastic.processes as processes
from pystochastic.processes import *
from scipy.stats import norm
from functools import partial


class MonteCarloEstimator:
    def __init__(self,samples,n_simulations=None):
        if n_simulations is None:
            n_simulations = len(samples)
        if n_simulations <= 1:
            raise ValueError("n_simulations cannot be less than or equal to 1.")
        if n_simulations > len(samples):
            raise ValueError("n_simulations cannot be greater than the number of samples provided.")
        self.samples = np.asarray(samples).flatten()
        self.n_simulations = n_simulations


    def estimate(self, n=None, function = lambda x: x):
        if n is None:
            n = self.n_simulations
        return np.mean(function(self.samples[:n]), axis=0)

    def mean_estimator(self, n= None, confidence = 0.95):
        if n is None:
            n = self.n_simulations
        mean_est = self.estimate(n)
        sd_estimate = np.std(self.samples[:n], axis=0)
        z = norm.ppf(0.5 + confidence / 2)
        half_width = z * sd_estimate / np.sqrt(n)
        return mean_est, half_width

    def confidence_interval(self, n = None,confidence = 0.95):
        if n is None:
            n = self.n_simulations
        mean_est, half_width = self.mean_estimator(n,confidence)
        return mean_est - half_width, mean_est + half_width

    def confidence_curve(self,n=None,confidence = 0.95):
        if n is None:
            n = self.n_simulations
        n_axis = np.arange(1, n + 1)
        S1 = np.cumsum(self.samples[:n])
        S2 = np.cumsum(self.samples[:n] ** 2)
        cum_mean = S1 / n_axis
        with np.errstate(invalid="ignore", divide="ignore"):
            cum_var = (S2 - S1 ** 2 / n_axis) / (n_axis - 1)
        cum_var = np.nan_to_num(cum_var, nan=0.0)
        z = norm.ppf(0.5 + confidence / 2)
        half_width = z * np.sqrt(cum_var) / np.sqrt(n_axis)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([n_axis, n_axis[::-1]]),
            y=np.concatenate([cum_mean + half_width, (cum_mean - half_width)[::-1]]),
            fill="toself", fillcolor="rgba(100,149,237,0.2)",
            line=dict(width=0), name=f"CI {int(confidence * 100)}%", showlegend=True,
        ))
        fig.add_trace(go.Scatter(x=n_axis, y=cum_mean, mode="lines", name="Cumulative estimator"))
        fig.show()

class MonteCarloProcess:
    def __init__(self,process,n_simulations=10000):
        self.process = process
        self.n_simulations = n_simulations
        self.t = self.process.t
        self.ech = self.process.simulate(self.n_simulations)

    def simulate(self):
        self.ech = self.process.simulate(self.n_simulations)
        return self.ech

    def estimate(self, t_0=None, function = lambda x: x[:,0]):
        if t_0 is None:
            t_0 = self.process.t_n
        if t_0 not in self.t:
            t_index = np.argmin(np.abs(t_0 - self.t))
        else:
            t_index = np.where(self.t == t_0)[0][0]
        return np.mean(function(self.ech[:,t_index]), axis=0)


    def mean_path(self, plot_sim=True):
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

    def values_at(self, t_0=None, function=lambda x: x[:,0]):
        if t_0 is None:
            t_0 = self.process.t_n
        if t_0 not in self.t:
            t_index = np.argmin(np.abs(t_0 - self.t))
        else:
            t_index = np.where(self.t == t_0)[0][0]
        if self.ech is None:
            self.simulate()
        return function(self.ech[:,t_index])
