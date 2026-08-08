import numpy as np
from sde.eulermaruyama import EulerMaruyama
import pyrandom.crandom
import plotly.graph_objects as go
from pyrandom.crandom import exponential

class Poisson:
    def __init__(self,intensity=1,t_0=0, t_n=10, n_steps=1000, n_simulations=1):
        self.intensity = intensity
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.path = None

    def simulate(self):
        self.path = np.zeros((self.n_simulations,self.n_steps+1))
        for sim in range(self.n_simulations):
            T = [0]
            while T[-1] < self.t_n:
                E = exponential(self.intensity).item()
                T.append(T[-1] + E)
            T = np.array(T)
            for i in range(self.n_steps+1):
                self.path[sim,i]= sum(T<= self.t[i])
        return self.path

    def plot(self):
        if self.path is None:
            raise ValueError("The path has not been simulated yet. Please run the simulate method first.")
        fig = go.Figure()
        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t, y=self.path[sim,:], mode="lines", line=dict(width=2,shape="hv")))
        fig.show()