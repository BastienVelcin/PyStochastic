import numpy as np
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom

class Poisson:
    def __init__(self,intensity=1,t_0=0, t_n=10, n_steps=1000):
        self.intensity = intensity
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.n_simulations = None
        self.t = np.linspace(t_0,t_n,n_steps+1)
        self.path = None

    def simulate(self,n_simulations=1):
        self.path = np.zeros((n_simulations,self.n_steps+1))
        for sim in range(n_simulations):
            T = [0]
            while T[-1] < self.t_n:
                E = crandom.exponential(self.intensity).item()
                T.append(T[-1] + E)
            for i in range(1,self.n_steps+1):
                self.path[sim,i]= sum(T<= self.t[i])

        self.n_simulations = n_simulations
        return self.path

    def plot(self):
        if self.path is None:
            raise ValueError("The path has not been simulated yet. Please run the simulate method first.")
        fig = go.Figure()
        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t, y=self.path[sim,:], mode="lines", line=dict(width=2,shape="hv")))
        fig.show()