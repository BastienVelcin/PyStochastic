import numpy as np
import plotly.graph_objects as go
import pystochastic.processes as processes

class MonteCarlo:
    def __init__(self,process,n_simulations=10000):
        self.process = process
        self.n_simulations = n_simulations

    def estimate(self, function = lambda x: x):
        return np.mean([function(self.process.simulate()) for _ in range(self.n_simulations)])

    def mean_path(self):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.process.t,y=self.process.path))
        return fig