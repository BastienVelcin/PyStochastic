import plotly.graph_objects as go
from abc import abstractmethod, ABC
from pystochastic.processes.process import Process

class JumpProcess(Process,ABC):

    @abstractmethod
    def simulate(self,n_simulations=1,plot=False):
        pass

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Poisson process.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        fig = go.Figure()

        for sim in range(self.n_simulations):
            fig.add_trace(go.Scatter(x=self.t,
                                     y=self.path[sim,:],
                                     mode="lines",
                                     line=dict(width=2,shape="hv"),
                                     name=f"Path {sim+1}"))
        fig.show()

    def covariance(self, t, i, j):
        pass

    def covariance_matrix(self, t):
        pass