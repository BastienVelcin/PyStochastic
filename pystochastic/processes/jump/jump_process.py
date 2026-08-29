import plotly.graph_objects as go
import numpy as np
from abc import abstractmethod, ABC
from pystochastic.processes.process import Process

class JumpProcess(Process,ABC):

    @abstractmethod
    def simulate(self,n_simulations=1,plot=False):
        pass

    def plot(self, density=True):

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim > 3:
            raise ValueError(
                "The path can be plotted only for 1D, 2D and 3D."
            )

        paths = self.path[:, :, 0]

        # Mean
        mean_path = np.mean(paths, axis=0)

        if density:
            from plotly.subplots import make_subplots
            # Figure
            fig = make_subplots(rows=1,
                                cols=2,
                                shared_yaxes=True,
                                horizontal_spacing=0.02,
                                column_widths=[0.8, 0.2],
                                subplot_titles=("Simulated paths", "Distribution"))

            # Simulated paths
            for sim in range(self.n_simulations):
                fig.add_trace(
                    go.Scatter(x=self.t,
                               y=paths[sim],
                               mode="lines",
                               line=dict(width=1, shape="hv"),
                               name=f"Path {sim + 1}",
                               opacity=0.5,
                               showlegend=False),
                    row=1,
                    col=1)

            # Mean path
            fig.add_trace(
                go.Scatter(x=self.t,
                           y=mean_path,
                           mode="lines",
                           line=dict(width=3),
                           name="Mean path"),
                row=1,
                col=1)

            initial_values = paths[:, -1]

            fig.add_trace(go.Histogram(y=initial_values, orientation="h",
                                       nbinsy=30,
                                       name="Histogram",
                                       histnorm="probability density",
                                       marker=dict(color="rgba(59, 130, 246, 0.65)"),
                                       showlegend=False),
                          row=1,
                          col=2)

            # If the density is available for the effective process, we will plot it on the histogram

            density_available = True

            # We want to know if the density is known for the process.
            try:
                x_density = np.linspace(np.min(initial_values), np.max(initial_values), 300)

                density_values = self.density(self.t[-1], x_density)

            except (AttributeError, NotImplementedError, ValueError):
                density_available = False

            if density_available:
                fig.add_trace(go.Scatter(x=density_values,
                                         y=x_density,
                                         mode="lines",
                                         line=dict(width=3),
                                         name="Density",
                                         marker=dict(color='#ff4b7d'),
                                         howlegend=True),
                              row=1,
                              col=2)

            frames = []

            for i, t in enumerate(self.t):

                values = paths[:, i]

                frame_data = [go.Histogram(y=values,
                                           orientation="h",
                                           nbinsy=30,
                                           histnorm="probability density",
                                           marker=dict(color="rgba(59, 130, 246, 0.65)"))]

                if density_available:
                    x_density = np.linspace(np.min(values), np.max(values), 300)

                    density_values = self.density(t, x_density)

                    frame_data.append(go.Scatter(x=density_values,
                                                 y=x_density,
                                                 mode="lines",
                                                 line=dict(width=3),
                                                 name="Density"))
                frames.append(go.Frame(name=str(i),
                                       data=frame_data,
                                       traces=([self.n_simulations + 1] if not density_available else [
                                           self.n_simulations + 1, self.n_simulations + 2])))

            fig.frames = frames

            sliders = [
                {"active": self.t.size - 1,
                 "currentvalue": {
                     "prefix": "t = ",
                     "visible": True
                 },
                 "pad": {"t": 50},
                 "steps": [{"label": f"{t:.2f}",
                            "method": "animate",
                            "args": [[str(i)],
                                     {"mode": "immediate",
                                      "frame": {
                                          "duration": 0,
                                          "redraw": True
                                      },
                                      "transition": {
                                          "duration": 0
                                      }
                                      }
                                     ]
                            }
                           for i, t in enumerate(self.t)
                           ]
                 }
            ]

            fig.update_layout(title=f"Simulations of {self.name}",
                              template="plotly_white",
                              sliders=sliders,
                              xaxis_title="t",
                              xaxis2_title="Frequency",
                              yaxis_title=self.name,
                              bargap=0.05)

            # Make the histogram appear horizontally
            fig.update_xaxes(autorange=True,
                             row=1,
                             col=2)
            fig.show()

        else:

            fig = go.Figure()

            for sim in range(self.n_simulations):
                fig.add_trace(
                    go.Scatter(
                        x=self.t,
                        y=self.path[sim, :, 0],
                        mode="lines",
                        line=dict(width=2),
                        name=f"Path {sim + 1}"
                    )
                )

            fig.update_layout(
                title=f"Simulations of {self.name}",
                scene=dict(
                    xaxis_title="t",
                    yaxis_title=self.name,
                ),
                template="plotly_white"
            )
            fig.show()

    def covariance(self, t, i, j):
        pass

    def covariance_matrix(self, t):
        pass