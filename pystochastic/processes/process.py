import numpy as np
import plotly.graph_objects as go
from abc import abstractmethod, ABC

class Process(ABC):

    def __init__(
            self,
            t_0=0,
            t_n=1,
            steps=1000):

        self.t_0 = t_0
        self.t_n = t_n
        self.steps = steps
        self.n_simulations = None
        self.path = None

        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        if steps <= 0:
            raise ValueError(
                "The number of steps must be strictly positive."
            )

        self.t = np.linspace(self.t_0, self.t_n, self.steps + 1)
        self.dt = (t_n - t_0) / steps

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def expectation(self,t):

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                f"The time must be between {self.t_0} and {self.t_n}."
            )

        pass

    @abstractmethod
    def covariance_matrix(self,t):

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                f"The time must be between {self.t_0} and {self.t_n}."
            )

        pass

    @abstractmethod
    def covariance(self,t,i,j):

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                f"The time must be between {self.t_0} and {self.t_n}."
            )

        if not 0 <= i < self.dim:
            raise ValueError(
                "The first coordinate must be between 0 and the dimension (excluded)."
            )

        if not 0 <= j < self.dim:
            raise ValueError(
                "The second coordinate must be between 0 and the dimension (excluded)."
            )
        pass

    @abstractmethod
    def variance(self,t):

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                f"The time must be between {self.t_0} and {self.t_n}."
            )

        pass

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Geometric Brownian Motion. The path can be plotted only in 1D, 2D or 3D.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim > 3:
            raise ValueError(
                "The path can be plotted only for 1D, 2D and 3D."
            )

        fig = go.Figure()

        if self.dim == 1:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.t,
                                         y=self.path[sim,:, 0],
                                         mode="lines",
                                         line=dict(width=2)))

        elif self.dim == 2:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.path[sim,:, 0],
                                         y=self.path[sim, :, 1],
                                         mode="lines",
                                         line=dict(width=2)))

        else:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter3d(x=self.path[sim,:, 0],
                                           y=self.path[sim,:, 1],
                                           z=self.path[sim,:, 2],
                                           mode="lines",
                                           line=dict(width=2)))

        fig.show()

    def final_position(self):

        """
        Final position method

        The function returns the state of the Brownian motion at the final time t_n.

        Returns
        -------
        float or np.ndarray
            State of the Brownian motion at the final time t_n.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        return self.path[:, -1]

    def max(self):
        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim != 1:
            raise ValueError(
                "The maximum value can only be computed in 1D. Please refer to the max_norm function."
            )

        argmax = np.argmax(self.path, axis=1)

        # RETURN FORM: (max_value, time_index, time)
        return np.max(self.path, axis=1), argmax, self.t[argmax]

    def min(self):

        """
        Min method

        The min method returns the minimum value of the Brownian motion, the time index at which the minimum value
        occurs, and the corresponding time.

        Returns
        -------
        tuple
            Value of the minimum value, time index at which the minimum occurs, and corresponding time.

        Notes
        -----
        This method can only be used in 1D since the notion of minimum value is not defined in higher dimensions.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim != 1:
            raise ValueError(
                "The minimum value can only be computed in 1D."
            )

        argmin = np.argmin(self.path, axis=1)

        # RETURN FORM: (max_value, time_index, time)
        return np.min(self.path, axis=1), argmin, self.t[argmin]


    def max_norm(self):

        """
        Max norm method

        The max norm method returns the maximum of the norm value of the Brownian motion, the time index at which the maximum norm value
        occurs, and the corresponding time.

        Returns
        -------
        tuple
            Value of the maximum norm value, time index at which the maximum norm occurs, and corresponding time.

        Notes
        -----
        The norm used in this function is the Euclidean norm.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        norms = np.sum(self.path**2, axis=2)
        arg_max = np.argmax(norms, axis=1)
        values = (self.path[np.arange(0,self.n_simulations),arg_max,:])
        return values, norms[arg_max], arg_max, self.t[arg_max]
