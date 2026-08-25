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

    @property
    def dt(self):
        return (self.t_n - self.t_0) / self.steps

    @property
    def t(self):
        return np.linspace(self.t_0, self.t_n, self.steps + 1)

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

        Plot the simulated path of the process. The path can be plotted only in 1D, 2D or 3D.
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
                                         line=dict(width=2),
                                         name=f"Path {sim+1}"))

        elif self.dim == 2:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter(x=self.path[sim,:, 0],
                                         y=self.path[sim, :, 1],
                                         mode="lines",
                                         line=dict(width=2),
                                         name=f"Path {sim+1}"))

        else:
            for sim in range(self.n_simulations):
                fig.add_trace(go.Scatter3d(x=self.path[sim,:, 0],
                                           y=self.path[sim,:, 1],
                                           z=self.path[sim,:, 2],
                                           mode="lines",
                                           line=dict(width=2),
                                           name=f"Path {sim+1}"))

        fig.update_layout(title=f"Simulations of {self.name}",
                          xaxis_title="t",
                          yaxis_title="Quadratic Variation",
                          template="plotly_white")
        fig.show()

    def final_position(self):

        """
        Final position method

        The function returns the state of the process at the final time t_n.

        Returns
        -------
        float or np.ndarray
            State of the process at the final time t_n.
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

        The min method returns the minimum value of the process, the time index at which the minimum value
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

        The max norm method returns the maximum of the norm value of the process, the time index at which the maximum norm value
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

    def hitting_time(self,value, inequality=">"):

        """
        Hitting time method

        The hitting time method returns an approximation of the first time at which the process reaches a given value.
        The methods support both the inequality ">" and "<".

        Parameters
        ----------
        value: float
            Value at which we want to get the hitting time.
        inequality : {"<", ">"}
            Direction of the inequality

        Returns
        -------
        np.ndarray
            Estimation of the hitting time for each simulation.
        """

        if self.dim != 1:
            raise ValueError(
                "The hitting time can only be computed in 1D."
            )

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if inequality not in ["<", ">"]:
            raise ValueError(
                "The inequality must be either '<' or '>'"
            )
        candidates = self.path >= value if inequality == ">" else self.path <= value
        return np.where(candidates.any(axis=1),candidates.argmax(axis=1), None)

    def hitting_norm_time(self,value, inequality=">", order=2):

        """
        Hitting norm time method

        The hitting norm time method returns an approximation of the first time at which the process reaches a given norm.
        The methods support both the inequality ">" and "<".

        Parameters
        ----------
        value: float
            Value at which we want to get the hitting time.
        inequality: {"<", ">"}
            Direction of the inequality
        order: int
            Order of the desired norm. If order = p, then || (x_1 , ... x,n) || = (x_1^p + ... x_n^p)^(1/p).
            Must be a strictly positive integer.

        Returns
        -------
        np.ndarray
            Estimation of the hitting norm time for each simulation.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if inequality not in ["<", ">"]:
            raise ValueError(
                "The inequality must be either '<' or '>'"
            )

        if not isinstance(order, int) and not order >= 1:
            raise ValueError(
                "The order must be a strictly positive integer."
            )

        candidates = np.linalg.norm(self.path,axis=2,ord=order) >= value if inequality == ">" else np.linalg.norm(self.path,axis=2,ord=order) <= value
        return np.where(candidates.any(axis=1),candidates.argmax(axis=1), None)

    def _quad_var_at_index(self, t_index):
        if t_index == 0:
            return 0
        quad_var_t = np.sum((self.path[:, 1:t_index+1] - self.path[:, 0:t_index]) ** 2, axis=1)
        return quad_var_t

    def quadratic_variation(self,t=None, mean = False, plot=False):

        """
        Quadratic Variation method

        The quadratic variation method returns an approximation of the first time at which the process reaches a given value.
        The methods support both the inequality ">" and "<".

        Parameters
        ----------
        value: float
            Value at which we want to get the hitting time.
        mean: bool
            Specify if the quadratic variation should be computed at the mean of the quadratic variation estimation of all simulated processes.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Estimation of the hitting time for each simulation, or average quadratic variation estimation.
        """

        if self.dim != 1:
            raise ValueError(
                "The quadratic variation can only be computed in 1D."
            )

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if t is None:
            quad_var = np.zeros((self.n_simulations, self.steps+1))
            for i in range(1,self.steps+1):
                quad_var[:,i] = self._quad_var_at_index(i).reshape((self.n_simulations,))
            if plot:
                fig = go.Figure()
                if not mean:
                    for sim in range(self.n_simulations):
                        fig.add_trace(go.Scatter(x=self.t,
                                                 y=quad_var[sim,:],
                                                 mode="lines",
                                                 line=dict(width=2),
                                                 name=f"Path {sim+1}"))


                else:
                    quad_var = np.mean(quad_var, axis=0)
                    fig.add_trace(go.Scatter(x=self.t,
                                             y=quad_var,
                                             mode="lines",
                                             line=dict(width=2),
                                             name=f"Mean Quadratic Variation"))

                fig.update_layout(title=f"Estimation of the Quadratic Variation for {self.name}",
                                  xaxis_title="t",
                                  yaxis_title="Quadratic Variation",
                                  template="plotly_white")
                fig.show()
        else:
            t_index = np.argmin(np.abs(self.t - t))
            quad_var = self._quad_var_at_index(t_index)
            if mean:
                quad_var = np.mean(quad_var)
        return quad_var

