import numpy as np
import plotly.graph_objects as go
from abc import abstractmethod, ABC
import pystochastic.processes.plot
def _validate_t(t):
    t = np.asarray(t)
    if not np.issubdtype(t.dtype, np.number):
        raise ValueError(
            f"The time must be a real number greater than 0."
        )
    if not np.all(0 <= t):
        raise ValueError(
            f"The time must be a real number greater than 0, but {t} was given."
        )
    return t

class Process(ABC):

    def __init__(
            self,
            T = 1,
            steps=1000,
            dim = 1,
            name = ""):

        if not isinstance(T, (int, np.integer, float, np.floating)) or not 0 < T :
            raise ValueError(
                "The final time must be a strictly positive number."
            )

        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError(
                "The number of steps must be strictly positive."
            )

        if not isinstance(dim, (int, np.integer)) or not dim >= 1:
            raise ValueError(
                "The dimension must be strictly positive."
            )
        self.T = T
    
        self.steps = steps
        self.n_simulations = None
        self.path = None
        self.dim = dim
        self.name = name



    @property
    def dt(self):
        return self.T / self.steps

    @property
    def t(self):
        return np.linspace(0, self.T, self.steps + 1)

    @abstractmethod
    def simulate(self):
        pass

    def plot(self, n_bins=30, points=1000, slider_steps=100):
        return pystochastic.processes.plot.plot(self, n_bins, points, slider_steps)


    def expectation(self,t):

        raise NotImplementedError(
            "There is no explicit formula for the expectation of this process yet."
        )

        pass

    def covariance_matrix(self,t):

        raise NotImplementedError(
            "There is no explicit formula for the covariance matrix of this process yet."
        )


    def covariance(self,t,i,j):
        raise NotImplementedError(
            "There is no explicit formula for the covariance of this process yet."
        )

    def variance(self,t):
        raise NotImplementedError(
            "There is no explicit formula for the variance of this process yet."
        )

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

        path = self.path[:, :, 0]

        argmax = np.argmax(path, axis=1)

        # RETURN FORM: (max_value, time_index, time)
        return np.max(path, axis=1), argmax, self.t[argmax]

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

        path = self.path[:, :, 0]

        argmin = np.argmin(path, axis=1)

        # RETURN FORM: (max_value, time_index, time)
        return np.min(path, axis=1), argmin, self.t[argmin]

    def mean_path(self, plot_sim=True):

        """
        Mean Path method.

        Compute the average path from the n_simulations simulated paths of the process.
        The average path can be plotted.

        Returns
        -------
        np.ndarray
            Average path deduced from the samples.
        """

        meanpath = np.mean(self.path,axis=0)
        fig = go.Figure()

        if self.dim == 1:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter(x=self.t,
                                             y=self.path[sim, :, 0],
                                             mode="lines",
                                             line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter(x=self.t,
                                     y=meanpath[:, 0],
                                     mode="lines",
                                     line=dict(width=2)))

        elif self.dim == 2:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(
                        go.Scatter(x=self.path[sim, :, 0],
                                   y=self.path[sim, :, 1],
                                   mode="lines",
                                   line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter(x=meanpath[:, 0],
                                     y=meanpath[:, 1],
                                     mode="lines",
                                     line=dict(width=2)))

        elif self.dim ==3:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter3d(x=self.path[sim, :, 0],
                                               y=self.path[sim, :, 1],
                                               z=self.path[sim, :, 2],
                                               mode="lines",
                                               line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter3d(x=meanpath[:, 0],
                                       y=meanpath[:, 1],
                                       z=meanpath[:, 2],
                                       mode="lines",
                                       line=dict(width=2)))

        fig.show()

        return meanpath

    def max_norm(self, order=2):

        """
        Max norm method

        The max norm method returns the maximum of the norm value of the process, the time index at which the maximum norm value
        occurs, and the corresponding time.

        Parameters
        ----------
        order : int or float or np.inf
            Order of the norm. If order = p, then || (x_1 , ... x,n) || = (x_1^p + ... x_n^p)^(1/p).
            If order = np.inf, then || (x_1 , ... x,n) || = max_i |x_i|.
            Must be strictly positive.

        Returns
        -------
        tuple
            Value of the maximum norm value, time index at which the maximum norm occurs, and corresponding time.
        """

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if not isinstance(order, (int, float)) or order < 1:
            raise ValueError(
                "The order must be a strictly positive integer."
            )

        norms = np.linalg.norm(self.path, ord=order, axis=2)
        arg_max = np.argmax(norms, axis=1)
        values = (self.path[np.arange(0,self.n_simulations),arg_max,:])

        return values, norms[:,arg_max], arg_max, self.t[arg_max]

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

        path = self.path[:,:,0]
        candidates = path >= value if inequality == ">" else path <= value

        selected = candidates.any(axis=1)

        first_indexes = candidates.argmax(axis=1)

        t_indexes = np.where(selected,first_indexes,np.nan)

        t_val = np.where(selected, self.t[first_indexes], np.nan)

        return t_indexes, t_val

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
        order: int or float or np.inf
            Order of the desired norm. If order = p, then || (x_1 , ... x,n) || = (x_1^p + ... x_n^p)^(1/p).
            If order = np.inf, then || (x_1 , ... x,n) || = max_i |x_i|.
            Must be strictly positive.

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

        if not isinstance(order, (int, float)) or order < 1:
            raise ValueError(
                "The order must be a strictly positive integer."
            )

        candidates = np.linalg.norm(self.path,axis=2,ord=order) >= value if inequality == ">" else np.linalg.norm(self.path,axis=2,ord=order) <= value

        selected = candidates.any(axis=1)

        first_indexes = candidates.argmax(axis=1)

        t_indexes = np.where(selected,first_indexes,np.nan)

        t_val = np.where(selected, self.t[first_indexes], np.nan)

        return t_indexes, t_val

    def _quad_var_at_index(self, t_index):
        if t_index == 0:
            return 0
        quad_var_t = np.sum((self.path[:, 1:t_index+1] - self.path[:, 0:t_index]) ** 2, axis=1)
        return quad_var_t

    def quadratic_variation(self,t=None, mean = False, plot=False):

        """
        Quadratic Variation method

        The quadratic variation method returns an approximation of the quadratic variation at a specified time t.

        Parameters
        ----------
        t : float or None
            Time at which we want to get the quadratic variation. If None, we compute the quadratic variation along [0,T].
            Must be between [0,T] if not None.
        mean : bool
            If few simulations are made, we can approximate the quadratic variation of the current process by computing the
            mean of all quadratic variation estimation.
        plot : bool
            Specify if the path should be plotted.

        Returns
        -------
        np.ndarray
            Estimation of the quadratic variation.
        """

        if self.dim != 1:
            raise ValueError(
                "The quadratic variation can only be computed in 1D."
            )

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if not 0 <= t <= self.T:
            raise ValueError(
                f"The time must be between {0} and {self.T}."
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

    def values_at(self, t=None, function=lambda x: x[:,0]):

        """
        Values At method.

        Returns the values of the process at a given time t.

        Returns
        -------
        np.ndarray
            Values of the process at a given time t_0.
        """

        t0 = t
        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if t0 is None:
            t0 = self.path.t_n

        if not 0<= t0 <= self.T:
            raise ValueError(
                f"The time must be between {0} and {self.T}."
            )

        # The specified time might not be in the time interval of the process. In this case, the closest time is used.
        if t0 not in self.t:
            t_index = np.argmin(np.abs(t0 - self.t))
        else:
            t_index = np.where(self.t == t0)[0][0]

        return function(self.path[:,t_index])

    def density(self, t, x):
        raise NotImplementedError(
            "The density function is not implemented for this process yet."
        )
