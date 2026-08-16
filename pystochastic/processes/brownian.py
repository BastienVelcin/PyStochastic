"""
============================================================
Module BROWNIAN
============================================================

Description
-----------
This module provides a way to simulate a Brownian motion with a given covariance matrix.

This module provides a general class "Brownian", which is an SDEs solver, built with the following methods:
    - .simulate() : Simulate a Brownian motion path.
    - .plot() : Plot the Brownian motion path.
    - .final_position() : Return the final position of the Brownian motion.
    - .max() : Return the maximum value of the Brownian motion (only in 1D).
    - .min() : Return the minimum value of the Brownian motion (only in 1D).
    - .max_norm() : Return the maximum norm value of the Brownian motion.
    - .mean() : Return the mean value of the Brownian motion at a given time.
    - .covariance_matrix() : Return the covariance matrix of the Brownian motion at a given time.
    - .covariance() : Return the covariance between two coordinates of the Brownian motion at a given time.
    - .variance() : Return the variance value of the Brownian motion at a given time.

as well as an external trajectory simulation function.

Examples
--------
>> W = Brownian(var=np.eye(2),dim=2,t_0=0,t_n=1,n_steps=1000) #Brownian motion with covariance matrix np.eye(2)
>>
>> W.simulate() #Simulate the Brownian motion path
>>
>> W.plot() #Plot the Brownian motion path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.pyrandom import crandom
from pystochastic.utils import is_pos_def

class Brownian:

    """
    Brownian motion class

    The Brownian motion (or Wiener process) is a Gaussian process that models noises and randomness mathematically.
    For more information, please refer to :
        - Stochastic Differential Equations: An Introduction with Applications - Bernt Øksendal
    or for a simpler explanation:
        - Stochastic calculus applied to some non-linear filtering problems (French version) - Bastien Velcin :
        https://bastienvelcin.github.io/
        - https://en.wikipedia.org/wiki/Brownian_motion#Mathematics

    Parameters
    ----------
    var : None, float, or np.ndarray
        Covariance matrix of the Brownian motion. If None, the covariance matrix is set to identity. The matrix must be positive-definite.
    dim : int
        Dimension of the brownian motion. The dimension must coincide with the dimension of the covariance matrix. Must be a strictly positive integer.
    t_0 : float
        Initial time.
    t_n : float
        Final time. Must be strictly greater than t_0.
    n_steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    var : None, float, or np.ndarray
        Covariance matrix of the Brownian motion. If None, the covariance matrix is set to identity.
    dim : int
        Dimension of the brownian motion.
    t_0 : float
        Initial time.
    t_n : float
        Final time.
    n_steps : int
        Number of time steps.
    sim : np.ndarray
        Simulation of a Brownian motion and its increments.
    path : np.ndarray
        Path of the simulated Brownian motion.
    increments : np.ndarray
        Increments of the simulated Brownian motion.
    t : np.ndarray
        Time interval on which we want to simulate the Brownian motion.

    Examples
    --------
    >> W = Brownian(var=np.eye(2),t_0=0,t_n=1,n_steps=1000)
    >> W.simulate()
    >> W.plot()
    """

    def __init__(self,
                 var=1,
                 t_0=0,
                 t_n=1,
                 n_steps=1000):

        self.var = np.atleast_2d(var)

        if not is_pos_def(self.var):
            raise ValueError(
                "The covariance matrix is not positive-definite."
            )

        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        if not is_pos_def(self.var):
            raise ValueError(
                "The covariance matrix is not positive-definite."
            )

        self.dim = np.shape(self.var)[0]
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.sim = None
        self.n_simulations = None
        self.path = None
        self.increments =None
        self.t = np.linspace(self.t_0,self.t_n,self.n_steps+1)


    def simulate(self,n_simulations=1,plot=False):

        """
        Simulate method.

        Simulate a brownian motion path and increments by using the brownian_motion() function.

        Returns
        -------
        np.ndarray
            Path of the simulated brownian motion.

        Notes
        -----
        The function only returns the path and not the increments. The increments can be accessed through the 'increments' attribute.
        """

        self.sim = brownian_motion(self.var, self.t_0,self.t_n,self.n_steps, n_simulations)
        self.path = self.sim[0]
        self.increments = self.sim[1]

        self.n_simulations = n_simulations

        if plot:
            self.plot()
        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the Geometric Brownian Motion. The path can be plotted only in 1D, 2D or 3D.
        """

        if self.sim == None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim > 3:
            raise ValueError(
                "The path can be plotted only for 1D, 2D and 3D."
            )

        if self.path is None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
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

        if self.sim == None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        return self.path[:,-1]

    def max(self):

        """
        Max method

        The max method returns the maximum value of the Brownian motion, the time index at which the maximum value
        occurs, and the corresponding time.

        Returns
        -------
        tuple
            Value of the maximum value, time index at which the maximum occurs, and corresponding time.

        Notes
        -----
        This method can only be used in 1D since the notion of maximum value is not defined in higher dimensions.
        """

        if self.sim == None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        if self.dim != 1:
            raise ValueError(
                "The maximum value can only be computed in 1D. Please refer to the max_norm function."
            )

        argmax = np.argmax(self.path, axis=1)

        # RETURN FORM: (max_value, time_index, time)
        return np.max(self.path, axis=1),argmax, self.t[argmax]

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

        if self.sim == None:
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

        if self.sim == None:
            raise ValueError(
                "The path has not been simulated yet. Please run the simulate method first."
            )

        norms = np.sum(self.path**2, axis=2)
        arg_max = np.argmax(norms, axis=1)
        values = (self.path[np.arange(0,self.n_simulations),arg_max,:])
        return values, arg_max, self.t[arg_max]

    def mean(self,t):

        """
        Mean method.

        Return the mean of the Brownian motion at a given time t.

        Parameters
        ----------
        t : float
            Time at which the mean is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        float
            0 : Mean of the Brownian motion at a time t

        Notes
        -----
        The mean of the Brownian motion at every time t is always 0, since W_t ~ N(0,t*Q), where Q is the covariance matrix.
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return 0

    def covariance_matrix(self,t):

        """
        Covariance matrix method.

        Return the covariance matrix of the Brownian motion at a given time t.

        Parameters
        ----------
        t : float
            Time at which the covariance matrix is evaluated. Must be between t_0 and t_n.

        Returns
        -------
        np.ndarray
            Covariance matrix of the Brownian motion at a time t

        Notes
        -----
        The covariance matrix of the Brownian motion at every time t is always t*Q, since W_t ~ N(0,t*Q), where Q is the Brownian matrix parameter
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return t*self.var

    def covariance(self, t,i,j):

        """
        Covariance Matrix method.

        Return the covariance between the i-th and j-th coordinates
        of the Brownian motion at a given time t.


        Parameters
        ----------
        t : float
            Time at which the covariance is evaluated.
        i : int
            Index of the first coordinate. It must verify 0 <= i < dim.
        j : int
            Index of the second coordinate. It must verify 0 <= j < dim.

        Returns
        -------
        np.ndarray
            Covariance between the i-th and j-th coordinates.
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return self.covariance_matrix(t)[i,j]


    def variance(self, t):

        """
        Variance method.

        Return the variance of the Brownian Motion coordinates at a given time t.

        Parameters
        ----------
        t : float
            Time at which the variance is evaluated.

        Returns
        -------
        np.ndarray
            Variance of the Brownian Motion path coordinates at a given time t.
        """

        if not self.t_0 <= t <= self.t_n:
            raise ValueError(
                "The time must be between t_0 and t_n."
            )

        return np.array([self.covariance(t,i,i) for i in range(self.dim)])


def brownian_motion(var=1, t_0 = 0, t_n = 1, n_steps = 1000,n_simulations=1):

    """
    Brownian motion function.

    The brownian motion functions simulate a Brownian motion path and increments by using the Cholesky decomposition.

    Returns
    -------
    tuple
        np.ndarray of the simulated brownian motion path and np.ndarray of the increments.

    """
    var = np.atleast_2d(var)
    d = np.shape(var)[0]

    W = np.zeros((n_simulations,n_steps+1, d))
    dW = np.zeros((n_simulations, n_steps, d))

    # Computation of the step length and Cholesky decomposition
    h = (t_n - t_0) / n_steps
    L = np.linalg.cholesky(var)  # The Cholesky decomposition of the covariance matrix is analogous to the square root for matrices.

    N = crandom.normal(0,1,n_simulations*n_steps*d)
    Z = np.reshape(N,(n_simulations,n_steps,d))
    dW[:,:] = Z @ L.T * np.sqrt(h)

    W[:, 1:, :] = np.cumsum(dW, axis=1)
    return W,dW
