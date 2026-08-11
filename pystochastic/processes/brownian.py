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
    >> W = Brownian(var=np.eye(2),dim=2,t_0=0,t_n=1,n_steps=1000)
    >> W.simulate()
    >> W.plot()
    """

    def __init__(self,
                 var=None,
                 dim=1,
                 t_0=0,
                 t_n=1,
                 n_steps=1000):

        if var is None:
            var = np.eye(dim)

        if not is_pos_def(var):
            raise ValueError(
                "The covariance matrix is not positive-definite."
            )

        if not t_0 < t_n:
            raise ValueError(
                "The final time must be strictly greater than the initial time."
            )

        if var.shape != (dim, dim):
            raise ValueError(
                "The dimension of the covariance matrix must coincide with the specified dimension."
            )

        if not is_pos_def(var):
            raise ValueError(
                "The covariance matrix is not positive-definite."
            )

        self.var = np.array(var)
        self.dim = dim
        self.t_0 = t_0
        self.t_n = t_n
        self.n_steps = n_steps
        self.sim = brownian_motion(self.var, self.dim,self.t_0,self.t_n,self.n_steps)
        self.path = self.sim[0]
        self.increments =self.sim[1]
        self.t = np.linspace(self.t_0,self.t_n,self.n_steps+1)

    def simulate(self):

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

        self.sim = brownian_motion(self.var, self.dim, self.t_0,self.t_n,self.n_steps)
        self.path = self.sim[0]
        self.increments = self.sim[1]
        return self.path

    def plot(self):

        """
        Plot method.

        Plot the simulated path of the brownian motion. The path can be plotted only in 1D, 2D or 3D.
        """

        if self.dim > 3:
            raise ValueError(
                "The path can be plotted only for 1D, 2D and 3D."
            )

        fig = go.Figure()

        if self.dim == 1:
            fig.add_trace(go.Scatter(x=self.t,
                                     y=self.path[:,0],
                                     mode="lines",
                                     line=dict(width=2)))

        elif self.dim == 2:
            fig.add_trace(go.Scatter(x=self.path[:, 0],
                                     y=self.path[:, 1],
                                     mode="lines",
                                     line=dict(width=2)))

        else:
            fig.add_trace(go.Scatter3d(x=self.path[:, 0],
                                       y=self.path[:, 1],
                                       z=self.path[:, 2],
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

        return self.path[-1]

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

        if self.dim != 1:
            raise ValueError(
                "The maximum value can only be computed in 1D. Please refer to the max_norm function."
            )

        argmax = np.argmax(self.path)

        # RETURN FORM: (max_value, time_index, time)
        return np.max(self.path),argmax, self.t[argmax]

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

        if self.dim != 1:
            raise ValueError(
                "The minimum value can only be computed in 1D."
            )

        argmin = np.argmin(self.path)

        # RETURN FORM: (max_value, time_index, time)
        return np.min(self.path), argmin, self.t[argmin]

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


        norms = np.sum(self.path**2, axis=1)
        arg_max = np.argmax(norms)
        return self.path[arg_max,:], arg_max, self.t[arg_max]

    def __repr__(self):

        """
        Representation method.

        The max norm provides some information about the effective brownian motion.

        Returns
        -------
        str
            Dimension, time horizon, time step, and covariance matrix of the brownian motion.

        """
        return (f"Brownian Motion\n------------------------\n "
                f"Dimension : {self.dim}\n "
                f"Time horizon: {self.t_n}\n "
                f"Time step: {(self.t_n-self.t_0)/self.n_steps}\n "
                f"Covariance matrix: \n"
                f"{self.var}")

def brownian_motion(var=np.array(1),d=1, t_0 = 0, t_n = 1, n_steps = 1000):

    """
    Brownian motion function.

    The brownian motion functions simulate a Brownian motion path and increments by using the Cholesky decomposition.

    Returns
    -------
    tuple
        np.ndarray of the simulated brownian motion path and np.ndarray of the increments.

    """

    # Computation of the step length and Cholesky decomposition
    h = (t_n-t_0)/n_steps
    L = np.linalg.cholesky(var) # The Cholesky decomposition of the covariance matrix is analogous to the square root for matrices.

    # Sampling of normal random variables

    N = [crandom.normal(0, 1, n_steps) for _ in range(d)]
    Z = np.stack(N,axis=0)

    W = np.zeros((n_steps+1,d))

    # The increments are independent and normally distributed, with a variance of h*L*L^T
    dW = np.sqrt(h) * L @ Z

    # Since W_0 = 0 and W_i = W_i - W_{i-1} + W_{i-1} - W_{i-2} + ... - W_0 = W_i - W_{i-1} + dW_{i-1} + ... + dW_0
    for k in range(n_steps):
        W[k + 1] = W[k] + dW[:, k]

    return W,np.transpose(dW)
