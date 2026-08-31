"""
============================================================
Module BROWNIAN
============================================================

Description
-----------
This module provides a way to simulate a Brownian motion with a given covariance matrix.

This module provides a general class "Brownian", which inherits from the methods of Process abstract class.

as well as an external trajectory simulation function.

Examples
--------
>> W = Brownian(variance=np.eye(2),T=1,steps=1000) #Brownian motion with covariance matrix np.eye(2)
>>
>> W.simulate() #Simulate the Brownian motion path
>>
>> W.plot() #Plot the Brownian motion path
"""

import numpy as np
from pystochastic.random import continuous
from pystochastic.utils import is_pos_def
from pystochastic.processes.process import Process
from pystochastic.processes.process import _validate_t
from pystochastic.dist import Normal

class Brownian(Process):

    """
    Brownian motion class

    The Brownian motion (or Wiener process) is a Gaussian process that models noises and randomness mathematically.

    Parameters
    ----------
    variance : None, float, or np.ndarray
        Covariance matrix of the Brownian motion. If None, the covariance matrix is set to identity. The matrix must be positive-definite.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    variance : None, float, or np.ndarray
        Covariance matrix of the Brownian motion. If None, the covariance matrix is set to identity.
    T : float
        Final time.
    steps : int
        Number of time steps.
    sim : np.ndarray
        Simulation of a Brownian motion and its increments.
    path : np.ndarray
        Path of the simulated Brownian motion.
    increments : np.ndarray
        Increments of the simulated Brownian motion.
    t : np.ndarray
        Time interval on which we want to simulate the Brownian motion.
    name : str
        Name of the process

    Examples
    --------
    >> W = Brownian(variance=np.eye(2),T=1,steps=1000)
    >> W.simulate()
    >> W.plot()
    """

    def __init__(self,
                 variance=1,
                 T=1,
                 steps=1000):

        self.variance = np.atleast_2d(variance)

        super().__init__(T=T,
                         steps=steps,
                         dim = np.shape(self.variance)[0],
                         name = "Brownian Motion")

        if not is_pos_def(self.variance):
            raise ValueError(
                "The covariance matrix is not positive-definite."
            )

        self.sim = None
        self.increments =None

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

        self.n_simulations = n_simulations

        self.sim = brownian_motion(self.variance, self.T,self.steps, n_simulations)
        self.path = self.sim[0]
        self.increments = self.sim[1]

        if plot:
            self.plot()
        return self.path

    def expectation(self,t):

        """
        Expectation method.

        Return the expectation of the Brownian motion at a given time t.

        Parameters
        ----------
        t : float
            Time at which the expectation is evaluated.

        Returns
        -------
        float
            0 : Expectation of the Brownian motion at a time t

        Notes
        -----
        The expectation of the Brownian motion at every time t is always 0, since W_t ~ N(0,t*Q), where Q is the covariance matrix.
        """

        t = _validate_t(t)

        return 0

    def covariance_matrix(self,t):

        """
        Covariance matrix method.

        Return the covariance matrix of the Brownian motion at a given time t.

        Parameters
        ----------
        t : float
            Time at which the covariance matrix is evaluated.

        Returns
        -------
        np.ndarray
            Covariance matrix of the Brownian motion at a time t

        Notes
        -----
        The covariance matrix of the Brownian motion at every time t is always t*Q, since W_t ~ N(0,t*Q), where Q is the Brownian matrix parameter
        """

        t = _validate_t(t)

        return t*self.variance

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

        t = _validate_t(t)

        if not 0 <= i < self.dim or not isinstance(i, (int, np.integer)):
            raise ValueError(
                "The first coordinate must be an integer between 0 and the dimension (excluded)."
            )

        if not 0 <= j < self.dim or not isinstance(j, (int, np.integer)):
            raise ValueError(
                "The second coordinate must be an integer between 0 and the dimension (excluded)."
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


        return np.array([self.covariance(t,i,i) for i in range(self.dim)])

    def density(self,t,x):



        if self.dim > 1:
            raise ValueError(
                "The density is only implemented for 1D processes yet."
            )

        t = _validate_t(t)

        if t == 0:
            return np.array([0])
        N = Normal(mu=0,var = t*self.variance)
        return N.pdf(x)

def brownian_motion(variance=1, T=1, steps = 1000,n_simulations=1):

    """
    Brownian motion function.

    The brownian motion functions simulate a Brownian motion path and increments by using the Cholesky decomposition.

    Returns
    -------
    tuple
        np.ndarray of the simulated brownian motion path and np.ndarray of the increments.

    """
    variance = np.atleast_2d(variance)
    d = np.shape(variance)[0]

    W = np.zeros((n_simulations,steps+1, d))
    dW = np.zeros((n_simulations, steps, d))

    # Computation of the step length and Cholesky decomposition
    h = T / steps
    L = np.linalg.cholesky(variance)  # The Cholesky decomposition of the covariance matrix is analogous to the square root for matrices.

    N = continuous.normal(0, 1, n_simulations * steps * d)
    Z = np.reshape(N,(n_simulations,steps,d))
    dW[:,:] = Z @ L.T * np.sqrt(h)

    W[:, 1:, :] = np.cumsum(dW, axis=1)
    return W,dW
