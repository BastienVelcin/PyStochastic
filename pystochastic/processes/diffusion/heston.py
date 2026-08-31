"""
============================================================
Module HESTON
============================================================

Description
-----------
This module provides a way to simulate a Heston model with given appropriate parameters. Heston model is used to determine an asset price where
the volatility is given by a Cox-Ingersoll-Ross process.

This module provides a general class "Heston", which inherits from the methods of Process and DiffusionProcess abstract classes.

Examples
--------
>> H = Heston(mu = 1, long_variance = 0.9, reverting_rate = 1, variance_volatility = 0.5, correlation = 0.3, initial_price = 10, initial_variance = 0.3, T = 5, steps = 500)
>>
>> H.simulate()
>>
>> H.plot()
"""
import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import Brownian
from pystochastic.processes.diffusion import DiffusionProcess
from pystochastic.sde import EulerMaruyama


class Heston(DiffusionProcess):

    """
    Heston class

    The Heston model describes the evolution of a stock price S_t under a volatility process sqrt(nu_t).
    The model equations are given by :
        dS_t = mu * S_t dt + sqrt(nu_t) * S_t * dW_t^1,
        dnu_t = reverting_rate * (long_variance - nu_t) dt + variance_volatility * sqrt(nu_t) * dW_t^2,

    where W^1 and W^2 are standard Brownian motion with a fixed correlation.

    Parameters
    ----------
    mu : float
        Drift of the stock price equation.
    initial_price : float
        Initial value of the stock price process.
    initial_variance : float
        Initial value of the variance process.
    long_variance : float
        Long variance parameter.
    correlation : float
        Correlation parameter of the two Browian motions
    reverting_rate : float
        Rate at which the variance reverts to the long variance.
    variance_volatility : float
        Volatility parameter of the variance process.
    T : float
        Final time. Must be strictly greater than 0.
    steps : int
        Number of time steps. Must be a strictly positive integer.

    Attributes
    ----------
    mu : float
        Drift of the stock price equation.
    initial_price : float
        Initial value of the stock price process.
    initial_variance : float
        Initial value of the variance process.
    long_variance : float
        Long variance parameter.
    correlation : float
        Correlation parameter of the two Browian motions
    reverting_rate : float
        Rate at which the variance reverts to the long variance.
    variance_volatility : float
        Volatility parameter of the variance process.
    T : float
        Final time.
    steps : int
        Number of time steps.
    n_simulations : None, or int
        Number of simulations.
    dim : int
        Dimension of the process.
    t : np.ndarray
        Time interval on which we want to simulate the process.
    dt : float
        Time step length.
    price : np.ndarray
        Path of the simulated stock price.
    var : np.ndarray
        Path of the simulated variance.
    path : np.ndarray
        Path of the simulated volatility.
    name : str
        Name of the process
    is_autonomous : bool
        Specify if the process SDE is autonomous.

    Examples
    --------
    >> H = Heston(mu = 1, long_variance = 0.9,  reverting_rate = 1, variance_volatility = 0.5, correlation = 0.3, initial_price = 10, initial_variance = 0.3, T = 5, steps = 500)
    >> H.simulate()
    >> H.plot()
    """
    def __init__(
            self,
            mu = 1,
            long_variance = 1,
            reverting_rate = 1,
            variance_volatility = 1,
            correlation=0,
            initial_price=1,
            initial_variance=1,
            T=1,
            steps=1000):

        super().__init__(T = T,
                         steps = steps,
                         dim = 1,
                         name = "Heston model")

        if initial_price <= 0:
            raise ValueError(
                "The initial price must be strictly positive."
            )
        if initial_variance <= 0:
            raise ValueError(
                "The initial variance must be strictly positive."
            )
        if long_variance <= 0:
            raise ValueError(
                "The long variance must be strictly positive."
            )
        if reverting_rate <= 0:
            raise ValueError(
                "The reverting rate must be strictly positive."
            )
        if variance_volatility <= 0:
            raise ValueError(
                "The variance volatility must be strictly positive."
            )
        if not -1 <= correlation <= 1:
            raise ValueError(
                "The correlation coefficient must be between -1 and 1."
            )

        self.mu = mu
        self.initial_price = initial_price
        self.initial_variance = initial_variance
        self.long_variance = long_variance
        self.correlation = correlation
        self.reverting_rate = reverting_rate
        self.variance_volatility = variance_volatility
        self.price = None
        self.var = None
        self.path = None #The volatility will be sqrt(self.variance), and stored in the path
        self.couple = None
        self.is_autonomous = True

    @property
    def feller_condition(self):

        """
        Feller condition property

        Returns if the Feller condition is satisfied or not. The Feller condition is said satisfied when the following equation is satisfied:
                            2 * reverting_rate * long_variance >= variance_volatility ** 2
        """

        return 2 * self.reverting_rate * self.long_variance >= self.variance_volatility ** 2


    def drift(self,x,t=None):

        """
        Drift function

        Evaluate the drift of the Heston process at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the drift is evaluated.
        t : float
            Time at which the drift is evaluated.

        Returns
        -------
        float :
            Drift evaluated at x and t.
        """

        price = x[0]
        variance = np.maximum(x[1], 0)

        drift = np.array([self.mu * price, self.reverting_rate*(self.long_variance- variance)])
        return drift

    def diffusion(self, x, t=None):

        """
        Diffusion function

        Evaluate the diffusion of the HullWhite process at a given point x and time t.

        Parameters
        ----------
        x : np.ndarray
            Point at which the diffusion is evaluated.
        t : float
            Time at which the diffusion is evaluated.

        Returns
        -------
        float :
            Diffusion evaluated at x and t.
        """

        # In this implementation, the form of x is x = (price, variance)
        price = x[0]
        variance = np.maximum(x[1], 0)

        diffusion = np.array([[price * np.sqrt(variance),0],
                              [0, self.variance_volatility * np.sqrt(variance)]])

        return diffusion


    def simulate(self,n_simulations=1,method=None,plot=False,parallel=False,n_workers=None):

        """
        Simulate method.

        Simulate a Heston model path with the truncated Euler-Maruyama method.

        Parameters
        ----------
        n_simulations : int
            Number of trajectories to simulate.
        method : str or None
            Method to choose for computing a Heston model. It only supports Euler Maruyama method.
        plot : bool
            Specify if the path should be plotted.
        parallel : bool
            Specify if the simulation should be parallelized. Useful on a huge number of simulations
        n_workers : int
            Number of cpu cores to use in parallel computing.

        Returns
        -------
        np.ndarray
            Path of the simulated Heston process.

        """
        self.n_simulations = n_simulations

        covariance_matrix = np.array([[1, self.correlation], [self.correlation, 1]])

        W = Brownian(cov = covariance_matrix, T = self.T, steps = self.steps)
        W.simulate(n_simulations = n_simulations)

        self.couple = EulerMaruyama(drift=self.drift,
                                  diffusion=self.diffusion,
                                  initial=np.array([self.initial_price, self.initial_variance]),
                                  T = self.T,
                                  steps=self.steps).solve(n_simulations=n_simulations,
                                                           plot=False,
                                                           parallel=parallel,
                                                           n_workers=n_workers,
                                                           brownian_increments=W.increments)
        self.price = self.couple[:, :, 0][:,:,None]
        self.var = self.couple[:, :, 1][:,:,None]

        # WARNING: Due to some approximation with the Euler-Maruyama method, the variance can be negative, so we need to truncate the variance with max(V_t, 0)
        self.path = np.sqrt(np.maximum(self.var, 0))

        if plot:
            fig_price = go.Figure()
            fig_vol = go.Figure()
            for sim in range(self.n_simulations):
                fig_price.add_trace(go.Scatter(x=self.t,
                                               y=self.price[sim,:],
                                               mode="lines",
                                               line=dict(width=2),
                                               name=f"Path {sim+1}"))
                fig_vol.add_trace(go.Scatter(x=self.t,
                                             y=self.path[sim, :],
                                             mode="lines",
                                             line=dict(width=2),
                                             name=f"Path {sim+1}"))

            fig_price.update_layout(title=f"Price evolution of the asset price",
                                    xaxis_title="t",
                                    yaxis_title="Price $S_t$",
                                    template="plotly_white")
            fig_vol.update_layout(title=f"Volatility evolution",
                                    xaxis_title="t",
                                    yaxis_title="Volatility $sqrt(\nu_t)$",
                                    template="plotly_white")

            fig_vol.show()
            fig_price.show()

        return self.couple
