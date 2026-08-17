"""
============================================================
Module MONTE CARLO
============================================================

Description
-----------
This module provides some Monte Carlo methods, for both random variables and processes.
This modules provides two classes, with the following methods:
    - MonteCarlo : Monte Carlo methods for samples from a given random variable.
        - .estimate() : Estimates the mean of a function of a given random variable.
        - .mean_estimator() : Estimates the mean and the half-width of the confidence interval.
        - .confidence_interval() : Estimates the confidence interval of the mean.
        - .confidence_curve() : Plots the cumulative distribution function of the mean.

    - MonteCarloProcess : Monte Carlo methods for stochastic processes
        - .simulate() : Resample a stochastic process path.
        - .estimate() : Plots the mean path of a function of the process at a given time.
        - .values_at() : Returns the values of the process at a given time.
        - .mean_path() : Plots the mean path of the process.



Examples
--------
>> R = CIR(a=2,b=3,sigma=1,r_0=0,t_0=0,t_n=1,n_steps=1000) #Cox-Ingersoll-Ross process with speed 2, mean 3 and volatility 1 and starting point 0.
>>
>> R.simulate() #Simulate the CIR process path
>>
>> R.plot() #Plot the CIR process path
"""

import numpy as np
import plotly.graph_objects as go
from pystochastic.processes import *
from scipy.stats import norm

class MonteCarlo:

    """
    MonteCarlo class

    A Monte Carlo class for independent samples that comes from a same distribution.
    It provides the main methods for estimating the mean of a function of a given random variable, and the
    confidence interval of the estimation.

    Parameters
    ----------
    samples : np.ndarray
        Independent samples from a given distribution.
    n_simulations : None or int
        Number of used simulations for the estimation. If None, it is set to the number of samples. Must be a integer greater than 2 and inferior to the number of samples.

    Attributes
    ----------
    samples : np.ndarray
        Independent samples from a given distribution.
    n_simulations : None or int
        Number of used simulations for the estimation.

    Examples
    --------
    >> mc = MonteCarlo(crandom.normal(mu=0,sigma=1,n=1000),n_simulations=100))
    >> mc.estimate(lambda x: x**2)
    >> mc.confidence_curve(confidence=0.90)
    """

    def __init__(self,samples,n_simulations=None):
        if n_simulations is None:
            n_simulations = len(samples)

        if n_simulations <= 1:
            raise ValueError(
                "n_simulations cannot be less than or equal to 1."
            )

        if n_simulations > len(samples):
            raise ValueError(
                "n_simulations cannot be greater than the number of samples provided."
            )

        self.samples = np.asarray(samples).flatten()
        self.n_simulations = n_simulations

    def estimate(self, n=None, function = lambda x: x):

        """
        Estimate method.

        Provides an estimation of the mean of a function of a given random variable with the Law of large numbers.
                        E[f(X)] ~~ (f(X_1) + ... + f(X_n)) / n

        Returns
        -------
        float
            Estimation of the mean.
        """

        if n is None:
            n = self.n_simulations
        return np.mean(function(self.samples[:n]), axis=0)

    def mean_estimator(self, n= None, confidence = 0.95):

        """
        Mean Estimator method.

        Provides an estimation of the mean of a function of a given random variable with the Law of large numbers,
                        E[f(X)] ~~ (f(X_1) + ... + f(X_n)) / n
        with the half width of the associated confidence interval.

        Returns
        -------
        tuple
            (mean, half width)
        """

        if n is None:
            n = self.n_simulations
        mean_est = self.estimate(n)
        sd_estimate = np.std(self.samples[:n], axis=0)

        #Quantile function of the standard normal distribution
        z = norm.ppf(0.5 + confidence / 2)
        half_width = z * sd_estimate / np.sqrt(n)

        return mean_est, half_width

    def confidence_interval(self, n = None,confidence = 0.95):

        """
        Confidence Interval method.

        Provides the confidence interval of the mean estimator of a given random variable.

        Returns
        -------
        tuple
            (Lower bound, Upper bound) : bounds of the confidence interval.
        """

        if n is None:
            n = self.n_simulations

        mean_est, half_width = self.mean_estimator(n,confidence)
        return mean_est - half_width, mean_est + half_width

    def confidence_curve(self,n=None,confidence = 0.95):

        """
        Confidence Curve method.

        Plot the mean estimator and the confidence interval of the mean estimator for all estimations, from 2 samples to
        the maximum number of samples specified in attributes
        """

        if n is None:
            n = self.n_simulations

        n_axis = np.arange(1, n + 1)

        #Computation of the mean and variance of the estimator

        S1 = np.cumsum(self.samples[:n])
        S2 = np.cumsum(self.samples[:n] ** 2)
        cum_mean = S1 / n_axis

        #Ignore the division by zero when the number of samples is 1
        with np.errstate(invalid="ignore", divide="ignore"):
            cum_var = (S2 - S1 ** 2 / n_axis) / (n_axis - 1)

        # Cumulative variance computation: replacing the NaN values by 0)
        cum_var = np.nan_to_num(cum_var, nan=0)

        #Quantile function of the standard normal distribution
        z = norm.ppf(0.5 + confidence / 2)
        half_width = z * np.sqrt(cum_var) / np.sqrt(n_axis)

        # Plot the mean estimator curve and the confidence interval with polygons
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([n_axis, n_axis[::-1]]),
                                 y=np.concatenate([cum_mean + half_width, (cum_mean - half_width)[::-1]]),
                                 fill="toself", fillcolor="rgba(100,149,237,0.2)",
                                 line=dict(width=0), name=f"CI {int(confidence * 100)}%", showlegend=True,
                                 ))

        fig.add_trace(go.Scatter(x=n_axis,
                                 y=cum_mean,
                                 mode="lines",
                                 name="Cumulative estimator"))
        fig.show()

class MonteCarloProcess:

    """
    MonteCarloProcess class

    A Monte Carlo class for stochastic processes.
    It provides the main methods for estimating parameters of the state of a stochastic process at a given time, and the
    overall average process.

    Parameters
    ----------
    process : np.ndarray
        Stochastic process. It must be a stochastic process of the sublibrary processes.
    n_simulations : None or int
        Number of used simulated path of the process

    Attributes
    ----------
    process : np.ndarray
        Stochastic process.
    n_simulations : None or int
        Number of used simulated path of the process
    t : np.ndarray
        Time interval on which we want to simulate the process.
    ech : np.ndarray
        Samples obtained from the process.

    Examples
    --------
    >> R = CIR(a=2,b=3,sigma=1,r_0=0,t_0=0,t_n=1,n_steps=1000)
    >> mc = MonteCarloProcess(R,100)
    >> mc.estimate(t_0=0.5, lambda x: x**2)
    >> mc.mean_path()
    """

    def __init__(self,process,n_simulations=100, method=None):

        self.process = process
        self.n_simulations = n_simulations
        self.t = self.process.t
        self.ech = self.process.simulate(self.n_simulations,method=method)
        self.dim = self.process.dim

    def simulate(self):

        """
        Simulate method.

        Simulate n_simulations times a stochastic process path.

        Returns
        -------
        np.ndarray
            Paths of the simulated process.
        """

        self.ech = self.process.simulate(self.n_simulations)
        return self.ech

    def estimate(self, t_0=None, function = lambda x: x,n=None):

        """
        Estimate method.

        Provides an estimation of the mean of a function of the process samples at a time t_0 with the Law of large numbers.
                        E[f(X)] ~~ (f(X_1) + ... + f(X_n)) / n

        Returns
        -------
        float
            Estimation of the mean.
        """

        if n is None:
            n = self.n_simulations

        n = np.atleast_1d(n)

        if n.any() <= 0:
            raise ValueError("n must be strictly positive.")

        if n.any() > self.n_simulations:

            raise ValueError(
                "n cannot be greater than the number of simulations."
            )

        if t_0 is None:
            t_0 = self.process.t_n

        # The specified time might not be in the time interval of the process. In this case, the closest time is used.
        if t_0 not in self.t:
            t_index = np.argmin(np.abs(t_0 - self.t))

        else:
            t_index = np.where(self.t == t_0)[0][0]

        ech = function(self.ech)

        means = np.zeros((np.size(n),self.dim))
        means[::] = np.mean(ech[:,t_index],axis=0)

        return means

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

        meanpath = np.mean(self.ech,axis=0)
        fig = go.Figure()

        if self.process.dim == 1:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter(x=self.t,
                                             y=self.ech[sim, :, 0],
                                             mode="lines",
                                             line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter(x=self.t,
                                     y=meanpath[:, 0],
                                     mode="lines",
                                     line=dict(width=2)))

        elif self.process.dim == 2:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(
                        go.Scatter(x=self.ech[sim, :, 0],
                                   y=self.ech[sim, :, 1],
                                   mode="lines",
                                   line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter(x=meanpath[:, 0],
                                     y=meanpath[:, 1],
                                     mode="lines",
                                     line=dict(width=2)))

        elif self.process.dim ==3:
            if plot_sim:
                for sim in range(self.n_simulations):
                    fig.add_trace(go.Scatter3d(x=self.ech[sim, :, 0],
                                               y=self.ech[sim, :, 1],
                                               z=self.ech[sim, :, 2],
                                               mode="lines",
                                               line=dict(width=1,color="#D1D9ED")))

            fig.add_trace(go.Scatter3d(x=meanpath[:, 0],
                                       y=meanpath[:, 1],
                                       z=meanpath[:, 2],
                                       mode="lines",
                                       line=dict(width=2)))

        fig.show()

        return meanpath

    def values_at(self, t_0=None, function=lambda x: x[:,0]):

        """
        Values At method.

        Returns the values of the process at a given time t_0.

        Returns
        -------
        np.ndarray
            Values of the process at a given time t_0.
        """

        if t_0 is None:
            t_0 = self.process.t_n

        # The specified time might not be in the time interval of the process. In this case, the closest time is used.
        if t_0 not in self.t:
            t_index = np.argmin(np.abs(t_0 - self.t))
        else:
            t_index = np.where(self.t == t_0)[0][0]

        if self.ech is None:
            self.simulate()

        return function(self.ech[:,t_index])

    def mean_error(self,t,N=(10, 100, 1000),error_type="absolute",n_experiments=1,plot=True,plottype=None):

        N = np.asarray(N)
        N_len = np.size(N)
        dim = self.process.dim

        mean_approximation = np.zeros((n_experiments,N_len,dim))

        for i in range(n_experiments):
            self.ech = self.process.simulate(self.n_simulations)
            mean_approximation[i, :] = self.estimate(t_0=t, n=N).reshape(N_len, dim)

        true_mean = self.process.mean(t)

        if error_type == "absolute":
            error = np.abs(mean_approximation - true_mean)

        elif error_type == "relative":

            if np.any(true_mean == 0):
                raise ValueError(
                    "The relative error is not defined when the process mean is zero."
                )

            error = np.abs((mean_approximation - true_mean) / true_mean)

        elif error_type == "mse":
            error = np.mean((mean_approximation - true_mean) ** 2, axis=0)

        else:
            raise ValueError(
                "error_type must be 'absolute', 'relative' or 'mse'."
            )

        if plot == True:
            fig = go.Figure()
            for d in range(dim):
                fig.add_trace(
                    go.Scatter(
                        x=N,
                        y=error[:, d],
                        mode="lines+markers",
                        name=f"Dimension {d + 1}"
                    )
                )
            if plottype == "log":
                fig.update_xaxes(type="log")
                fig.update_yaxes(type="log")
            fig.show()
