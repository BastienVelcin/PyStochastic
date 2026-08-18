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
import sys, inspect
from scipy.stats import norm, t
from pystochastic.dist.dist import *

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

    def __init__(self,samples):

        self.samples = np.atleast_3d(samples)
        self.n_simulations = self.samples.shape[0]
        self.n_pool_values = self.samples.shape[1]



    ##################################
    #    I.  ESTIMATORS & MOMENTS    #
    ##################################

    def estimate(self, n=None, function = lambda x: x):

        """
        Estimate method.

        Provides an estimation of the mean of a function of samples from a random variable with the Law of large numbers.
                        E[f(X)] ~~ (f(X_1) + ... + f(X_n)) / n
        and eventually the half-width of the confidence interval.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the mean E[f(X)].

        Returns
        -------
        float or np.ndarray
            Estimation of the mean.
        """

        if n is None:
            n = self.n_pool_values
        return np.mean(function(self.samples[:,:n]), axis=1)

    def half_width(self, n=None, function = lambda x: x, confidence=0.95, type="normal"):

        """
        Half Width method.

        Provide the Half Width estimation of the confidence interval of specified confidence level. The half width can
        be computed with two laws :
            - Normal law : when the variance is known.
            - Student law : when the variance is unknown.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the half width for the mean estimator of E[f(X)].
        confidence : float
            Confidence level. Must be a float between 0 and 1.
        type : {"normal", "student"}
            Law used for the confidence interval.

        Returns
        -------
        float or np.ndarray
            Estimation of the half width.
        """

        if n is None:
            n = self.n_pool_values

        sd_estimate = self.std(n,function,correction=True)

        if type == "normal":
            #Quantile function of the standard normal distribution
            z = norm.ppf(0.5 + confidence / 2)

        elif type == "student":
            # Quantile function of the Student distribution with n-1 degrees of freedom
            z = t.ppf(0.5 + confidence / 2, df=n-1)

        else:
            raise ValueError(
                "type must be 'normal' or 'student'."
            )

        hw = z * sd_estimate / np.sqrt(n)
        return hw

    def moment(self, order=1, n=None, function = lambda x: x):

        """
        Moment method.

        Determines an estimation of a moment of a given order of a function of a samples from a random variable.
        This methods does not check if the specified moment exists.

        Parameters
        ----------
        order : int
            Moment order. Must be a strictly positive integer.
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the moment E[f(X)^order].

        Returns
        -------
        float or np.ndarray
            Estimation of the moment of specified order.
        """

        if n is None:
            n = self.n_pool_values

        return self.estimate(n, lambda x: np.power(function(x), order))


    def variance(self,n=None, function = lambda x: x, correction=True):

        """
        Variance method.

        Provides an estimation of the variance of a function of samples from a random variable with the
        Konig-Huygens formula :
                                       Var(f(X)) = E(f(X)^2) - E(f(X))^2
        Both corrected and uncorrected variance can be computed with this method.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the variance Var[f(X)].
        correction : bool
            Specify if the Bessel's correction should be applied to the variance.

        Returns
        -------
        float or np.ndarray
            Estimation of the variance.
        """

        if n is None:
            n = self.n_pool_values

        if not n > 1:
            raise ValueError(
                "n must be strictly greater than 1."
            )

        var = self.estimate(n, lambda x : function(x)**2) - (self.estimate(n, lambda x : function(x)))**2

        if correction:
            return n*var/(n-1)

        return var

    def std(self,n=None, function = lambda x: x, correction=True):

        """
        Standard Deviation method.

        Provides an estimation of standard deviation from the variance estimation :
                                std(f(X)) = sqrt(Var(f(X)))
        Both corrected and uncorrected variance can be used to compute the standard deviation.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the standard deviation std[f(X)].
        correction : bool
            Specify if the Bessel's correction should be applied to the variance.

        Returns
        -------
        float or np.ndarray
            Estimation of the standard deviation.
        """

        return np.sqrt(self.variance(n,function,correction))

    def standard_error(self, n = None, function = lambda x: x, correction = True):

        """
        Standard Error method.

        Provides an estimation of standard error from the standard deviation estimation :
                                ste[f(X)] = std[f(X)] / sqrt(n)
        Both corrected and uncorrected standard deviation can be used to compute the standard error.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the standard error sde[f(X)].
        correction : bool
            Specify if the Bessel's correction should be applied to the standard deviation.

        Returns
        -------
        float or np.ndarray
            Estimation of the standard error.
        """

        if n is None:
            n = self.n_pool_values

        return self.std(n,function,correction)/np.sqrt(n)


    ############################################
    #    II.  CONFIDENCE INTERVALS & CURVES    #
    ############################################

    def confidence_interval(self, n = None, function = lambda x : x, confidence = 0.95,type="normal"):


        """
        Confidence Interval method.

        Provides the confidence interval of the mean estimator of a given random variable.

        Returns
        -------
        tuple
            (Lower bound, Upper bound) : bounds of the confidence interval.
        """

        if n is None:
            n = self.n_pool_values

        if n <= 1:
            raise ValueError(
                "n must be strictly greater than 1."
            )

        mean_est = self.estimate(n, function=function)
        half_width = self.half_width(n,function=function,confidence=confidence,type=type)

        return mean_est - half_width, mean_est + half_width

    def confidence_curve(self,n=None,n_pool = 0,function = lambda x : x, confidence = 0.95,type="normal"):

        """
        Confidence Curve method.

        Plot the mean estimator and the confidence interval of the mean estimator for all estimations, from 2 samples to
        the maximum number of samples specified in attributes
        """

        if n is None:
            n = self.n_pool_values

        n_axis = np.arange(1, n + 1)

        # Computation of the mean and variance of the estimator
        samples = function(self.samples[n_pool])
        S1 = np.cumsum(samples[:n])
        S2 = np.cumsum(samples[:n] ** 2)
        cum_mean = S1 / n_axis

        # Ignore the division by zero when the number of samples is 1
        with np.errstate(invalid="ignore", divide="ignore"):
            cum_var = (S2 - S1 ** 2 / n_axis) / (n_axis - 1)

        # Cumulative variance computation: replacing the NaN values by 0)
        cum_var = np.nan_to_num(cum_var, nan=0)

        if type == "normal":
            q = norm.ppf(0.5 + confidence / 2)

        elif type == "student":
            q = t.ppf(0.5 + confidence / 2, df=n_axis - 1)

        half_width = q * np.sqrt(cum_var) / np.sqrt(n_axis)

        # Plot the mean estimator curve and the confidence interval with polygons
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([n_axis, n_axis[::-1]]),
                                 y=np.concatenate([cum_mean + half_width, (cum_mean - half_width)[::-1]]),
                                 fill="toself", fillcolor="rgba(100,149,237,0.2)",
                                 line=dict(width=0),
                                 name=f"CI {int(confidence * 100)}%",
                                 showlegend=True,))

        fig.add_trace(go.Scatter(x=n_axis,
                                 y=cum_mean,
                                 mode="lines",
                                 name=f"Cumulative estimator with {type} law"))

        fig.update_layout(
            title=f"Confidence curve with sample pool {n_pool}.",
            xaxis_title="Number of samples",
            yaxis_title="Estimation",
            template="plotly_white",
        )

        fig.show()

    ##################################
    #    III.  STATISTICAL ERRORS    #
    ##################################

    def bias(self, reference, n=None, function = lambda x: x):

        """
        Bias method

        Provides the bias of the likelihood estimator of E[f(X)] when E[f(X)] is known.

        Parameters
        ----------
        reference :
            Known estimated value.
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to estimate the bias of E[f(X)].

        Returns
        -------
        float or np.ndarray
            Bias of the likelihood estimator.
        """

        return self.estimate(n, function=function) - reference

    def mse(self, reference, n=None, function = lambda x: x):

        """
        MSE method

        Provides the mean squared error of the likelihood estimator of E[f(X)] when E[f(X)] is known.

        Parameters
        ----------
        reference :
            Known estimated value.
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to compute the MSE.

        Returns
        -------
        float or np.ndarray
            Bias of the likelihood estimator.
        """

        if n is None:
            n = self.n_pool_values

        return self.estimate(n, function = lambda x : (reference - function(x))**2)

    def rmse(self, reference, n=None, function = lambda x: x):

        """
        RMSE method

        Provides the root mean squared error of the likelihood estimator of E[f(X)] when E[f(X)] is known.

        Parameters
        ----------
        reference :
            Known estimated value.
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to compute the RMSE.

        Returns
        -------
        float or np.ndarray
            Bias of the likelihood estimator.
        """

        if n is None:
            n = self.n_pool_values

        return np.sqrt(self.mse(reference,n,function))


    ######################################
    #    III.  DESCRIPTIVE STATISTICS    #
    ######################################

    def quantile(self, q, n=None, function = lambda x: x):

        """
        Quantile method

        Provides the quantile of order q of a specified array of values.

        Parameters
        ----------
        q : float
            Quantile. Must be a float between 0 and 1.
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the quantile.

        Returns
        -------
        float or np.ndarray
            Quantile of order q.
        """

        if n is None:
            n = self.n_pool_values

        return np.quantile(function(self.samples[:,:n]),q, axis=1)

    def min(self,n = None,function = lambda x: x):

        """
        Min method

        Provides the minimum value from samples.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the minimum value.

        Returns
        -------
        float or np.ndarray
            Minimum value(s).
        """

        if n is None:
            n = self.n_pool_values

        return np.min(function(self.samples[:,:n]), axis=1)

    def max(self,n = None, function = lambda x: x):

        """
        Max method

        Provides the maximum value from samples.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the maximum value.

        Returns
        -------
        float or np.ndarray
            Maximum value(s).
        """

        if n is None:
            n = self.n_pool_values

        return np.max(function(self.samples[:,:n]), axis=1)

    def median(self, n = None, function = lambda x: x):

        """
        Median method

        Provides the median value from samples.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the median value.

        Returns
        -------
        float or np.ndarray
            Median value(s).
        """

        if n is None:
            n = self.n_pool_values

        return np.median(function(self.samples[:,:n]), axis=1)

    def skewness(self,n=None, function = lambda x: x):

        """
        Skewness method

        Provides the skewness coefficient from samples.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the skewness coefficient.

        Returns
        -------
        float or np.ndarray
            Skewness coefficient(s).
        """

        std = self.std(n, function, correction=True)
        mean = self.estimate(n, function)

        return self.estimate(n, lambda x : np.power((self.samples - mean) / std,3))

    def kurtosis(self, n=None, function = lambda x: x):

        """
        Kurtosis method

        Provides the unormalized kurtosis coefficient from samples.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to search the unormalized kurtosis coefficient.

        Returns
        -------
        float or np.ndarray
            Unormalized kurtosis coefficient(s).
        """

        std = self.std(n, function, correction=True)
        mean = self.estimate(n, function)

        return self.estimate(n, lambda x: np.power((self.samples - mean) / std, 4))


    #################################
    #    IV.  EMPIRICAL ANALYSIS    #
    #################################

    def histogram(self,n=None, function = lambda x: x, dim=0, bins = 10, normalized = True, plot = True, distribution = None):

        """
        Histogram method

        Plot an histogram of samples values.

        Parameters
        ----------
        n : int
            Number of considered samples from each sample pool. Must be a strictly positive integer.
        function : function
            Functional of the samples of which we want to plot the histogram.
        dim: int
            Considered and plotted dimension of the samples. Must be a positive integer.
        bins: int
            Number of bars in the histogram. Must be a strictly positive integer.
        normalized: bool
            Specify if the histogram should be normalized, so it can provides an estimation of the probability density function.
        plot: bool
            Specify if the histogram should be plotted.
        distribution: Distribution
            Distribution of which we want to plot the PDF.

        Returns
        -------
        np.ndarray
            Histogram of samples values.
        """

        if n is None:
            n = self.n_pool_values

        # If the samples array contains samples from different simulations, we get the number of simulations.
        size = self.samples.shape[0]
        histograms = np.empty((size, bins))
        bins_val = np.empty((size, bins+1))


        for i in range(0,size):
            histograms[i], bins_val[i] = np.histogram(function(self.samples[i,:n,dim]), bins=bins, density=normalized)

        if plot:
            fig = go.Figure()

            for i in range(size):
                fig.add_trace(go.Bar(x=(bins_val[i, :-1] + bins_val[i, 1:]) / 2,
                                     y=histograms[i],
                                     width=np.diff(bins_val[i]),
                                     name=f"Sample pool {i}"))

            fig.update_layout(
                title=f"Distribution of samples values",
                xaxis_title="Values",
                yaxis_title="Frequency",
                template="plotly_white",
            )

            if distribution is not None :
                t = np.linspace(bins_val[0,0],bins_val[0,-1], int(100*np.floor(bins_val[0,-1]-bins_val[0,0])))
                fig.add_trace(go.Scatter(x=t,
                                         y=distribution.pdf(t),
                                         mode="lines",
                                         line=dict(width=2),
                                         name=f"Target PDF ({type(distribution).__name__})"))
            fig.show()

        return histograms

    def ecdf(self,n=None, function = lambda x: x, dim=0, distribution=None):

        if n is None:
            n = self.n_pool_values

        # If the samples array contains samples from different simulations, we get the number of simulations.
        size = self.samples.shape[0]

        min_val = np.min(self.min(n, function))
        max_val = np.max(self.max(n, function))

        fig = go.Figure()
        for i in range(size):

            x = np.sort(self.samples[i, :, dim])
            f = np.arange(1, len(x) + 1) / len(x)
            fig.add_trace(go.Scatter(x=x,
                                     y=f,
                                     mode="lines",
                                     line=dict(width=2),
                                     name=f"eCDE for sample pool {i}"))
        fig.update_layout(
            title=f"Empirical Cumulative Distribution Function",
            xaxis_title="Values",
            yaxis_title="Probability",
            template="plotly_white",
        )

        if distribution is not None:
            t = np.linspace(min_val, max_val, int(1000 * np.floor(max_val - min_val)))
            fig.add_trace(go.Scatter(x=t,
                                     y=distribution.cdf(t),
                                     mode="lines",
                                     line=dict(width=2),
                                     name=f"Target CDF ({type(distribution).__name__})"))


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

    def __init__(self,process,n_simulations=100, method="euler-maruyama"):

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

        if n <= 0:
            raise ValueError("n must be strictly positive.")

        if n > self.n_simulations:

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

        # Apply function to the process samples
        samples = self.ech[:n, t_index]

        # Apply f to X(t_0)
        samples = function(samples)

        # Monte Carlo estimate
        means = np.mean(samples, axis=0)

        return np.atleast_1d(means)

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
