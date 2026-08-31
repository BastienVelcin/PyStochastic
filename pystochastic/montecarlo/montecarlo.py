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
        - .half_width() : Estimates the half-width of the confidence interval of the mean estimator.
        - .moment() : Estimates a moment of a given order of a function of a samples from a random variable.
        - .variance() : Estimates the variance of a function of samples from a random variable.
        - .std() : Estimates the standard deviation of a function of samples from a random variable.
        - .standard_error() : Estimates the standard error of a function of samples from a random variable.
        - .confidence_interval() : Estimates the confidence interval of the mean estimator of a given random variable.
        - .confidence_curve() : Plots the mean estimator and the confidence interval of the mean estimator for all estimations, from 2 samples to the maximum number of samples specified in attributes.
        - .bias_estimator() : Estimates the bias of the likelihood estimator of E[f(X)] when E[f(X)] is known.
        - .mse_estimator() : Estimates the mean squared error of the likelihood estimator of E[f(X)] when E[f(X)] is known.
        - .rmse_estimator() : Estimates the root mean squared error of the likelihood estimator of E[f(X)] when E[f(X)] is known.
        - .quantile() : Estimates the quantile of order q of a specified array of values.
        - .min() : Estimates the minimum value from samples.
        - .max() : Estimates the maximum value from samples.
        - .median() : Estimates the median value from samples.
        - .skewness() : Estimates the skewness of a function of samples from a random variable.
        - .kurtosis() : Estimates the kurtosis of a function of samples from a random variable.
        - .histogram() : Plots the histogram of samples.
        - .ecdf() : Plots the empirical cumulative distribution function of samples.


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
from scipy.stats import norm, t
from pystochastic.dist import Distribution


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
        self.dim = self.samples.shape[2]

    def _validate_n(self, n):

        if n is None:
            n = self.n_pool_values

        if not n >= 2 or not isinstance(n, (int, np.integer)):
            raise ValueError(
                "The number of considered samples n must be strictly an integer strictly greater than 2."
            )

        if not n <= self.n_pool_values:
            raise ValueError(
                "The number of considered samples n must be inferior to the number of samples."
            )
        return n


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

        n = self._validate_n(n)

        return np.mean(function(self.samples[:,:n]), axis=1)

    def half_width(self, n=None, function = lambda x: x, confidence=0.95, type="normal", variance = None):

        """
        Half Width method.

        Provide the Half Width estimation of the confidence interval of specified confidence level. We can use the normal (with specified or
        approximed variance) and Student confidence intervals.

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

        n = self._validate_n(n)

        if not 0 < confidence < 1:
            raise ValueError(
                "The confidence level must be a float between 0 and 1."
            )

        if type == "normal":
            #Quantile function of the standard normal distribution
            z = norm.ppf(0.5 + confidence / 2)
            if variance is None:
                sd_estimate = self.std(n, function, correction=True)
            else:
                sd_estimate = np.sqrt(variance)

        elif type == "student":
            # Quantile function of the Student distribution with n-1 degrees of freedom
            z = t.ppf(0.5 + confidence / 2, df=n-1)
            sd_estimate = self.std(n, function, correction=True)

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

        n = self._validate_n(n)

        if order < 1 or not isinstance(order, int):
            raise ValueError(
                "The order must be a strictly positive integer."
            )

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

        mean_est = self.estimate(n, function=function)
        half_width = self.half_width(n,function=function,confidence=confidence,type=type)

        return mean_est - half_width, mean_est + half_width

    def confidence_curve(self,n=None,n_pool = 0,function = lambda x : x, confidence = 0.95,type="normal"):

        """
        Confidence Curve method.

        Plot the mean estimator and the confidence interval of the mean estimator for all estimations, from 2 samples to
        the maximum number of samples specified in attributes
        """

        n = self._validate_n(n)

        if not 0 < confidence < 1:
            raise ValueError(
                "The confidence level must be a float between 0 and 1."
            )

        if not 0 <= n_pool < self.n_pool_values:
            raise ValueError(
                "The number of considered sample pool must be inferior to the number of sample pools."
            )

        n_axis = np.arange(2, n + 1)

        # Computation of the mean and variance of the estimator
        samples = function(self.samples[n_pool])
        S1 = np.cumsum(samples[:])[1:n]
        S2 = np.cumsum(samples[:n] ** 2)[1:n]
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
        else:
            raise ValueError(
                "The confidence curve type must be 'normal' or 'student'."
            )

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

    def bias_estimator(self, reference, n=None, function = lambda x: x):

        """
        Bias method

        Provides the bias estimation of the likelihood estimator of E[f(X)] when E[f(X)] is known.

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

        n = self._validate_n(n)

        return self.estimate(n, function=function) - reference

    def mse_estimator(self, reference, n=None, function = lambda x: x):

        """
        MSE method

        Provides the mean squared error estimation of the likelihood estimator of E[f(X)] when E[f(X)] is known.

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

        n = self._validate_n(n)

        return self.estimate(n, function = lambda x : (reference - function(x))**2)

    def rmse_estimator(self, reference, n=None, function = lambda x: x):

        """
        RMSE method

        Provides the root mean squared error estimation of the likelihood estimator of E[f(X)] when E[f(X)] is known.

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

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

        n = self._validate_n(n)

        std = self.std(n, function, correction=True)
        mean = self.estimate(n, function)
        return self.estimate(n, lambda x : np.power((function(x) - mean) / std,3))

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

        n = self._validate_n(n)

        std = self.std(n, function, correction=True)
        mean = self.estimate(n, function)

        return self.estimate(n, lambda x: np.power((function(x)- mean) / std, 4))


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

        n = self._validate_n(n)

        if not (0 <= dim < self.dim and isinstance(dim, (int, np.integer))):
            raise ValueError(
                "The considered dimension must be a strictly positive integer that is inferior to the number of dimensions of the samples."
            )

        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(
                "The number of bins must be a strictly positive integer."
            )
        # If the samples array contains samples from different simulations, we get the number of simulations.
        histograms = np.empty((self.n_simulations, bins))
        bins_val = np.empty((self.n_simulations, bins+1))


        for i in range(0,self.n_simulations):
            histograms[i], bins_val[i] = np.histogram(function(self.samples[i,:n,dim]), bins=bins, density=normalized)

        if plot:
            fig = go.Figure()

            for i in range(self.n_simulations):
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
                t = np.linspace(bins_val[0,0],bins_val[0,-1], 500)
                fig.add_trace(go.Scatter(x=t,
                                         y=distribution.pdf(t),
                                         mode="lines",
                                         line=dict(width=2),
                                         name=f"Target PDF ({type(distribution).__name__})"))
            fig.show()

        return histograms

    def ecdf(self,n=None, function = lambda x: x, dim=0, plot=True, distribution=None):

        n = self._validate_n(n)

        if not (0 <= dim < self.dim and isinstance(dim, (int, np.integer))):
            raise ValueError(
                "The considered dimension must be a strictly positive integer that is inferior to the number of dimensions of the samples."
            )

        if not isinstance(isinstance(N, Distribution)):
            raise ValueError(
                "The distribution must be a Distribution object (pystochastic.dist.Distribution)."
            )

        # If the samples array contains samples from different simulations, we get the number of simulations.

        min_val = np.min(self.min(n, function))
        max_val = np.max(self.max(n, function))


        if plot :
            fig = go.Figure()
            for i in range(self.n_simulations):

                x = np.sort(function(self.samples[i, :n, dim]))
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
        else:
            for i in range(self.n_simulations):
                x = np.sort(function(self.samples[i, :n, dim]))
                f = np.arange(1, len(x) + 1) / len(x)
        return x,f
