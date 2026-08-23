import numpy as np
import plotly.graph_objects as go
from abc import abstractmethod, ABC

class Distribution(ABC):

    @abstractmethod
    def cdf(self, x=None):

        """
        Cumulative Distribution function method.

        Print or evaluate the cumulative distribution function at a point x.
        The cumulative distribution function is defined by
                            F(x) = P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate the cumulative distribution function.

        Returns
        -------
        float
            Image of x by the cumulative distribution function.
        """

        pass

    @abstractmethod
    def sample(self,n=1):

        """
        Sampling method.

        Sample n points from the effective distribution.

        Parameters
        ----------
        n : int
            Number of samples. Must be a strictly positive integer.

        Returns
        -------
        np.ndarray
            n samples of the effective distribution.
        """

        pass

    @abstractmethod
    def mean(self):

        """
        Mean method.

        Return the mean of the effective distribution.

        Returns
        -------
        float
            Mean of the effective distribution.
        """

        pass

    @abstractmethod
    def variance(self):

        """
        Variance method.

        Return the variance of the effective distribution.

        Returns
        -------
        float
            Variance of the effective distribution.
        """

        pass

    @abstractmethod
    def entropy(self):

        """
        Entropy method.

        Return the Shannon Entropy of the effective distribution.

        Returns
        -------
        float
            Shannon Entropy of the effective distribution.
        """

        pass

    @abstractmethod
    def support(self):
        """
        Support method.

        Return the support of the effective distribution.
        The support of a distribution is the set of real numbers where the probability density function is nonzero.

        Returns
        -------
        tuple
            (lower bound, upper bound) of the support.
        """

        pass


    def plot_cdf(self):

        """
        Plot CDF method.

        Plot the graph of the cumulative distribution function.
        """

        supp = self.support()
        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if callable(self.variance) else 1

        lo_bound = supp[0] if supp[0] > -np.inf else mu - 8 * sd
        up_bound = supp[1] if supp[1] < np.inf else mu + 8 * sd

        supp_length = up_bound - lo_bound
        n_points = 1000
        x_axis = np.linspace(lo_bound, up_bound, n_points)
        y_axis = [self.cdf(x) for x in x_axis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="CDF", line=dict(width=2)))
        fig.update_layout(title="Cumulative Distribution Function",xaxis_title="x",yaxis_title="f(x)")
        fig.show()

    @abstractmethod
    def info(self):
        pass