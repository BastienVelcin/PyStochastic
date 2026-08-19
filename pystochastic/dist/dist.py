"""
============================================================
Module DIST
============================================================

Description
-----------
This module provides a set of classes to work with probability distributions.

This module provides a general class "Distribution" for continuous distributions, which yields to general methods :
    - .pdf() : Probability density function, which can be evaluated at any point with an optional argument
    - .cdf() : Cumulative distribution function, which can be evaluated at any point with an optional argument
    - .plot_pdf() : Plot the probability density function of the distribution
    - .plot_cdf() : Plot the cumulative distribution function of the distribution
    - .sample() : Sample from the distribution
    - .mean() : Mean of the distribution
    - .variance() : Variance of the distribution
    - .entropy() : Entropy of the distribution
    - .support() : Support of the distribution
    - .infos() : Recap of all information about the current distribution (pdf, cdf, mean, variance, entropy, support)

This module provides a general class "DiscreteDistribution" for discrete distributions, which yields to general methods :
    - .pmf() : Probability mass function, which can be evaluated at any integer point with an optional argument
    - .cdf() : Cumulative distribution function, which can be evaluated at any point with an optional argument
    - .plot_pmf() : Plot the probability mass function of the distribution
    - .plot_cdf() : Plot the cumulative distribution function of the distribution
    - .sample() : Sample from the distribution
    - .mean() : Mean of the distribution
    - .variance() : Variance of the distribution
    - .entropy() : Entropy of the distribution
    - .support() : Support of the distribution
    - .infos() : Recap of all information about the current distribution (pdf, cdf, mean, variance, entropy, support)

From this general class, we introduce various subclasses for each implemented probability distribution.
The available distributions are :

    CONTINUOUS DISTRIBUTIONS
    - Continuous-Time Uniform
    - Exponential
    - Normal
    - Gamma
    - Beta
    - Weibull
    - Frechet
    - Cauchy
    - Gumbel
    - Kumaraswamy
    - Fisher
    - Pareto
    - Rayleigh

    DISCRETE DISTRIBUTIONS
    - Discrete-Time Uniform
    - Bernoulli
    - Rademacher
    - Binomial
    - Poisson
    - Hypergeometric
    - Geometric
    - Negative Binomial
    - Yule Simon

Examples
--------
>> N = Normal(0,1,10) #Normal distribution with mean 0 and standard deviation 1
>>
>> N.sample(10) #Sample 10 random numbers from the distribution
>>
>> N.plot_pdf() #Plot the probability density function of the distribution
"""
from math import comb

import numpy as np
import scipy
import plotly.graph_objects as go
from abc import abstractmethod, ABC
from pystochastic.pyrandom import crandom, drandom
from sympy import harmonic
from mpmath import hyp3f2

class Distribution(ABC):

    @abstractmethod
    def pdf(self, x=None):

        """
        Probability Density Function method.

        Print or evaluate the probability density function at a point x.

        Parameters
        ----------
        x : float
            Point at which to evaluate the the probability density function.

        Returns
        -------
        float
            Image of x by the probability density function.
        """

        pass

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

    def plot_pdf(self):

        """
        Plot PDF method.

        Plot the graph of the probability density function.
        """

        supp = self.support()
        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if (callable(self.variance) or self.variance is not None) else 1

        lo_bound = supp[0] if supp[0] > -np.inf else mu - 8 * sd
        up_bound = supp[1] if supp[1] < np.inf else mu + 8 * sd

        supp_length = up_bound - lo_bound

        x_axis = np.linspace(lo_bound, up_bound, int(1000 * supp_length))
        y_axis = [self.pdf(x) for x in x_axis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="PDF", line=dict(width=2)))
        fig.update_layout(title="Probability Density Function",xaxis_title="x",yaxis_title="f(x)")
        fig.show()

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

    def info(self):
        """
        Info method.

        Print a recap of the effective distribution.
        """
        print(f"| Distribution : {self.__class__.__name__}")
        print(f"| Parameters : {self.__dict__}")
        print(f"| Probability Density Function : {self.pdf()}")
        print(f"| Cumulative Distribution Function : {self.cdf()}")
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

class Uniform(Distribution):

    """
    Continuous-Time Uniform probability distribution.

    The Continuous-Time Uniform distribution is parameterized by two bounds parameterss ``a`` and ``b``.

    Parameters
    ----------
    a : float
        Lower bound. Must be strictly less than ``b``.
    b : float
        Upper bound. Must be strictly greater than ``a``.

    Attributes
    ----------
    lobound : float
        Lower bound.
    upbound : float
        Upper bound.

    Examples
    --------
    >> U = Uniform(a=0,b=1)
    >> U.sample(10)

    Notes
    -----
    If a > b, the constructor automatically swaps the bounds.
    """

    def __init__(self,a=0,b=1):

        if a == b:
            raise ValueError(
                "The lower and upper bound should be different."
            )

        self.lobound = min(a,b)
        self.upbound = max(a,b)

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| {(1/{self.upbound-self.lobound})} for {self.lobound} <= x <= {self.upbound}" )
            print(f"| 0 else")

        else:
            if (self.lobound <= x <= self.upbound):
                return (1/(self.upbound-self.lobound))
            else:
                return 0

    def cdf(self, x=None):

        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < {self.lobound}")
            print(f"| (x-{self.lobound})/{self.upbound-self.lobound} for {self.lobound} <= x <= {self.upbound}")
            print(f"| 1 for x > {self.upbound}")

        else:
            if x < self.lobound:
                return 0
            elif self.lobound <= x <= self.upbound:
                return (x-self.lobound)/(self.upbound-self.lobound)
            else:
                return 1

    def sample(self,n=1):
        return crandom.uniform(self.lobound, self.upbound, n)

    def mean(self):
        return (self.lobound + self.upbound)/2

    def variance(self):
        return ((self.upbound - self.lobound)**2)/12

    def entropy(self):
        return np.log(self.upbound-self.lobound)

    def support(self):
        return (self.lobound, self.upbound)

class Exponential(Distribution):

    """
    Exponential probability distribution.

    The Exponential distribution is parameterized by an intensity parameter ''alpha''.

    Parameters
    ----------
    alpha : float
        Intensity parameter, or scale parameter inverse. Must be strictly positive.

    Attributes
    ----------
    alpha : float
        Intensity parameter, or scale parameter inverse.

    Examples
    --------
    >> E = Exponential(alpha=2)
    >> E.sample(10)
    """

    def __init__(self,alpha=1):

        if alpha <=0:
            raise ValueError(
                "The parameter should be greater than 0."
            )

        self.alpha = alpha

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| {self.alpha} * exp(-{self.alpha}*x) for x >= 0")
        else:
            if x < 0:
                return 0
            else:
                return self.alpha*np.exp(-self.alpha*x)

    def cdf(self, x=None):

        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1- exp(-{self.alpha}*x) for x >= 0")

        else:
            if x < 0:
                return 0
            else:
                return 1- np.exp(-self.alpha*x)

    def sample(self,n=1):
        return crandom.exponential(self.alpha, n)

    def mean(self):
        return 1/self.alpha

    def variance(self):
        return 1/self.alpha**2

    def entropy(self):
        return 1-np.log(self.alpha)

    def support(self):
        return (0, np.inf)

class Normal(Distribution):

    """
    Normal probability distribution.

    The Normal distribution is parameterized by a mean parameter ``mu`` and a standard deviation parameter ``sd``.

    Parameters
    ----------
    mu : float
        Mean parameter.
    sd : float
        Standard deviation parameter. Must be strictly positive.

    Attributes
    ----------
    mu : float
        Mean parameter.

    sd : float
        Standard deviation parameter. Must be strictly positive.

    Examples
    --------
    >> N = Normal(mu=0,sd=1)
    >> N.sample(10)
    """

    def __init__(self,mu=0,sd=1):

        if sd <=0:
            raise ValueError(
                "The standard deviation should be greater than 0."
            )

        self.mu = mu
        self.sd = sd

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| (1/{self.sd}*sqrt(2*pi)) * exp(-(x-{self.mu})^2 / 2*{self.sd}^2)")
        else:
            return (1/(self.sd*np.sqrt(2*np.pi)))*np.exp(-(x - self.mu)**2 / (2*self.sd**2))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| (1+erf((x-{self.mu})/({self.sd}*sqrt(2))))/2")
        else:
            return (1+scipy.special.erf((x-self.mu)/(self.sd*np.sqrt(2))))/2

    def sample(self,n=1):
        return crandom.normal(self.mu, self.sd, n)

    def mean(self):
        return self.mu

    def variance(self):
        return self.sd**2

    def entropy(self):
        return np.log(self.sd*np.sqrt(2*np.pi*np.e))

    def support(self):
        return (-np.inf ,np.inf)

class Gamma(Distribution):

    """
    Gamma probability distribution.

    The Gamma distribution is parameterized by a shape parameter
    ``k`` and a rate parameter ``theta``.

    Parameters
    ----------
    k : float
        Shape parameter. Must be strictly positive.
    theta : float
        Rate parameter. Must be strictly positive.

    Attributes
    ----------
    k : float
        Shape parameter.
    theta : float
        Rate parameter.

    Examples
    --------
    >> G = Gamma(k=2,theta=1)
    >> G.sample(10)
    """

    def __init__(self,k=1,theta=1):

        if k <=0:
            raise ValueError(
                "The form parameter should be greater than 0."
            )

        if theta <=0:
            raise ValueError(
                "The rate parameter should be greater than 0."
            )

        self.k = k
        self.theta = theta

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| ({self.theta**self.k} * x^{self.k-1} *  exp(-{self.theta}*x))/Gamma({self.k}) for x>= 0")
        else:
            if x<0:
                return 0
            return ((self.theta**self.k)*(x**(self.k-1))*np.exp(-self.theta*x))/scipy.special.gamma(self.k)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| IncGamma({self.k}, {self.theta}*x)/Gamma({self.k}) for x >= 0")
        else:
            if x<0:
                return 0
            return scipy.special.gammainc(self.k,self.theta*x)

    def sample(self,n=1):
        return crandom.gamma(self.k, self.theta, n)

    def mean(self):
        return self.k/self.theta

    def variance(self):
        return self.k/(self.theta**2)

    def entropy(self):
        return self.k/self.theta + (1-self.k)*np.log(1/self.theta) + np.log(scipy.special.gamma(self.k)) + (1+self.k)*scipy.special.digamma(self.k)

    def support(self):
        return (0,np.inf)

class Beta(Distribution):

    """
    Beta probability distribution.

    The Beta distribution is parameterized by two shape parameters
    ``a`` and ''b''.

    Parameters
    ----------
    a : float
        First shape parameter. Must be strictly positive.
    b : float
        Second shape parameter. Must be strictly positive.

    Attributes
    ----------
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Examples
    --------
    >> B = Beta(a=2,b=3/2)
    >> B.sample(10)
    """

    def __init__(self,a=1,b=1):

        if a <= 0:
            raise ValueError(
                "The first shape parameter should be greater than 0."
            )
        if b <= 0:
            raise ValueError(
                "The second shape parameter should be greater than 0."
            )

        self.a = a
        self.b = b

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0 or x > 1")
            print(f"| (x^{self.a-1} * (1-x)^{self.b-1})/{scipy.special.beta(self.a,self.b)} for 0 <= x <= 1")
        else:
            if not (0 <= x <= 1):
                return 0
            return ((x**(self.a-1)) * ((1-x)**(self.b-1)))/scipy.special.beta(self.a,self.b)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| IncBeta(x,{self.a}, {self.b})/{scipy.special.beta(self.a,self.b)} if 0 <= x <= 1")
            print(f"| 1 for x > 1")
        else:
            if x < 0:
                return 0
            elif x > 1:
                return 1
            else:
                return scipy.special.betainc(self.a, self.b, x)

    def sample(self,n=1):
        return crandom.beta(self.a, self.b, n)

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        return (self.a * self.b) / ((self.a + self.b)**2 * (self.a + self.b + 1))

    def entropy(self):
        return np.log(scipy.special.beta(self.a,self.b)) - (self.a-1)*scipy.special.digamma(self.a) - (self.b-1)*scipy.special.digamma(self.b) + (self.a+self.b-2)*scipy.special.digamma(self.a+self.b)

    def support(self):
        return (0,1)

class Weibull(Distribution):

    """
    Weibull probability distribution.

    The Weibull distribution is parameterized by a shape parameter
    ``k`` and a scale parameter ''l''.

    Parameters
    ----------
    k : float
        Shape parameter. Must be strictly positive.
    l : float
        Scale parameter. Must be strictly positive.

    Attributes
    ----------
    k : float
        Shape parameter.
    l : float
        Scale parameter.

    Examples
    --------
    >> W = Weibull(k=2,l=3/2)
    >> W.sample(10)
    """

    def __init__(self,k=1,l=1):

        if k <=0:
            raise ValueError(
                "The shape parameter should be greater than 0."
            )

        if l <=0:
            raise ValueError(
                "The scale parameter should be greater than 0."
            )

        self.k = k
        self.l = l

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| {self.k/self.l}*(x/{self.l})^{self.k-1}*exp(-(x/{self.l})^{self.k}) for x >= 0")
        else:
            if x < 0:
                return 0
            return (self.k/self.l)*((x/self.l)**(self.k-1))*np.exp(-(x/self.l)**self.k)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1 - exp(-(x/{self.l})^{self.k}) for x >= 0")
        else:
            return 1 - np.exp(-(x/self.l)**self.k)

    def sample(self, n=1):
        return crandom.weibull(self.k, self.l, n)

    def mean(self):
        return self.l * scipy.special.gamma(1 + 1 / self.k)

    def variance(self):
        return self.l ** 2 * scipy.special.gamma(1 + 2 / self.k) - self.mean()**2

    def entropy(self):
        return scipy.special.gammainc(1-1/self.k)+(self.l/self.k)**self.k+np.log(self.l)-np.log(self.l/self.k)

    def support(self):
        return (0, np.inf)

class Frechet(Distribution):

    """
    Fréchet probability distribution.

    The Fréchet distribution is parameterized by a shape parameter
    ``a``, a scale parameter ''s'' and a position parameter ``m``.

    Parameters
    ----------
    a : float
        Shape parameter. Must be strictly positive.
    s : float
        Scale parameter. Must be strictly positive.
    m : float
        Position parameter.

    Attributes
    ----------
    a : float
        Shape parameter.
    s : float
        Scale parameter.
    m : float
        Position parameter.

    Examples
    --------
    >> F = Frechet(a=2,s=3/2,m=-3)
    >> F.sample(10)
    """

    def __init__(self,a=1,s=1,m=0):

        if a <= 0:
            raise ValueError(
                "The shape parameter should be greater than 0."
            )

        if s <= 0:
            raise ValueError(
                "The scale parameter should be greater than 0."
            )

        self.a = a
        self.s = s
        self.m = m

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < {self.m}")
            print(f"| {self.a/self.s}*((x-{self.m})/{self.s})^{-1-self.a} * exp(-((x-{self.m})/{self.s})^{-self.a}) for x >= {self.m}")
        else:
            if x <= self.m:
                return 0
            return (self.a/self.s)*((x-self.m)/(self.s))**(-1-self.a) * np.exp(-((x-self.m)/(self.s))**(-self.a))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| exp(-((x-{self.m})/{self.s})^{-self.a})")
        else:
            if x <=self.m:
                return 0
            return  np.exp(-((x-self.m)/(self.s))**(-self.a))

    def sample(self, n=1):
        return crandom.frechet(self.a, self.s, self.m, n)

    def mean(self):
        if self.a > 1:
            return self.m + self.s * scipy.special.gamma(1 - 1 / self.a)
        else:
            return np.inf

    def variance(self):
        if self.a > 2:
            return self.s ** 2 * (scipy.special.gamma(1 - 2 / self.a) - scipy.special.gamma(1 - 1 / self.a)**2)
        else:
            return np.inf
    def entropy(self):
        return 1 + np.euler_gamma/self.a + np.euler_gamma + np.log(self.s/self.a)

    def support(self):
        return (self.m, np.inf)

class Cauchy(Distribution):

    """
    Cauchy probability distribution.

    The Cauchy distribution is parameterized by a position parameter
    ``x`` and a scale parameter ''a''.

    Parameters
    ----------
    x : float
        Position parameter.
    a : float
        Scale parameter. Must be strictly positive.

    Attributes
    ----------
    x : float
        Position parameter.
    a : float
        Scale parameter.

    Examples
    --------
    >> C = Cauchy(x=0,a=1)
    >> C.sample(10)
    """

    def __init__(self,x=0,a=1):

        if a <= 0:
            raise ValueError(
                "The scale parameter should be greater than 0."
            )

        self.x = x
        self.a = a

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 1/(pi*{self.a}*(1+(x-{self.x})/{self.a})^2)")
        else:
            return 1/(np.pi*self.a*(1+(x-self.x)/self.a)**2)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1/pi * Arctan((x-{self.x})/{self.a}) + 1/2")
        else:
            return 1/np.pi * np.arctan((x-self.x)/self.a) + 1/2

    def sample(self, n=1):
        return crandom.cauchy(self.x, self.a, n)

    def mean(self):
        return None

    def variance(self):
        return None

    def entropy(self):
        return np.log(4*np.pi*self.a)

    def support(self):
        return (-np.inf, np.inf)

class Gumbel(Distribution):

    """
    Gumbel probability distribution.

    The Gumbel distribution is parameterized by a position parameter
    ``mu`` and a scale parameter ''beta''.

    Parameters
    ----------
    mu : float
        Position parameter.
    beta : float
        Scale parameter. Must be strictly positive.

    Attributes
    ----------
    mu : float
        Position parameter.
    beta : float
        Scale parameter.

    Examples
    --------
    >> G = Gumbel(mu=-1/2,beta=2)
    >> G.sample(10)
    """

    def __init__(self,mu=0,beta=1):

        if beta <= 0:
            raise ValueError(
                "The scale parameter should be greater than 0."
            )

        self.mu = mu
        self.beta = beta

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| {1/self.beta} * exp(-exp(-(x-{self.mu})/{self.beta})) * exp(-(x-{self.mu})/{self.beta})")
        else:
            return (1/self.beta) * np.exp(-np.exp(-(x-self.mu)/self.beta)) * np.exp(-(x-self.mu)/self.beta)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| exp(-exp(-(x-{self.mu})/{self.beta}))")
        else:
            return np.exp(-np.exp(-(x-self.mu)/self.beta))

    def sample(self, n=1):
        return crandom.gumbel(self.mu, self.beta, n)

    def mean(self):
        return self.mu+self.beta*np.euler_gamma

    def variance(self):
        return (np.pi**2)*(self.beta**2)/6

    def entropy(self):
        return np.log(self.beta) + np.euler_gamma + 1

    def support(self):
        return (-np.inf, np.inf)


class Kumaraswamy(Distribution):

    """
    Kumaraswamy probability distribution.

    The Kumaraswamy distribution is parameterized by a two shape parameters
    ``a`` and ''b''.

    Parameters
    ----------
    a : float
        First shape parameter. Must be strictly positive.
    b : float
        Second shape parameter. Must be strictly positive.

    Attributes
    ----------
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Examples
    --------
    >> K = Kumaraswamy(a=3/2,b=5)
    >> K.sample(10)
    """

    def __init__(self,a=1,b=1):

        if a <= 0:
            raise ValueError(
                "The first shape parameter should be greater than 0."
            )

        if b <= 0:
            raise ValueError(
                "The second shape parameter should be greater than 0."
            )

        self.a = a
        self.b = b

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0 or x > 1")
            print(f"| {self.a*self.b}* x^{self.a-1} * (1-x^{self.a})^{self.b-1} for 0 <= x <= 1")
        else:
            if not 0 <=x <= 1:
                return 0
            return (self.a*self.b)*(x**(self.a-1)) * (1-x**self.a)**(self.b-1)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1-(1-x^{self.a})**{self.b} for 0 <= x <= 1")
            print(f"| 1 for x > 1")
        else:
            if x < 0:
                return 0
            elif x > 1:
                return 1
            else:
                return 1-(1-x**self.a)**self.b

    def sample(self, n=1):
        return crandom.kumaraswamy(self.a, self.b, n)

    def mean(self):
        return (self.b*scipy.special.gamma(1+1/self.a)*scipy.special.gamma(self.b))/scipy.special.gamma(1 + self.b + 1/self.a)

    def variance(self):
        return self.b * scipy.special.beta(1+2/self.a, self.b) - self.mean()**2

    def entropy(self):
        return (1-1/self.b)+(self.a-1)*harmonic(self.b) - np.log(self.a*self.b)

    def support(self):
        return (0,1)

class Fisher(Distribution):

    """
    Fisher probability distribution.

    The Fisher distribution is parameterized by a two degree of freedom parameters
    ``d1`` and ''d2''.

    Parameters
    ----------
    d1 : float
        First degree of freedom. Must be strictly positive.
    d1 : float
        Second degree of freedom. Must be strictly positive.

    Attributes
    ----------
    d1 : float
        First degree of freedom.
    d1 : float
        Second degree of freedom.

    Examples
    --------
    >> F = Fisher(d1=2,d2=5)
    >> F.sample(10)
    """

    def __init__(self,d1=1,d2=1):

        if d1 <= 0:
            raise ValueError(
                "The first degree of freedom should be greater than 0."
            )

        if d2 <= 0:
            raise ValueError(
                "The second degree of freedom should be greater than 0."
            )

        self.d1 = d1
        self.d2 = d2

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| sqrt[(({self.d1}*x)^{self.d1}*{self.d2}^{self.d2})/(({self.d1}*x+{self.d2})^{self.d1+self.d2})]/(x*Beta({self.d1/2}, {self.d2/2})) for x >= 0")
        else:
            if x <= 0:
                return 0
            return np.sqrt((((self.d1*x)**self.d1)*self.d2**self.d2)/((self.d1*x+self.d2)**(self.d1+self.d2)))/(x*scipy.special.beta(self.d1/2, self.d2/2))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| RegIncBeta(({self.d1}*x)/({self.d1}*x+{self.d2}),{self.d1/2},{self.d2/2}) for x >= 0")
        else:
            if x <= 0:
                return 0
            return scipy.special.betainc(self.d1/2,self.d2/2,(self.d1*x)/(self.d1*x+self.d2))

    def sample(self, n=1):
        return crandom.fisher(self.d1, self.d2, n)

    def mean(self):
        if self.d2 > 2:
            return self.d2 / (self.d2 - 2)
        return None
    def variance(self):
        if self.d2 > 4:
            return (2*self.d2**2)*(self.d1+self.d2-2)/(self.d1*((self.d2-2)**2)*(self.d2-4))
        return None
    def entropy(self):
       return np.log(scipy.special.gamma(self.d1/2))+np.log(scipy.special.gamma(self.d2/2)) - np.log(scipy.special.gamma((self.d1+self.d2)/2)) + (1-self.d1/2)*scipy.special.digamma(1+self.d1/2) - (1+self.d2/2)*scipy.special.digamma(1+self.d2/2) + ((self.d1+self.d2)/2)*scipy.special.digamma((self.d1+self.d2)/2) + np.log(self.d2/self.d1)

    def support(self):
        return (0,np.inf)

class Pareto(Distribution):

    """
    Pareto probability distribution.

    The Pareto distribution is parameterized by a position parameter ''x_m'' and
    a shape parameter ''k''.

    Parameters
    ----------
    x_m : float
        Position parameter. Must be strictly positive.
    k : float
        Shape parameter. Must be strictly positive.

    Attributes
    ----------
    x_m : float
        Position parameter. Must be strictly positive.
    k : float
        Shape parameter. Must be strictly positive.

    Examples
    --------
    >> P = Pareto(x_m=2,k=1/2)
    >> P.sample(10)
    """

    def __init__(self,x_m=1,k=1):

        if x_m <= 0:
            raise ValueError(
                "The position parameter should be greater than 0."
            )

        if k <= 0:
            raise ValueError(
                "The shape parameter should be greater than 0."
            )

        self.x_m = x_m
        self.k = k

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < {self.x_m}")
            print(f"| {self.k*(self.x_m)**(self.k)}/(x^{self.k+1}) if x >= {self.x_m}")
        else:
            if x < self.x_m:
                return 0
            return (self.k*(self.x_m)**(self.k))/(x**(self.k+1))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < {self.x_m}")
            print(f"| 1 - ({self.x_m}/x)^{self.k} for x >= {self.x_m}")
        else:
            if x <= self.x_m:
                return 0
            return  1 - (self.x_m/x)**(self.k)

    def sample(self, n=1):
        return crandom.pareto(self.x_m, self.k, n)

    def mean(self):
        if self.k > 1:
            return self.x_m * self.k / (self.k - 1)
        return None

    def variance(self):
        if self.k > 2:
            return (self.x_m**2) * self.k / (((self.k - 1)**2) * (self.k-2))
        return None
    def entropy(self):
        return np.log(self.k/self.x_m)-1/self.k -1

    def support(self):
        return (self.x_m, np.inf)

class Rayleigh(Distribution):

    """
    Rayleigh probability distribution.

    The Rayleigh distribution is parameterized by a scale parameter ''s''.

    Parameters
    ----------
    s : float
        Scale parameter. Must be strictly positive.

    Attributes
    ----------
    s : float
        Scale parameter.

    Examples
    --------
    >> R = Rayleigh(s=4)
    >> R.sample(10)
    """

    def __init__(self,s=1):

        if s <= 0:
            raise ValueError(
                "The scale parameter should be greater than 0."
            )

        self.s = s

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| {1/self.s**2}*x*exp(-x^2/(2*s^2)) for x >= 0")
        else:
            if x < 0:
                return 0
            return (1/self.s**2)*x*np.exp(-(x**2)/(2*self.s**2))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1 - exp(-x^2/(2*s^2)) for x >= 0")
        else:
            if x < 0:
                return 0
            return  1 - np.exp(-(x**2)/(2*self.s**2))

    def sample(self, n=1):
        return crandom.rayleigh(self.s, n)

    def mean(self):
        return self.s*np.sqrt(np.pi/2)

    def variance(self):
        return (self.s**2)*(4-np.pi)/2

    def entropy(self):
        return 1 + np.log(self.s/np.sqrt(2))+np.euler_gamma/2

    def support(self):
        return (0, np.inf)


class DiscreteDistribution(ABC):

    @abstractmethod
    def pmf(self, x=None):

        """
        Probability Mass Function method.

        Print or evaluate the probability mass function at a point x.

        Parameters
        ----------
        x : float
            Point at which to evaluate the probability mass function.

        Returns
        -------
        float
            Image of x by the probability mass function.
        """

        pass

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

    def plot_pmf(self):

        """
        Plot PDF method.

        Plot the graph of the probability density function.
        """

        supp = self.support()

        if type(supp) != set:
            lo_bound = -0.5
            up_bound = 20
        else:

            lo_bound = min(supp)
            up_bound = max(supp) + 1

        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if callable(self.variance) else 1

        supp_length = up_bound - lo_bound
        n_points = 1000

        x_axis = np.arange(np.floor(lo_bound), np.floor(up_bound),dtype=int)
        pmf = [self.pmf(x) for x in x_axis]

        fig = go.Figure()

        for xi, pi in zip(x_axis, pmf):
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi],
                    y=[0, pi],
                    mode="lines",
                    line=dict(color="#EF553B"),
                    showlegend=False
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=pmf,
                mode="markers",
                marker=dict(size=8,color="#EF553B"),
                name="PMF"
            )
        )

        fig.update_layout(
            title=f"Probability mass function",
            xaxis_title="k",
            yaxis_title="P(X = k)",
            template="plotly_white"
        )

        fig.show()

    def plot_cdf(self):

        """
        Plot CDF method.

        Plot the graph of the cumulative distribution function.
        """

        supp = self.support()
        if type(supp) != set:
            lo_bound = -0.5
            up_bound = 20
        else:

            lo_bound = min(supp)-0.5
            up_bound = max(supp)+0.5

        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if callable(self.variance) else 1


        supp_length = up_bound - lo_bound
        n_points = 1000
        x_axis = np.linspace(lo_bound, up_bound, n_points)
        y_axis = [self.cdf(x) for x in x_axis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="CDF", line=dict(width=2,shape="hv")))
        fig.update_layout(title="Cumulative Distribution Function", xaxis_title="x", yaxis_title="f(x)")
        fig.show()

    def info(self):
        """
        Info method.

        Print a recap of the effective distribution.
        """
        print(f"| Distribution : {self.__class__.__name__}")
        print(f"| Parameters : {self.__dict__}")
        print(f"| Probability Density Function : {self.pdf()}")
        print(f"| Cumulative Distribution Function : {self.cdf()}")
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")


class DUniform(DiscreteDistribution):

    """
    Discrete-Time Uniform probability distribution.

    The Discrete-Time Uniform distribution is parameterized by a number of outcome values ''n''.

    Parameters
    ----------
    N : int
        Number of outcome values. Must be a integer strictly greater that 2.

    Attributes
    ----------
    N : int
        Number of outcome values.

    Examples
    --------
    >> U = DUniform(N=3)
    >> U.sample(12)
    """

    def __init__(self,N=2):

        if N <= 1 or not isinstance(N, (int, np.integer)):
            raise ValueError(
                "The number of outcome values must be an integer strictly greater than 1."
            )

        self.N = N

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| {1/self.N} if 1 <= k <= {self.N}")
            print(f"| 0 otherwise")
        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            return 1/self.N if k in np.arange(1, self.N+1) else 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 if x < 1")
            print(f"| floor(x)/{self.N} if 1 <= x < {self.N}")
            print(f"| 1 for x >= {self.N}")
        else:
            if x < 1:
                return 0
            elif 1 <= x < self.N:
                return np.floor(x)/self.N
            else:
                return 1

    def sample(self, n=1):
        return drandom.duniform(self.N,n)

    def mean(self):
        return (self.N+1)/2

    def variance(self):
        return (self.N**2 - 1)/12

    def entropy(self):
        return np.log(self.N)

    def support(self):
        return set(range(1, self.N+1))

class Bernoulli(DiscreteDistribution):

    """
    Bernoulli probability distribution.

    The Bernoulli distribution is parameterized by a probability of success parameter ``p``.

    Parameters
    ----------
    p : float
        Probability of success. Must be in the interval [0,1].

    Attributes
    ----------
    p : float
        Probability of success.
    q : float
        Probability of failure.

    Examples
    --------
    >> B = Bernoulli(p=0.2)
    >> B.sample(5)
    """

    def __init__(self,p=0.5):

        if not 0 <= p <= 1:
            raise ValueError(
                "The success probability must be between 0 and 1."
            )

        self.p = p
        self.q = 1- p

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| {self.q} if k = 0")
            print(f"| {self.p} if k = 1")
            print(f"| 0 otherwise")
        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k not in [0,1]:
                return 0
            else:
                return self.p if k == 1 else self.q

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 if x < 0")
            print(f"| {self.q} if 0 <= x < 1")
            print(f"| 1 for x >= 1")
        else:
            if x < -1:
                return 0
            elif 0 <= x < 1:
                return self.q
            else:
                return 1

    def sample(self, n=1):
        return drandom.bernoulli(self.p,n)

    def mean(self):
        return self.p

    def variance(self):
        return self.p * self.q

    def entropy(self):
        return -self.q*np.log(self.q) - self.p*np.log(self.p)

    def support(self):
        return {0,1}


class Rademacher(DiscreteDistribution):

    """
    Rademacher probability distribution.

    The Rademacher distribution is parameterized by a probability of gain parameter ``p``.

    Parameters
    ----------
    p : float
        Probability of gain. Must be in the interval [0,1].

    Attributes
    ----------
    p : float
        Probability of gain.
    q : float
        Probability of loss.

    Examples
    --------
    >> R = Rademacher(p=0.7)
    >> R.sample(10)
    """

    def __init__(self,p=0.5):

        if not 0 <= p <= 1:
            raise ValueError(
                "The success probability must be between 0 and 1."
            )

        self.p = p
        self.q = 1- p

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| {self.q} if k = -1")
            print(f"| {self.p} if k = 1")
            print(f"| 0 otherwise")

        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k not in [-1,1]:
                return 0
            else:
                return self.p if k == 1 else self.q

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 if x < -1")
            print(f"| {self.q} if -1 <= x < 1")
            print(f"| 1 for x >= 1")
        else:
            if x < 0:
                return 0
            elif -1 <= x < 1:
                return self.q
            else:
                return 1

    def sample(self, n=1):
        return drandom.bernoulli(self.p,n)

    def mean(self):
        return 2*self.p - 1

    def variance(self):
        return 1

    def entropy(self):
        return -self.q*np.log(self.q) - self.p*np.log(self.p)

    def support(self):
        return {-1,1}

class Binomial(DiscreteDistribution):

    """
    Binomial probability distribution.

    The Binomial distribution is parameterized by a probability of success parameter ``p`` an
    a repetition parameter ''n''.

    Parameters
    ----------
    p : float
        Probability of gain. Must be in the interval [0,1].
    n : int
        Number of repetitions. Must be a strictly positive integer.

    Attributes
    ----------
    p : float
        Probability of gain.
    q : float
        Probability of failure.
    n : int
        Number of repetitions. Must be a strictly positive integer.

    Examples
    --------
    >> B = Binomial(p=0.6,n=7)
    >> B.sample(5)
    """

    def __init__(self,p=0.5,n=2):

        if not 0 <= p <= 1:
            raise ValueError(
                "The success probability must be between 0 and 1."
            )

        if  n < 1 or not isinstance(n, (int, np.integer)):
            raise ValueError(
                "The number of repetition must be a strictly positive integer."
            )

        self.p = p
        self.q = 1 - p
        self.n = n

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| binom(n,k) * p^k * (1-p)^(n-k) if 0 <= k <= n")
            print(f"| 0 otherwise")

        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k not in range(self.n+1):
                return 0
            else:
                return comb(self.n,k)*self.p**k*(self.q)**(self.n-k)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 if x < 0")
            print(f"| IncBeta({self.q}, {self.n} - floor(x), 1 + floor(x) ) if 0 <= x < n")
            print(f"| 1 for x >= n")
        else:
            if 0 <= x:
                return scipy.special.betainc(self.n-np.floor(x), 1+np.floor(x), self.q)
            else:
                return 1 if x >= self.n else 0

    def sample(self, n=1):
        return drandom.binomial(self.p,self.n,n)

    def mean(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * self.q

    def entropy(self):
        return 0.5*np.log(2*np.pi*self.n*np.e*self.p*self.q)

    def support(self):
        return set(range(self.n+1))


class Poisson(DiscreteDistribution):

    """
    Poisson probability distribution.

    The Poisson distribution is parameterized by a intensity parameter ''lam''

    Parameters
    ----------
    lam : float
        Intensity paramter. Must be strictly positive.

    Attributes
    ----------
    lam : float
        Intensity paramter. Must be strictly positive.

    Examples
    --------
    >> L = Poisson(lam=2)
    >> L.sample(11)
    """

    def __init__(self,lam=0.5):

        if not lam > 0:
            raise ValueError(
                "The intensity must be strictly positive."
            )

        self.lam = lam

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| exp(-{self.lam}) * {self.lam}^k / k! if k >= 0")
            print(f"| 0 otherwise")

        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k >= 0:
                return np.exp(-self.lam) * ((self.lam)**k)/scipy.special.factorial(k)
            else:
                return 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 if x < 0")
            print(f"| RegUpGamma(floor(x)+1, {self.lam})/(floor(x)!) if x >= 0")
        else:
            if x >= 0:
                return scipy.special.gammaincc(np.floor(x)+1,self.lam)
            return 0

    def sample(self, n=1):
        return drandom.poisson(self.lam,n)

    def mean(self):
        return self.lam

    def variance(self):
        return self.lam

    def entropy(self):
        return self.lam * (1 - np.log(self.lam)) + np.exp(-self.lam) * np.sum([self.lam**k * np.log(scipy.special.factorial(k)) / scipy.special.factorial(k) for k in range(1, 100)])

    def support(self):
        return "N"

class Hypergeometric(DiscreteDistribution):

    """
    Hypergeometric probability distribution.

    The Hypergeometric distribution is parameterized by a the population size ''N'', a number
    of successes ''m'' and a number of draws ''k''.

    Parameters
    ----------
    N : int
        Population size. Must be a strictly positive integer.
    K : int
        Number of draws. Must be a strictly positive integer such that 0 <= k <= N.
    m : int
        Number of success states in the considered population. Must be a strictly positive integer such that 0 <= m <= N.

    Attributes
    ----------
    N : int
        Population size.
    K : int
        Number of draws.
    m : int
        Number of success states in the considered population.


    Examples
    --------
    >> H = HyperGeometric(N=12,K=6,m=3)
    >> H.sample(10)
    """

    def __init__(self,N=2,K=1,m=1):

        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError(
                "The population size must be a strictly positive integer."
            )

        if type(m) != int or m < 0:
            raise ValueError(
                "The number of success states in the considered population must be a positive integer."
            )

        if not isinstance(K, (int, np.integer)) or K < 0:
            raise ValueError(
                "The number of draws must be a positive integer."
            )

        if not (0 <= K <= N and 0 <= m <= N):
            raise ValueError(
                "The parameters must satisfy 0 <= k <= N and 0 <= m < N."
            )

        self.N = N
        self.m = m
        self.K = K

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| comb({self.N},k) * comb({self.N-self.m},{self.m}-k) / {comb(self.N,self.K)} if {max(0,self.m+self.K-self.N)} <= k <= {min(self.m,self.K)}")
            print(f"| 0 otherwise")

        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if max(0,self.m+self.K-self.N) <= k <= min(self.m,self.K):
                return comb(self.N,k) * comb(self.N-self.m,self.m-k) / comb(self.N,self.K)
            else:
                return 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1 - comb({self.m},k+1)*comb({self.N-self.m},{self.K-1}-k) / {comb(self.N,self.K)} * GenHypFct(3,2,1,k+{1-self.K},k+{1-self.m},k+2,{self.N+2-self.K-self.m}+k,1)")
            print(f"| for k = floor(x) and x > {max(0,self.m+self.K-self.N)}")
            print(f"| 0 otherwise")
        else:
            if max(0,self.m+self.K-self.N) <= x:
                k = np.floor(x)
                return 1 - comb(self.m,k+1)*comb(self.N-self.m,self.K-1-k) / comb(self.N,self.K) * hyp3f2(1,k+1-self.K,k+1-self.m,k+2,self.N+2-self.K-self.m+k,1)
            else:
                return 0

    def sample(self, n=1):
        return drandom.hypergeometric(self.N,self.K,self.m,n)

    def mean(self):
        return self.m * self.N / self.K

    def variance(self):
        return self.m * (self.K/self.N) * ((self.N-self.K)/self.N) * ((self.N-self.m)/(self.N-1))

    def entropy(self):
        return "Unknown"

    def support(self):
        return set(i for i in range(max(0,self.m+self.K-self.N), min(self.m,self.K)+1))

class Geometric(DiscreteDistribution):

    """
    Geometric probability distribution.

    The Geometric distribution is parameterized by a success probability parameter ''p''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 and 1.

    Attributes
    ----------
    p : float
        Success probability.
    q : float
        Failure probability.

    Examples
    --------
    >> G = Geometric(p=0.3)
    >> G.sample(12)
    """

    def __init__(self,p=0.5):

        if not 0 < p <= 1:
            raise ValueError(
                "The success probability must be between 0 (strictly) and 1."
            )

        self.p = p
        self.q = 1 - p

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| {self.p} * {self.q}^(k-1) if k >= 1")
            print(f"| 0 otherwise")

        else:
            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k >= 1:
                return self.p * (self.q)**(k-1)
            else:
                return 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1 - {self.q}^floor(x) if x >= 1")
            print(f"| 0 otherwise")

        else:
            if x >= 1:
                k = np.floor(x)
                return 1-self.q**k
            else:
                return 0

    def sample(self, n=1):
        return drandom.geometric(self.p,n)

    def mean(self):
        return 1/self.p

    def variance(self):
        return self.q/(self.p**2)

    def entropy(self):
        return (-self.q*np.log(self.q) - self.p * np.log(self.p))/self.p

    def support(self):
        return "N*"

class NegativeBinomial(DiscreteDistribution):

    """
    Negative Binomial probability distribution.

    The Negative Binomial distribution is parameterized by a success probability parameter ''p'' and
    a target success occurrence parameter ''k''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 (strictly) and 1.
    k : int
        Target success occurrence parameter. Must be a strictly positive integer.

    Attributes
    ----------
    p : float
        Success probability.
    p : float
        Failure probability.
    k : int
        Target success occurrence parameter.

    Examples
    --------
    >> B = NegativeBinomial(p=0.4,k=5)
    >> B.sample(5)
    """

    def __init__(self,p=0.5,n=1):

        if not 0 < p <= 1:
            raise ValueError(
                "The success probability must be between 0 and 1."
            )

        if type(n) != int or n < 1:
            raise ValueError(
                "The number of target success occurrence must be a strictly positive integer."
            )

        self.p = p
        self.q = 1 - p
        self.n = n

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| comb(k+{self.n-1}, k)*{self.p**self.n}*{self.q}^k if k >= 1")
            print(f"| 0 otherwise")

        else:
            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k >= 1:
                return comb(k+self.n-1, k)*self.p**self.n*(self.q)**k
            else:
                return 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| IncRegGamma({self.n}, floor(x)+1) if x >= 1")
            print(f"| 0 otherwise")

        else:
            if x >= 1:
                return scipy.special.betainc(self.n,np.floor(x)+1,self.p)
            else:
                return 0

    def sample(self, n=1):
        return drandom.negative_binomial(self.p,self.n,n)

    def mean(self):
        return self.n * self.q / self.p

    def variance(self):
        return self.n * self.q / (self.p ** 2)

    def entropy(self):
        return "Unknown"

    def support(self):
        return "N*"

class YuleSimon(DiscreteDistribution):

    """
    Yule Simon probability distribution.

    The Yule-Simon distribution is parameterized by a shape parameter ''rho''.

    Parameters
    ----------
    rho : float
        Form parameter. Must be strictly positive.

    Attributes
    ----------
    rho : float
        Form parameter.

    Examples
    --------
    >> Y = YuleSimon(rho=2)
    >> Y.sample(15)
    """

    def __init__(self,rho=1):

        if rho <= 0:
            raise ValueError(
                "The shape parameter must be strictly positive."
            )

        self.rho = rho

    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| {self.rho} * Beta(k, {self.rho+1}) if k >= 1")
            print(f"| 0 otherwise")

        else:
            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if k >= 1:
                return scipy.special.beta(k, self.rho+1) * self.rho
            else:
                return 0

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1 - floor(x)*Beta(floor(x), {self.rho+1}) if x >= 1")
            print(f"| 0 otherwise")

        else:
            if x >= 1:
                k = np.floor(x)
                return 1 - k*scipy.special.beta(k, self.rho+1)
            else:
                return 0

    def sample(self, n=1):
        return drandom.yule_simon(self.rho,n)

    def mean(self):
        return self.rho/(self.rho-1)

    def variance(self):
        if self.rho > 2:
            return (self.rho**2)/(((self.rho-1)**2)*(self.rho-2))
        return None

    def entropy(self):
        return "Unknown"

    def support(self):
        return "N*"