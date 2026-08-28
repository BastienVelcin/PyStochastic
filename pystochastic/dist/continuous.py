"""
============================================================
Module DIST
============================================================

Description
-----------
This module provides a set of classes to work with probability distributions.

This module provides a general class "Distribution" for elementary distributions, which yields to general methods :
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
from pystochastic.random import continuous
from sympy import harmonic
from pystochastic.dist.distribution import Distribution

class ContinuousDistribution(Distribution, ABC):

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

    def info(self):
        """
        Info method.

        Print a recap of the effective distribution.
        """
        print(f"Distribution : {self.__class__.__name__}")
        print(f"Parameters : {self.__dict__}")
        print(f"{self.pdf()}")
        print(f"{self.cdf()}")
        print(f"Support : {self.support()}")
        print(f"Mean : {self.mean()}")
        print(f"Variance : {self.variance()}")
        print(f"Entropy : {self.entropy()}")

class Uniform(ContinuousDistribution):

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
                "The lower and upper bound must be different."
            )

        self.lobound = min(a,b)
        self.upbound = max(a,b)

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| {1/(self.upbound-self.lobound)} for {self.lobound} <= x <= {self.upbound}" )
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
        return continuous.uniform(self.lobound, self.upbound, n)

    def mean(self):
        return (self.lobound + self.upbound)/2

    def variance(self):
        return ((self.upbound - self.lobound)**2)/12

    def entropy(self):
        return np.log(self.upbound-self.lobound)

    def support(self):
        return (self.lobound, self.upbound)

class Exponential(ContinuousDistribution):

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
                "The parameter must be greater than 0."
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
        return continuous.exponential(self.alpha, n)

    def mean(self):
        return 1/self.alpha

    def variance(self):
        return 1/self.alpha**2

    def entropy(self):
        return 1-np.log(self.alpha)

    def support(self):
        return (0, np.inf)

class Normal(ContinuousDistribution):

    """
    Normal probability distribution.

    The Normal distribution is parameterized by a mean parameter ``mu`` and a standard deviation parameter ``sd``.

    Parameters
    ----------
    mu : float
        Mean parameter.
    var : float
        Variance parameter. Must be strictly positive.

    Attributes
    ----------
    mu : float
        Mean parameter.
    var : float
        Variance parameter. Must be strictly positive.

    Examples
    --------
    >> N = Normal(mu=0,var=1)
    >> N.sample(10)
    """

    def __init__(self,mu=0,var=1):

        if var <=0:
            raise ValueError(
                "The variance must be greater than 0."
            )

        self.mu = mu
        self.var = var

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| (1/sqrt(2*{self.var}*pi)) * exp(-(x-{self.mu})^2 / 2*{self.var})")
        else:
            return (1/(np.sqrt(2*self.var*np.pi)))*np.exp(-(x - self.mu)**2 / (2*self.var))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| (1+erf((x-{self.mu})/(sqrt({2*self.var}))))/2")
        else:
            return (1+scipy.special.erf((x-self.mu)/(np.sqrt(2*self.var))))/2

    def sample(self,n=1):
        return continuous.normal(self.mu, self.var, n)

    def mean(self):
        return self.mu

    def variance(self):
        return self.var

    def entropy(self):
        return np.log(np.sqrt(2*np.pi*np.e*self.var))

    def support(self):
        return (-np.inf ,np.inf)

class Gamma(ContinuousDistribution):

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
                "The shape parameter must be greater than 0."
            )

        if theta <=0:
            raise ValueError(
                "The rate parameter must be greater than 0."
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
        return continuous.gamma(self.k, self.theta, n)

    def mean(self):
        return self.k/self.theta

    def variance(self):
        return self.k/(self.theta**2)

    def entropy(self):
        return self.k/self.theta + (1-self.k)*np.log(1/self.theta) + np.log(scipy.special.gamma(self.k)) + (1+self.k)*scipy.special.digamma(self.k)

    def support(self):
        return (0,np.inf)

class Beta(ContinuousDistribution):

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
                "The first shape parameter must be greater than 0."
            )
        if b <= 0:
            raise ValueError(
                "The second shape parameter must be greater than 0."
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
        return continuous.beta(self.a, self.b, n)

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        return (self.a * self.b) / ((self.a + self.b)**2 * (self.a + self.b + 1))

    def entropy(self):
        return np.log(scipy.special.beta(self.a,self.b)) - (self.a-1)*scipy.special.digamma(self.a) - (self.b-1)*scipy.special.digamma(self.b) + (self.a+self.b-2)*scipy.special.digamma(self.a+self.b)

    def support(self):
        return (0,1)

class Weibull(ContinuousDistribution):

    """
    Weibull probability distribution.

    The Weibull distribution is parameterized by a shape parameter
    ``k`` and a scale parameter ''lam''.

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
                "The shape parameter must be greater than 0."
            )

        if l <=0:
            raise ValueError(
                "The scale parameter must be greater than 0."
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
        return continuous.weibull(self.k, self.l, n)

    def mean(self):
        return self.l * scipy.special.gamma(1 + 1 / self.k)

    def variance(self):
        return self.l ** 2 * scipy.special.gamma(1 + 2 / self.k) - self.mean()**2

    def entropy(self):
        return scipy.special.digamma(1-1/self.k)+(self.l/self.k)**self.k+np.log(self.l)-np.log(self.l/self.k)

    def support(self):
        return (0, np.inf)

class Frechet(ContinuousDistribution):

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
                "The shape parameter must be greater than 0."
            )

        if s <= 0:
            raise ValueError(
                "The scale parameter must be greater than 0."
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
        return continuous.frechet(self.a, self.s, self.m, n)

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

class Cauchy(ContinuousDistribution):

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

    def __init__(self,x_0=0,a=1):

        if a <= 0:
            raise ValueError(
                "The scale parameter must be greater than 0."
            )

        self.x_0 = x_0
        self.a = a

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 1/(pi*{self.a}*(1+(x-{self.x_0})/{self.a})^2)")
        else:
            return 1/(np.pi*self.a*(1+(x-self.x_0)/self.a)**2)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1/pi * Arctan((x-{self.x_0})/{self.a}) + 1/2")
        else:
            return 1/np.pi * np.arctan((x-self.x_0)/self.a) + 1/2

    def sample(self, n=1):
        return continuous.cauchy(self.x_0, self.a, n)

    def mean(self):
        return None

    def variance(self):
        return None

    def entropy(self):
        return np.log(4*np.pi*self.a)

    def support(self):
        return (-np.inf, np.inf)

class Gumbel(ContinuousDistribution):

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
                "The scale parameter must be greater than 0."
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
        return continuous.gumbel(self.mu, self.beta, n)

    def mean(self):
        return self.mu+self.beta*np.euler_gamma

    def variance(self):
        return (np.pi**2)*(self.beta**2)/6

    def entropy(self):
        return np.log(self.beta) + np.euler_gamma + 1

    def support(self):
        return (-np.inf, np.inf)


class Kumaraswamy(ContinuousDistribution):

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
                "The first shape parameter must be greater than 0."
            )

        if b <= 0:
            raise ValueError(
                "The second shape parameter must be greater than 0."
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
        return continuous.kumaraswamy(self.a, self.b, n)

    def mean(self):
        return (self.b*scipy.special.gamma(1+1/self.a)*scipy.special.gamma(self.b))/scipy.special.gamma(1 + self.b + 1/self.a)

    def variance(self):
        return self.b * scipy.special.beta(1+2/self.a, self.b) - self.mean()**2

    def entropy(self):
        return (1-1/self.b)+(self.a-1)*harmonic(self.b) - np.log(self.a*self.b)

    def support(self):
        return (0,1)

class Fisher(ContinuousDistribution):

    """
    Fisher probability distribution.

    The Fisher distribution is parameterized by a two degree of freedom parameters
    ``d1`` and ''d2''.

    Parameters
    ----------
    d1 : int
        First degree of freedom. Must be a strictly positive integer.
    d1 : int
        Second degree of freedom. Must be a strictly positive integer.

    Attributes
    ----------
    d1 : int
        First degree of freedom.
    d1 : int
        Second degree of freedom.

    Examples
    --------
    >> F = Fisher(d1=2,d2=5)
    >> F.sample(10)
    """

    def __init__(self,d1=1,d2=1):
        
        if d1 < 1 or not isinstance(d1, (int, np.integer)):
            raise ValueError(
                "The first degree of freedom must be a strictly positive integer."
            )

        if d2 < 1 or not isinstance(d1, (int, np.integer)):
            raise ValueError(
                "The second degree of freedom must be a strictly positive integer."
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
        return continuous.fisher(self.d1, self.d2, n)

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

class Pareto(ContinuousDistribution):

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
                "The position parameter must be greater than 0."
            )

        if k <= 0:
            raise ValueError(
                "The shape parameter must be greater than 0."
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
        return continuous.pareto(self.x_m, self.k, n)

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

class Rayleigh(ContinuousDistribution):

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
                "The scale parameter must be greater than 0."
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
        return continuous.rayleigh(self.s, n)

    def mean(self):
        return self.s*np.sqrt(np.pi/2)

    def variance(self):
        return (self.s**2)*(4-np.pi)/2

    def entropy(self):
        return 1 + np.log(self.s/np.sqrt(2))+np.euler_gamma/2

    def support(self):
        return (0, np.inf)
