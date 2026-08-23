"""
============================================================
Module DISCRETE DISTRIBUTIONS
============================================================

Description
-----------
This module provides a set of classes to work with discrete probability distributions.

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
The available discrete distributions are :

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
>> B = Binomial(0.26,10) #Binomial distribution with success probability 0.26 and 10 repetitions
>>
>> B.sample(10) #Sample 10 random numbers from the distribution
>>
>> B.plot_pmf() #Plot the probability mass function of the distribution
"""

import numpy as np
import scipy
import plotly.graph_objects as go

from abc import abstractmethod, ABC
from pystochastic.random import discrete
from mpmath import hyp3f2
from math import comb
from pystochastic.dist.distribution import Distribution

class DiscreteDistribution(Distribution, ABC):

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

    def info(self):
        """
        Info method.

        Print a recap of the effective distribution.
        """
        print(f"Distribution : {self.__class__.__name__}")
        print(f"Parameters : {self.__dict__}")
        print(f"{self.pmf()}")
        print(f"{self.cdf()}")
        print(f"Support : {self.support()}")
        print(f"Mean : {self.mean()}")
        print(f"Variance : {self.variance()}")
        print(f"Entropy : {self.entropy()}")

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
        return discrete.duniform(self.N, n)

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
        return discrete.bernoulli(self.p, n)

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
        return discrete.bernoulli(self.p, n)

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
        return discrete.binomial(self.p, self.n, n)

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
        return discrete.poisson(self.lam, n)

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
        Number of draws. Must be a strictly positive integer such that 0 <= K <= N.
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
        self.K = K
        self.m = m
    def pmf(self, k=None):

        if k is None:
            print("Probability mass function:")
            print(f"| comb({self.K},k) * comb({self.N-self.K},{self.m}-k) / {comb(self.N,self.m)} if {max(0,self.m+self.K-self.N)} <= k <= {min(self.m,self.K)}")
            print(f"| 0 otherwise")

        else:

            if not isinstance(k, (int, np.integer)):
                raise ValueError(
                    "The evaluation point must be an integer."
                )

            if max(0,self.m+self.K-self.N) <= k <= min(self.m,self.K):
                return comb(self.K,k) * comb(self.N-self.K,self.m-k) / comb(self.N,self.m)
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
                k = np.floor(x).astype(int)
                return float(1 - comb(self.m,k+1)*comb(self.N-self.m,self.K-1-k) / comb(self.N,self.K) * hyp3f2(1,k+1-self.K,k+1-self.m,k+2,self.N+2-self.K-self.m+k,1))
            else:
                return 0

    def sample(self, n=1):
        return discrete.hypergeometric(self.N, self.K, self.m, n)

    def mean(self):
        return self.m * self.K / self.N

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
        return discrete.geometric(self.p, n)

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
    n : int
        Target success occurrence parameter. Must be a strictly positive integer.

    Attributes
    ----------
    p : float
        Success probability.
    n : int
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
        return discrete.negative_binomial(self.p, self.n, n)

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
        Shape parameter. Must be strictly positive.

    Attributes
    ----------
    rho : float
        Shape parameter.

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
        return discrete.yule_simon(self.rho, n)

    def mean(self):
        if self.rho > 1:
            return self.rho/(self.rho-1)
        return None

    def variance(self):
        if self.rho > 2:
            return (self.rho**2)/(((self.rho-1)**2)*(self.rho-2))
        return None

    def entropy(self):
        return "Unknown"

    def support(self):
        return "N*"



