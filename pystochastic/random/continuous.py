"""
============================================================
Module CONTINUOUS RANDOM
============================================================

Description
-----------
This module provides a set of sample generators for elementary-time random variables.
The following distributions are implemented :

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

These functions are directly linked to the elementary.py classes.

Examples
--------
>> normal(0,1,10)
array([ 0.61189587, -1.42362161, -0.36381574,  0.37421544,  1.20040897,
        0.63604695, -1.51159324, -0.38354233,  1.98387139, -0.300739  ])

>> fisher(5,5/2,6)
array([0.50638435, 2.45166231, 0.07610754, 6.88851094, 4.02989885,
       0.49415416])
"""

import numpy as np
import sys
from pystochastic.random.setseed import get_rng, seed

def uniform(a=0,b=1,n=1):

    """
    Continuous-Time Uniform sampling function.

    The Continuous-Time Uniform distribution is parameterized by two bounds parameterss ``a`` and ``b``.

    Parameters
    ----------
    a : float
        Lower bound. Must be strictly less than ``b``.
    b : float
        Upper bound. Must be strictly greater than ``a``.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> uniform(a=0,b=1,n=10)

    Notes
    -----
    If a > b, the function automatically swaps the bounds.
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if a > b:
        raise ValueError(
            "The lower bound must be inferior or equal to the upper bound."
        )

    rng = get_rng()

    return rng.uniform(0, 1, size=n)*(b-a)+a

def exponential(alpha=1,n=1):

    """
    Exponential sampling function.

    The Exponential distribution is parameterized by an intensity parameter ''alpha''.

    Parameters
    ----------
    alpha : float
        Intensity parameter, or scale parameter inverse. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> exponential(alpha=2,n=10)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if alpha <= 0:
        raise ValueError(
            "The parameter must be greater than 0."
        )

    return (-1 / alpha)*np.log(1 - uniform(0,1,n))

def normal(mean=0, var=1, n=1):

    """
    Normal sampling function.

    The Normal distribution is parameterized by a mean parameter ``mu`` and a standard deviation parameter ``sd``.

    Parameters
    ----------
    mean : float
        Mean parameter.
    var : float
        Variance parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> normal(mean=1,var=3/2,n=5)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if var <= 0:
        raise ValueError(
            "The variance must be greater than 0."
        )

    # U must not be equal to 0 when we implement Box-Muller, because we can't compute log(0).
    U = uniform(sys.float_info.epsilon,1,n)
    V = uniform(0,1,n)

    return (np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V))*np.sqrt(var) + mean

def _gamma_frac_reject(p, size):


    """
    Gamma reject function.

    The Gamma reject function is used to generate samples from the Gamma distribution.
    It is not used directly by the user.
    """


    t = np.e / (np.e + p) #Reject constant
    out = np.empty(0)
    while out.size < size:
        m = size - out.size
        U0 = np.random.uniform(0, 1, m)
        U1 = np.random.uniform(0, 1, m)
        W  = np.random.uniform(0, 1, m)
        branch1 = U0 <= t

        X = np.where(branch1, U1**(1/p), 1 - np.log(U1))
        accept = np.where(branch1, W <= np.exp(-X), W <= X**(p - 1))

        out = np.concatenate([out, X[accept]])
    return out[:size]

def gamma(p=1, theta=1, n=1):

    """
    Gamma sampling function.

    The Gamma distribution is parameterized by a shape parameter
    ``k`` and a rate parameter ``theta``.

    Parameters
    ----------
    k : float
        Shape parameter. Must be strictly positive.
    theta : float
        Rate parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> gamma(k=2,theta=1,n=12)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if p <= 0:
        raise ValueError(
            "The shape parameter must be greater than 0."
        )

    if theta <= 0:
        raise ValueError(
            "The rate parameter must be greater than 0."
        )

    p_int, p_frac = int(np.floor(p)), p - int(np.floor(p))

    int_part = np.zeros(n)
    for _ in range(p_int):
        int_part += exponential(1, n)

    frac_part = _gamma_frac_reject(p_frac, n) if p_frac > 1e-12 else np.zeros(n)

    return (int_part + frac_part)/theta

def beta(a=1,b=1,n=1):

    """
    Beta sampling function.

    The Beta distribution is parameterized by two shape parameters
    ``a`` and ''b''.

    Parameters
    ----------
    a : float
        First shape parameter. Must be strictly positive.
    b : float
        Second shape parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> beta(a=2,b=3/2,n=6)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if a <= 0:
        raise ValueError(
            "The first shape parameter must be greater than 0."
        )

    if b <= 0:
        raise ValueError(
            "The second shape parameter must be greater than 0."
        )

    X = gamma(a,1,n)
    Y = gamma(b,1,n)

    return X/(X+Y)

def weibull(k=1, l=1, n=1):

    """
    Weibull sampling function.

    The Weibull distribution is parameterized by a shape parameter
    ``k`` and a scale parameter ''l''.

    Parameters
    ----------
    k : float
        Shape parameter. Must be strictly positive.
    l : float
        Scale parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> weibull(k=2,l=3/2,n=3)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if k <= 0:
        raise ValueError(
            "The shape parameter must be greater than 0."
        )

    if l <= 0:
        raise ValueError(
            "The scale parameter must be greater than 0."
        )

    U = uniform(0,1,n)
    return l*(-np.log(1-U))**(1/k)

def frechet(a=1,s=1,m=0,n=1):

    """
    Fréchet sampling function.

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
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> frechet(a=2,s=3/2,m=-3,n=7)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if a <= 0:
        raise ValueError(
            "The shape parameter must be greater than 0."
        )

    if s <= 0:
        raise ValueError(
            "The scale parameter must be greater than 0."
        )

    U = uniform(0, 1, n)
    return m+s*(-np.log(U))**(-1/a)

def cauchy(x=0,a=1,n=1):

    """
    Cauchy sampling function.

    The Cauchy distribution is parameterized by a position parameter
    ``x`` and a scale parameter ''a''.

    Parameters
    ----------
    x : float
        Position parameter.
    a : float
        Scale parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> cauchy(x=0,a=1,n=5)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if a <= 0:
        raise ValueError(
            "The scale parameter must be greater than 0."
        )

    U = uniform(0, 1, n)
    return a*np.tan(np.pi*U - np.pi/2)+x

def gumbel(mu=0,beta=1,n=1):

    """
    Gumbel sampling function.

    The Gumbel distribution is parameterized by a position parameter
    ``mu`` and a scale parameter ''beta''.

    Parameters
    ----------
    mu : float
        Position parameter.
    beta : float
        Scale parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> gumbel(mu=-1/2,beta=2,n=4)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if beta <= 0:
        raise ValueError(
            "The scale parameter must be greater than 0."
        )

    U = uniform(0, 1, n)
    return mu-beta*np.log(-np.log(U))

def kumaraswamy(a=1,b=1,n=1):

    """
    Kumaraswamy sampling function.

    The Kumaraswamy distribution is parameterized by a two shape parameters
    ``a`` and ''b''.

    Parameters
    ----------
    a : float
        First shape parameter. Must be strictly positive.
    b : float
        Second shape parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> kumaraswamy(a=3/2,b=5,n=10)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if a <= 0:
        raise ValueError(
            "The first shape parameter must be greater than 0."
        )

    if b <= 0:
        raise ValueError(
            "The second shape parameter must be greater than 0."
        )

    U = uniform(0, 1, n)
    return (1-(1-U)**(1/b))**(1/a)

def fisher(d1=1,d2=1,n=1):

    """
    Fisher sampling function.

    The Fisher distribution is parameterized by a two degree of freedom parameters
    ``d1`` and ''d2''.

    Parameters
    ----------
    d1 : float
        First degree of freedom. Must be strictly positive.
    d1 : float
        Second degree of freedom. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> fisher(d1=2,d2=5,n=15)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if d1 <= 0:
        raise ValueError(
            "The first degree of freedom must be greater than 0."
        )

    if d2 <= 0:
        raise ValueError(
            "The second degree of freedom must be greater than 0."
        )

    U = gamma(d1/2,0.5,n)
    V = gamma(d2 / 2, 0.5, n)
    return (U*d2)/(V*d1)

def pareto(x_m=1,k=1,n=1):

    """
    Pareto sampling function.

    The Pareto distribution is parameterized by a position parameter ''x_m'' and
    a shape parameter ''k''.

    Parameters
    ----------
    x_m : float
        Position parameter. Must be strictly positive.
    k : float
        Shape parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> pareto(x_m=2,k=1/2,n=12)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if x_m <= 0:
        raise ValueError(
            "The position parameter must be greater than 0."
        )

    if k <= 0:
        raise ValueError(
            "The shape parameter must be greater than 0."
        )

    U = uniform(0,1,n)
    return x_m * (U)**(-1/k)

def rayleigh(s=1,n=1):

    """
    Rayleigh sampling function.

    The Rayleigh distribution is parameterized by a scale parameter ''s''.

    Parameters
    ----------
    s : float
        Scale parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Examples
    --------
    >> rayleigh(s=4,n=10)
    """

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    if s <= 0:
        raise ValueError(
            "The scale parameter must be greater than 0."
        )

    U = uniform(0, 1, n)
    return s*np.sqrt(-2*np.log(U))
