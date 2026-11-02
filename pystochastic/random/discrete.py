"""
============================================================
Module DISCRETE RANDOM
============================================================

Description
-----------
This module provides a set of sample generators for discrete-time random variables.
The following distributions are implemented :

    - Discrete-Time Uniform
    - Binomial
    - Poisson
    - Hypergeometric
    - Geometric
    - Negative Binomial
    - Yule-Simon

These functions are directly linked to the elementary.py classes.

Examples
--------
>> binomial(0.6,5,10)
array([2, 2, 1, 3, 1, 4, 2, 2, 3, 0])

>> geometric(0.2,15)
array([12,  4,  6,  4,  8,  8,  3,  6,  2,  7,  2, 12,  3,  3, 22])
"""

import numpy as np
from pystochastic.random.continuous import uniform, exponential
from math import comb

def _validate_n(n):
    if not isinstance(n, (int, np.integer)) or not np.isfinite(n) or np.isnan(n) or n <= 0:
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )


def duniform(N=2,n=1):

    """
    Discrete-Time Uniform sampling function.

    The Discrete-Time Uniform distribution is parameterized by a number of equal-sized subintervals of [0,1] ''N''.

    Parameters
    ----------
    N : int
        Number of equal-sized subintervals of [0,1]. Must be a strictly positive integer.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given discrete-time uniform distribution.

    Examples
    --------
    >> duniform(N=3,n=10)
    """

    if not isinstance(N, (int, np.integer)) or not np.isfinite(N) or np.isnan(N) or N <= 1:
        raise ValueError(
            "The number of outcome values must be an integer strictly greater than 1."
        )

    _validate_n(n)

    return get_rng().integers(1, N + 1, size=n)

def bernoulli(p=0.5,n=1):

    """
    Bernoulli sampling function.

    The Bernoulli distribution is parameterized by a success probability parameter ''p''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 and 1.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given Bernoulli distribution.

    Examples
    --------
    >> bernoulli(p=0.6,n=8)
    """

    if not isinstance(p, (int, np.integer, float, np.floating)) or not np.isfinite(p) or np.isnan(p) or not 0 <= p <= 1:
        raise ValueError(
            "The success probability must be a real number between 0 and 1."
        )

    _validate_n(n)

    return (uniform(0,1, n) < p).astype(int)

def rademacher(p=0.5, n=1):

    """
    Rademacher sampling function.

    The Rademacher distribution is parameterized by a gain probability parameter ''p''.

    Parameters
    ----------
    p : float
        Gain probability. Must be between 0 and 1.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given Rademacher distribution.

    Examples
    --------
    >> rademacher(p=0.6,n=8)

    Notes
    -----
    This function provides an extended version of the Rademacher distribution, since it allows any p in [0,1] instead
    of just p=1/2.
    """

    if not isinstance(p, (int, np.integer, float, np.floating)) or not np.isfinite(p) or np.isnan(p) or not 0 <= p <= 1:
        raise ValueError(
            "The success probability must be a real number between 0 and 1."
        )

    _validate_n(n)

    return bernoulli(p,n)*2-1

def binomial(p,k,n=1):

    """
    Binomial sampling function.

    The Binomial distribution is parameterized by a success probability parameter ''p'' and a
    repetition parameter ''k''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 and 1.
    k : int
        Repetition parameter. Must be a strictly positive integer.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given binomial distribution.

    Examples
    --------
    >> binomial(p=0.6,k=10,n=3)
    """

    if not isinstance(p, (int, np.integer, float, np.floating)) or not np.isfinite(p) or np.isnan(p) or not 0 <= p <= 1:
        raise ValueError(
            "The success probability must be a real number between 0 and 1."
        )

    if not isinstance(k, (int, np.integer)) or not np.isfinite(k) or np.isnan(k) or k < 1:
        raise ValueError(
            "The number of repetition must be a strictly positive integer."
        )

    _validate_n(n)

    ber = (uniform(0, 1, k*n).reshape(k, n) < p)
    return np.sum(ber,axis=0)

def poisson(lam=1,n=1):

    """
    Poisson sampling function.

    The Poisson distribution is parameterized by an expectation parameter ''lam''.

    Parameters
    ----------
    lam : float
        Expectation parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given Poisson distribution.

    Examples
    --------
    >> poisson(lam=4,n=12)
    """

    if not isinstance(lam, (int, np.integer, float, np.floating)) or not np.isfinite(lam) or np.isnan(
            lam) or not lam > 0:
        raise ValueError(
            "The intensity must be a strictly positive real number."
        )

    _validate_n(n)


    count = np.zeros(n, dtype=int)
    total_time = np.zeros(n)
    active = np.arange(n) #Index of non simulated elements

    while active.size > 0:
        # One uniform call for all non simulated elements
        U = uniform(0, 1, active.size)

        # Adding an exponential distribution of parameter lambda to the total time
        new_total = total_time[active] + (-np.log(U) / lam) # Adding an exponential distribution of parameter lambda to the total time

        # We consider only the elements that are still below 1
        still_below = new_total <= 1

        # We update the count and the total time of the elements that are still below 1
        idx_below = active[still_below]
        total_time[idx_below] = new_total[still_below]
        count[idx_below] += 1

        # Active elements are the ones that are still below 1
        active = idx_below
    return count

def hypergeometric(N=2,k=1,m=1,n=1):

    """
    Hypergeometric sampling function.

    The Hypergeometric distribution is parameterized by a the population size ''N'', a number
    of successes ''m'' and a number of draws ''k''.

    Parameters
    ----------
    N : int
        Population size. Must be a strictly positive integer.
    m : int
        Number of success states in the considered population. Must be a strictly positive integer such that 0 <= m <= N.
    k : int
        Number of draws. Must be a strictly positive integer such that 0 <= k <= N.

    Returns
    -------
    np.ndarray
        Array of n samples of the given hypergeometric distribution.

    Examples
    --------
    >> hypergeometric(N=12,k=6,m=3,n=10)
    """

    if not isinstance(N, (int, np.integer)) or not np.isfinite(N) or np.isnan(N) or N < 1:
        raise ValueError(
            "The population size must be a strictly positive integer."
        )

    if not isinstance(m, (int, np.integer)) or not np.isfinite(m) or np.isnan(m) or m < 0:
        raise ValueError(
            "The number of success states in the considered population must be a positive integer."
        )

    if not isinstance(k, (int, np.integer)) or not np.isfinite(k) or np.isnan(k) or k < 0:
        raise ValueError(
            "The number of draws must be a positive integer."
        )

    if not (0 <= k <= N and 0 <= m < N):
        raise ValueError(
            "The parameters must satisfy 0 <= k <= N and 0 <= m < N."
        )

    _validate_n(n)

    probas = np.array([comb(m,i)*comb(N-m,k-i)/comb(N,k) for i in range(k+1)])

    # Due to possible approximations errors, we need to normalize the probabilities.
    probas = probas/sum(probas)
    cum_probas = np.cumsum(probas)
    U = uniform(0,1,n)

    return np.searchsorted(cum_probas, U)

def geometric(p=0.5,n=1):

    """
    Geometric sampling function.

    The Geometric distribution is parameterized by a success probability parameter ''p''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 and 1 strictly.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given geometric distribution.

    Examples
    --------
    >> geometric(p=0.4,n=10)
    """

    if not isinstance(p, (int, np.integer, float, np.floating)) or not np.isfinite(p) or np.isnan(p) or not 0 < p < 1:
        raise ValueError(
            "The success probability must be a real number strictly between 0 and 1."
        )

    _validate_n(n)

    U = 1 - uniform(0,1,n)

    # We use log1p(x) instead of log(1+x) for better approximations.
    return np.ceil(np.log(U) / np.log1p(-p)).astype(int)

def negative_binomial(p=0.5,k=1,n=1):

    """
    Negative Binomial sampling function.

    The Negative Binomial distribution is parameterized by a success probability parameter ''p'' and
    a target success occurrence parameter ''k''.

    Parameters
    ----------
    p : float
        Success probability. Must be between 0 and 1.
    k : int
        Target success occurrence parameter. Must be a strictly positive integer.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given negative binomial distribution.

    Examples
    --------
    >> negative_binomial(p=0.4,k=5,n=10)
    """

    if not isinstance(p, (int, np.integer, float, np.floating)) or not np.isfinite(p) or np.isnan(p) or not 0 < p < 1:
        raise ValueError(
            "The success probability must be a real number strictly between 0 and 1."
        )

    if not isinstance(k, (int, np.integer)) or not np.isfinite(k) or np.isnan(k) or k < 1:
        raise ValueError(
            "The number of target success occurrence must be a strictly positive integer."
        )

    _validate_n(n)

    G = geometric(p,n*k).reshape(n,k)
    return np.sum(G,axis=0)

def yule_simon(rho=1,n=1):

    """
    Yule-Simon sampling function.

    The Yule-Simon distribution is parameterized by a shape parameter ''rho''.

    Parameters
    ----------
    rho : float
        Form parameter. Must be strictly positive.
    n : int
        Number of samples. Must be a strictly positive integer.

    Returns
    -------
    np.ndarray
        Array of n samples of the given Yule-Simon distribution.

    Examples
    --------
    >> yule_simon(rho=2,n=7)
    """

    if not isinstance(rho, (int, np.integer, float, np.floating)) or not np.isfinite(rho) or np.isnan(rho) or rho <= 0:
        raise ValueError(
            "The shape parameter must be a strictly positive real number."
        )

    _validate_n(n)
    W = exponential(rho,n)
    U = 1 - uniform(0, 1, n)

    return np.maximum(np.ceil(np.log(U) / np.log(-np.expm1(-W))), 1).astype(int)