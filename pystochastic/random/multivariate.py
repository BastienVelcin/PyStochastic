"""
============================================================
Module CRANDOM
============================================================

Description
-----------
This module provides a set of sample generators for multivariate elementary-time random variables.
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
from pystochastic.random import normal
from pystochastic.utils import is_pos_def


def multivariate_normal(mu = [0,0], cov = np.eye(2), n = 1):

    mu = np.atleast_1d(mu)
    cov = np.atleast_2d(cov)

    if not mu.size == cov.shape[0] == cov.shape[1]:
        raise ValueError(
            "The mean vector and the covariance matrix must have the same size."
        )
    if not is_pos_def(cov):
        raise ValueError(
            "The covariance matrix must be positive definite."
        )

    if n < 1 or not isinstance(n, (int, np.integer)):
        raise ValueError(
            "The number of samples must be a strictly positive integer."
        )

    N = normal(0,1,n*mu.size).reshape(mu.size,n)

    diag = np.linalg.eig(cov)
    diag_matrix_sqrt = np.diag(np.sqrt(np.real(diag[0])))
    eigv_matrix = np.real(diag[1])

    B = eigv_matrix * diag_matrix_sqrt

    return np.repeat([mu], n, axis=0) - (B @ N).T

