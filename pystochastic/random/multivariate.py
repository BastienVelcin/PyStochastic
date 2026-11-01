"""
============================================================
Module CRANDOM
============================================================

Description
-----------
This module provides a set of sample generators for multivariate elementary-time random variables.
The following distributions are implemented :

    - Normal Multivariate
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

    diag = np.linalg.eigh(cov)
    diag_matrix_sqrt = np.diag(np.sqrt(np.real(diag[0])))
    eigv_matrix = np.real(diag[1])

    B = eigv_matrix @ diag_matrix_sqrt

    return np.repeat([mu], n, axis=0) + (B @ N).T

