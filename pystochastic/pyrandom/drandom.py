import numpy as np
from .crandom import uniform, exponential
from math import comb

def duniform(N=2,n=1,dim=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    if dim == 1:
        return (np.floor(uniform(0, N, n )))
    return np.atleast_1d(np.reshape(np.floor(uniform(0,N,n*d)),(n,dim)))

def bernoulli(p=0.5,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    return (duniform(2, n) < p).astype(int)

def rademacher(p=0.5, n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    return bernoulli(p,n)*2-1

def binomial(p,k,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    ber = np.array([bernoulli(p,n) for _ in range(k)])
    return np.sum(ber,axis=0)

def poisson(lam=1,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    if lam <= 0:
        raise ValueError("The rate parameter should be greater than 0.")
    P = np.zeros(n)
    for k in range(n):
        total_time  = 0
        i = -1
        verif = 0
        while verif == 0:
            U = uniform(0,1)
            total_time  += -np.log(U)/lam
            if total_time  > 1:
                verif = 1
            i += 1
        P[k] = i
    return P

def hypergeometric(N=2,k=1,m=1,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    if not (0 <= k <= N and 0 <= m < N):
        raise ValueError("The parameters should satisfy 0 <= k <= N and 0 <= m < N.")

    probas = np.array([comb(m,i)*comb(N-m,k-i)/comb(N,k) for i in range(k+1)])

    # Due to possible approximations errors, we need to normalize the probabilities.
    probas = probas/sum(probas)
    cum_probas = np.cumsum(probas)
    U = uniform(0,1,n)

    return np.searchsorted(cum_probas, U)

def geometric(p=0.5,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    U = np.random.rand(n)
    return np.ceil(np.log(1 - U) / np.log(1 - p)).astype(int)

def negative_binomial(p=0.5,k=1,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    G = np.array([geometric(p,n) for _ in range(k)])
    return np.sum(G,axis=0)

def yule_simon(rho=1,n=1):
    if n < 1 or type(n) != int:
        raise ValueError("The number of samples should be a strictly positive integer.")
    W = exponential(rho,n)
    return geometric(np.exp(-W))