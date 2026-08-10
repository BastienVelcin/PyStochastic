import numpy as np
from crandom import uniform, exponential
from math import comb

def duniform(N=2,n=1,d=1):
    if d == 1:
        return (np.floor(uniform(0, N, n )))
    return np.atleast_1d(np.reshape(np.floor(uniform(0,N,n*d)),(n,d)))

def bernoulli(p=0.5,n=1):
    return (duniform(2,b) < p).astype(int)

def rademacher(p=0.5, n=1):
    return bernoulli(p,n)*2-1

def binomial(p,k,n=1):
    ber = np.array([bernoulli(p,n) for _ in range(k)])
    return np.sum(ber,axis=0)

def poisson(lam=1,n=1):
    P = np.zeros(n)
    for k in range(n):
        sum = 0
        i = -1
        verif = 0
        while verif == 0:
            U = uniform(0,1)
            sum += -np.log(U)/lam
            if sum > 1:
                verif = 1
            i += 1
        P[k] = i
    return P

def hypergeometric(N=2,k=1,m=1,n=1):
    probas = np.array([comb(m,i)*comb(N-m,k-i)/comb(N,k) for i in range(k+1)])

    # Due to possible approximations errors, we need to normalize the probabilities.
    probas = probas/sum(probas)
    cum_probas = np.cumsum(probas)
    U = uniform(0,1,n)

    return np.searchsorted(cum_probas, U)

def geometric(p=0.5,n=1):
    U = np.random.rand(n)
    return np.ceil(np.log(1 - U) / np.log(1 - p)).astype(int)

def negative_binomial(p=0.5,k=1,n=1):
    G = np.array([geometric(p,n) for _ in range(k)])
    return np.sum(G,axis=0)

def yule_simon(rho=1,n=1):
    W = exponential(rho,n)
    return geometric(np.exp(-W))