import numpy as np
import sys

import scipy
import scipy as sp

from dist import density


#######################################################################################################
# CRANDOM MODULE                                                                                      #
#                                                                                                     #
# This library allows to generate samples of different continuous probability distributions.          #
# List of the available distributions :                                                               #
# - Uniform                                                                                           #
# - Exponential                                                                                       #
# - Normal                                                                                            #
# - Gamma                                                                                             #
# - Beta                                                                                              #
# - Weibull                                                                                           #
# - Frechet                                                                                           #
#                                                                                                     #
# TO DO :                                                                                             #
#                                                                                                     #
# - Cauchy                                                                                            #
# - Gumbel                                                                                            #
# - Kumaraswamy                                                                                       #
# - Fisher                                                                                            #
# - Wigner                                                                                            #
# - Logit-Normal                                                                                      #
# - Power                                                                                             #
# - Wald                                                                                              #
# - Lévy                                                                                              #
# - Pareto                                                                                            #
#######################################################################################################

def uniform(a=0,b=1,n=1):

    '''
    :param a: lower bound
    :param b: upper bound
    :param n: number of samples

    The uniform function returns n samples of a uniform distribution on (a,b).
    '''

    (a,b) = (min(a,b),max(a,b))
    u = np.array([np.random.uniform(0,1)*(b-a)+a for i in range(n)])
    return u

def exponential(alpha=1,n=1):
    if alpha < 0:
        raise ValueError("The parameter should be greater than 0.")

    '''

    :param alpha: parameter for the exponential distribution
    :param n: number of samples

    The exponential function returns n samples of an alpha-exponential distribution, with the generalized inverse density function.
    '''

    return (-1 / alpha)*np.log(1 - uniform(0,1,n))

def normal(mean=0, sd=1,n=1):

    '''
        :param mean: mean of the normal distribution
        :param sd: standard deviation of the normal distribution
        :param n: number of samples

        The normal function returns n samples of a normal distribution of parameters (mean, sd^2) with the Box-Muller method
    '''

    if sd <= 0:
        raise ValueError("The standard deviation should be greater than 0.")

    # U should not be equal to 0 when we implement Box-Muller, because we can't compute log(0).
    U = uniform(sys.float_info.epsilon,1,n)
    V = uniform(0,1,n)

    return (np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V))*sd + mean

def gamma(p=1, theta=1,n=1):

    '''

    :param p: shape parameter
    :param theta: inverse intensity parameter
    :param n: number of samples

    The normal function returns n samples of a gamma distribution of parameters (p, theta).
    '''

    if p <= 0:
        raise ValueError("The first parameter should be greater than 0.")
    if theta <= 0:
        raise ValueError("The second parameter should be greater than 0.")

    E = exponential(1/theta,n)
    return np.array(sum(E))

def beta(a=1,b=1,n=1):

    '''

    :param a: first shape parameter
    :param b: second shape parameter
    :param n: number of samples

    The beta function returns n samples of a beta distribution of parameters (a,b), according to the following result :
    X ~ Gamma(a,1), Y ~ Gamma(b,1) ==> X/(X+Y) ~ Beta(a,b)
    '''

    X = gamma(a,1,n)
    Y = gamma(b,1,n)
    return X/(X+Y)

def weibull(k=1, l=1, n=1):

    '''

    :param k: shape parameter
    :param l: scale parameter
    :param n: number of samples

    The weibull function returns n samples of a Weibull distribution of parameters (k,l).
    '''

    U = uniform(0,1,n)
    return l*(-np.log(1-U))**(1/k)

def frechet(a=1,s=1,m=0,n=1):

    '''

    :param a: shape parameter
    :param s: scale parameter
    :param m: position parameter
    :param n: number of samples

    The frechet function returns n samples of a Fréchet distribution of parameters (a,s,m).
    '''

    U = uniform(0, 1, n)
    return m+s*(-np.log(U))**(-1/a)

