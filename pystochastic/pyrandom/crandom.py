import numpy as np
import sys


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
# - Cauchy                                                                                            #
# - Gumbel                                                                                            #
# - Kumaraswamy                                                                                       #
# - Fisher                                                                                            #
# - Pareto                                                                                            #
# - Rayleigh                                                                                          #
#######################################################################################################

def uniform(a=0,b=1,n=1):

    '''
    :param a: lower bound
    :param b: upper bound
    :param n: number of samples

    The uniform function returns n samples of a uniform distribution on (a,b).
    '''

    if a > b:
        raise ValueError("The lower bound should be inferior or equal to the upper bound.")

    return np.random.uniform(0, 1, size=n)*(b-a)+a

def exponential(alpha=1,n=1):
    '''
    :param alpha: parameter for the exponential distribution
    :param n: number of samples

    The exponential function returns n samples of an alpha-exponential distribution, with the generalized inverse density function.
    '''

    if alpha <= 0:
        raise ValueError("The parameter should be greater than 0.")

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

def _gamma_frac_reject(p, size):

    '''
    :param p: shape parameter
    :param theta: rate parameter
    :param n: number of samples

    The _gamma_frac_reject provides the rejection sampling for the gamma function.
    '''

    t = np.e / (np.e + p)
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

    '''

    :param p: shape parameter
    :param theta: rate parameter
    :param n: number of samples

    The gamma function returns n samples of a gamma distribution of parameters (p,theta),
    '''

    if p <= 0:
        raise ValueError("The shape parameter should be greater than 0.")
    if theta <= 0:
        raise ValueError("The rate parameter should be greater than 0.")

    p_int, p_frac = int(np.floor(p)), p - int(np.floor(p))

    int_part = np.zeros(n)
    for _ in range(p_int):
        int_part += exponential(1, n)

    frac_part = _gamma_frac_reject(p_frac, n) if p_frac > 1e-12 else np.zeros(n)

    return (int_part + frac_part)/theta

def beta(a=1,b=1,n=1):

    '''
    :param a: first shape parameter
    :param b: second shape parameter
    :param n: number of samples

    The beta function returns n samples of a beta distribution of parameters (a,b), according to the following result :
    X ~ Gamma(a,1), Y ~ Gamma(b,1) ==> X/(X+Y) ~ Beta(a,b)
    '''

    if a <= 0:
        raise ValueError("The first shape parameter should be greater than 0.")
    if b <= 0:
        raise ValueError("The second shape parameter should be greater than 0.")

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

    if k <= 0:
        raise ValueError("The shape parameter should be greater than 0.")
    if l <= 0:
        raise ValueError("The scale parameter should be greater than 0.")

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

    if a <= 0:
        raise ValueError("The shape parameter should be greater than 0.")
    if s <= 0:
        raise ValueError("The scale parameter should be greater than 0.")

    U = uniform(0, 1, n)
    return m+s*(-np.log(U))**(-1/a)

def cauchy(x=0,a=1,n=1):

    '''
    :param x: position parameter
    :param a: scale parameter
    :param n: number of samples

    The cauchy function returns n samples of a Cauchy distribution of parameters (x,a).
    '''

    if a <= 0:
        raise ValueError("The scale parameter should be greater than 0.")

    U = uniform(0, 1, n)
    return a*np.tan(np.pi*U - np.pi/2)+x

def gumbel(mu=0,beta=1,n=1):

    '''
    :param mu: position parameter
    :param beta: scale parameter
    :param n: number of samples

    The gumbel function returns n samples of a Gumbel distribution of parameters (mu, beta).
    '''

    if beta <= 0:
        raise ValueError("The scale parameter should be greater than 0.")

    U = uniform(0, 1, n)
    return mu-beta*np.log(-np.log(U))

def kumaraswamy(a=1,b=1,n=1):

    '''
    :param a: first shape parameter
    :param b: second shape parameter
    :param n: number of samples

    The kumaraswamy function returns n samples of a Kumaraswamy distribution of parameters (a,b).
    '''

    if a <= 0:
        raise ValueError("The first shape parameter should be greater than 0.")
    if b <= 0:
        raise ValueError("The second shape parameter should be greater than 0.")

    U = uniform(0, 1, n)
    return (1-(1-U)**(1/b))**(1/a)

def fisher(d1=1,d2=1,n=1):

    '''
    :param a: first degree of freedom
    :param b: second degree of freedom
    :param n: number of samples

    The fisher function returns n samples of a Fisher distribution of parameters (d1,d2).
    '''

    if d1 <= 0:
        raise ValueError("The first degree of freedom should be greater than 0.")
    if d2 <= 0:
        raise ValueError("The second degree of freedom should be greater than 0.")

    U = gamma(d1/2,0.5,n)
    V = gamma(d2 / 2, 0.5, n)
    return (U*d2)/(V*d1)

def pareto(a=1,b=1,n=1):

    '''
    :param a: position parameter
    :param b: shape parameter
    :param n: number of samples

    The pareto function returns n samples of a Pareto distribution of parameters (a,b).
    '''

    if a <= 0:
        raise ValueError("The position parameter should be greater than 0.")
    if b <= 0:
        raise ValueError("The shape parameter should be greater than 0.")

    U = uniform(0,1,n)
    return b/(U**(1/a))

def rayleigh(s=1,n=1):

    '''
    :param s: scale parameter
    :param n: number of samples

    The rayleigh function returns n samples of a Rayleigh distribution of parameter s.
    '''

    if s <= 0:
        raise ValueError("The scale parameter should be greater than 0.")

    U = uniform(0, 1, n)
    return s*np.sqrt(-2*np.log(U))
