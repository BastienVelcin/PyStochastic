import scipy

import pyrandom.crandom
import numpy as np
from scipy import special
class Uniform:
    def __init__(self,a,b):
        self.lobound = min(a,b)
        self.upbound = max(a,b)

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| {self.upbound-self.lobound/2} for {self.lobound} <= x <= {self.upbound}" )
            print(f"| 0 else")

        else:
            if (self.lobound <= x <= self.upbound):
                return self.upbound-self.lobound/2
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
        return pyrandom.crandom.uniform(self.lobound,self.upbound,n)

    def expectation(self):
        return (self.lobound + self.upbound)/2

    def variance(self):
        return ((self.upbound - self.lobound)**2)/12

    def entropy(self):
        return np.log(self.upbound-self.lobound)

    def infos(self):
        print(f"| Support : [{self.lobound}, {self.upbound}]")
        print(f"| Expectation : {self.expectation()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")


class Exponential:
    def __init__(self,alpha):
        if alpha <=0:
            raise ValueError("The parameter should be greater than 0.")
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
        return pyrandom.crandom.exponential(self.alpha,n)

    def expectation(self):
        return 1/self.alpha

    def variance(self):
        return 1/self.alpha**2

    def entropy(self):
        return 1-np.log(self.alpha)

    def infos(self):
        print(f"| Support : [0, +inf)")
        print(f"| Expectation : {self.expectation()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

class Normal:
    def __init__(self,mu,sd):
        self.mu = mu
        self.sd = sd
        if sd <=0:
            raise ValueError("The standard deviation should be greater than 0.")

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| (1/{self.sd}*sqrt(2*pi)) * exp(-(x-{self.mu})^2 / 2*{self.sd}^2)")
        else:
            return (1/(self.sd*np.sqrt(2*np.pi)))*np.exp(-(x-self.mu)**2 / 2*(self.sd)**2)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| (1+erf((x-{self.mu})/({self.sd}*sqrt(2))))/2")
        else:
            return (1+scipy.special.erf((x-self.mu)/(self.sd*np.sqrt(2))))/2

    def sample(self,n=1):
        return pyrandom.crandom.normal(self.mu, self.sd,n)

    def expectation(self):
        return self.mu

    def variance(self):
        return self.sd**2

    def entropy(self):
        return np.log(self.sd*np.sqrt(2*np.pi*np.e))

    def infos(self):
        print(f"| Support : (-inf, +inf)")
        print(f"| Expectation : {self.expectation()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

class Gamma:
    def __init__(self,k,theta):
        if k <=0:
            raise ValueError("The form parameter should be greater than 0.")
        if theta <=0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.k = k
        self.theta = theta

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| ({self.theta**self.k} * x^{self.k-1} *  exp(-{self.theta}*x))/Gamma({self.k})")
        else:
            return ((self.theta**self.k)*(x**(self.k-1)*np.exp(-self.theta*x))/scipy.special.gamma(self.k))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| IncGamma({self.k}, {self.theta}*x)/Gamma({self.k})")
        else:
            return scipy.special.gammainc(self.k,self.theta*x)

    def sample(self,n=1):
        return pyrandom.crandom.gamma(self.k, self.theta, n)

    def expectation(self):
        return self.k/self.theta

    def variance(self):
        return self.k/(self.theta**2)

    def entropy(self):
        return self.k/self.theta + (1-self.k)*np.log(1/self.theta) + np.log(scipy.special.gamma(self.k)) + (1+self.k)*scipy.special.digamma(self.k)

    def infos(self):
        print(f"| Support : [0, +inf)")
        print(f"| Expectation : {self.expectation()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

