from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import scipy

import pyrandom.crandom
import numpy as np
from scipy import special

class Distribution(ABC):

    @abstractmethod
    def pdf(self, x=None):
        pass

    @abstractmethod
    def cdf(self, x=None):
        pass

    @abstractmethod
    def sample(self,n=1):
        pass

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    @property
    def entropy(self):
        pass

    @property
    def infos(self):
        pass

    @property
    def support(self):
        pass

    def plot_pdf(self):
        lo_bound = -5
        up_bound = 5
        supp = self.support()
        if supp[0] > -np.inf:
            lo_bound = supp[0]
        if supp[1] < np.inf:
            lo_bound = supp[1]
        supp_lenght = up_bound-lo_bound

        x_axis = np.linspace(lo_bound-0.1*supp_lenght,up_bound+0.1*supp_lenght,int(1000*(supp_lenght)))
        y_axis = [self.pdf(x) for x in x_axis]
        plt.plot(x_axis,y_axis, label="Probability Density Function (PDF)")
        plt.legend()
        plt.grid(True)
        plt.show()



class Uniform(Distribution):
    def __init__(self,a,b):
        self.lobound = min(a,b)
        self.upbound = max(a,b)

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| {(self.upbound-self.lobound)/2} for {self.lobound} <= x <= {self.upbound}" )
            print(f"| 0 else")

        else:
            if (self.lobound <= x <= self.upbound):
                return (self.upbound-self.lobound)/2
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

    def mean(self):
        return (self.lobound + self.upbound)/2

    def variance(self):
        return ((self.upbound - self.lobound)**2)/12

    def entropy(self):
        return np.log(self.upbound-self.lobound)

    def support(self):
        return (self.lobound, self.upbound)

    def infos(self):
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")


class Exponential(Distribution):
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

    def mean(self):
        return 1/self.alpha

    def variance(self):
        return 1/self.alpha**2

    def entropy(self):
        return 1-np.log(self.alpha)

    def support(self):
        return (0, np.inf)

    def infos(self):
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")


class Normal(Distribution):
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

    def mean(self):
        return self.mu

    def variance(self):
        return self.sd**2

    def entropy(self):
        return np.log(self.sd*np.sqrt(2*np.pi*np.e))

    def support(self):
        return (-np.inf ,np.inf)

    def infos(self):
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")


class Gamma(Distribution):
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

    def mean(self):
        return self.k/self.theta

    def variance(self):
        return self.k/(self.theta**2)

    def entropy(self):
        return self.k/self.theta + (1-self.k)*np.log(1/self.theta) + np.log(scipy.special.gamma(self.k)) + (1+self.k)*scipy.special.digamma(self.k)

    def support(self):
        return (0,np.inf)
    def infos(self):
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

