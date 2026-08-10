from encodings import search_function

import numpy as np
import scipy
import plotly.graph_objects as go
import sympy
from abc import abstractmethod, ABC
from pystochastic.pyrandom import crandom



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
        supp = self.support()
        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if callable(self.variance) else 1

        lo_bound = supp[0] if supp[0] > -np.inf else mu - 8 * sd
        up_bound = supp[1] if supp[1] < np.inf else mu + 8 * sd

        supp_length = up_bound - lo_bound

        x_axis = np.linspace(lo_bound, up_bound, int(1000 * supp_length))
        y_axis = [self.pdf(x) for x in x_axis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="PDF", line=dict(width=2)))
        fig.update_layout(title="Probability Density Function",xaxis_title="x",yaxis_title="f(x)")
        fig.show()

    def plot_cdf(self):
        supp = self.support()
        mu = self.mean() if callable(self.mean) else 0
        sd = np.sqrt(self.variance()) if callable(self.variance) else 1

        lo_bound = supp[0] if supp[0] > -np.inf else mu - 8 * sd
        up_bound = supp[1] if supp[1] < np.inf else mu + 8 * sd

        supp_length = up_bound - lo_bound

        x_axis = np.linspace(lo_bound, up_bound, int(1000 * supp_length))
        y_axis = [self.cdf(x) for x in x_axis]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines", name="CDF", line=dict(width=2)))
        fig.update_layout(title="Probability Density Function",xaxis_title="x",yaxis_title="f(x)")
        fig.show()

    def infos(self):
        print(f"| Support : {self.support()}")
        print(f"| mean : {self.mean()}")
        print(f"| Variance : {self.variance()}")
        print(f"| Entropy : {self.entropy()}")

class Uniform(Distribution):
    def __init__(self,a=0,b=1):
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
        return crandom.uniform(self.lobound, self.upbound, n)

    def mean(self):
        return (self.lobound + self.upbound)/2

    def variance(self):
        return ((self.upbound - self.lobound)**2)/12

    def entropy(self):
        return np.log(self.upbound-self.lobound)

    def support(self):
        return (self.lobound, self.upbound)

class Exponential(Distribution):
    def __init__(self,alpha=1):
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
        return crandom.exponential(self.alpha, n)

    def mean(self):
        return 1/self.alpha

    def variance(self):
        return 1/self.alpha**2

    def entropy(self):
        return 1-np.log(self.alpha)

    def support(self):
        return (0, np.inf)

class Normal(Distribution):
    def __init__(self,mu=0,sd=1):
        self.mu = mu
        self.sd = sd
        if sd <=0:
            raise ValueError("The standard deviation should be greater than 0.")

    def pdf(self, x=None):

        if x is None:
            print("Probability density function :")
            print(f"| (1/{self.sd}*sqrt(2*pi)) * exp(-(x-{self.mu})^2 / 2*{self.sd}^2)")
        else:
            return (1/(self.sd*np.sqrt(2*np.pi)))*np.exp(-(x - self.mu)**2 / (2*self.sd**2))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| (1+erf((x-{self.mu})/({self.sd}*sqrt(2))))/2")
        else:
            return (1+scipy.special.erf((x-self.mu)/(self.sd*np.sqrt(2))))/2

    def sample(self,n=1):
        return crandom.normal(self.mu, self.sd, n)

    def mean(self):
        return self.mu

    def variance(self):
        return self.sd**2

    def entropy(self):
        return np.log(self.sd*np.sqrt(2*np.pi*np.e))

    def support(self):
        return (-np.inf ,np.inf)

class Gamma(Distribution):
    def __init__(self,k=1,theta=1):
        if k <=0:
            raise ValueError("The form parameter should be greater than 0.")
        if theta <=0:
            raise ValueError("The rate parameter should be greater than 0.")

        self.k = k
        self.theta = theta

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| ({self.theta**self.k} * x^{self.k-1} *  exp(-{self.theta}*x))/Gamma({self.k}) for x>= 0")
        else:
            if x<0:
                return 0
            return ((self.theta**self.k)*(x**(self.k-1))*np.exp(-self.theta*x))/scipy.special.gamma(self.k)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| IncGamma({self.k}, {self.theta}*x)/Gamma({self.k}) for x >= 0")
        else:
            if x<0:
                return 0
            return scipy.special.gammainc(self.k,self.theta*x)

    def sample(self,n=1):
        return crandom.gamma(self.k, self.theta, n)

    def mean(self):
        return self.k/self.theta

    def variance(self):
        return self.k/(self.theta**2)

    def entropy(self):
        return self.k/self.theta + (1-self.k)*np.log(1/self.theta) + np.log(scipy.special.gamma(self.k)) + (1+self.k)*scipy.special.digamma(self.k)

    def support(self):
        return (0,np.inf)

class Beta(Distribution):
    def __init__(self,a=1,b=1):
        if a <= 0:
            raise ValueError("The first shape parameter should be greater than 0.")
        if b <= 0:
            raise ValueError("The second shape parameter should be greater than 0.")

        self.a = a
        self.b = b

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0 or x > 1")
            print(f"| (x^{self.a-1} * (1-x)^{self.b-1})/{scipy.special.beta(self.a,self.b)} for 0 <= x <= 1")
        else:
            if not (0 <= x <= 1):
                return 0
            return ((x**(self.a-1)) * ((1-x)**(self.b-1)))/scipy.special.beta(self.a,self.b)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| IncBeta(x,{self.a}, {self.b})/{scipy.special.beta(self.a,self.b)} if 0 <= x <= 1")
            print(f"| 1 for x > 1")
        else:
            if x < 0:
                return 0
            elif x > 1:
                return 1
            else:
                return scipy.special.betainc(self.a,self.b,x)/scipy.special.beta(self.a,self.b)

    def sample(self,n=1):
        return crandom.beta(self.a, self.b, n)

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        return (self.a * self.b) / ((self.a + self.b)**2 * (self.a + self.b + 1))

    def entropy(self):
        return np.log(scipy.special.beta(self.a,self.b)) - (self.a-1)*scipy.special.digamma(self.a) - (self.b-1)*scipy.special.digamma(self.b) + (self.a+self.b-2)*scipy.special.digamma(self.a+self.b)

    def support(self):
        return (0,1)

class Weibull(Distribution):
    def __init__(self,k=1,l=1):
        if k <=0:
            raise ValueError("The shape parameter should be greater than 0.")
        if l <=0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.k = k
        self.l = l

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| {self.k/self.l}*(x/{self.l})^{self.k-1}*exp(-(x/{self.l})^{self.k}) for x >= 0")
        else:
            if x < 0:
                return 0
            return (self.k/self.l)*((x/self.l)**{self.k-1})*np.exp(-(x/self.l)**self.k)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1 - exp(-(x/{self.l})^{self.k}) for x >= 0")
        else:
            return 1 - np.exp(-(x/self.l)**self.k)

    def sample(self, n=1):
        return crandom.weibull(self.k, self.l, n)

    def mean(self):
        return self.l * scipy.special.gamma(1 + 1 / self.k)

    def variance(self):
        return self.l ** 2 * scipy.special.gamma(1 + 2 / self.k) - self.mean()**2

    def entropy(self):
        return scipy.special.gammainc(1-1/self.k)+(self.l/self.k)**self.k+np.log(self.l)-np.log(self.l/self.k)

    def support(self):
        return (0, np.inf)

class Frechet(Distribution):
    def __init__(self,a=1,s=1,m=0):

        if a <= 0:
            raise ValueError("The shape parameter should be greater than 0.")
        if s <= 0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.a = a
        self.s = s
        self.m = m

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < {self.m}")
            print(f"| {self.a/self.s}*((x-{self.m})/{self.s})^{-1-self.a} * exp(-((x-{self.m})/{self.s})^{-self.a}) for x >= {self.m}")
        else:
            if x < self.m:
                return 0
            return (self.a/self.s)*((x-self.m)/(self.s))^(-1-self.a) * np.exp(-((x-self.m)/(self.s))^(-self.a))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| exp(-((x-{self.m})/{self.s})^{-self.a})")
        else:
            if x <=self.m:
                return 0
            return  np.exp(-((x-self.m)/(self.s))^(-self.a))

    def sample(self, n=1):
        return crandom.frechet(self.a, self.s, self.m, n)

    def mean(self):
        if self.a > 1:
            return self.m + self.s * scipy.special.gamma(1 - 1 / self.a)
        else:
            return np.inf

    def variance(self):
        if self.a > 2:
            return self.s ** 2 * (scipy.special.gamma(1 - 2 / self.a) - scipy.special.gamma(1 - 1 / self.a)**2)
        else:
            return np.inf
    def entropy(self):
        return 1 + np.euler_gamma/self.a + np.euler_gamma + np.log(self.s/self.a)

    def support(self):
        return (self.m, np.inf)

class Cauchy(Distribution):

    def __init__(self,x=0,a=1):

        if a <= 0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.x = x
        self.a = a

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 1/(pi*{self.a}*(1+(x-{self.x})/{self.a})^2)")
        else:
            return 1/(np.pi*self.a*(1+(x-self.x)/self.a)**2)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 1/pi * Arctan((x-{self.x})/{self.a}) + 1/2")
        else:
            return 1/np.pi * np.arctan((x-self.x)/self.a) + 1/2

    def sample(self, n=1):
        return crandom.cauchy(self.x, self.a, n)

    def mean(self):
        return None

    def variance(self):
        return None

    def entropy(self):
        return np.log(4*np.pi*self.a)

    def support(self):
        return (-np.inf, np.inf)

class Gumbel(Distribution):

    def __init__(self,mu=0,beta=1):

        if beta <= 0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.mu = mu
        self.beta = beta

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| {1/self.beta} * exp(-exp(-(x-{self.mu})/{self.beta})) * exp(-(x-{self.mu})/{self.beta})")
        else:
            return (1/self.beta) * np.exp(-np.exp(-(x-self.mu)/self.beta)) * np.exp(-(x-self.mu)/self.beta)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| exp(-exp(-(x-{self.mu})/{self.beta}))")
        else:
            return np.exp(-np.exp(-(x-self.mu)/self.beta))

    def sample(self, n=1):
        return crandom.gumbel(self.mu, self.beta, n)

    def mean(self):
        return self.mu+self.beta*np.euler_gamma

    def variance(self):
        return (np.pi**2)*(self.beta**2)/6

    def entropy(self):
        return np.log(self.beta) + np.euler_gamma + 1

    def support(self):
        return (-np.inf, np.inf)


class Kumaraswamy(Distribution):

    def __init__(self,a=1,b=1):

        if a <= 0:
            raise ValueError("The first shape parameter should be greater than 0.")
        if b <= 0:
            raise ValueError("The second shape parameter should be greater than 0.")

        self.a = a
        self.b = b

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0 or x > 1")
            print(f"| {self.a*self.b}* x^{self.a-1} * (1-x^{self.a})^{self.b-1} for 0 <= x <= 1")
        else:
            if not 0 <=x <= 1:
                return 0
            return (self.a*self.b)*(x**(self.a-1)) * (1-x^self.a)**(self.b-1)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1-(1-x^{self.a})**{self.b} for 0 <= x <= 1")
            print(f"| 1 for x > 1")
        else:
            if x < 0:
                return 0
            elif x > 1:
                return 1
            else:
                return 1-(1-x**self.a)**self.b

    def sample(self, n=1):
        return crandom.kumaraswamy(self.a, self.b, n)

    def mean(self):
        return (self.b*scipy.special.gamma(1+1/self.a)*scipy.special.gamma(self.b))/scipy.special.gamma(1 + self.b + 1/self.a)

    def variance(self):
        return self.b * scipy.special.beta(1+2/self.a, self.b) - self.mean()**2

    def entropy(self):
        return (1-1/self.b)+(self.a-1)*sympy.harmonic(self.b) - np.log(self.a*self.b)

    def support(self):
        return (0,1)

class Fisher(Distribution):

    def __init__(self,d1=1,d2=1):

        if d1 <= 0:
            raise ValueError("The first degree of freedom should be greater than 0.")
        if d2 <= 0:
            raise ValueError("The second degree of freedom should be greater than 0.")

        self.d1 = d1
        self.d2 = d2

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| sqrt[(({self.d1}*x)^{self.d1}*{self.d2}^{self.d2})/(({self.d1}*x+{self.d2})^{self.d1+self.d2})]/(x*Beta({self.d1/2}, {self.d2/2})) for x >= 0")
        else:
            if x <= 0:
                return 0
            return np.sqrt[(((self.d1*x)**self.d1)*self.d2^self.d2)/((self.d1*x+self.d2)^(self.d1+self.d2))]/(x*scipy.special.beta({self.d1/2}, {self.d2/2}))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| RegIncBeta(({self.d1}*x)/({self.d1}*x+{self.d2}),{self.d1/2},{self.d2/2}) for x >= 0")
        else:
            if x <= 0:
                return 0
            return scipy.special.betainc(self.d1/2,self.d2/2,(self.d1*x)/(self.d1*x+self.d2))

    def sample(self, n=1):
        return crandom.fisher(self.d1, self.d2, n)

    def mean(self):
        if self.d2 > 2:
            return self.d2 / (self.d2 - 2)
        return None
    def variance(self):
        if self.d2 > 4:
            return (2*self.d2**2)*(self.d1+self.d2-2)/(self.d1*((self.d2-2)**2)*(self.d2-4))
        return None
    def entropy(self):
       return np.log(scipy.special.gamma(self.d1/2))+np.log(scipy.special.gamma(self.d2/2)) - np.log(scipy.special.gamma((self.d1+self.d2)/2)) + (1-{self.d1}/2)*scipy.special.digamma(1+self.d1/2) - (1+{self.d2}/2)*scipy.special.digamma(1+self.d2/2) + ((self.d1+self.d2)/2)*scipy.special.digamma((self.d1+self.d2)/2) + np.log(self.d2/self.d1)

    def support(self):
        return (0,np.inf)

class Pareto(Distribution):
    def __init__(self,x_m=1,k=1):

        if x_m <= 0:
            raise ValueError("The position parameter should be greater than 0.")
        if k <= 0:
            raise ValueError("The shape parameter should be greater than 0.")

        self.x_m = x_m
        self.k = k

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < {self.x_m}")
            print(f"| {self.k*(self.x_m)**(self.k)}/(x^{self.k+1}) if x >= {self.x_m}")
        else:
            if x < self.x_m:
                return 0
            return (self.k*(self.x_m)**(self.k))/(x**self.k+1)

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < {self.x_m}")
            print(f"| 1 - ({self.x_m}/x)^{self.k} for x >= {self.x_m}")
        else:
            if x <= self.x_m:
                return 0
            return  1 - (self.x_m/x)**(self.k)

    def sample(self, n=1):
        return crandom.pareto(self.x_m, self.k, n)

    def mean(self):
        if self.k > 1:
            return self.x_m * self.k / (self.k - 1)
        return None

    def variance(self):
        if self.k > 2:
            return (self.x_m**2) * self.k / (((self.k - 1)**2) * (self.k-2))
        return None
    def entropy(self):
        return np.log(self.k/self.x_m)-1/self.k -1

    def support(self):
        return (self.x_m, np.inf)

class Rayleigh(Distribution):
    def __init__(self,s=1):

        if s <= 0:
            raise ValueError("The scale parameter should be greater than 0.")

        self.s = s

    def pdf(self, x=None):
        if x is None:
            print("Probability density function :")
            print(f"| 0 for x < 0")
            print(f"| {1/self.s**2}*x*exp(-x^2/(2*s^2)) for x >= 0")
        else:
            if x < 0:
                return 0
            return (1/self.s**2)*x*np.exp(-(x**2)/(2*self.s**2))

    def cdf(self, x=None):
        if x is None:
            print("Cumulative distribution function :")
            print(f"| 0 for x < 0")
            print(f"| 1 - exp(-x^2/(2*s^2)) for x >= 0")
        else:
            if x < 0:
                return 0
            return  1 - np.exp(-(x**2)/(2*self.s**2))

    def sample(self, n=1):
        return crandom.pareto(self.s, n)

    def mean(self):
        return self.s*np.sqrt(np.pi/2)

    def variance(self):
        return (self.s**2)*(4-np.pi)/2
    def entropy(self):
        1 + np.log(self.s/np.sqrt(2))+np.euler_gamma/2

    def support(self):
        return (0, np.inf)

