# Normal distribution


## Import line
You can import the Normal distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Normal
```

## Description
```python
pystochastic.dist.Normal(mu = 0, sd = 1)
```
**Type :** Class

Create an instance of the Normal distribution. A normal distribution is a continuous probability distribution used to model random natural phenomena.

The probability density function of the Normal distribution of mean $\mu$ and standard deviation $\sigma$ is given by:

\begin{equation*}
f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
\end{equation*}


### Attributes

`mu` : _float_
: Mean parameter of the normal distribution.

`sd` : _float_
: Standard deviation parameter of the normal distribution. Must be strictly positive.

> [!WARNING]
> The second parameter represents the standard deviation. Don't confuse it with the variance parameter of the normal distribution.
> The call `N = Normal(mu,sigma)` generates a normal distribution with mean `mu` and variance `sigma**2`.


### Methods
The Normal distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Normal
>>> N = Normal(-1,3)
>>> N.sample(12)
array([-1.54071764, -3.31218451, -0.4471541 ,  1.55421478, -2.31592963,
       -5.1728659 , -4.38693714, -0.43997557,  3.99003161, -2.18267661,
        0.53702751, -2.44383932])
>>> N.pdf(-3)
np.float64(0.10648266850745075)
>>> N.cdf(0.5)
np.float64(0.691462461274013)
>>> N.variance()
9
>>> N.info()
Distribution : Normal
Parameters : {'mu': -1, 'sd': 3}
Probability density function :
| (1/3*sqrt(2*pi)) * exp(-(x--1)^2 / 2*3^2)
Cumulative distribution function :
| (1+erf((x--1)/(3*sqrt(2))))/2
Support : (-inf, inf)
Mean : -1
Variance : 9
Entropy : 2.5175508218727822