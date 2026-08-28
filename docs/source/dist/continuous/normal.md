# Normal distribution


## Import line
You can import the Normal distribution from the `dist` module as follows:

```python
from pystochastic.dist import Normal
```

## Description
```python
pystochastic.dist.Normal(mu = 0, var = 1)
```
**Type :** Class

Creates an instance of the Normal distribution. A normal distribution is a continuous probability distribution used to model random natural phenomena.

The probability density function of the Normal distribution of mean $\mu$ and variance $\sigma^2$ is given by:

\begin{equation*}
f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
\end{equation*}


### Attributes

`mu` : _float_
: Mean parameter of the normal distribution.

`var` : _float_
: Standard deviation parameter of the normal distribution. Must be strictly positive.

> [!WARNING]
> The second parameter represents the variance. Don't confuse it with the standard deviation parameter of the normal distribution.
> The call `N = Normal(mu,sigma)` generates a normal distribution with mean `mu` and variance `sigma`.


### Methods
The Normal distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Normal
>>> N = Normal(-1,3)
>>> N.sample(12)
array([ 0.02198102,  0.47720099, -3.60885847, -1.73686551, -1.36815971,
       -0.95119961, -3.51687189, -2.63351054, -0.88179052, -2.37367196,
       -3.2103292 , -3.5364264 ])
>>> N.pdf(-3)
np.float64(0.1182550739094592)
>>> N.cdf(0.5)
np.float64(0.8067618846143836)
>>> N.variance()
3
>>> N.info()
Distribution : Normal
Parameters : {'mu': -1, 'var': 3}
Probability density function :
| (1/sqrt(2*3*pi)) * exp(-(x--1)^2 / 2*3)
Cumulative distribution function :
| (1+erf((x--1)/(sqrt(6))))/2
Support : (-inf, inf)
Mean : -1
Variance : 3
Entropy : 1.9682446775387274