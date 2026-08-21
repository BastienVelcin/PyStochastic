# Weibull distribution


## Import line
You can import the Weibull distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Weibull
```

## Description
```python
pystochastic.dist.Weibull(a = 1, b = 1)
```
**Type :** Class

Create an instance of the Weibull distribution. A Weibull distribution is a continuous probability distribution used for reliability modeling.

The probability density function of the Weibull distribution of parameters $k$ and $\lambda$ is given by:

\begin{equation*}
f(x) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1}e^{-\left(\frac{x}{\lambda}\right)^k}\chi_{\mathbb{R}_+}(x).
\end{equation*}

### Attributes

`k` : _float_
: Scale parameter of the Weibull distribution. Must be strictly positive.

`l` : _float_
: Shape parameter of the Weibull distribution. Must be strictly positive.

### Methods
The Weibull distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Weibull
>>> W = Weibull(2.5,6)
>>> W.sample(9)
array([7.7116568 , 7.6449409 , 4.56348225, 6.29896091, 5.71629054,
       6.45260193, 3.29415687, 5.33925405, 4.05378596])
>>> W.pdf(1)
np.float64(0.02803088975131713)
>>> W.cdf(1.5)
np.float64(0.03076676552365587)
>>> W.mean()
np.float64(5.323582905018451)
>>> W.info()
Distribution : Weibull
Parameters : {'k': 2.5, 'l': 6}
Probability density function :
| 0 for x < 0
| 0.4166666666666667*(x/6)^1.5*exp(-(x/6)^2.5) for x >= 0
Cumulative distribution function :
| 0 for x < 0
| 1 - exp(-(x/6)^2.5) for x >= 0
Support : (0, inf)
Mean : 5.323582905018451
Variance : 5.1892808086840425
Entropy : 8.299025147642851
