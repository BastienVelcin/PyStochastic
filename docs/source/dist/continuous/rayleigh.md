# Rayleigh distribution


## Import line
You can import the Rayleigh distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Rayleigh
```

## Description
```python
pystochastic.dist.Rayleigh(s = 1)
```
**Type :** Class

Create an instance of the Rayleigh distribution. A Rayleigh distribution is a continuous probability distribution used to describe the amplitude of random phenomena.

The probability density function of the Rayleigh distribution of parameter $\sigma$ is given by:

\begin{equation*}
f(x) = \frac{x}{\sigma^2}\exp\left(-\frac{x^2}{2\sigma^2}\right)\chi_{\mathbb{R}_+}(x).
\end{equation*}


### Attributes

`s` : _int_
: Scale parameter of the Rayleigh distribution. Must be strictly positive.

### Methods
The Rayleigh distribution inherits all methods from the [Continuous-Time Distribution](<project:../intro.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Rayleigh
>>> R = Rayleigh(3)
>>> R.sample(10)
array([3.31071439, 1.16699297, 1.39041003, 1.20251137, 3.54765386,
       3.30139312, 1.11240684, 5.3951555 , 3.7597183 , 1.73849304])
>>> R.pdf(3)
np.float64(0.2021768865708778)
>>> R.cdf(3)
np.float64(0.3934693402873666)
>>> R.variance()
3.862833058845931
>>> R.info()
Distribution : Rayleigh
Parameters : {'s': 3}
Probability density function :
| 0 for x < 0
| 0.1111111111111111*x*exp(-x^2/(2*s^2)) for x >= 0
Cumulative distribution function :
| 0 for x < 0
| 1 - exp(-x^2/(2*s^2)) for x >= 0
Support : (0, inf)
Mean : 3.7599424119465006
Variance : 3.862833058845931
Entropy : 2.0406465308389032
