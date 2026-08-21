# Fréchet distribution


## Import line
You can import the Fréchet distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Frechet
```

## Description
```python
pystochastic.dist.Frechet(a = 1, s = 1, m = 0)
```
**Type :** Class

Create an instance of the Fréchet distribution. A Fréchet distribution is a continuous probability distribution used to model the maximum value of heavy-tailed samples.

The probability density function of the Fréchet distribution of parameters $a$, $s$ and $m$ is given by:

\begin{equation*}
f(x) = \frac{a}{s} \left(\frac{x-m}{s} \right)^{k-1} e^{-\left(\frac{x-m}{s}\right)^{-a}}\chi_{(m,+\infty)}(x).
\end{equation*}

### Attributes

`a` : _float_
: Shape parameter of the Fréchet distribution. Must be strictly positive.

`s` : _float_
: Scale parameter of the Fréchet distribution. Must be strictly positive.

`m` : _float_
: Position parameter of the Fréchet distribution.

### Methods
The Fréchet distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Fréchet
>>> F = Frechet(4,2,3.2)
>>> F.sample(8)
array([5.12926362, 5.58321052, 5.11142306, 4.77993502, 5.596934  ,
       6.1838556 , 5.07168606, 5.19091624])
>>> F.pdf(6)
np.float64(0.2866417194222772)
>>> F.cdf(4.6)
np.float64(0.0155307821599998)
>>> F.variance()
np.float64(1.083231024899547)
>>> F.info()
Distribution : Frechet
Parameters : {'a': 4, 's': 2, 'm': 3.2}
Probability density function :
| 0 for x < 3.2
| 2.0*((x-3.2)/2)^-5 * exp(-((x-3.2)/2)^-4) for x >= 3.2
Cumulative distribution function :
| exp(-((x-3.2)/2)^-4)
Support : (3.2, inf)
Mean : 5.650833404930355
Variance : 1.083231024899547
Entropy : 1.0283724005669708