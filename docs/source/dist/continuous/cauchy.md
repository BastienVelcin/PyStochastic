# Cauchy distribution


## Import line
You can import the Cauchy distribution from the `dist` module as follows:

```python
from pystochastic.dist import Cauchy
```

## Description
```python
pystochastic.dist.Cauchy(x_0 = 0, a = 1)
```
**Type :** Class

Creates an instance of the Cauchy distribution. A Cauchy distribution is a continuous probability distribution used to describe phenomena with extreme unpredictable fluctuations.

The probability density function of the Cauchy distribution of parameters $x_0$ and $a$ is given by:

\begin{equation*}
f(x) = \frac{1}{\pi a \left[1 + \left( \frac{x-x_0}{a}\right)^2 \right]}.
\end{equation*}

### Attributes

`x_0` : _float_
: Position parameter of the Cauchy distribution.

`a` : _float_
: Scale parameter of the Cauchy distribution. Must be strictly positive.

### Methods
The Cauchy distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Cauchy
>>> C = Cauchy(0,1)
>>> C.sample(9)
array([  0.71633811,   1.00722029, -12.04420328,   1.38184178,
         0.64025223,  15.20672029,   2.5047318 ,  11.90794291,
         0.68826505])
>>> C.pdf(0)
0.3183098861837907
>>> C.cdf(0)
np.float64(0.5)
>>> C.mean()
None
>>> C.info()
Distribution : Cauchy
Parameters : {'x': 0, 'a': 1}
Probability density function :
| 1/(pi*1*(1+(x-0)/1)^2)
Cumulative distribution function :
| 1/pi * Arctan((x-0)/1) + 1/2
Support : (-inf, inf)
Mean : None
Variance : None
Entropy : 2.5310242469692907
