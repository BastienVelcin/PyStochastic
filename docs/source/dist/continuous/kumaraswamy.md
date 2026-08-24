# Kumaraswamy distribution


## Import line
You can import the Kumaraswamy distribution from the `dist` module as follows:

```python
from pystochastic.dist import Kumaraswamy
```

## Description
```python
pystochastic.dist.Kumaraswamy(a = 0, b = 1)
```
**Type :** Class

Creates an instance of the Kumaraswamy distribution. A Kumaraswamy distribution is a continuous probability distribution used to model bounded random variables.

The probability density function of the Kumaraswamy distribution of parameters $a$ and $b$ is given by:

\begin{equation*}
f(x) = ab x^{a-1} \left(1-x^a\right)^{b-1}\chi_{[0,1]}(x).
\end{equation*}

### Attributes

`a` : _float_
: First shape parameter of the Kumaraswamy distribution.  Must be strictly positive.

`b` : _float_
: Second shape parameter of the Kumaraswamy distribution. Must be strictly positive.

### Methods
The Kumaraswamy distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Kumaraswamy
>>> K = Kumaraswamy(2,2)
>>> K.sample(10)
array([0.96773412, 0.77111767, 0.23769089, 0.87287474, 0.45161526,
       0.18700304, 0.11476253, 0.79525024, 0.8008753 , 0.53600337])
>>> K.pdf(0.3)
1.092
>>> K.cdf(0.2)
0.07840000000000003
>>> K.mean()
np.float64(0.5333333333333332)
>>> K.info()
Distribution : Kumaraswamy
Parameters : {'a': 2, 'b': 2}
Probability density function :
| 0 for x < 0 or x > 1
| 4* x^1 * (1-x^2)^1 for 0 <= x <= 1
Cumulative distribution function :
| 0 for x < 0
| 1-(1-x^2)**2 for 0 <= x <= 1
| 1 for x > 1
Support : (0, 1)
Mean : 0.5333333333333332
Variance : 0.04888888888888898
Entropy : 0.613705638880109
