# Yule-Simon distribution


## Import line
You can import the Yule-Simon distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import YuleSimon
```

## Description
```python
pystochastic.dist.YuleSimon(p = 0.5)
```
**Type :** Class

Create an instance of the Yule-Simon distribution. A Yule-Simon distribution is a discrete-time distribution that models preferential growth phenomena.
The probability distribution of the Yule-Simon distribution of parameter $\rho$ is given by:
\begin{equation*}
\mathbb{P}_{YS} = \sum_{k=1}^{+\infty} \rho B(k,\rho + 1)
\end{equation*}

### Attributes

`rho` : _float_
: Shape parameter. Must be strictly positive.

### Methods
The geometric distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import YuleSimon
>>> YS = YuleSimon(0.5)
>>> YS.sample(11)
array([ 61,   6, 471,   6,   4,  18,  39,   2,  27,   2,   2])
>>> YS.pmf(12)
np.float64(0.010340389632353555)
>>> YS.cdf(60)
np.float64(0.8862975138214279)
>>> YS.variance()
None
>>> YS.info()
Distribution : YuleSimon
Parameters : {'rho': 0.5}
Probability mass function:
| 0.5 * Beta(k, 1.5) if k >= 1
| 0 otherwise
Cumulative distribution function :
| 1 - floor(x)*Beta(floor(x), 1.5) if x >= 1
| 0 otherwise
Support : N*
Mean : None
Variance : None
Entropy : Unknown
