# Poisson distribution


## Import line
You can import the Poisson distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import Poisson
```

## Description
```python
pystochastic.dist.Poisson(lam = 0.5)
```
**Type :** Class

Create an instance of the Poisson distribution. A Poisson distribution is a discrete-time distribution that counts the occurrences of an event on a fixed time interval.
The probability distribution of the Poisson distribution of parameter $\lambda$ is given by:
\begin{equation*}
\mathbb{P}_P = e^{-\lambda} \sum_{k=0}^{+\infty} \frac{\lambda^k}{k!} \delta_k
\end{equation*}

### Attributes

`lam` : _float_
: Intensity parameter. Must be strictly positive.

### Methods
The Poisson distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Poisson
>>> P = Poisson(5)
>>> P.sample(12)
array([7, 9, 6, 9, 3, 5, 3, 4, 3, 6, 9, 5])
>>> P.pmf(8)
np.float64(0.06527803934815875)
>>> P.cdf(0.4)
np.float64(0.006737946999085468)
>>> P.variance()
5
>>> P.info()
Distribution : Poisson
Parameters : {'lam': 5}
Probability mass function:
| exp(-5) * 5^k / k! if k >= 0
| 0 otherwise
Cumulative distribution function :
| 0 if x < 0
| RegUpGamma(floor(x)+1, 5)/(floor(x)!) if x >= 0
Support : N
Mean : 5
Variance : 5
Entropy : 2.204395243428368