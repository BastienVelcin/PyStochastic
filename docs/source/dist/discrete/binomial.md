# Binomial distribution


## Import line
You can import the Binomial distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import Binomial
```

## Description
```python
pystochastic.dist.Binomial(p = 0.5, n = 1)
```
**Type :** Class

Create an instance of the Binomial distribution. A Binomial distribution is a discrete-time distribution which represents `n` independant repetitions of the same experience with two issues: a success and a failure.
The probability distribution of the Binomial distribution of parameters $p$ and $n$ is given by:
\begin{equation*}
\mathbb{P}_B = \sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k}
\end{equation*}

> [!NOTE]
> A sample from the binomial distribution of parameters $p$ and $n$ can be obtained by sum `n` samples of the Bernoulli distribution of parameters $p$.
### Attributes

`p` : _float_
: Success probability. Must be in the interval [0,1].

`n` : _int_
: Number of repetitions. Must be an integer equal or greater than 1.


> [!NOTE]
> Another attribute is `q` which is the complementary probability. It is not specified by the user when instantiating the distribution.
### Methods
The Binomial distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Binomial
>>> B = Binomial(0.7,5)
>>> B.sample(10)
array([1, 0, 1, 2, 3, 3, 4, 1, 1, 2, 2, 5])
>>> B.pmf(4)
0.36014999999999997
>>> B.cdf(3.2)
np.float64(0.4717800000000001)
>>> B.mean()
3.5
>>> U.info()
Distribution : Binomial
Parameters : {'p': 0.7, 'q': 0.30000000000000004, 'n': 5}
Probability mass function:
| binom(n,k) * p^k * (1-p)^(n-k) if 0 <= k <= n
| 0 otherwise
None
Cumulative distribution function :
| 0 if x < 0
| IncBeta(0.30000000000000004, 5 - floor(x), 1 + floor(x) ) if 0 <= x < n
| 1 for x >= n
None
Support : {0, 1, 2, 3, 4, 5}
Mean : 3.5
Variance : 1.0500000000000003
Entropy : 1.4433336152893887
