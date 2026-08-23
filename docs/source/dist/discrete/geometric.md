# Geometric distribution


## Import line
You can import the geometric distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import Geometric
```

## Description
```python
pystochastic.dist.Geometric(p = 0.5)
```
**Type :** Class

Create an instance of the geometric distribution. A geometric distribution is a discrete-time distribution that counts the number of repetitions of a Bernoulli experiment needed to get a success.
The probability distribution of the geometric distribution of parameter $p$ is given by:
\begin{equation*}
\mathbb{P}_G = p\sum_{k=1}^{+\infty} (1-p)^{k-1}
\end{equation*}

### Attributes

`p` : _float_
: Success probability. Must be strictly positive.

> [!NOTE]
> Another attribute is `q` which is the complementary probability. It is not specified by the user when instantiating the distribution.

### Methods
The geometric distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Geometric
>>> G = Geometric(0.3)
>>> G.sample(9)
array([3, 3, 3, 6, 2, 3, 2, 1, 3])
>>> G.pmf(11)
0.008474257469999994
>>> G.cdf(10)
np.float64(0.9717524751000001)
>>> G.mean()
3.3333333333333335
>>> G.info()
Distribution : Geometric
Parameters : {'p': 0.3, 'q': 0.7}
Probability mass function:
| 0.3 * 0.7^(k-1) if k >= 1
| 0 otherwise
Cumulative distribution function :
| 1 - 0.7^floor(x) if x >= 1
| 0 otherwise
Support : N*
Mean : 3.3333333333333335
Variance : 7.777777777777778
Entropy : 2.0362143401829784
