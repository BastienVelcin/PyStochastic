# Discrete-Time Uniform distribution


## Import line
You can import the Discrete-Time Uniform distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import DUniform
```

## Description
```python
pystochastic.dist.DUniform(N = 3)
```
**Type :** Class

Create an instance of the Discrete-Time Uniform distribution. A Discrete-Time Uniform distribution is a discrete-time distribution that picks an integer between 1 and $N$ with the same probability.
The probability distribution of the Discrete-Time Uniform distribution of parameter $N$ is given by:

\begin{equation*}
\mathbb{P}_U = \sum_{i=1}^N \frac{1}{N}\delta_i
\end{equation*}

### Attributes

`N` : _float_
: Number of outcome values. Must be an integer greater than 1.

### Methods
The Discrete-Time Uniform distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import DUniform
>>> U = DUniform(5)
>>> U.sample(10)
array([3., 2., 0., 1., 4., 3., 3., 3., 0., 2.])
>>> U.pmf(1)
0.2
>>> U.cdf(3.5)
0.6
>>> U.mean()
3.0
>>> U.info()
Distribution : DUniform
Parameters : {'N': 5}
Probability mass function:
| 0.2 if 1 <= k <= 5
| 0 otherwise
Cumulative distribution function :
| 0 if x < 1
| floor(x)/5 if 1 <= x < 5
| 1 for x >= 5
Support : {1, 2, 3, 4, 5}
Mean : 3.0
Variance : 2.0
Entropy : 1.6094379124341003