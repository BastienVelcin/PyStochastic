# Bernoulli distribution


## Import line
You can import the Bernoulli distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Bernoulli
```

## Description
```python
pystochastic.dist.Bernoulli(p = 0.5)
```
**Type :** Class

Create an instance of the Bernoulli distribution. A Bernoulli distribution is a discrete-time distribution which represents an experience with two issues: a success and a failure.
The probability distribution of the Bernouilli distribution of parameter $p$ is given by:
\begin{equation*}
\mathbb{P}_B = (1-p)\delta_0 + p\delta_1
\end{equation*}

### Attributes

`p` : _float_
: Success probability. Must be in the interval [0,1].

> [!NOTE]
> Another attribute is `q` which is the complementary probability. It is not specified by the user when instantiating the distribution.
### Methods
The Bernoulli distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Bernoulli
>>> B = Bernoulli(0.3)
>>> B.sample(10)
array([0, 1, 1, 0, 0, 0, 1, 0, 1, 1])
>>> B.pmf(1)
0.3
>>> B.cdf(0.4)
0.7
>>> B.variance()
0.21
>>> B.info()
Distribution : Bernoulli
Parameters : {'p': 0.3, 'q': 0.7}
Probability mass function:
| 0.7 if k = 0
| 0.3 if k = 1
| 0 otherwise
Probability Mass Function : None
Cumulative distribution function :
| 0 if x < 0
| 0.7 if 0 <= x < 1
| 1 for x >= 1
Cumulative Distribution Function : None
Support : {0, 1}
Mean : 0.3
Variance : 0.21
Entropy : 0.6108643020548935
