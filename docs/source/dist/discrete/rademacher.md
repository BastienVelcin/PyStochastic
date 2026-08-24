# Rademacher distribution


## Import line
You can import the Rademacher distribution from the `dist` module as follows:

```python
from pystochastic.dist import Rademacher
```

## Description
```python
pystochastic.dist.Rademacher(p = 0.5)
```
**Type :** Class

Creates an instance of the Rademacher distribution. A Rademacher distribution is a discrete-time distribution which represents an experience with a gain and a loss.
The probability distribution of the Rademacher distribution of parameter $p$ is given by:
\begin{equation*}
\mathbb{P}_R = (1-p)\delta_{-1} + p\delta_1
\end{equation*}

> [!NOTE]
> The Rademacher distribution implementation generalizes the standard Rademacher distribution, which has no parameter $p$. The standard Rademacher distribution can be obtained by setting $p=0.5$.
### Attributes

`p` : _float_
: Gain probability. Must be in the interval [0,1].

> [!NOTE]
> Another attribute is `q` which is the complementary probability. It is not specified by the user when instantiating the distribution.
### Methods
The Rademacher distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Rademacher
>>> R = Rademacher(0.3)
>>> R.sample(8)
array([0, 0, 0, 0, 1, 0, 0, 0])
>>> R.pmf(-1)
0.7
>>> R.cdf(0.5)
0.7
>>> R.mean()
-0.4
>>> R.info()
Distribution : Rademacher
Parameters : {'p': 0.3, 'q': 0.7}
Probability mass function:
| 0.7 if k = -1
| 0.3 if k = 1
| 0 otherwise
Probability Mass Function : None
Cumulative distribution function :
| 0 if x < -1
| 0.7 if -1 <= x < 1
| 1 for x >= 1
Cumulative Distribution Function : None
Support : {1, -1}
Mean : -0.4
Variance : 1
Entropy : 0.6108643020548935
