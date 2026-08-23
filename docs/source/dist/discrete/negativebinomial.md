# Negative Binomial distribution


## Import line
You can import the Negative Binomial distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import NegativeBinomial
```

## Description
```python
pystochastic.dist.NegativeBinomial(p = 0.5, n = 1)
```
**Type :** Class

Create an instance of the Negative Binomial distribution. A Negative Binomial distribution is a discrete-time distribution which counts the number of failures until a specified number of successes occur for a repetition Bernoulli experiment of parameter $p$. 
The probability distribution of the Binomial distribution of parameters $p$ and $n$ is given by:
\begin{equation*}
\mathbb{P}_{NB} = \sum_{k=0}^{+\infty} \binom{k+n-1}{k} p^n (1-p)^{k}
\end{equation*}


### Attributes

`p` : _float_
: Success probability of the Bernoulli experiment. Must be in the interval [0,1].

`n` : _int_
: Number of desired successes. Must be a strictly positive integer.

> [!NOTE]
> Another attribute is `q` which is the complementary probability. It is not specified by the user when instantiating the distribution.
### Methods
The Binomial distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import NegativeBinomial
>>> NB = NegativeBinomial(0.3,7)
>>> NB.sample(11)
array([20, 29, 21, 26, 12, 21, 18, 14, 19, 21, 25])
>>> NB.pmf(12)
0.05619488967958085
>>> NB.cdf(30)
np.float64(0.9560328332450693)
>>> NB.variance()
54.44444444444444
>>> NB.info()
Distribution : NegativeBinomial
Parameters : {'p': 0.3, 'q': 0.7, 'n': 7}
Probability mass function:
| comb(k+6, k)*0.00021869999999999995*0.7^k if k >= 1
| 0 otherwise
Cumulative distribution function :
| IncRegGamma(7, floor(x)+1) if x >= 1
| 0 otherwise
Support : N*
Mean : 16.333333333333332
Variance : 54.44444444444444
Entropy : Unknown