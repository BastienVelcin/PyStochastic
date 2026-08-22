# Hypergeometric distribution


## Import line
You can import the hypergeometric distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Hypergeometric
```

## Description
```python
pystochastic.dist.Hypergeometric(N=2,K=1,m=1)
```
**Type :** Class

Create an instance of the hypergeometric distribution. A hypergeometric distribution is a discrete-time distribution that models the probability of obtaining a $m$ number of successes in a fixed number $K$ draws without replacement from a finite population of size $N$.
The probability distribution of the hypergeometric distribution of parameters $N$, $K$ and $m$ is given by:
\begin{equation*}
\mathbb{P}_HG = \sum_{k=\max(0,m+K-N)}^{\min(m,K)}\frac{\binom{K}{k} \binom{N-K}{m-k}}{\binom{N}{m}}
\end{equation*}
\end{equation*}

### Attributes

`N` : _int_
: Size of the population. Must be a strictly positive integer.

`K` : _int_
: Number of draws. Must be a positive integer.

`m` : _int_
: Number of desired successes. Must be a positive integer.

> [!WARNING]
> The parameters must satisfy $0 \leq K \leq N$ and $0 <= m <= N$.

### Methods
The hypergeometric distribution inherits all methods from the [Discrete-Time Distribution](<project:./index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Hypergeometric
>>> HG = Hypergeometric(15,8,4)
>>> HG.sample(8)
array([2, 1, 0, 1, 1, 2, 2, 1])
>>> HG.pmf(2)
0.4307692307692308
>>> HG.cdf(3.2)
0.9487179487179487
>>> HG.mean()
2.1333333333333333
>>> HG.info()
Distribution : Hypergeometric
Parameters : {'N': 15, 'K': 8, 'm': 4}
Probability mass function:
| comb(8,k) * comb(7,4-k) / 1365 if 0 <= k <= 4
| 0 otherwise
Cumulative distribution function :
| 1 - comb(4,k+1)*comb(11,7-k) / 6435 * GenHypFct(3,2,1,k+-7,k+-3,k+2,5+k,1)
| for k = floor(x) and x > 0
| 0 otherwise
Support : {0, 1, 2, 3, 4}
Mean : 2.1333333333333333
Variance : 0.7822222222222222
Entropy : Unknown