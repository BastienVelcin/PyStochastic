# Gumbel distribution


## Import line
You can import the Gumbel distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import Gumbel
```

## Description
```python
pystochastic.dist.Gumbel(mu = 0, beta = 1)
```
**Type :** Class

Create an instance of the Gumbel distribution. A Gumbel distribution is a continuous probability distribution used to model extreme values from low-tailed phenomena. 

The probability density function of the Gumbel distribution of parameters $\mu$ and $\beta$ is given by:

\begin{equation*}
f(x) = \frac{1}{\beta}\exp\left(-\exp\left(-\frac{x-\mu}{\beta}\right)\right)\exp\left(-\frac{x-\mu}{\beta}\right).
\end{equation*}

### Attributes

`mu` : _float_
: Position parameter of the Gumbel distribution.

`beta` : _float_
: Scale parameter of the Gumbel distribution. Must be strictly positive.

### Methods
The Gumbel distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Gumbel
>>> G = Gumbel(-1,1.5)
>>> G.sample(6)
array([-1.9208259 , -1.48819944,  3.11563336, -1.60648112,  0.06074798,
       -1.30882311])
>>> G.pdf(1.5)
np.float64(0.10424541734274533)
>>> G.cdf(-0.5)
np.float64(0.4884435800065159)
>>> G.mean()
-0.13417650264770065
>>> G.info()
Distribution : Gumbel
Parameters : {'mu': -1, 'beta': 1.5}
Probability density function :
| 0.6666666666666666 * exp(-exp(-(x--1)/1.5)) * exp(-(x--1)/1.5)
Cumulative distribution function :
| exp(-exp(-(x--1)/1.5))
Support : (-inf, inf)
Mean : -0.13417650264770065
Variance : 3.7011016504085092
Entropy : 1.9826807730096974
