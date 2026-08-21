# Pareto distribution


## Import line
You can import the Pareto distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Pareto
```

## Description
```python
pystochastic.dist.Pareto(x_m = 2, k = 3)
```
**Type :** Class

Create an instance of the Pareto distribution. A Pareto distribution is a continuous probability distribution used to model haivy-tailed phenomena where extreme values dominates the overall behavior.

The probability density function of the Pareto distribution of parameters $x_m$ and $k$ is given by:

\begin{equation*}
f(x) = \frac{k x_m^k}{x^{k+1}}\chi_{[x_m, +\infty)}(x),
\end{equation*}


### Attributes

`x_m` : _int_
: Position parameter of the Pareto distribution. Must be strictly positive.

`k` : _int_
: Shape parameter of the Pareto distribution. Must be strictly positive.

### Methods
The Pareto distribution inherits all methods from the [Continuous-Time Distribution](<project:../intro.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Pareto
>>> P = Pareto(1,1)
>>> P.sample(5)
array([1.22739427, 1.33464216, 4.76626616, 2.02557854, 2.6693788 ])
>>> P.pdf(10)
0.01
>>> P.cdf(7)
0.8571428571428572
>>> P.mean()
None
>>> P.info()
Distribution : Pareto
Parameters : {'x_m': 1, 'k': 1}
Probability density function :
| 0 for x < 1
| 1/(x^2) if x >= 1
Cumulative distribution function :
| 0 for x < 1
| 1 - (1/x)^1 for x >= 1
Support : (1, inf)
Mean : None
Variance : None
Entropy : -2.0
