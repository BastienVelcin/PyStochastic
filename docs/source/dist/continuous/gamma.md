# Gamma distribution


## Import line
You can import the Gamma distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Gamma
```

## Description
```{py:class} 
pystochastic.dist.Gamma(k = 1, theta = 1)
```
**Type :** Class

Create an instance of the Gamma distribution. A gamma distribution is a continuous probability distribution used to model econometrics or insurance problems. 

The probability density function of the Gamma distribution is given by:

\begin{equation*}
f(x) = \frac{\theta^k}{\Gamma(k)} x^{k-1}e^{-\theta x}\chi_{\mathbb{R}_+}(x).
\end{equation*}


### Attributes

`k` : _float_
: Shape parameter of the gamma distribution. Must be strictly positive.

`theta` : _float_
: Rate parameter of the gamma distribution. Must be strictly positive.

### Methods
The Gamma distribution inherits all methods from the [Continuous-Time Distribution](<project:../intro.md>) class.

### Examples

```python
>>> from pystochastic.dist.dist import Gamma
>>> G = Gamma(2,3.5)
>>> G.sample(8)
array([0.5225304 , 0.74659204, 0.20366063, 0.40455494, 0.31522825,
       0.76491317, 0.20232503, 0.63915118])
>>> G.pdf(0)
np.float64(0.0)
>>> G.cdf(1)
np.float64(0.8641117745995668)
>>> G.mean()
0.5714285714285714
>>> G.info()
Distribution : Gamma
Parameters : {'k': 2, 'theta': 3.5}
Probability density function :
| 0 for x < 0
| (12.25 * x^1 *  exp(-3.5*x))/Gamma(2) for x>= 0
Cumulative distribution function :
| 0 for x < 0
| IncGamma(2, 3.5*x)/Gamma(2) for x >= 0
Support : (0, inf)
Mean : 0.5714285714285714
Variance : 0.16326530612244897
Entropy : 3.092544545219341
