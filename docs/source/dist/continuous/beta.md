# Beta distribution


## Import line
You can import the Beta distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Beta
```

## Description
```python
pystochastic.dist.Beta(a = 1, b = 1)
```
**Type :** Class

Create an instance of the Beta distribution. A beta distribution is a continuous probability distribution used to study the variation of a probability, a percentage, or a rate. 

The probability density function of the Beta distribution of parameters $a$ and $b$ is given by:

\begin{equation*}
f(x) = \frac{1}{B(a,b)} x^{a-1}(1-x)^{b-1}\chi_{(0,1)}(x).
\end{equation*}

### Attributes

`a` : _float_
: First shape parameter of the beta distribution. Must be strictly positive.

`b` : _float_
: Second shape parameter of the beta distribution. Must be strictly positive.

> [!NOTE]
> The parameters `a` and `b` are symmetric, which means that if $X \sim \text{Beta}(a,b)$ and $Y \sim \text{Beta}(b,a)$.
> then $X$ and $Y$ follow the same distribution.

### Methods
The Beta distribution inherits all methods from the [Continuous-Time Distribution](<project:../intro.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Beta
>>> B = Beta(0.5,1.5)
>>> B.sample(10)
array([0.03102851, 0.1052254 , 0.01341274, 0.17452682, 0.53467125,
       0.02697998, 0.62019722, 0.57246912, 0.00321012, 0.33072112])
>>> B.pdf(0.5)
np.float64(0.6366197723675816)
>>> B.cdf(0.5)
np.float64(0.8183098861837911)
>>> B.mean()
0.25
>>> B.info()
Distribution : Beta
Parameters : {'a': 0.5, 'b': 1.5}
Probability density function :
| 0 for x < 0 or x > 1
| (x^-0.5 * (1-x)^0.5)/1.5707963267948963 for 0 <= x <= 1
Cumulative distribution function :
| 0 for x < 0
| IncBeta(x,0.5, 1.5)/1.5707963267948963 if 0 <= x <= 1
| 1 for x > 1
Support : (0, 1)
Mean : 0.25
Variance : 0.0625
Entropy : -0.5484172947105452
