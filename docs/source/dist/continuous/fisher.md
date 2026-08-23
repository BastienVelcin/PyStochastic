# Fisher distribution


## Import line
You can import the Fisher distribution from the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import Fisher
```

## Description
```python
pystochastic.dist.Fisher(d1 = 2, d2 = 3)
```
**Type :** Class

Create an instance of the Fisher distribution. A Fisher distribution is a continuous probability distribution used in test statistics, in variance analysis, for example.

The probability density function of the Fisher distribution of parameters $d_1$ and $d_2$ is given by:

\begin{equation*}
f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1}d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}}{x \text{B}\left(\frac{d_1}{2},\frac{d_2}{2}\right)}\chi_{\mathbb{R}_+}(x).
\end{equation*}

> [!NOTE]
> The Fisher distribution is also known as the Fisher-Snedecor distribution.


### Attributes

`d1` : _int_
: First degree of freedom of the Fisher distribution. Must be strictly positive.

`d2` : _int_
: Second degree of freedom of the Fisher distribution. Must be strictly positive.

### Methods
The Fisher distribution inherits all methods from the [Continuous-Time Distribution](<project:/index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Fisher
>>> F = Fisher(3,6)
>>> F.sample(11)
array([0.91491482, 4.59181099, 0.65895937, 6.67934441, 0.62146571,
       1.83082035, 0.7162071 , 2.99598474, 1.38854416, 0.43777812,
       1.41993972])
>>> F.pdf(0.7)
np.float64(0.5030039374814714)
>>> F.cdf(0.7)
np.float64(0.4144953646422315)
>>> F.variance()
5.25
>>> F.info()
Distribution : Fisher
Parameters : {'d1': 3, 'd2': 6}
Probability density function :
| 0 for x < 0
| sqrt[((3*x)^3*6^6)/((3*x+6)^9)]/(x*Beta(1.5, 3.0)) for x >= 0
Cumulative distribution function :
| 0 for x < 0
| RegIncBeta((3*x)/(3*x+6),1.5,3.0) for x >= 0
Support : (0, inf)
Mean : 1.5
Variance : 5.25
Entropy : -0.31435427278974004
