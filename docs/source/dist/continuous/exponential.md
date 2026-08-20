# Exponential distribution


## Import line
You can import the Continuous-Time Uniform distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Exponential
```

## Description
```{py:class} 
pystochastic.dist.Exponential(alpha = 2)
```
**Type :** Class

Create an instance of the Exponential distribution. An exponential distribution is a continuous probability distribution that represents a memoryless phenomenon with an intensity parameter `alpha`.

The probability density function of the Exponential distribution is given by:

\begin{equation*}
f(x) = \alpha e^{-\alpha x}\chi_{\mathbb{R}_+}(x).
\end{equation*}

> [!NOTE]
> The Exponential distribution is characterized by the following property:
> 
> If $X$ follows an exponential distribution, then, for all $s,t \in \mathbb{R},~~ \mathbb{P}(X>t+s ~|~ X>t)=\mathbb{P}(X>s)$.

### Attributes

`alpha` : _float_
: Intensity parameter of the exponential distribution. Must be a **strictly positive real number**.


### Methods
The Exponential distribution inherits all methods from the [Continuous-Time Distribution](<project:../intro.md>) class.

### Examples

```python
>>> from pystochastic.dist.dist import Exponential
>>> E = Exponential(2.5)
>>> E.sample(7)
array([1.11649568, 0.17226749, 0.06755954, 0.17814998, 0.03381519,
       0.44523216, 0.37164071])
>>> E.pdf(6)
np.float64(7.647558012545645e-07)
>>> E.cdf(3)
np.float64(0.9994469156298522)
>>> E.variance()
0.16
>>> E.info()
Distribution : Exponential
Parameters : {'alpha': 2.5}
Probability density function :
| 0 for x < 0
| 2.5 * exp(-2.5*x) for x >= 0
Cumulative distribution function :
| 0 for x < 0
| 1- exp(-2.5*x) for x >= 0
Support : (0, inf)
Mean : 0.4
Variance : 0.16
Entropy : 0.0837092681258449