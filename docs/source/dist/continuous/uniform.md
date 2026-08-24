# Continuous-Time Uniform distribution


## Import line
You can import the Continuous-Time Uniform distribution from the `dist` module as follows:

```python
from pystochastic.dist import Uniform
```

## Description
```python
pystochastic.dist.Uniform(a = 0, b = 1)
```
**Type :** Class

Creates an instance of the Continuous-Time Uniform distribution. A Continuous-Time Uniform distribution is a continuous-time distribution that takes values in the interval [a,b],
where a and b are real numbers such that $a \leq b$.

The probability density function of the Continuous-Time Uniform distribution of parameters $a$ and $b$ is given by:

\begin{equation*}
f(x) = \frac{1}{b-a} \chi_{a \leq x \leq b}(x).
\end{equation*}

> [!NOTE]
> The Continuous-Time Uniform distribution is characterized by the following property:
> 
> Every same-length real interval included in the support interval [a,b] have the same probability.

### Attributes

`a` : _float_
: Lower bound of the distribution.

`b` : _float_
: Upper bound of the distribution.

Note that if a > b, the constructor will immediately change the roles of a and b.

### Methods
The Continuous-Time Uniform distribution inherits all methods from the [Continuous-Time Distribution](<project:.../index.md>) class.

## Examples

```python
>>> from pystochastic.dist.dist import Uniform
>>> U = Uniform(a=0, b=1)
>>> U.sample(10)
array([0.21551514, 0.67418772, 0.72580384, 0.44785791, 0.16345073,
       0.06059345, 0.6648661 , 0.74893065, 0.4546897 , 0.17916723])
>>> U.pdf(1)
1.0
>>> U.cdf(0.5)
0.5
>>> U.mean()
0.5
>>> U.info()
Distribution : Uniform
Parameters : {'lobound': 0, 'upbound': 1}
Probability density function :
| 1.0 for 0 <= x <= 1
| 0 else
Cumulative distribution function :
| 0 for x < 0
| (x-0)/1 for 0 <= x <= 1
| 1 for x > 1
Support : (0, 1)
Mean : 0.5
Variance : 0.08333333333333333
Entropy : 0.0