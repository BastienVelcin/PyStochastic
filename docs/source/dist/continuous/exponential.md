# Exponential distribution


## Import line
You can import the Continuous-Time Uniform distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Exponential
```

## Description
```{py:class} 
pystochastic.dist.Uniform(a=0, b=1)
```
**Type :** Class

Create an instance of the Continuous-Time Uniform distribution. A Continuous-Time Uniform distribution is a continuous-time distribution that takes values in the interval [a,b],
where a and b are real numbers such that a <= b.

The probability density function of the Continuous-Time Uniform distribution is given by:

\begin{equation*}
f(x) = \frac{1}{b-a} \chi_{a \leq x \leq b}
\end{equation*}

> [!NOTE]
> The Continuous-Time Uniform distribution is characterized by the following property:
> 
> Every same-length real interval within the support interval [a,b] have the same probability.

### Attributes

- `a` : _float_
: Lower bound of the distribution.

- `b` : _float_
: Upper bound of the distribution.

Note that if a > b, the constructor will immediately change the roles of a and b.

### Methods
The Continuous-Time Uniform distribution inherits all methods from the Continuous-Time Distribution class.