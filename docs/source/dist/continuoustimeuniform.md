# Continuous-Time Uniform distribution


## Import line
You can import the Continuous-Time Uniform distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import Uniform
```

## Description
```{py:class} 
pystochastic.dist.Uniform(a=0, b=1)
```
**Type :** Class

Create an instance of the Continuous-Time Uniform distribution. A Continuous-Time Uniform distribution is a continuous-time distribution that takes values in the interval [a,b],
where a and b are real numbers such that a <= b.

### Attributes

- `a` : _float_
: Lower bound of the distribution.

- `b` : _float_
: Upper bound of the distribution.

Note that if a > b, the constructor will immediately change the roles of a and b.

### Methods
The Continuous-Time Uniform distribution inherits all methods from the Continuous-Time Distribution class.