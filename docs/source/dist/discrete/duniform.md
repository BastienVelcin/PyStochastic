# Discrete-Time Uniform distribution


## Import line
You can import the Descrete-Time Uniform distribution from the `dist` module as follows:
```python
from pystochastic.dist.dist import DUniform
```

## Description
```python
pystochastic.dist.DUniform(N = 3)
```
**Type :** Class

Create an instance of the Descrete-Time Uniform distribution. A Descrete-Time Uniform distribution is a descrete-time distribution which pick a integer number 1 and $N$ with the same probability.
The probability distribution of the Descrete-Time Uniform distribution of parameter $N$ is given by:

\begin{equation*}
\mathbb{P}_U = \sum_{i=1}^N \frac{1}{N}\delta_i
\end{equation*}

### Attributes

`N` : _float_
: Number of outcome values. Must be an integer greater than 1.

### Methods
The Descrete-Time Uniform distribution inherits all methods from the [Descrete-Time Distribution](<project:/index.md>) class.

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