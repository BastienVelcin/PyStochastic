# Dist Module

The dist module provides a set of classic discrete and continuous probability distributions, with
a common interface. This page explores all the distributions available and their properties.

```{toctree}
:maxdepth: 2
:caption: Distribution module

ContinuousDistribution
:maxdepth: 1
:caption: Continuous distributions

continuous/index

```

## Import line
You can import all the distributions listed in the `dist` module as follows:
```python
from pystochastic.dist.dist import *
```

## Implemented distributions
### Continuous distributions

- Continuous-Time Uniform
- Exponential
- Normal 
- Gamma
- Beta
- Weibull
- Fréchet
- Cauchy
- Gumbel
- Kumaraswamy
- Fisher
- Pareto
- Rayleigh

### Discrete distributions

- Discrete-Time Uniform
- Bernoulli
- Rademacher
- Binomial
- Poisson
- Geometric
- Hypergeometric
- Negative Binomial
- Yule Simon
