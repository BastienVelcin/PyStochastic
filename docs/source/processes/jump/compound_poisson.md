# Compound Poisson Process

## Import line
You can import the Compound Poisson process class from the `processes` module as follows:
```python
from pystochastic.processes import CompoundPoisson
```
## Description

```python
pystochastic.processes.CompoundPoisson(intensity = 1, distribution = Normal(0,1) , T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a compound Poisson process on the interval $[0, T]$, with a specified distribution $\mathfrak{D}$ and intensity parameter $\lambda$.
A Compound Poisson process of distribution $\mathfrak{D}$ is a stochastic process characterized by the following assertion:

If $(X_t)_{t \in \mathbb{N}} \sim \mathfrak{D}$, and $(N_t)_{t \geq 0}$ is a Poisson process of intensity $\lambda$, the stochastic process $(C_t)_{t \geq 0}$
defined for all $t \geq 0$ by
\begin{equation*}
C_t = \sum_{i=1}^{N(t)} X_i
\end{equation*}
is a compound Poisson process of intensity $\lambda$ and distribution $\mathfrak{D}$.

A Poisson process sums random-sized jumps that are distributed according to a same distribution $\mathfrak{D}$ and occur independently and at a constant rate $\lambda$, so that the number of events in any interval of length $t$ follows a Poisson distribution with parameter $\lambda t$.

### Parameters

`intensity` : _float_ 
: Intensity parameter $\lambda$ of the Poisson process. It represents the rate at which events occur.

`distribution` : _pystochastic.dist_
: Distribution of the jump sizes. Must be a distribution from the [Distributions](<project:/distributions/index.md>) module.

`T` : _float_
: Final time of the Compound Poisson process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Compound Poisson process is simulated. Must be strictly greater than 0.

### Attributes
The Compound Poisson process class inherits all attributes from the [Processes](<project:/processes/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Compound Poisson process class inherits all methods from the [Processes](<project:/processes/index.md>) class.
## Examples

```python
>>> from pystochastic.dist import Gamma
>>> C = CompoundPoisson(intensity = 3, distribution = Gamma(1.2,3.6), T = 1, steps = 500)
>>> C.simulate(100)
array([[[0.        ],
        [0.        ],
        [0.        ],
        ...,
        [0.88956339],
        [0.88956339],
        [0.88956339]],
       [[0.        ],
        [0.        ],
        [0.        ],
        ...,
        [0.58722488],
        [0.58722488],
        [0.58722488]],
       ...,
       [[0.        ],
        [0.        ],
        [0.        ],
        ...,
        [1.70458235],
        [1.70458235],
        [1.70458235]],
       [[0.        ],
        [0.        ],
        [0.        ],
        ...,
        [1.05428974],
        [1.05428974],
        [1.05428974]],
       [[0.        ],
        [0.        ],
        [0.        ],
        ...,
        [0.89657774],
        [0.89657774],
        [0.89657774]]], shape=(100, 501, 1))

>>> C.plot()
```

## References
- Mikosch, T. (2009). Non-Life Insurance Mathematics. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-88233-6
