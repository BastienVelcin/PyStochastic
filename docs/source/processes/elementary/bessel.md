# Bessel Process

## Import line
You can import the Bessel process class from the `processes` module as follows:
```python
from pystochastic.processes import Bessel
```
## Description

```python
pystochastic.processes.Bessel(order = 2, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Bessel process on the interval $[0, T]$, with a specified norm order matrix.
A Bessel process of order $n\in\mathbb{N}^\star$  $\left(\mathrm{BES}_t(n)\right)_{t\geq 0}$ is defined as the Euclidean norm of a standard $n$-dimensional Brownian motion:

\begin{equation*}
\forall t\geq 0, ~~ \mathrm{BES}_t(n) = ||W_t|| = \sqrt{\sum_{i=1}^n \left(W_t^i\right)^2}
\end{equation*}

where $(W_t)_{t\geq 0}$ denotes a standard $n$-dimensional Brownian motion.

>[!NOTE]
> A $d$-dimensional Brownian motion is said standard if $Q = \mathrm{Id}_d$.

The Brownian motion is one of the fundamental objects of stochastic calculus, since it mathematically formalizes the concept of random motion.
It appears naturally in stochastic differential equations and so, within the diffusion process definitions.

### Parameters

`order` : _int_
: Order of the Bessel process. In the previous example, the order is represented by $n$.

`T` : _float_
: Final time of the Bessel process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Bessel process is simulated. Must be strictly greater than 0.

### Attributes
The Bessel class inherits all attributes from the [Processes](<project:/processes/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Bessel class inherits all methods from the [Processes](<project:/processes/index.md>) class.
## Examples

```python
>>> BSL = Bessel(order = 5, T = 10, steps= 750)
>>> BSL.simulate(5, plot = False)
array([[[0.        ],
        [0.28245147],
        [0.39893963],
        ...,
        [5.6968423 ],
        [5.7301934 ],
        [5.9047602 ]],
       [[0.        ],
        [0.33095952],
        [0.62998883],
        ...,
        [8.81073229],
        [9.05938807],
        [9.02016293]],
       [[0.        ],
        [0.25659784],
        [0.53961491],
        ...,
        [3.36714455],
        [3.33645985],
        [3.42682631]],
       [[0.        ],
        [0.3264745 ],
        [0.4678982 ],
        ...,
        [3.24248475],
        [3.16006872],
        [3.10188941]],
       [[0.        ],
        [0.34346597],
        [0.36442699],
        ...,
        [7.31806438],
        [7.33390633],
        [7.41006382]]], shape=(5, 751, 1))
>>> BSL.max()
(array([[ 8.59509264],
       [10.70444174],
       [ 6.22984211],
       [ 5.51000105],
       [ 7.52481944]]), array([[590],
       [435],
       [480],
       [354],
       [745]]), array([[7.86666667],
       [5.8       ],
       [6.4       ],
       [4.72      ],
       [9.93333333]]))