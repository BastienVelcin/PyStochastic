# Poisson Process

## Import line
You can import the Poisson process class from the `processes` module as follows:
```python
from pystochastic.processes import PoissonProcess
```
## Description

```python
pystochastic.processes.PoissonProcess(intensity = 1, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a homogeneous Poisson process on the interval $[0, T]$, with a specified intensity (or mean parameter) $\lambda$.
A Poisson process of intensity parameter $\lambda$ is a stochastic process characterized by the following assertion:

If $(X_t)_{t \in \mathbb{N}} \sim \mathrm{Exp}(\lambda)$, and if we define $T_0 = 0$ and $T_n = \sum_{i=1}^n X_i$, then the stochastic process
$(N_t)_{t \geq 0}$ defined for all $t \geq 0$ by
\begin{equation}
N_t = \sup \left\{n\in\mathbb{N} | T_n \leq t \right\}
\end{equation}

>[!NOTE]
> Note that, for all $t \geq 0$, $N_t \sim \mathrm{Poisson}(\lambda t)$.

A Poisson process counts events that occur independently and at a constant rate $\lambda$, so that the number of events in any interval of length $t$ follows a Poisson distribution with parameter $\lambda t$.

### Parameters

`intensity` : _float_ 
: Intensity parameter $\lambda$ of the Poisson process. It represents the rate at which events occur.

`T` : _float_
: Final time of the Poisson process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Poisson process is simulated. Must be strictly greater than 0.

### Attributes
The Poisson process class inherits all attributes from the [Processes](<project:/processes/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Poisson process class inherits all methods from the [Processes](<project:/processes/index.md>) class.
## Examples

```python
>>> P = PoissonProcess(intensity = 2.7, T = 1, steps = 365)
>>> P.simulate(100)
array([[[0],
        [0],
        [0],
        ...,
        [1],
        [1],
        [1]],
       [[0],
        [0],
        [0],
        ...,
        [2],
        [2],
        [2]],
        ...,
       [[0],
        [0],
        [0],
        ...,
        [4],
        [4],
        [4]]], shape=(100, 1001, 1))

>>> P.plot()
```

## References
- Mikosch, T. (2009). Non-Life Insurance Mathematics. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-88233-6
