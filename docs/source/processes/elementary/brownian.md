# Brownian motion

## Import line
You can import the Brownian motion class from the `processes` module as follows:
```python
from pystochastic.processes import Brownian
```
## Description

```python
pystochastic.processes.Brownian(variance = 1, t_0 = 0, t_n = 1, steps = 1000)
```
**Type :** Class

Creates an instance of a Brownian motion process on the interval $[t_0, t_n]$, with a specified covariance matrix.
Let $Q$ be a symmetric positive definite matrix. A $d$-dimensional $Q$-Brownian motion $(W_t)_{t\geq 0}$ is an $\mathbb{R}^d$-valued Gaussian process such that
- $W_0 = 0 ~~a.s.$ 
- $\forall s,t \in \mathbb{R}_+, ~~ \mathbb{E}[W_t] = 0 ~~$ and $~~ \mathbb{E}[W_s W_t^\star] = (s\wedge t)Q$
- $(W_t)_{t\geq 0}$ is continuous process

>[!NOTE]
> A $d$-dimensional Brownian motion is said standard if $Q = \mathrm{Id}_d$.

The Brownian motion is one of the fundamental object of stochastic calculus, since it mathematically formalizes the concept of random motion.
It appears naturally in stochastic differential equations, and so, within the diffusion process definitions.

### Parameters

`variance` : _float_ or _np.ndarray_
: Variance-covariance matrix of the Brownian motion. In the previous example, it is represented by $Q$.

`t_0` : _float_
: Initial time of the Brownian motion simulation.

`t_n` : _float_
: Final time of the Brownian motion simulation. Must be greater than `t_0`.

`steps` : _int_
: Number of time steps between `t_0` and `t_n` on which the Brownian motion is simulated. Must be strictly greater than 0.

### Attributes
The Brownian motion inherits all attributes from the [Processes](<project:/index.md>) class.
### Methods
The Cauchy distribution inherits all methods from the [Processes](<project:/index.md>) class.

## Examples

```python
>>> W = Brownian(variance = np.eye(2),t_0 = 0, t_n = 1, steps = 1000)
>>> W.simulate(2, plot = True)
>>> dW = W.increments