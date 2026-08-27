from pystochastic.processes import GeometricBrownianMotion

# Diffusion Process class

The `DiffusionProcess` module provides an abstract class for all diffusion processes. A diffusion process is a stochastic process that is defined as a solution of a stochastic differential equation of the following form:

\begin{equation*}
dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
\end{equation*}

where:
- $(X_t)_{t\geq 0}$ is a $n$-dimensional stochastic process,
- $(W_t)_{t\geq 0}$ is a standard $d$-dimensional Brownian motion.
- $\mu : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^n$ is a continuous mapping, called **drift**,
- $\sigma : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^{n\times d}$ is a mapping, called **diffusion**,


## Import line
You can import all the diffusion processes as follows:

```python
from pystochastic.processes.diffusion import *
```


## Attributes
This section lists all the attributes that are common to all implemented diffuprocesses.

`T` : _float_
: Final time of the process. The process is simulated on the interval $[0,T]$.

`steps` : _int_
: Number of time steps on $[0,T]$. Must be greater than 0.


## Methods

#### .simulate()
```python
.simulate(n_simulations = 1, method = "euler-maruyama", plot = False, parallel = False, n_workers = None)
```
Simulates the effective process on the interval $[0,T]$.

**Parameters**

`n_simulations` : _int_
: Number of desired simulations.

`method` : _str_
: Desired simulation method. Must be one of the following: `euler-maruyama`, `milstein`, `runge-kutta` or `exact`.

> [!WARNING]
> - The Milstein and Runge-Kutta methods are only implemented for unidimensional diffusion processes that satisfy an autonomous stochastic differential equation.
>
>
> - The `exact` method is not implemented for all diffusion processes. If the exact method is chosen and is not implemented, PyStochastic raises an `NotImplementedError` error.


`plot` : _bool_
: Specifies whether to plot the simulation.

`parallel` : _bool_
: Specifies whether to parallelize the simulation. Only available if the numpy vectorization is not possible.

`n_workers` : _int_ or `None`
: Number of workers to use for parallel simulation.

> [!NOTE]
> PyStochastic tries to use the numpy vectorization to speed up the simulation when it's possible (for example, with scalar or diagonal drift and diffusion functions). 
>
>However, if it can't provide
a vectorized implementation, PyStochastic falls back to a sequential implementation. The `parallel` and `n_workers` parameters allow speeding up sequential simulation by using multiple workers. It is only useful
on a huge number of simulations and/or of time steps.


**Returns**

_np.ndarray_
: Path of `n_simulations` effective process on $[0,T]$. The final array is a three-dimensional array such that :
- 1st dimension is the number of simulations
- 2nd dimension is the number of time steps
- 3rd dimension is the dimension of the effective process 

For example, if we run 4 simulations of a 3D standard Geometric Brownian Motion on the interval $[0,1]$ with 1000 time steps with the following code:
```python
S = GeometricBrownianMotion(mu = [3,2,1], volatility = np.eye(3), initial = [5,12,1], T = 1, steps = 1000)
S.simulate(n_simulations = 4, plot = False)
```
then we can access to the first simulation with:
```python
S.path[0]
```
or to the second coordinates of the third simulation with:
```python
S.path[2,:,1]
```
or to the value of each path at the 201th time step with:
```python
S.path[:,200,:]
```

> [!IMPORTANT]
> Due to the different simulation methods, the `.simulation()` method of the `DiffusionProcess` class needs more arguments than the `.simulate()` method from the `Process` class.
>

The `DiffusionProcess` class also inherits all other methods from the [Process](<project:/processes/index.md>) class.

## Implemented processes

```{toctree}
:maxdepth: 1
:caption: Diffusion processes
geometric_brownian_motion
ornstein_uhlenbeck
vasicek
constant_elasticity_variance
cox_ingersoll_ross
heston
hull_white

:maxdepth: 1
:caption: Jump processes
jump/index
jump/poisson
jump/compound_poisson
```