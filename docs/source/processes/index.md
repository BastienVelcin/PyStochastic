# Processes Module

The `processes` module provides a set of classic stochastic processes methods for simulation.

## Import line
You can import all the distributions listed in the `continuous.py` module as follows:

```python
from pystochastic.dist.continuous import *
```

## Attributes
This section lists all the attributes that are common to all implemented processes.

`T` : _float_
: Final time of the process. The process is simulated on the interval $[0,T]$.

`steps` : _int_
: Number of time steps on $[0,T]$. Must be greater than 0.


## Methods

#### .simulate(n_simulations = 1, plot = False)
Simulate a $d$-dimensional Brownian motion of covariance matrix `variance` on the interval $[0,T]$.
**Parameters**

`n_simulations` : _int_
: Number of desired simulations.

`plot` : _bool_
: Specifies whether to plot the simulation.

**Returns**

_np.ndarray_
: Path of `n_simulations` Brownian motions on $[0,T]$. The final array is a three-dimensional array such that :
- 1st dimension is the number of simulations
- 2nd dimension is the number of time steps
- 3rd dimension is the dimension of the Brownian motion

For example, if we run 4 simulations of a 2D standard Brownian motion on the interval $[0,1]$ with 1000 time steps with the following code:
```python
W = Brownian(variance = np.eye(2), T = 1, steps = 1000)
W.simulate(n_simulations = 4, plot = False)
```
then we can access to the first simulation with:
```python
W.path[0]
```
or to the second coordinates of the third simulation with:
```python
W.path[2,:,1]
```
or to the value of each path at the 201th time step with:
```python
W.path[:,200,:]
```

> [!IMPORTANT]
> Simulation method for diffusion processes is implemented with more parameters. Please refer to the documentation of the [Diffusion Process](<project:/processes/diffusion/index.md>) class to learn more about the additional parameters.
> 

#### .plot()
Plot all the previous simulations of the effective process.

> [!NOTE]
> This method plots the simulation only if the process is 1D, 2D or 3D. The plot is made with `plotly` for better interactive figures.

**Parameters**

No parameters.

**Returns**

No return value.

#### .expectation(t = 0)
Returns the expectation of the effective process at time `t`. The expectation of a stochastic process $(X_t)_{t\geq 0}$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ is given by
\begin{equation}
\mathbb{E}[X_t] = \int_{\Omega} X_t d\mathbb{P} = \int_\mathbb{R} x d\mathbb{P}_{X_t}(x)
\end{equation}

> [!NOTE]
> For some processes, the expectation is not defined. In this case, PyStochastic raises an `NonImplementedError` error.

**Parameters**
`t` : _float_
: Time at which the expectation is computed. Must be in the interval $[0,T]$.

**Returns**
_float_ or _np.ndarray_
: Expectation of the effective process at time `t`.


#### .variance(t = 0)
Returns the variance of the effective process at time `t`. The expectation of a stochastic process $(X_t)_{t\geq 0}$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ is given by
\begin{equation}
\mathbb{V}[X_t] = \mathbb{E}\left[(X_t-\mathbb{E}[X_t])^2\right]
\end{equation}

> [!NOTE]
> For some processes, the variance is not defined. In this case, PyStochastic raises an `NonImplementedError` error.

**Parameters**
`t` : _float_
: Time at which the variance is computed. Must be in the interval $[0,T]$.

**Returns**
_float_ or _np.ndarray_
: Variance of the effective process at time `t`.


#### .covariance(t = 0, i = 0, j = 0)
For a multidimensional process, the `covariance` method returns the covariance of the $i$-th and $j$-th coordinates of the effective process at time `t`. The covariance of the $i$-th and $j$-th coordinates of a $d$-dimensional stochastic process $(X_t)_{t\geq 0}$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ is given by
\begin{equation}
\mathrm{Cov}(X_t^i, X_t^j) = \mathbb{E}\left[(X_t^i-\mathbb{E}[X_t^i])(X_t^j-\mathbb{E}[X_t^j])\right]
\end{equation}

> [!NOTE]
> For some processes, the covariance is not defined. In this case, PyStochastic raises an `NonImplementedError` error.
>
> For unidimensional processes, the `covariance` method calls the `variance` method.

**Parameters**
`t` : _float_
: Time at which the covariance is computed. Must be in the interval $[0,T]$.

`i` : _int_
: First coordinate index of the process. It must verify $0 \leq i < \mathrm{dim}_X$.

`j` : _int_
: Second coordinate index of the process. It must verify $0 \leq i < \mathrm{dim}_X$


**Returns**
_float_
: Covariance of the effective process $i$-th and $j$-th coordinates at time `t`.


#### .covariance_matrix(t = 0)
For a multidimensional process, the `covariance_matrix` method returns the covariance matrix the effective process at time `t`. The covariance matrix of a $d$-dimensional stochastic process $(X_t)_{t\geq 0}$ is given by
\begin{equation}
\Gamma(i,j)=\mathrm{Cov}(X_t^i,X_t^j)~~ \forall 0 \leq i,j < \mathrm{dim}_X
\end{equation}

> [!NOTE]
> For some processes, the covariance matrix is not defined. In this case, PyStochastic raises an `NonImplementedError` error.
>
> For unidimensional processes, the `covariance_matrix` method calls the `variance` method.


**Parameters**
`t` : _float_
: Time at which the covariance matrix is computed. Must be in the interval $[0,T]$.

**Returns**
_np.ndarray_
: Covariance matrix of the effective process at time `t`.


#### .final_position()
Returns the final position of the effective process.

**Parameters**

No parameters.

**Returns**

_np.ndarray_
: Final values of the effective process.


#### .max()
Returns the maximum value of the effective process.

> [!NOTE]
> This method is only implemented for unidimensional processes. For multidimensional processes, please refer to the `max_norm` method.

**Parameters**

No parameters.

**Returns**

_tuple_ : (np.ndarray, np.ndarray, np.ndarray)

- The first element is the array of maximum value for each simulation.
- The second element is the array of time indexes where the maximum value occurs for each simulation.
- The third element is the array of time values where the maximum value occurs for each simulation.

#### .min()
Returns the minimum value of the effective process.

> [!NOTE]
> This method is only implemented for unidimensional processes.

**Parameters**

No parameters.

**Returns**

_tuple_ : (np.ndarray, np.ndarray, np.ndarray)

- The first element is the array of minimum value for each simulation.
- The second element is the array of time indexes where the minimum value occurs for each simulation.
- The third element is the array of time values where the minimum value occurs for each simulation.

#### .max_norm(order = 2)
Returns the maximum norm value of the effective process. The available norms are p-norms for $1 \leq p < +\infty$.

**Parameters**

`order` : _int_
: Order of the norm. If `order = p`, then the norm is defined by : $||x||_p = \sqrt[p]{\sum_{i=1}^d x_i^p}$.

**Returns**

_tuple_ : (np.ndarray, np.ndarray, np.ndarray)

- The first element is the array of maximum norm for each simulation.
- The second element is the array of time indexes where the maximum norm occurs for each simulation.
- The third element is the array of time values where the maximum norm occurs for each simulation.

#### .hitting_time(value, inequality = ">")

Returns the first time when each simulation reaches the value `value`. The hitting time can be defined as
\begin{equation*}
\inf \left\{ t \mid X_t \geq value \right\} ~~~\text{ or }~~~ \inf \left\{ t \mid X_t \leq value \right\}
\end{equation*}
depending on the value of the `inequality` parameter.

> [!NOTE]
> This method is only implemented for unidimensional processes.

**Parameters**

`value` : _int_
: Target value.

`inequality` : _str_ in {"<", ">"}
: Inequality direction.

**Returns**

_tuple_ : (np.ndarray, np.ndarray)

- The first element is the array of time indexes where the process reaches the target value for each simulation.
- The second element is the array of time values where the process reaches the target value for each simulation.

> [!IMPORTANT]
> If the process is not reaching the target value at any time, the method returns `np.nan` for the corresponding time index and time value.

#### .hitting_norm_time(value, inequality = ">", order = 2)

Returns the first time when each simulation reaches the norm `value` with a specified $p$-norm. The hitting time can be defined as
\begin{equation*}
\inf \left\{ t \mid ||X_t||_p \geq value \right\} ~~~\text{ or }~~~ \inf \left\{ t \mid ||X_t||_p \leq value \right\}
\end{equation*}
depending on the value of the `inequality` parameter.

**Parameters**

`value` : _int_
: Target norm value.

`inequality` : _str_ in {"<", ">"}
: Inequality direction.

`order` : _int_
: Order of the norm. If `order = p`, then the norm is defined by : $||x||_p = \sqrt[p]{\sum_{i=1}^d x_i^p}$.
**Returns**

_tuple_ : (np.ndarray, np.ndarray)

- The first element is the array of time indexes where the process reaches the target norm for each simulation.
- The second element is the array of time values where the process reaches the target norm for each simulation.

> [!IMPORTANT]
> If the process is not reaching the target value at any time, the method returns `np.nan` for the corresponding time index and time value.

#### .quadratic_variation(t = None, mean = False, plot = False)

Returns an estimation of the quadratic variation process deduced from all simulations of the process. The quadratic variation of process $(X_t)_{t\geq 0}$ is defined as
\begin{equation*}
<X>_t = \lim_{n\to\+infty} \sum_{i=1}^n \left(X_{t_i^n}-X_{t_{i-1}^n}\right)^2
\end{equation*}
where (t_i^n)_{1\leq i \leq n} is a mesh of $[0,T]$.

> [!NOTE]
> This method is only implemented for unidimensional processes.

**Parameters**

`t` : _int_ or `None`
: Time value at which the quadratic variation is computed. If `None`, the quadratic variation is computed all over the interval $[0,T]$.

`mean` : _bool_
: Specifies whether to compute each quadratic variation separately or to compute the mean of all quadratic variations.

`plot` : _bool_
: Specifies whether to plot the quadratic variation.

**Returns**

_np.ndarray_ :
    Estimation of the quadratic variation of the process at time `t`.

## Implemented processes
```{toctree}
:maxdepth: 2
:caption: Elementary processes
elementary/brownian
elementary/brownian_bridge

:maxdepth: 2
:caption: Diffusion processes
diffusion/index
diffusion/geometric_brownian_motion
diffusion/ornstein_uhlenbeck
diffusion/vasicek
diffusion/cox_ingersoll_ross
diffusion/heston
diffusion/hull_white
diffusion/constant_elasticity_variance

:maxdepth: 2
:caption: Jump processes
jump/index
jump/poisson
jump/compound_poisson
```