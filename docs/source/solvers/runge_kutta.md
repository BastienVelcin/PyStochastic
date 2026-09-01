# Runge-Kutta Solver

## Import line
You can import the Runge-Kutta class from the `sde` module as follows:
```python
from pystochastic.sde import RungeKutta 
```

## Description

```python
pystochastic.sde.RungeKutta(drift = lambda x,t : 2*x, diffusion = lambda x,t : 1, initial = 0.2, T = 1, steps = 1000)
```

**Type :** Class

**Multidimensional support :** ❌

The Runge-Kutta scheme is a way to solve an autonomous and unidimensional stochastic differential equation of the following form:

\begin{equation*}
dX_t = \mu(X_t) dt + \sigma(X_t) dW_t
\end{equation*}

where:
- $(X_t)_{t\geq 0}$ is a unidimensional stochastic process,
- $(W_t)_{t\geq 0}$ is a standard unidimensional Brownian motion,
- $\mu : \mathbb{R} \to \mathbb{R}$ is a continuous mapping, called **drift**,
- $\sigma : \mathbb{R} \to \mathbb{R}$ is a differentiable mapping, called **diffusion**.

> [!IMPORTANT]
> The drift and diffusion functions must be batch-compatible with NumPy.

To proceed, the Runge-Kutta scheme introduces a partition of the time interval `[0,T]` into `steps+1` time steps $0=t_0 < t_1 < \dots < t_{\text{steps}} = T$.
By letting $\Delta t = \frac{T}{\text{steps}}$ and $\Delta W_{t_i} = W_{t_{i+1}} - W_{t_i} $, the following recursively-built stochastic process $(\tilde X_{t_i})_{i\in \{0, \cdots, \text{steps}\}}$ defined by

\begin{equation*}
    \left\{ \begin{array}{l}
        \tilde{X}_0 = X_0 \\
        \tilde{X}_{t_{i+1}} = \tilde{X}_{t_i} + \mu(\tilde{X}_{t_i}) \Delta t + \sigma(\tilde{X}_{t_i}) \Delta W_{t_i} + \frac{1}{2\sqrt{\Delta t}} \left(\sigma\left(\tilde{X}_{t_i} + \mu(\tilde{X}_{t_i})\Delta t + \sigma(\tilde{X}_{t_i})\sqrt{\Delta t}\right) - \sigma(\tilde{X}_{t_i})\right) \left((\Delta W_{t_i})^2-\Delta t\right)
    \end{array} \right.
\end{equation*}

is an approximation of the solution of the considered SDE.

> [!NOTE]
> The Runge-Kutta method is used to numerically simulate unidimensional autonomous diffusion processes implemented by the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

### Parameters

`drift` : _function_
: Drift function $\mu$ of the model. Must be batch-compatible with NumPy.

`diffusion` : _function_
: Diffusion function $\sigma$ of the model. Must be batch-compatible with NumPy.

`initial` : _float_
: Initial value of the process. 

`T` : _float_
: Final time of the SDE simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the SDE is simulated. Must be strictly greater than 0.

> [!WARNING]
> The drift and diffusion functions must take one argument as input: the position point `x`. So, if you want to use the same drift and diffusion functions
> as an Euler-Maruyama scheme, you need to remove the `t` argument, or set an arbitrary default value.


### Methods

#### .solve()
```python
.solve(n_simulations = 1, plot=True, brownian_increments = None, diffusion_derivative = None)
```

The `solve` method returns an approximation of the solution of the considered SDE on the interval $[0,T]$. 

**Parameters**

`n_simulations` : _int_
: Number of desired simulations. Must be strictly positive.

`plot` : _bool_
: Specifies whether to plot the simulation.

`brownian_increments` : _array_like_ or `None`
: Brownian increments $(\Delta W_{t_i})_{i\in \{0, \cdots, \text{steps}-1\}}$to use for the simulation. If None, the Brownian increments are automatically computed.

**Returns**

_np.ndarray_
: Path of `n_simulations` solutions of the SDE on $[0,T]$. The final array is a three-dimensional array such that :

- 1st dimension is the number of simulations
- 2nd dimension is the number of time steps
- 3rd dimension is the dimension of the effective process 

> [!NOTE]
> The NumPy vectorization always works for the unidimensional Runge-Kutta solver.

## Examples

### Simulation of an autonomous unidimensional SDE

This example provides a step-by-step explanation on how to simulate 100 solutions of the following unidimensional SDE:

\begin{equation*}
dX_t = \min(X_t, 1) dt - \exp(-|X_t|) dW_t
\end{equation*}

on the interval $[0,1]$, with the initial value $X_0 = 0.7$.
```python
import numpy as np
from pystochastic.sde import RungeKutta

solver = RungeKutta(drift = lambda x : np.minimum(x,1), diffusion = lambda x : -np.exp(-np.abs(x)), initial = 0.7, T = 1, steps = 1000)
solutions = solver.solve(n_simulations = 100, plot = True, diffusion_derivative = lambda x : -1)

print(solutions)
```
## References

- Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential Equations. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-12616-5
- Roberts, A. J. (2012). Modify the Improved Euler scheme to integrate stochastic differential equations (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1210.0933

