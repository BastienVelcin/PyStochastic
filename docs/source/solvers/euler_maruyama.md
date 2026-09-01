# Euler-Maruyama Solver

## Import line
You can import the Euler-Maruyama class from the `sde` module as follows:
```python
from pystochastic.sde import EulerMaruyama
```

## Description

```python
pystochastic.sde.EulerMaruyama(drift = lambda x,t : 2*x, diffusion = lambda x,t : 1, initial = 0.2, T = 1, steps = 1000)
```

**Type :** Class

**Multidimensional support :** ✅

The Euler-Maruyama scheme is a way to solve a stochastic differential equation of the following form:

\begin{equation*}
dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
\end{equation*}

where:
- $(X_t)_{t\geq 0}$ is a $n$-dimensional stochastic process,
- $(W_t)_{t\geq 0}$ is a standard $d$-dimensional Brownian motion,
- $\mu : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^n$ is a continuous mapping, called **drift**,
- $\sigma : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^{n\times d}$ is a mapping, called **diffusion**.

> [!IMPORTANT]
> The drift and diffusion functions must be batch-compatible with NumPy.

To proceed, the Euler-Maruyama scheme introduces a partition of the time interval `[0,T]` into `steps+1` time steps $0=t_0 < t_1 < \dots < t_{\text{steps}} = T$.
By letting $\Delta t = \frac{T}{\text{steps}}$ and $\Delta W_{t_i} = W_{t_{i+1}} - W_{t_i} $, the following recursively-built stochastic process $(\tilde X_{t_i})_{i\in \{0, \cdots, \text{steps}\}}$ defined by

\begin{equation*}
    \left\{ \begin{array}{l}
        \tilde{X}_0 = X_0 \\
        \tilde{X}_{t_{i+1}} = \tilde{X}_{t_i} + \mu(\tilde{X}_{t_i}, t_i) \Delta t + \sigma(\tilde{X}_{t_i}, t_i) \Delta W_{t_i} \\
    \end{array} \right.
\end{equation*}

is an approximation of the solution of the considered SDE.

> [!NOTE]
> The Euler-Maruyama method is used to numerically simulate diffusion processes implemented by the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

### Parameters

`drift` : _function_
: Drift function $\mu$ of the model. Must be batch-compatible with NumPy.

`diffusion` : _function_
: Diffusion function $\sigma$ of the model. Must be batch-compatible with NumPy.

`initial` : _float_ or _array_like_
: Initial value of the process. The dimension of the initial value must be equal to the dimension of the output of the drift and diffusion functions.

`T` : _float_
: Final time of the SDE simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the SDE is simulated. Must be strictly greater than 0.

### Methods

#### .solve()
```python
.solve(n_simulations = 1, plot=True, parallel=False, n_workers=None, brownian_increments = None)
```

The `solve` method returns an approximation of the solution of the considered SDE on the interval $[0,T]$. 

**Parameters**

`n_simulations` : _int_
: Number of desired simulations. Must be strictly positive.

`plot` : _bool_
: Specifies whether to plot the simulation.

`parallel` : _bool_
: For the non-vectorizable case, specifies whether to parallelize the simulation.

`n_workers` : _int_ or `None`
: If parallel is set to True, specifies the number of workers to use.

`brownian_increments` : _array_like_ or `None`
: Brownian increments $(\Delta W_{t_i})_{i\in \{0, \cdots, \text{steps}-1\}}$to use for the simulation. If None, the Brownian increments are automatically computed.

**Returns**

_np.ndarray_
: Path of `n_simulations` solutions of the SDE on $[0,T]$. The final array is a three-dimensional array such that :

- 1st dimension is the number of simulations
- 2nd dimension is the number of time steps
- 3rd dimension is the dimension of the effective process 

> [!IMPORTANT]
> If the SDE is unidimensional or the drift and diffusion functions are diagonal, the algorithm uses NumPy vectorization to speed up the simulation.
> NumPy vectorization is not available for multidimensional SDEs where the drift and diffusion functions are not diagonal.
> In this case, you can use the `parallel` parameter to parallelize the simulation.
>
> Notice that the parallel simulation is only useful on a huge number of simulations.

## Examples

### Simulation of a unidimensional SDE

This example provides a step-by-step explanation on how to simulate 100 solutions of the following unidimensional SDE:

\begin{equation*}
dX_t = 2X_t dt + \sqrt{t} dW_t
\end{equation*}

on the interval $[0,1]$, with the initial value $X_0 = 0.2$.
```python
import numpy as np
from pystochastic.sde import EulerMaruyama

solver = EulerMaruyama(drift = lambda x,t : 2*x, diffusion = lambda x,t : np.sqrt(t), initial = 0.2, T = 1, steps = 1000)
solutions = solver.solve(n_simulations = 100, plot = True)

print(solutions)
```

Since we consider a unidimensional SDE, the NumPy vectorization is automatically used to speed up the simulation.

### Simulation of a multidimensional SDE

This example provides a step-by-step explanation on how to simulate 100 solutions of the following 2-dimensional SDE:

\begin{equation*}
dX_t =  \begin{pmatrix}
    1 & 0.2 \\
    0.2 & 2
\end{pmatrix} X_t dt + \begin{pmatrix}
    1 & 0 \\
    0 & 1
\end{pmatrix} dW_t
\end{equation*}

on the interval $[0,1]$, with the initial value $X_0 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.

```python
import numpy as np
from pystochastic.sde import EulerMaruyama

def drift_fct(x,t):
    return np.array([[1,0.2],[0.2,2]]) @ x

def diffusion_fct(x,t):
    return np.eye(2)

solver = EulerMaruyama(drift = drift_fct, diffusion = diffusion_fct, initial = [1,1], T = 1, steps = 1000)
solutions = solver.solve(n_simulations = 100, plot = True)

print(solutions)
```

In this case, since the drift function is not diagonal, a sequential simulation is done.

### Simulation of a system of unidimensional SDEs with correlated Brownian increments

This example provides a step-by-step explanation on how to simulate 100 solutions of the following 2-dimensional SDE:

\begin{equation*}
    \left\{ \begin{array}{l}
        dS_t = 2 S_t dt + \sqrt{\nu_t} S_t dW_t^{(1)}\\
        d\nu_t = 0.7 (1 - \nu_t) dt + 0.3 \sqrt{\nu_t} dW_t^{(2)} \\
    \end{array} \right.
\end{equation*}

on the interval $[0,1]$, with the initial value $\begin{pmatrix} S_0 \\ \nu_0 \end{pmatrix} = \begin{pmatrix} 10 \\ 0.3 \end{pmatrix}$,  where $\left(W_t^{(1)}\right)_{t\geq 0}$ and $\left(W_t^{(2)}\right)_{t\geq 0}$ are two unidimensional standard Brownian motions, with a correlation coefficient $\rho$.

First, we can remark that the previous system can be written as a 2-dimensional SDE of the following form:
\begin{equation*}
dX_t = \begin{pmatrix}
     2X_t^{(1)}\\
     0.7(1-X_t^{(2)})
\end{pmatrix} dt + \begin{pmatrix}
     X_t^{(1)}\sqrt{X_t^{(2)}} & 0 \\
     0 & 0.3 \sqrt{X_t^{(2)}} 
\end{pmatrix} d \begin{pmatrix}
     W_t^{(1)} \\ W_t^{(2)}
\end{pmatrix}
\end{equation*}

where $X_t = \begin{pmatrix} X_t^{(1)}  \\  X_t^{(2)}  \end{pmatrix} = \begin{pmatrix} S_t \\ \nu_t \end{pmatrix}$, and  $W_t$ is a two-dimensional Brownian motion of covariance matrix $Q = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$.

So, we can run the following code to simulate 100 solutions of this system:
```python
import numpy as np
from pystochastic.sde import EulerMaruyama
from pystochastic.processes import Brownian

def drift_fct(x,t):
    return np.array([2*x[0], 0.7*(1-x[1])])

def diffusion_fct(x,t):
    
    #Since we take the square root of the second variable and Euler-Maruyama can provide negative values, we need to truncate the values to positive values only.
    x_1_max = np.maximum(x[1],0)
    return np.array([[x[0]*np.sqrt(x_1_max), 0], [0, 0.3*np.sqrt(x_1_max)]])

cov_matrix = np.array([[1,0.7],[0.7,1]])

brownian = Brownian(cov = cov_matrix, T = 1, steps = 1000)
brownian.simulate(n_simulations = 100)
dW = brownian.increments

solver = EulerMaruyama(drift = drift_fct, diffusion = diffusion_fct, initial = [1,1], T = 1, steps = 1000)
solutions = solver.solve(n_simulations = 100, plot = False, parallel = False, brownian_increments = dW)

# We print the price, the variance and the volatility of the simulated stocks, and we reshape it to fit with the PyStochastic format.
print("Price:", solutions[:,:,0][:,:,None])
print("Variance:", solutions[:,:,1][:,:,None])
print("Volatility:", np.sqrt(solutions[:,:,1])[:,:,None])
```