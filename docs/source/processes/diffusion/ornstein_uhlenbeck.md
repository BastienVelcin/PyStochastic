# Ornstein-Uhlenbeck process

## Import line
You can import the Ornstein-Uhlenbeck process class from the `processes` module as follows:
```python
from pystochastic.processes import OrnsteinUhlenbeck
```
## Description

```python
pystochastic.processes.OrnsteinUhlenbeck(speed = 1, volatility = 1, initial = 0, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ✅

Creates an instance of an Ornstein-Uhlenbeck process on the interval $[0, T]$, with specified speed $\theta$ and volatility $\sigma$.
An Ornstein-Uhlenbeck process $(X_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dX_t = -\theta X_t dt + \sigma dW_t.
\end{equation*}

> [!NOTE]
> The Ornstein-Uhlenbeck process is a special case of the [Constant Elasticity of Variance (CEV)](<project:/processes/diffusion/constant_elasticity_of_variance.md>) process, with elasticity parameter $\gamma = 0$.

It is generally used to model interest rates in financial markets.

### Parameters

`speed` : _float_ or _array_like_
: Reverting speed parameter $\theta$ of the Ornstein-Uhlenbeck process. It determines the speed at which the process reverts to zero. Must be strictly positive.

`volatility` : _float_ or _array_like_
: Volatility parameter $\sigma$ of the Ornstein-Uhlenbeck process. It determines the amplitude of the random fluctuations in the process. Must be strictly positive.

`initial` : _float_ or _array_like_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the Ornstein-Uhlenbeck process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Geometric Brownian Motion is simulated. Must be strictly greater than 0.

### Attributes
The Ornstein-Uhlenbeck class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Ornstein-Uhlenbeck Motion class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> X = OrnsteinUhlenbeck(speed = np.eye(2), volatility = np.array([[2,5],[3,4]]), initial = [2,6])
>>> X.simulate(1)
array([[[2.        , 6.        ],
        [1.95918102, 6.02677194],
        [2.12793376, 6.19767134],
        ...,
        [2.04272298, 3.069482  ],
        [2.24033825, 3.24786612],
        [2.0551122 , 3.08653224]]], shape=(1, 1001, 2))
>>> X.max_norm()
(array([[2.38713004, 6.51915862]]), array([[42.87975187]]), array([40]), array([0.04]))
>>> X.variance(1)
array([12.53681424, 10.80759848])

```

## References

- Øksendal, B. (1992). Stochastic Differential Equations. In Universitext. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-02847-6
- Brigo, D., Dalessandro, A., Neugebauer, M., & Triki, F. (2008). A Stochastic Processes Toolkit for Risk Management. Elsevier BV. https://doi.org/10.2139/ssrn.1109160
