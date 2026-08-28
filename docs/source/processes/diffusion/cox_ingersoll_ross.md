# Cox-Ingersoll-Ross Model

## Import line
You can import the Cox-Ingersoll-Ross class from the `processes` module as follows:
```python
from pystochastic.processes import CIR
```
## Description

```python
pystochastic.processes.CIR(mean = 0, speed = 1, volatility = 1, initial = 0, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Cox-Ingersoll-Ross Model process on the interval $[0, T]$, with specified mean $\mu$, speed $\theta$, and volatility $\sigma$.
A Cox-Ingersoll-Ross Model process $(X_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dX_t = \theta(\mu - X_t) dt + \sigma \sqrt{X_t}dW_t.
\end{equation*}

It is generally used to model short interest rates in financial markets.

> [!NOTE]
> The Cox-Ingersoll-Ross Model addresses one of the drawbacks of the Vasicek Model: the allowance negative interest rates.

### Parameters

`mean` : _float_ or _array_like_
: Long term mean parameter $\mu$ of the Cox-Ingersoll-Ross model. It determines the mean quantity of the trajectory in the long term.

`speed` : _float_ or _array_like_
: Reverting speed parameter $\theta$ of the Cox-Ingersoll-Ross model. It determines the speed at which the process reverts to the mean $\mu$. Must be strictly positive.

`volatility` : _float_ or _array_like_
: Volatility parameter $\sigma$ of the Cox-Ingersoll-Ross model. It determines the amplitude of the random fluctuations in the process. Must be strictly positive.

`initial` : _float_ or _array_like_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the Cox-Ingersoll-Ross process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Geometric Brownian Motion is simulated. Must be strictly greater than 0.

### Attributes
The Cox-Ingersoll-Ross class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

`.feller_condition` : _bool_
: Returns `True` if the Feller condition is satisfied. The Feller is satisfied when $2\theta \mu > \sigma^2$.

It also possesses the attributes deduced from its parameters.

### Methods

The Cox-Ingersoll-Ross class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> X = CIR(mean = 2, speed = 1, volatility = 1, initial = 10, T = 5, steps = 500)
>>> X.simulate(3)
array([[[10.        ],
        [ 9.53345982],
        [ 9.51792188],
        ...,
        [ 1.0483384 ],
        [ 0.95885013],
        [ 0.89860721]],
       [[10.        ],
        [10.26313479],
        [10.03713495],
        ...,
        [ 1.42776155],
        [ 1.5599068 ],
        [ 1.40419922]],
       [[10.        ],
        [10.21398201],
        [10.08133925],
        ...,
        [ 1.93135884],
        [ 1.67464881],
        [ 1.59798213]]], shape=(3, 501, 1))
>>> X.max()
(array([[10.        ],
       [10.26313479],
       [11.01600017]]), array([[0],
       [1],
       [9]]), array([[0.  ],
       [0.01],
       [0.09]]))
```

## References

- Interest Rate Models — Theory and Practice. (2006). In Springer Finance. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-34604-3
- Musiela, M., & Rutkowski, M. (1997). Martingale Methods in Financial Modelling. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-22132-7
