# Vasicek Model

## Import line
You can import the Vasicek class from the `processes` module as follows:
```python
from pystochastic.processes import Vasicek
```
## Description

```python
pystochastic.processes.Vasicek(mean = 0, speed = 1, volatility = 1, initial = 0, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ✅

Creates an instance of a Vasicek Model process on the interval $[0, T]$, with specified mean $\mu$, speed $\theta$, and volatility $\sigma$.
A Vasicek Model process $(X_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dX_t = \theta(\mu - X_t) dt + \sigma dW_t.
\end{equation*}

It is generally used to model short interest rates in financial markets.

### Parameters

`mean` : _float_ or _array_like_
: Long term mean parameter $\mu$ of the Vasicek model. It determines the mean quantity of the trajectory on the long term.

`speed` : _float_ or _array_like_
: Reverting speed parameter $\theta$ of the Vasicek model. It determines the speed at which the process reverts to zero. Must be strictly positive.

`volatility` : _float_ or _array_like_
: Volatility parameter $\sigma$ of the Vasicek model. It determines the amplitude of the random fluctuations in the process. Must be strictly positive.

`initial` : _float_ or _array_like_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the Vasicek model simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Vasicek model is simulated. Must be strictly greater than 0.

### Attributes
The Vasicek class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Vasicek class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> X = Vasicek(mean = 10, speed = 3, volatility = 0.5, initial = 40, T = 5, steps = 10000)
>>> X.simulate(100)
array([[[40.        ],
        [39.94448305],
        [39.89952887],
        ...,
        [10.431671  ],
        [10.43989243],
        [10.40889791]],
       [[40.        ],
        [39.95725489],
        [39.92717797],
        ...,
        [10.13915863],
        [10.14102572],
        [10.15492723]],
       [[40.        ],
        [39.92376841],
        [39.88124779],
        ...,
        [10.0413545 ],
        [10.0566543 ],
        [10.05021694]],
       ...,
       [[40.        ],
        [39.94484921],
        [39.90224448],
        ...,
        [10.26217644],
        [10.24610028],
        [10.24144753]]], shape=(100, 10001, 1))
>>> X.plot()

```

## References

- Interest Rate Models — Theory and Practice. (2006). In Springer Finance. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-34604-3
- Musiela, M., & Rutkowski, M. (1997). Martingale Methods in Financial Modelling. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-22132-7
