# Hull-White Model

## Import line
You can import the Hull-White class from the `processes` module as follows:
```python
from pystochastic.processes import HullWhite
```
## Description

```python
pystochastic.processes.HullWhite(reverting_speed = 1, calibration = lambda t : t, volatility = lambda t : 1, initial = 0, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Hull-White Model process on the interval $[0, T]$, with specified mean $\mu$, calibration function $\theta(t)$, and volatility function $\sigma(t)$.
A Cox-Ingersoll-Ross Model process $(X_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dX_t = (\theta(t)- \mu X_t)dt + \sigma(t)dW_t
\end{equation*}

It is generally used to model short interest rates in financial markets.

### Parameters

`reversion_speed` : _float_ 
: Reversion speed parameter $\mu$ of the Hull-White model. It determines the speed at which the process reverts to its equilibrium position. Must be strictly positive.

`calibration` : _float_ or _function_
: Calibration parameter $\theta$ of the Hull-White model. It allows fitting the process to specific data.

`volatility` : _float_ or _function_
: Volatility parameter $\sigma$ of the Hull-White model. It determines the amplitude of the random fluctuations in the process.

`initial` : _float_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the Hull-White process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Geometric Brownian Motion is simulated. Must be strictly greater than 0.

### Attributes
The Hull-White class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Hull-White class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> H = HullWhite(reversion_speed = 1.5, calibration = lambda t : t**2/2, volatility = lambda t : np.exp(-2*t), T = 10, steps = 1000)
>>> H.simulate(100)
array([[[ 0.        ],
        [ 0.03256804],
        [ 0.05480962],
        ...,
        [ 0.139716  ],
        [ 0.13913871],
        [ 0.13609225]],
       [[ 0.        ],
        [-0.05249945],
        [-0.03553723],
        ...,
        [ 0.12243423],
        [ 0.13072681],
        [ 0.13895758]],
       [[ 0.        ],
        [-0.01806683],
        [-0.03425346],
        ...,
        [ 0.06473323],
        [ 0.05954243],
        [ 0.06165395]],
       ...,
       [[ 0.        ],
        [ 0.01344162],
        [ 0.00336492],
        ...,
        [ 0.01234835],
        [ 0.00834775],
        [ 0.00814536]],
       [[ 0.        ],
        [ 0.01702888],
        [ 0.078319  ],
        ...,
        [ 0.10238168],
        [ 0.10244703],
        [ 0.11004844]],
       [[ 0.        ],
        [ 0.0194431 ],
        [-0.00566558],
        ...,
        [ 0.09231365],
        [ 0.09305966],
        [ 0.09163218]]], shape=(100, 1001, 1))
>>> H.quadratic_variation(plot = True)
array([[0.00000000e+00, 1.06067747e-03, 1.55536508e-03, ...,
        2.66073338e-01, 2.66073672e-01, 2.66082952e-01],
       [0.00000000e+00, 2.75619268e-03, 3.04390959e-03, ...,
        2.15188208e-01, 2.15256975e-01, 2.15324720e-01],
       [0.00000000e+00, 3.26410398e-04, 5.88417476e-04, ...,
        2.32747136e-01, 2.32774080e-01, 2.32778539e-01],
       ...,
       [0.00000000e+00, 1.80677226e-04, 2.82217105e-04, ...,
        2.40997539e-01, 2.41013544e-01, 2.41013585e-01],
       [0.00000000e+00, 2.89982636e-04, 4.04646226e-03, ...,
        2.66579082e-01, 2.66579086e-01, 2.66636867e-01],
       [0.00000000e+00, 3.78034330e-04, 1.00848037e-03, ...,
        2.45269666e-01, 2.45270222e-01, 2.45272260e-01]],
      shape=(100, 1001))
```

## References

- Interest Rate Models — Theory and Practice. (2006). In Springer Finance. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-34604-3
- Musiela, M., & Rutkowski, M. (1997). Martingale Methods in Financial Modelling. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-22132-7
