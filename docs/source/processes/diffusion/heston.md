# Heston Model

## Import line
You can import the Heston class from the `processes` module as follows:
```python
from pystochastic.processes import Heston
```
## Description

```python
pystochastic.processes.Heston(mu = 1, long_variance = 1, reverting_rate = 1, vol_volatility = 1, correlation = 0, initial_price = 1, initial_variance = 1, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Heston Model process on the interval $[0, T]$, with specified long-term variance $\theta$, reveting rate $\kappa$, volatility of variance $\xi$, and correlation coefficient $\rho$.
A Heston Model process $(S_t, \nu_t)_{t\geq0}$ is a solution of the following stochastic differential equation system:
\begin{equation*}
    \left\{ \begin{array}{l}
        dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_t^{(1)}\\
        d\nu_t = \kappa (\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW_t^{(2)} \\
    \end{array} \right.
\end{equation*}


where $\left(W_t^{(1)}\right)_{t\geq 0}$ and $\left(W_t^{(2)}\right)_{t\geq 0}$ are two unidimensional standard Brownian motions, with a correlation coefficient $\rho$.
It is generally used to model short interest rates in financial markets.
The sub-process $(S_t)_{t\geq0}$ represents the asset price, and the sub-process $(\sqrt{\nu_t})_{t\geq0}$ represents the volatility of the asset price.

> [!NOTE]
> The variance process $(\nu_t)_{t\geq0}$ follows a CIR process. The volatility is defined as the square root of the variance.

### Parameters

`mu` : _float_
: Drift coefficient $\mu$ of the asset price. It determines the average evolution of the asset price.

`long_variance` : _float_
: Long-term variance $\theta$ of the variance process.  It determines the average variance in the long term.

`reverting_rate` : _float_
: Reverting rate parameter $\kappa$ of the variance process. It determines the speed at which the process reverts to the mean $\mu$. Must be strictly positive.

`variance_volatility` : _float_
: Volatility parameter $\sigma$ of the variance process. It determines the amplitude of the random fluctuations in the process. Must be strictly positive.

`correlation` : _float_
: Correlation parameter $\rho$ of the two Brownian Motions. Must be in the interval $(-1,1)$.

`initial_price` : _float_
: Initial value of the asset price process at time $t=0$.

`initial_variance` : _float_
: Initial value of the variance process at time $t=0$.

`T` : _float_
: Final time of the Heston model simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Heston model is simulated. Must be strictly greater than 0.

### Attributes
The Heston class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

`.path` : _np.ndarray_
: Path of the simulated volatility process.

`.price` : _np.ndarray_
: Path of the simulated asset price process.

`.var` : _np.ndarray_
: Path of the simulated variance process.

`.couple` : (_np.ndarray_ , _np.ndarray_)
: Couple of paths of the simulated asset price and volatility processes. 

`.feller_condition` : _bool_
: Returns `True` if the Feller condition is satisfied. The Feller is satisfied when $2\kappa \theta > \xi^2$


It also possesses the attributes deduced from its parameters.

### Methods

The Heston class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> H = Heston(mu = 1, long_variance = 0.9,  reverting_rate = 1, variance_volatility = 0.5, correlation = 0.3, initial_price = 10, initial_variance = 0.3, T = 5, steps = 500)
path = H.simulate(100)
>>> np.mean(H.max())
np.float64(112.02673067086337)
>>> H.plot()
```

## References

- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. Review of Financial Studies, 6(2), 327–343. https://doi.org/10.1093/rfs/6.2.327