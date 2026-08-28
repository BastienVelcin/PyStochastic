# Constant Elasticity of Variance Model

## Import line
You can import the Bessel process class from the `processes` module as follows:
```python
from pystochastic.processes import CEV
```
## Description

```python
pystochastic.processes.CEV(speed = 1, volatility = 1, elasticity = 1, initial = 1, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Constant Elasticity of Variance model on the interval $[0, T]$, with specified speed, volatility, and elasticity parameters.
A Constant Elasticity of Variance (CEV) process $(S_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t.
\end{equation*}

> [!NOTE]
>The CEV process is an extension of the [Geometric Brownian Motion](<project:/processes/diffusion/geometric_brownian_motion.md>) ($\gamma = 1$), and of the
>[Ornstein-Uhlenbeck](<project:/processes/diffusion/ornstein_uhlenbeck.md>) process ($\gamma = 0$).

It is generally used to model the price of a stock or a commodity.
### Parameters

`speed` : _float_
: Drift parameter $\mu$ of the CEV process. It determines the tendency of the process to increase or decrease.

`volatility` : _float_
: Volatility parameter $\sigma$ of the CEV process. It determines the amplitude of the random fluctuations in the process. Must be positive.

`elasticity` : _float_
: Elasticity parameter $\gamma$ of the CEV process. It controls the relation between the price and the volatility. Must be positive.
: - If `elasticity < 1`, the volatility increases when the price decreases.
: - If `elasticity = 1`, the model is exactly the Black-Sholes model.
: - If `elasticity > 1`, the volatility increases when the price increases.
: - If `elasticity = 0`, the model is exactly the Bachelier model.

`initial` : _float_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the CEV process simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the CEV process is simulated. Must be strictly greater than 0.



### Attributes
The CEV class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The CEV class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> S = CEV(speed=1, volatility=0.3, elasticity=0.7, initial=3, T = 1, steps=1000)
>>> path = S.simulate(10)
>>> path.shape
(10, 1001, 1)
>>> S.expectation(1)
np.float64(8.154845485377136)
```

## References

- BECKERS, S. (1980). The Constant Elasticity of Variance Model and Its Implications For Option Pricing. The Journal of Finance, 35(3), 661–673. https://doi.org/10.1111/j.1540-6261.1980.tb03490.x
- COX, J. (1975). Notes on Option Pricing I: Constant Elasticity of Diffusions. Unpublished draft, Stanford University.
- Emanuel, D. C., & MacBeth, J. D. (1982). Further Results on the Constant Elasticity of Variance Call Option Pricing Model. The Journal of Financial and Quantitative Analysis, 17(4), 533. https://doi.org/10.2307/2330906
