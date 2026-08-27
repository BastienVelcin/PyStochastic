# Constant Elasticity Variance Model

## Import line
You can import the Bessel process class from the `processes` module as follows:
```python
from pystochastic.processes import CEV
```
## Description

```python
pystochastic.processes.CEV(speed=1, volatility=1, elasticity=1, initial=1, T = 1, steps=1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Constant Elasticity Variance model on the interval $[0, T]$, with a speed, volatility, and elasticity parameters.
A Constant Elasticity Variance (CEV) process $(S_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t.
\end{equation*}

The CEV process is an extension of the [Geometric Brownian Motion](<project:/processes/diffusion/geometric_brownian_motion.md>) ($\gamma = 1$), and of the
[Ornstein-Uhlenbeck](<project:/processes/diffusion/ornstein_uhlenbeck.md>) process ($\gamma = 0$).
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
The Bessel class inherits all attributes from the [Processes](<project:/processes/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Bessel class inherits all methods from the [Processes](<project:/processes/index.md>) class.
## Examples

```python
>>> BSL = Bessel(order = 5, T = 10, steps= 750)
>>> BSL.simulate(5, plot = False)
array([[[0.        ],
        [0.28245147],
        [0.39893963],
        ...,
        [5.6968423 ],
        [5.7301934 ],
        [5.9047602 ]],
       [[0.        ],
        [0.33095952],
        [0.62998883],
        ...,
        [8.81073229],
        [9.05938807],
        [9.02016293]],
       [[0.        ],
        [0.25659784],
        [0.53961491],
        ...,
        [3.36714455],
        [3.33645985],
        [3.42682631]],
       [[0.        ],
        [0.3264745 ],
        [0.4678982 ],
        ...,
        [3.24248475],
        [3.16006872],
        [3.10188941]],
       [[0.        ],
        [0.34346597],
        [0.36442699],
        ...,
        [7.31806438],
        [7.33390633],
        [7.41006382]]], shape=(5, 751, 1))
>>> BSL.max()
(array([[ 8.59509264],
       [10.70444174],
       [ 6.22984211],
       [ 5.51000105],
       [ 7.52481944]]), array([[590],
       [435],
       [480],
       [354],
       [745]]), array([[7.86666667],
       [5.8       ],
       [6.4       ],
       [4.72      ],
       [9.93333333]]))
```

## References

- BECKERS, S. (1980). The Constant Elasticity of Variance Model and Its Implications For Option Pricing. The Journal of Finance, 35(3), 661–673. https://doi.org/10.1111/j.1540-6261.1980.tb03490.x
- COX, J. (1975). Notes on Option Pricing I: Constant Elasticity of Diffusions. Unpublished draft, Stanford University.
- Emanuel, D. C., & MacBeth, J. D. (1982). Further Results on the Constant Elasticity of Variance Call Option Pricing Model. The Journal of Financial and Quantitative Analysis, 17(4), 533. https://doi.org/10.2307/2330906
