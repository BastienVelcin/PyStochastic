# Geometric Brownian Motion

## Import line
You can import the Geometric Brownian Motion process class from the `processes` module as follows:
```python
from pystochastic.processes import GeometricBrownianMotion
```
## Description

```python
pystochastic.processes.GeometricBrownianMotion(mu = 1, volatility = 1, initial = 1, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ✅

Creates an instance of a Geometric Brownian Motion on the interval $[0, T]$, with specified drift and volatility parameters.
A Geometric Brownian Motion (GBM) process $(S_t)_{t\geq0}$ is a solution of the following stochastic differential equation:

\begin{equation*}
dS_t = \mu S_t dt + \sigma S_t dW_t.
\end{equation*}

> [!NOTE]
> The Geometric Brownian Motion process is a special case of the [Constant Elasticity of Variance (CEV)](<project:/processes/diffusion/constant_elasticity_of_variance.md>) process, with elasticity parameter $\gamma = 1$.
### Parameters

`mu` : _float_ or _array_like_
: Drift parameter $\mu$ of the Geometric Brownian Motion. It determines the average evolution of the process.

`volatility` : _float_ or _array_like_
: Volatility parameter $\sigma$ of the Geometric Brownian Motion. It determines the amplitude of the random fluctuations in the process. Must be strictly positive.

`initial` : _float_ or _array_like_
: Initial value of the process at time $t=0$.

`T` : _float_
: Final time of the Geometric Brownian Motion simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Geometric Brownian Motion is simulated. Must be strictly greater than 0.

### Attributes
The Geometric Brownian Motion class inherits all attributes from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods

The Geometric Brownian Motion class inherits all methods from the [Diffusion Process](<project:/processes/diffusion/index.md>) class.

## Examples

```python
>>> S = GeometricBrownianMotion(mu = [2,5,1], volatility = np.ones((3,3)), initial = [1,8,5])
>>> S.simulate(10, plot = True)
array([[[ 1.00000000e+00,  8.00000000e+00,  5.00000000e+00],
        [ 5.54292960e-01,  7.60452146e+00,  5.45918510e+00],
        [ 1.55933409e+00,  8.22455418e+00,  6.16272954e+00],
        ...,
        [-1.43817300e+02,  3.51069471e+02,  4.76665002e+01],
        [-1.47541984e+02,  3.63045495e+02,  5.44180505e+01],
        [-1.44450669e+02,  3.52734644e+02,  6.75789810e+01]],
       [[ 1.00000000e+00,  8.00000000e+00,  5.00000000e+00],
        [ 1.04926875e+00,  8.48988034e+00,  5.04409268e+00],
        [ 8.86492180e-01,  7.96581220e+00,  4.71274239e+00],
        ...,
        [-1.13719853e+03,  2.56609657e+03,  5.50404630e+02],
        [-1.22215128e+03,  2.64813874e+03,  5.93996358e+02],
        [-1.07323726e+03,  2.65064292e+03,  6.99445651e+02]],
       [[ 1.00000000e+00,  8.00000000e+00,  5.00000000e+00],
        [ 9.77953399e-01,  8.48799841e+00,  5.09515895e+00],
        [ 1.12571261e+00,  8.26065425e+00,  5.92669385e+00],
        ...,
        [-1.95968252e+02,  1.37071889e+02,  3.49224328e+02],
        [-2.08459053e+02,  1.37937903e+02,  3.47531744e+02],
        [-2.21003262e+02,  1.34958778e+02,  3.39453298e+02]]],
      shape=(3, 1001, 3))
>>> S.max_norm()
(array([[ -92.58849453, 1189.37327022,  -39.7365541 ],
       [ -13.5062623 , -268.46189496,   47.84349125],
       [  -7.24957361,  -42.93621163,    9.22397146]]), array([[1.41463563e+06, 1.23878531e+06, 1.23885960e+06],
       [1.11794639e+04, 7.21083599e+04, 6.21647669e+04],
       [4.09178818e+02, 1.55845518e+03, 1.84622877e+03]]), array([892, 976, 983]), array([0.892, 0.976, 0.983]))
```

## References

- Øksendal, B. (1992). Stochastic Differential Equations. In Universitext. Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-02847-6
- Ross, S. M. (2014). Introduction to Probability Models. Elsevier Science & Technology Books. http://openlibrary.org/books/OL35774950M/Introduction_to_Probability_Models