# Brownian motion

## Import line
You can import the Brownian motion class from the `processes` module as follows:
```python
from pystochastic.processes import Brownian
```
## Description

```python
pystochastic.processes.Brownian(variance = 1, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ✅

Creates an instance of a Brownian motion process on the interval $[0, T]$, with a specified covariance matrix.
Let $Q$ be a symmetric positive definite matrix. A $d$-dimensional $Q$-Brownian motion $(W_t)_{t\geq 0}$ is an $\mathbb{R}^d$-valued Gaussian process such that
- $W_0 = 0 ~~a.s.$ 
- $\forall s,t \in \mathbb{R}_+, ~~ \mathbb{E}[W_t] = 0 ~~$ and $~~ \mathbb{E}[W_s W_t^\star] = (s\wedge t)Q$
- $(W_t)_{t\geq 0}$ is continuous process

>[!NOTE]
> A $d$-dimensional Brownian motion is said standard if $Q = \mathrm{Id}_d$.

The Brownian motion is one of the fundamental objects of stochastic calculus, since it mathematically formalizes the concept of random motion.
It appears naturally in stochastic differential equations and so, within the diffusion process definitions.

When
### Parameters

`variance` : _float_ or _np.ndarray_
: Variance-covariance matrix of the Brownian motion. In the previous example, it is represented by $Q$.

`T` : _float_
: Final time of the Brownian motion simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Brownian motion is simulated. Must be strictly greater than 0.

### Attributes
The Brownian class inherits all attributes from the [Processes](<project:/processes/index.md>) class.
### Methods

#### .simulate(n_simulations = 1, plot = False)
Simulate a $d$-dimensional Brownian motion of covariance matrix `variance` on the interval $[0,T]$.
**Parameters**

`n_simulations` : _int_
: Number of desired simulations.

`plot` : _bool_
: Specifies whether to plot the simulation.

**Returns**

_np.ndarray_
: Path of `n_simulations` Brownian motions on $[0,T]$. The final array is a three-dimensional array such that :
- 1st dimension is the number of simulations
- 2nd dimension is the number of time steps
- 3rd dimension is the dimension of the Brownian motion

For example, if we run 4 simulations of a 2D standard Brownian motion on the interval $[0,1]$ with 1000 time steps with the following code:
```python
W = Brownian(variance = np.eye(2), T = 1, steps = 1000)
W.simulate(n_simulations = 4, plot = False)
```
then we can access to the first simulation with:
```python
W.path[0]
```
or to the second coordinates of the third simulation with:
```python
W.path[2,:,1]
```
or to the value of each path at the 201th time step with:
```python
W.path[:,200,:]
```
The Brownian class inherits all methods from the [Processes](<project:/processes/index.md>) class.
## Examples

```python
>>> W = Brownian(variance = np.eye(2), T = 1, steps = 1000)
>>> W.simulate(2, plot = True)
array([[[ 0.        ,  0.        ],
        [-0.00944504, -0.0103207 ],
        [-0.01089965, -0.06896138],
        ...,
        [-0.26183923, -0.04326542],
        [-0.32149696, -0.0815789 ],
        [-0.36599909, -0.04858352]],
       [[ 0.        ,  0.        ],
        [ 0.07904865, -0.01548704],
        [ 0.03610401, -0.05401726],
        ...,
        [-1.79254653, -1.4591608 ],
        [-1.7628466 , -1.44618102],
        [-1.7475036 , -1.40262064]]], shape=(2, 1001, 2))
>>> dW = W.increments
>>> dW
array([[[-0.00944504, -0.0103207 ],
        [-0.00145461, -0.05864069],
        [ 0.01951358, -0.04491318],
        ...,
        [ 0.00941946,  0.00232427],
        [-0.05965772, -0.03831348],
        [-0.04450214,  0.03299538]],
       [[ 0.07904865, -0.01548704],
        [-0.04294464, -0.03853022],
        [ 0.00371223, -0.01490523],
        ...,
        [-0.04196104, -0.0015993 ],
        [ 0.02969993,  0.01297978],
        [ 0.015343  ,  0.04356038]]], shape=(2, 1000, 2))