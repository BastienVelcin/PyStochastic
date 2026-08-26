# Brownian Bridge 

## Import line
You can import the Brownian Bridge class from the `processes` module as follows:
```python
from pystochastic.processes import BrownianBridge
```
## Description

```python
pystochastic.processes.BrownianBridge(dim = 1, t_0 = 0, t_n = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ✅

Creates an instance of a $d$-dimensional Brownian Bridge on the interval $[0, T]$. A Brownian Bridge $(B_t)_{t\in[0,T]}$ is a particular example of a standard Brownian motion, where $B_T = 0$ a.s.

>[!NOTE]
> A $d$-dimensional Brownian bridge can be defined, for all $t\in[0,T] by the following equation:
>
> \begin{equation}
> B_t = W_t + \frac{t}{T}W_T
> \begin{*equation}
>
> where $(W_t)_{t\geq 0}$ is a standard $d$-dimensional Brownian motion. 


### Parameters

`dim` : _int_
: Dimension of the Brownian bridge. Must be strictly a strictly positive integer.


`T` : _float_
: Final time of the Brownian motion simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Brownian motion is simulated. Must be strictly greater than 0.

### Attributes
The Brownian Bridge class inherits all attributes from the [Processes](<project:/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods
The Brownian Bridge class inherits all methods from the [Processes](<project:/index.md>) class.

## Examples

```python
>>> B = BrownianBridge(dim = 1, T = 5, steps = 500)
>>> B.simulate(3, plot = True)
array([[[ 0.        ],
        [-0.01655659],
        [-0.12330862],
        ...,
        [-0.19210069],
        [-0.01051594],
        [ 0.        ]],
       [[ 0.        ],
        [ 0.09441971],
        [ 0.22554915],
        ...,
        [-0.00463925],
        [-0.11652599],
        [ 0.        ]],
       [[ 0.        ],
        [ 0.09192548],
        [ 0.15507396],
        ...,
        [-0.00224112],
        [ 0.06494432],
        [ 0.        ]]], shape=(3, 501, 1))
>>> B.variance(1)
0.8