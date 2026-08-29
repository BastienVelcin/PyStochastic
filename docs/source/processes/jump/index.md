# Jump Processes

The `JumpProcess` module provides an abstract class for all jump processes. A diffusion process is a stochastic process that admits discrete movements (jumps) of its state.



## Import line
You can import all the jump processes as follows:

```python
from pystochastic.processes.jump import *
```


## Attributes
This section lists all the attributes that are common to all implemented jump processes.

`T` : _float_
: Final time of the process. The process is simulated on the interval $[0,T]$.

`steps` : _int_
: Number of time steps on $[0,T]$. Must be greater than 0.


## Methods

The `JumpProcess` class inherits all methods from the [Process](<project:/processes/index.md>) class.

## Implemented processes

```{toctree}
:maxdepth: 1
:caption: Jump processes
poisson
compounded_poisson
```