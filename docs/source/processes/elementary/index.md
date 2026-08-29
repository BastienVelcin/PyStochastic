# Elementary Processes

Elementary processes are the fundamental processes that are used to model the dynamics of a system. It contains, for example, the Brownian motion and all its variants.

> [!IMPORTANT]
> Every elementary process is a subclass of the [Process](<project:/processes/index.md>) class. There is no abstract class for elementary processes.

## Import line
You can import all the elementary processes as follows:

```python
from pystochastic.processes.elementary import *
```

## Attributes
This section lists all the attributes that are common to all implemented elementary processes.

`T` : _float_
: Final time of the process. The process is simulated on the interval $[0,T]$.

`steps` : _int_
: Number of time steps on $[0,T]$. Must be greater than 0.


## Methods

All the elementary processes inherit all methods from the [Process](<project:/processes/index.md>) class.

## Implemented processes

```{toctree}
:maxdepth: 1
:caption: Elementary processes
brownian
brownian_bridge
fractional_brownian_motion
bessel
```