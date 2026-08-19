# Getting Started

## Brownian Motion

```python
from pystochastic.processes import Brownian

brownian = Brownian(
    t_0=0,
    t_n=1,
    n_steps=100,
    n_simulations=10
)

paths = brownian.simulate()
```