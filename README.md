# PyStochastic

PyStochastic is a Python library for probability, stochastic calculus and stochastic modelling, Monte Carlo methods and numerical methods for
stochastic differential equations.

The project aims to provide a simple and consistent interface for
simulating, analyzing and visualizing stochastic models.

---

## ✨ Features

PyStochastic currently provides tools for:

- Probability distributions
  - Continuous distributions
  - Discrete distributions
- Random number generation
- Stochastic processes
- Numerical SDE solvers
- Monte Carlo analysis
- Plotting and visualisation

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/BastienVelcin/PyStochastic.git
cd PyStochastic
```

Then install the package:

```bash
pip install .
```

Or directly from PyPI:

```bash
pip install pystochastic
```

---

## 🚀 Quick start

### Probability distributions

PyStochastic provides a common interface for probability distributions.

For example:

```python
from pystochastic.dist import Normal

distribution = Normal(mu = 0, var = 1)

samples = distribution.sample(10000)

mean = distribution.mean()
variance = distribution.variance()
```

Probability distributions also provide functions such as `pdf` and
`cdf` when appropriate.

---

### Discrete distributions

Discrete probability distributions use the `DiscreteDistribution`
interface and provide a `pmf` method.

For example:

```python
from pystochastic.dist import Bernoulli

distribution = Bernoulli(p=0.3)

samples = distribution.sample(10000)

probability = distribution.pmf(1)
```

---

## 🎲 Random number generation

PyStochastic provides random number generators for both continuous
and discrete distributions through the `random` module.

```python
from pystochastic.random import continuous

samples = continuous.normal(
  mean = 0,
  var = 1,
  n = 10000,
)
```

A global seed can also be configured for reproducible simulations.

```python
from pystochastic.random.setseed import seed

seed(42)
```

---

# 📈 Stochastic processes

PyStochastic provides several classical stochastic processes, distributed into three categories:

- Elementary processes
  - Brownian motion
  - Brownian bridge
  - Fractional Brownian motion
  - Bessel process

- Diffusion processes 
  - Geometric Brownian motion
  - Ornstein-Uhlenbeck process
  - Vasicek model
  - Cox-Ingersoll-Ross model
  - Constant Elasticity of Variance model
  - Heston model
  - Hull-White model
  
- Jump processes
  - Poisson process
  - Compound Poisson process

For example:

```python
import numpy as np
from pystochastic.processes import Brownian

process = Brownian(
    cov = np.eye(2),
    T = 1,
    steps = 1000,
)

process.simulate(n_simulations = 100)
```

The simulated paths can then be analysed and visualised.

---

# 🧮 Stochastic differential equations

PyStochastic provides numerical solvers for stochastic differential
equations.

Currently implemented methods include:

- Euler-Maruyama
- Milstein (1D only)
- Runge-Kutta (1D only)

These solvers support vectorised simulations and can be applied to
multidimensional stochastic systems.

For example:

```python
import numpy as np
from pystochastic.sde import EulerMaruyama

solver = EulerMaruyama(
    drift = lambda x,t : x,
    diffusion = lambda x,t : np.exp(-2*t)*0.5*x,
    initial = 5,
    T = 1,
    steps = 1000
)

solution = solver.solve()
```

---

# 🎯 Monte Carlo

The `MonteCarlo` class provides statistical tools for analyzing
collections of simulated samples.

For example:

```python
from pystochastic.montecarlo import MonteCarlo

mc = MonteCarlo(samples)

estimate = mc.estimate()
variance = mc.variance()
standard_error = mc.standard_error()
```

Confidence intervals can also be computed:

```python
lower, upper = mc.confidence_interval(
    confidence=0.95,
    type="student",
)
```

PyStochastic also provides tools for statistical visualisation,
including histograms, empirical CDFs and confidence curves.

```python
mc.histogram()

mc.ecdf()

mc.confidence_curve()
```

---

# 📊 Vectorisation

A major focus of PyStochastic is efficient simulation.

Whenever possible, simulations are vectorised using NumPy rather than
performing independent Python-level loops.

This is particularly useful when a large number of Monte Carlo
simulations is required.

---

# 🧪 Testing

PyStochastic uses `pytest` for its test suite.

Run all tests with:

```bash
pytest
```

The project currently contains tests covering:

- Probability distributions
- Discrete distributions
- Random number generators
- Stochastic processes
- SDE solvers
- Monte Carlo methods
- Public APIs

The test suite also includes mathematical consistency and statistical
tests.

---

# ⚡ Benchmarks

Performance benchmarks are included to compare different simulation
approaches and evaluate the benefit of vectorisation.

For example, vectorised simulations can provide substantial speedups
compared with sequential matrix-based implementations for large
numbers of simulations.

---

# 📚 Project structure

```text
pystochastic/
├── dist/
│   ├── continuous.py
│   ├── discrete.py
│   └── distribution.py
│
├── montecarlo/
│   └── montecarlo.py
│
├── processes/
│   ├──  diffusion/
│   │   ├── constant_elasticity_variance.py
│   │   ├── cox_ingersoll_ross.py
│   │   ├── diffusion_process.py
│   │   ├── geometric_brownian_motion.py
│   │   ├── heston.py
│   │   ├── hull_white.py
│   │   ├── ornstein_uhlenbeck.py
│   │   └── vasicek.py
│   │
│   ├──  elementary/
│   │   ├── bessel.py
│   │   ├── brownian.py
│   │   ├── brownian_bridge.py
│   │   └── fractional_brownian_motion.py
│   │
│   └──  jump/
│       ├── compound_poisson.py
│       ├── jump_process.py
│       └── poisson.py
│
├── random/
│   ├── continuous.py
│   ├── discrete.py
│   ├── multivariate.py
│   └── setseed.py
│
└── sde/
    ├── eulermaruyama.py
    ├── milstein.py
    └── rungekutta.py
```

---

# 🛠️ Development

Clone the repository and install the project in editable mode:

```bash
git clone https://github.com/BastienVelcin/PyStochastic.git
cd PyStochastic

pip install -e .
```

Install the development dependencies and run the tests:

```bash
pytest
```

---

# 🗺️ Roadmap

Possible future developments include:

- Additional probability distributions
- Additional stochastic processes
- Additional SDE numerical schemes
- Additional performance optimizations
- More statistical analysis tools
- Expanded benchmarking
- Improved visualization capabilities

---

# 🤝 Contributing

Contributions, suggestions and bug reports are welcome.

If you find a bug or have an idea for a new feature, please open an
issue on GitHub.

Pull requests are also welcome.

---

# 📄 License

PyStochastic is released under the MIT License.

See the `LICENSE` file for more information.
