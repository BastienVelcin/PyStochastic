# PyStochastic

**PyStochastic** is a Python library dedicated to probability, random variables, stochastic processes, stochastic differential equations (SDEs), and Monte Carlo simulation.

The project aims to provide a simple and modular framework for studying and simulating classical stochastic models, while keeping the mathematical structure of the underlying models explicit.
Visit my website for more informations: https://bastienvelcin.github.io/projets/pystochastic.html

> **Status:** Work in progress — first development version.

---

## Features

PyStochastic currently provides tools for:

* Probability distributions

  * Probability density functions (PDF)
  * Cumulative distribution functions (CDF)
  * Random sampling
  * Moments and entropy
  * Support
    
* Random number generation
  
* Stochastic processes

  * Brownian motion
  * Poisson process
  * Geometric Brownian motion
  * Ornstein–Uhlenbeck process
  * Vasicek model
  * Cox–Ingersoll–Ross (CIR) model
    
* Numerical methods for stochastic differential equations

  * Euler–Maruyama scheme
  * Milstein scheme
    
* Exact simulation methods for selected stochastic processes

    * Geometric Brownian motion
    * Ornstein–Uhlenbeck process (1D only)
    * Vasicek model (1D only)
    * Cox–Ingersoll–Ross (CIR) model (1D only)
    
* Monte Carlo estimation

  * Empirical means
  * Confidence intervals
  * Process-level Monte Carlo simulation
    
* Interactive visualization through Plotly

* Multidimensional stochastic processes

---

## Installation

Clone the repository:

```bash
git clone https://github.com/BastienVelcin/PyStochastic.git
cd PyStochastic
```

Then install the package in editable mode:

```bash
pip install -e .
```

The main dependencies are:

* NumPy
* SciPy
* SymPy
* Plotly

---

## Project structure

The project is organized into several subpackages according to the mathematical objects and numerical methods they implement.

```text
pystochastic/
│
├── dist/
│   └── Probability distributions
│
├── montecarlo/
│   └── Monte Carlo estimators and process simulation
│
├── processes/
│   └── Stochastic process models
│
├── pyrandom/
│   └── Random number generation
│
├── sde/
│   └── Numerical methods for SDEs
│
└── utils.py
    └── Utility functions and default SDE coefficients
```

### `pystochastic.dist`

This module contains implementations of classical probability distributions.

Each distribution provides a common interface including, when available:

```python
distribution.pdf(x)
distribution.cdf(x)
distribution.sample(n)
distribution.mean()
distribution.variance()
distribution.entropy()
distribution.support()
```

Examples include:

* Uniform
* Normal
* Exponential
* Gamma
* Beta
* Weibull
* Fréchet
* Cauchy
* and other classical distributions.

Example:

```python
from pystochastic.dist import Normal

X = Normal(mu=0, sd=1)

print(X.mean())
print(X.variance())

samples = X.sample(10_000)
```

The distributions can also be visualized using their PDF or CDF.

```python
X.plot_pdf()
X.plot_cdf()
```

---

### `pystochastic.pyrandom`

`pyrandom` provides the random number generation layer used by the rest of the library.

The goal is to keep random sampling separated from the mathematical description of probability distributions and stochastic processes.

This layer is used internally by distributions and stochastic process simulations.

---

### `pystochastic.processes`

This module contains implementations of classical stochastic processes.

Currently implemented models include:

* Brownian motion
* Poisson process
* Geometric Brownian motion
* Ornstein–Uhlenbeck process
* Vasicek model
* Cox–Ingersoll–Ross (CIR) model

For example, a Brownian motion can be simulated with:

```python
from pystochastic.processes import Brownian

brownian = Brownian(
    var=1,
    t_0=0,
    t_n=1,
    n_steps=1000
)

path = brownian.simulate()
```

Multidimensional processes are also supported where mathematically appropriate. For example:

```python
from pystochastic.processes import Brownian

R = OrnsteinUhlenbeck(
    mean = [1,2],
    sigma = np.ones((2,2)),
    theta = np.array([[1,1/2],[1/2,1]]),
    r_0 = [2,3],
    t_0 = 0,
    t_n = 1,
    n_steps = 1000,
)

path = R.simulate()

path = brownian.simulate()
```
---

### `pystochastic.sde`

The `sde` module contains numerical methods for solving stochastic differential equations.

The general SDE considered is

$$
dX_t = \mu(X_t,t)dt + \sigma(X_t,t)dW_t.
$$

Currently implemented numerical schemes include:

* Euler–Maruyama
* Milstein

Example:

```python
from pystochastic.sde import EulerMaruyama

def drift(x, t):
    return -x

def diffusion(x, t):
    return 1

solver = EulerMaruyama(
    mu=drift,
    sigma=diffusion,
    x_0=1,
    t_0=0,
    t_n=1,
    n_steps=1000,
    n_simulations=100
)

paths = solver.solve()
```

The solver can also generate interactive plots for one-, two- and three-dimensional processes.

---

### `pystochastic.montecarlo`

The `montecarlo` module provides tools for Monte Carlo estimation.

The basic Monte Carlo estimator approximates an expectation with the empirical mean

$$
\mathbb{E}[f(X)]
$$

by

$$ \hat{\mathbb{E}}[f(X)] = 
\frac{1}{N}
\sum_{i=1}^{N} f(X_i).
$$

The module also provides confidence intervals based on the empirical variance.

Example:

```python
from pystochastic.montecarlo import MonteCarloEstimator
import numpy as np

samples = np.random.normal(size=100_000)

mc = MonteCarloEstimator(samples)

print(mc.estimate())
print(mc.confidence_interval())
```

A process-level Monte Carlo interface is also provided for studying quantities obtained from simulated stochastic processes.

---

## Stochastic process models

A major objective of PyStochastic is to provide both numerical and, whenever possible, exact simulation methods for classical stochastic processes.

For example, the CIR model

$$
dX_t = a(b-X_t)dt + \sigma\sqrt{X_t}dW_t
$$

can be simulated using either Euler–Maruyama or its exact transition distribution.

```python
from pystochastic.processes import CIR

cir = CIR(
    a=1,
    b=1,
    sigma=0.5,
    r_0=1,
    t_0=0,
    t_n=1,
    n_steps=1000
)

paths = cir.simulate(
    n_simulations=1000,
    method="exact"
)
```

This makes it possible to compare numerical schemes with exact simulation when an exact transition distribution is available.

---

## Visualization

PyStochastic uses [Plotly](https://plotly.com/python/) for interactive visualization.

Depending on the dimension of the process, simulated trajectories can be represented in one, two or three dimensions.

For example:

```python
process.plot()
```

produces an interactive visualization of the simulated paths.

---

## Mathematical scope

The project is primarily focused on computational probability and stochastic modelling.

The current development follows the following structure:

```text
Probability distributions
        ↓
Random variables
        ↓
Stochastic processes
        ↓
Stochastic differential equations
        ↓
Numerical SDE schemes
        ↓
Monte Carlo methods
        ↓
Quantitative applications
```

This structure is intended to keep the mathematical foundations of the library closely connected to their numerical implementation.

---

## Future developments

Possible future developments include:

* More probability distributions
* Additional stochastic processes
* Additional SDE numerical schemes
* Improved multidimensional support
* Statistical and numerical validation tests
* Convergence studies for numerical SDE schemes
* Variance reduction techniques for Monte Carlo simulation
* Monte Carlo pricing of financial derivatives
* Black–Scholes comparison and validation
* Additional quantitative finance models
* Improved documentation and examples

---

## Motivation

PyStochastic is a personal project developed to explore the computational side of probability and stochastic calculus through Python.

The project is particularly interested in the connection between mathematical theory and numerical implementation: a stochastic model should not only be represented mathematically, but also simulated, visualized, and numerically investigated.

The long-term goal is to develop PyStochastic into a small, coherent framework for stochastic modelling and Monte Carlo methods, with applications to quantitative finance.

---

## References

The implementation of the mathematical models is based on standard results from probability theory, stochastic calculus and numerical stochastic analysis.

Some useful references include:

* B. Øksendal, *Stochastic Differential Equations: An Introduction with Applications*
* P. E. Kloeden and E. Platen, *Numerical Solution of Stochastic Differential Equations*
* I. Karatzas and S. E. Shreve, *Brownian Motion and Stochastic Calculus*
* J. C. Hull, *Options, Futures, and Other Derivatives*

---

## License

PyStochastic is distributed under the MIT License.

See [`LICENSE`](LICENSE) for more information.
