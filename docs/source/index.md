# PyStochastic documentation


Welcome to the PyStochastic documentation.

PyStochastic is a Python library for probability, stochastic calculus and stochastic modelling, Monte Carlo methods and numerical methods for stochastic differential equations.
The project aims to provide a simple and consistent interface for simulating, analysing and visualising stochastic models.

Getting Started with PyStochastic

## 📦 Installation

### 🐍 Installation with pip


The easiest way to install PyStochastic is using `pip`, by running the following command:

```bash
pip install pystochastic
```

### 🗃️ Installation from source

In case you want to install PyStochastic from the repository source, you can do so by running the following command:

```bash
git clone https://github.com/BastienVelcin/PyStochastic.git
cd PyStochastic
```

and then running:
```bash
pip install .
```

Once PyStochastic library is installed on your computer, you can import it in your python project with the following line:
```python
import pystochastic
```
   
Now, you're ready to use PyStochastic!

## Conventions & Notations

In PyStochastic, vectors are represented by a 1D array-like. This means that a vector is a line one, and each matrix-vector operation $AX$ must be implemented
like

```python
>>> X @ A
```
where `@` denotes the NumPy matrix multiplication operator.

```{toctree}
:maxdepth: 2
:caption: Contents
dist/index

:maxdepth: 3
processes/index

:maxdepth: 3
solvers/index

