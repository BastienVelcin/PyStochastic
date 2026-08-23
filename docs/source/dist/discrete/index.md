# Discrete Distribution class

The `DiscreteDistribution` class provides the common
interface for discrete probability distributions in PyStochastic.

> [!WARNING]
> `DiscreteDistribution` is an abstract class and cannot be instantiated directly. It is intended to be subclassed by concrete discrete distributions.

## Description

## Attributes

The `DiscreteDistribution` class has no attributes.

## Methods

#### pdf()

```python
.pmf(x=None)
```

The `pmf` method returns the probability mass function of the distribution. If no argument is provided,
the method prints the probability mass function. Otherwise, if the parameter `x` is provided,
the method returns the probability mass function evaluated at the point `x`.
By default, the argument `x` is set to `None`, so that the method prints the probability mass function.

For a discrete random variable $X$ with discrete support $D$, the probability mass function of $X$ is defined, for all $k\in D$, by
\begin{equation*}
f_X(k) = \mathbb{P}(X=k).
\end{equation*}

**Parameters**

`k` : _int_, _np.int_ or `None`
: Point at which the probability mass function is evaluated. If `None`, the method prints the expression of the probability mass function.

**Returns**

_np.ndarray_
: When `k` is provided, the method returns the probability mass function evaluated at the point `k`.

___

#### cdf()

```python
.cdf(x = None)
```

The `cdf` method returns the cumulative distribution function of the distribution. If no argument is provided,
the method prints the cumulative distribution function. Otherwise, if the parameter `x` is provided,
the method returns the cumulative distribution function evaluated at the point `x`.
By default, the argument `x` is set to `None`, so that the method prints the cumulative distribution function.

The cumulative distribution function of a random variable $X$ is the function defined, for all $t\in\mathbb{R}$, by
\begin{equation*}
F_X(t) = \mathbb{P}(X\leq t).
\end{equation*}

**Parameters**

`x` : _float_, _np.float_ or `None`
: Point at which the cumulative distribution function is evaluated. If `None`, the method prints the expression of the cumulative distribution function.

**Returns**

_np.ndarray_
: When `x` is provided, the method returns the cumulative distribution function evaluated at the point `x`.

___

#### plot_pdf()

```python
.plot_pmf()
```

The `plot_pmf` method plots the probability density mass of the distribution. The plotting
is done using the `plotly` library.

**Parameters**

No parameters.

**Returns**

No return value.
___

#### plot_cdf()

```python
.plot_cdf()
```

The `plot_pdf` method plots the cumulative distribution function of the distribution. The plotting
is done using the `plotly` library.

**Parameters**

No parameters.

**Returns**

No return value.
___

#### samples()
```python
.samples(n = 1)
```
The `samples` method returns `n` samples from the current distribution. By default, the number of samples is set to 1.

**Parameters**

`n` : _int_ or _np.integer_
: Number of desired samples. If `n` is not provided, the method returns 1 sample.

**Returns**

_np.ndarray_
: Array of `n` samples from the current distribution.

#### mean()

```python
.mean()
```
The `mean` method returns the exact mean of the distribution. For a discrete random variable $X$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, such that $\mathbb{P}_X= \displaystyle\sum_{k\in D} \mathbb{P}(X=k)\delta_k$, then the mean of $X$ is defined as
\begin{equation*}
\mathbb{E}[X] = \sum_{k\in D} k\mathbb{P}(X=k)
\end{equation*}

**Parameters**

No parameters.

**Returns**

_float_ or _np.float_
: Mean of the distribution.

#### variance()

```python
.variance()
```
The `variance` method returns the exact variance of the distribution. For a random variable $X$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, the variance of $X$ is defined as
\begin{equation*}
\mathbb{V}[X] = \mathbb{E}[(X-\mathbb{E}[X])^2].
\end{equation*}

Moreover, if $X$ is a square-integrable random variable, then the variance can be obtained by the König-Huygens formula:
\begin{equation*}
\mathbb{V}[X] = \mathbb{E}[X^2]-\mathbb{E}[X]^2.
\end{equation*}

**Parameters**

No parameters.

**Returns**

_float_ or _np.float_
: Variance of the distribution.

#### entropy()

```python
.entropy()
```
The `entropy` method returns the exact Shannon entropy of the distribution.
The Shannon entropy of a random variable $X$ admitting a density $f$ is defined as
\begin{equation*}
H(X) = -\sum_{k\in D} \mathbb{P}(X=k)\log(\mathbb{P}(X=k))
\end{equation*}

**Parameters**

No parameters.

**Returns**

_float_ or _np.float_
: Shannon Entropy of the distribution.

#### support()

```python
.support()
```
The `support` method returns the support of the distribution. The support of a random variable $X$ is the smallest subset $F$ of $D$ such that
\begin{equation*}
\mathbb{P}(X \in F) = 1.
\end{equation*}

**Parameters**

No parameters.

**Returns**

_tuple_ or _str_
: Support of the distribution.

#### info()

```python
.info()
```
The `info` method summarizes the information of the current distribution.

**Parameters**

No parameters.

**Returns**

No return value.

## Implemented discrete distributions

```{toctree}
:maxdepth: 1

duniform
bernoulli
rademacher
binomial
poisson
geometric
hypergeometric
negativebinomial
yulesimon
```