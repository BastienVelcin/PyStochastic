# Continuous Distribution class

```{toctree}
:maxdepth: 1

uniform
exponential
normal
gamma
beta
weibull

```
The `ContinuousDistribution` class provides the common
interface for continuous probability distributions in PyStochastic.

> [!WARNING]
> `ContinuousDistribution` is an abstract class and cannot be instantiated directly. It is intended to be subclassed by concrete continuous distributions.

## Description

## Attributes

The `ContinuousDistribution` class has no attributes.

## Methods

#### pdf()

```python
.pdf(x=None)
```

The `pdf` method returns the probability density function of the distribution. If no argument is provided,
the method prints the probability density function. Otherwise, if the parameter `x` is provided,
the method returns the probability density function evaluated at the point `x`.
By default, the argument `x` is set to `None`, so that the method prints the probability density function.

**Parameters**

`x` : _float_, _np.float_ or `None`
: Point at which the probability density function is evaluated. If `None`, the method prints the expression of the probability density function.

**Returns**

_np.ndarray_
: When `x` is provided, the method returns the probability density function evaluated at the point `x`.

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
.plot_pdf()
```

The `plot_pdf` method plots the probability density function of the distribution. The plotting
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
The `mean` method returns the exact mean of the distribution. For a random variable $X$ defined on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, the mean of $X$ is defined as
\begin{equation*}
\mathbb{E}[X] = \int_\Omega Xd\mathbb{P}.
\end{equation*}

Moreover, if $X$ follows a continuous distribution that admits an integrable density $f$, then
\begin{equation*}
\mathbb{E}[X] = \int_\mathbb{R} xf(x)dx
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
H(X) = -\int_\mathbb{R} f(x)\log(f(x))dx
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
The `support` method returns the support of the distribution. The support of a random variable $X$ is the smallest closed $F$ set of $\mathbb{R}$ such that
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