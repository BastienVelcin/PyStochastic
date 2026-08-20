# Continuous Distribution class

The `ContinuousDistribution` class provides the common
interface for continuous probability distributions in PyStochastic.

> [!WARNING]
> `ContinuousDistribution` is an abstract class and cannot be
    instantiated directly. It is intended to be subclassed by
    concrete continuous distributions.

## Description

## Attributes

The `ContinuousDistribution` class has no attributes.

## Methods

#### pdf(x = None)

The `pdf` method returns the probability density function of the distribution. If no argument is provided,
the method prints the probability density function. Otherwise, if the parameter `x` is provided,
the method returns the probability density function evaluated at the point `x`.
By default, the argument `x` is set to `None`, so that the method prints the probability density function.

**Parameters**

`x` : _float_ or `None`
: Point at which the probability density function is evaluated. If `None`, the method prints the expression of the probability density function.

**Returns**

_np.ndarray_
: When `x` is provided, the method returns the probability density function evaluated at the point `x`.

___

####cdf(x = None)

The `cdf` method returns the cumulative distribution function of the distribution. If no argument is provided,
the method prints the cumulative distribution function. Otherwise, if the parameter `x` is provided,
the method returns the cumulative distribution function evaluated at the point `x`.
By default, the argument `x` is set to `None`, so that the method prints the cumulative distribution function.

**Parameters**

`x` : _float_ or `None`
: Point at which the cumulative distribution function is evaluated. If `None`, the method prints the expression of the cumulative distribution function.

**Returns**

_np.ndarray_
: When `x` is provided, the method returns the cumulative distribution function evaluated at the point `x`.

___

#### plot_pdf()

The `plot_pdf` method plots the probability density function of the distribution. The plotting
is done using the `plotly` library.

**Parameters**

No parameters.

**Returns**

No return value.
___

#### plot_cdf()

The `plot_pdf` method plots the cumulative distribution function of the distribution. The plotting
is done using the `plotly` library.

**Parameters**

No parameters.

**Returns**

No return value.
___

#### samples(n = 1)

The `samples` method returns `n` samples from the current distribution. By default, the number of samples is set to 1.

**Parameters**

`n` : _int_ or _np.integer_
: Number of desired samples. If `n` is not provided, the method returns 1 sample.

**Returns**

_np.ndarray_
: Array of `n` samples from the current distribution.

