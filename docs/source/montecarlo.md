# Monte-Carlo Methods

The Monte-Carlo class provides a common interface for all Monte-Carlo methods. This page explores the Monte-Carlo interface.


## Import line
You can import the Monte-Carlo class as follows:

```python
from pystochastic.montecarlo import *
```

## Description

```python
pystochastic.montecarlo.MonteCarlo(samples = sample_pool)
```
**Type :** Class

**Multi sample pools support :** ✅

Monte-Carlo methods are a set of numerical methods used to approximate numerical values from independent and identically distributed samples, which follow a given probability distribution $X$.
The aim of this class is to provide statistical methods for Monte-Carlo simulations, from estimation of moments, confidence intervals, to empirical density.

## Attributes
This section lists all the attributes that are common to all implemented processes.

`samples` : _array_like_
: 3-dimensional array of samples from the process.

- 1st dimension is the number of sample pools
- 2nd dimension is the sample at each time step
- 3rd dimension is the coordinate of the sample of the current time step

> [!NOTE]
> If the user provides a 2-dimensional array (with one sample pool), PyStochastic automatically converts it into a 3-dimensional array with one sample pool, to fit with the dimension convention.

## Methods

The built-in methods are divided into five categories.

### Estimators & Moments

#### .estimate()

```python
.estimate(n = None, function = lambda x: x)
```

The `estimate()` method estimates the value $\mathbb{E}[f(X)]$ with the Law of Large Numbers.

If $(X_n)_{n\in\mathbb{N}}$ is a sequence of independent and identically distributed samples from an integrable distribution $X$, then,
\begin{equation*}
\lim_{n \to + \infty} \frac{1}{n}\sum_{n=1}^n f(X_n) \xrightarrow{\mathbb{P}} \mathbb{E}[f(X)]
\end{equation*}

Furthermore, if $X$ is square-integrable, then
\begin{equation*}
\lim_{n \to + \infty} \frac{1}{n}\sum_{n=1}^n f(X_n) \xrightarrow{a.s.} \mathbb{E}[f(X)]
\end{equation*}

So, the `estimate()` method estimates the value $\mathbb{E}[f(X)]$ with the empirical estimator $\mu_n = \frac{1}{n}\sum_{n=1}^n f(X_n)$.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{E}[f(X)]$ for each sample pool.

#### .half_width()
```python
.half_width(n = None, function = lambda x: x, confidence = 0.95, type = "normal", variance = None)
```

The `half_width()` method returns an approximation of the confidence interval half-width, deduced from normality assumptions.
The confidence interval can be computed with a known or an unknown variance of normally distributed samples, with the normal or Student's t-distribution quantile.

Let $X_n$ be the mean estimator deduced from the first $n$ samples.

If the variance $\sigma^2$ of the samples is known, the confidence interval of confidence level $1-\alpha$ is deduced from the Normal distribution:

\begin{equation*}
I_\alpha = \left[\bar{X}_n - q_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}}~,~ \bar{X}_n + q_{1-\frac{\alpha}{2}} \frac{\sigma}{\sqrt{n}} \right]
\end{equation*}

where $q_{1-\frac{\alpha}{2}}$ is the quantile of order $1-\frac{\alpha}{2}$ of the standard Normal distribution.

If the variance $\sigma^2$ of the samples is unknown, the confidence interval of confidence level $1-\alpha$ is deduced from the Student's t-distribution:

\begin{equation*}
I_\alpha = \left[\bar{X}_n - t_{1-\frac{\alpha}{2},n-1} \frac{\hat\sigma}{\sqrt{n}}~,~ \bar{X}_n + t_{1-\frac{\alpha}{2}, n-1} \frac{\hat\sigma}{\sqrt{n}} \right]
\end{equation*}

where $t_{1-\frac{\alpha}{2},n-1}$ is the quantile of order $1-\frac{\alpha}{2}$ of the Student's t-distribution with $n-1$ degrees of freedom, and $\hat\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X}_n)^2}$ is estimated standard deviation.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

`confidence` : _float_
: Confidence level $1-\alpha$ of the confidence interval. Must be a float between 0 and 1.

`type` : _str_
: Type of confidence interval. Must be one of the following: `normal` or `student`.

`variance` : _float_ or `None`
: Exact variance of the samples. If `None`, the variance is estimated from the samples.

>[!NOTE]
> If `type` is set as `"normal"` and `variance` is not set, then the method returns an approximation of the half-width of the confidence interval of confidence level $1-\alpha$ deduced from the Normal distribution with
an estimated variance.

**Returns**

_np.ndarray_
: Estimated value of the half-width of the confidence interval of confidence level $1-\alpha$ for each sample pool.

#### .moment()

```python
.moment(order=1, n=None, function = lambda x: x)
```

The `moment()` method estimates the moment of a given order with the Law of Large Numbers.

The moment of order $r\in \mathbb{N}$ of a function of a random variable $f(X)$ is defined as $\mathbb{E}[f(X)^n]$.

**Parameters**

`order` : _int_
: Moment order $r$. Must be a positive integer.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{E}[f(X)^n]$ for each sample pool.

#### .variance()

```python
.variance(n=None, function = lambda x: x, correction=True)
```

The `variance()` method estimates the variance of the samples with the König-Huygens formula.

\begin{equation*}
\mathbb{V}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the variance. If `true`, the estimated variance is multiplied by $\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{V}[f(X)^n]$ for each sample pool.

#### .std()

```python
.std(n=None, function = lambda x: x, correction=True)
```

The `std()` method estimates the standard deviation of the samples.

\begin{equation*}
\sigma[X] = \sqrt{\mathbb{V}[X]}
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the standard deviation. If `true`, the estimated standard deviation is multiplied by $\sqrt\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\sigma[f(X)^n]$ for each sample pool.

#### .std()

```python
.standard_error(n=None, function = lambda x: x, correction=True)
```

The `standard_error()` method estimates the standard error of the samples.

\begin{equation*}
\bar\sigma[X] = \frac{\sigma[X]}{\sqrt{n}}
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the standard error. If `true`, the estimated standard error is multiplied by $\sqrt\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\bar\sigma[f(X)^n]$ for each sample pool.


### Confidence Intervals & Curves

### Statistical errors

### Descriptive statistics

### Empirical analysis

