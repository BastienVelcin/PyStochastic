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
: 3-dimensional array of samples.

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
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{E}[f(X)]$ for each sample pool.

#### .moment()

```python
.moment(order = 1, n = None, function = lambda x: x)
```

The `moment()` method estimates the moment of a given order with the Law of Large Numbers.

The moment of order $r\in \mathbb{N}$ of a function of a random variable $f(X)$ is defined as $\mathbb{E}[f(X)^n]$.

**Parameters**

`order` : _int_
: Moment order $r$. Must be a positive integer.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{E}[f(X)^n]$ for each sample pool.

#### .variance()

```python
.variance(n = None, function = lambda x: x, correction = True)
```

The `variance()` method estimates the variance of the samples with the König-Huygens formula.

\begin{equation*}
\mathbb{V}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the variance. If `true`, the estimated variance is multiplied by $\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\mathbb{V}[f(X)^n]$ for each sample pool.

#### .std()

```python
.std(n = None, function = lambda x: x, correction = True)
```

The `std()` method estimates the standard deviation of the samples.

\begin{equation*}
\sigma[X] = \sqrt{\mathbb{V}[X]}
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the standard deviation. If `true`, the estimated standard deviation is multiplied by $\sqrt\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\sigma[f(X)^n]$ for each sample pool.

#### .standard_error()

```python
.standard_error(n = None, function = lambda x: x, correction = True)
```

The `standard_error()` method estimates the standard error of the samples.

\begin{equation*}
\bar\sigma[X] = \frac{\sigma[X]}{\sqrt{n}}
\end{equation*}

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`correction` : _bool_
: Specify if the Bessel correction should be applied to the standard error. If `true`, the estimated standard error is multiplied by $\sqrt\frac{n}{n-1}$.

**Returns**

_np.ndarray_
: Estimated value of $\bar\sigma[f(X)^n]$ for each sample pool.


### Confidence Intervals & Curves

#### .half_width()
```python
.half_width(n = None, function = lambda x: x, confidence = 0.95, type = "normal", variance = None)
```

The `half_width()` method returns an approximation of the confidence interval half-width of the estimator of $\mathbb{E}[f(X)]$, deduced from normality assumptions.
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
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

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

#### .confidence_interval()

```python
.confidence_interval(n=None, function = lambda x: x, confidence = 0.95,type = "normal", variance = None)
```

The `confidence_interval()` method estimates the confidence interval of a given confidence level and type of the estimator of $\mathbb{E}[f(X)]$.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`confidence` : _float_
: Confidence level $1-\alpha$ of the confidence interval. Must be a float between 0 and 1.

`type` : _str_
: Type of confidence interval. Must be one of the following: `normal` or `student`.

`variance` : _float_ or `None`
: Exact variance of the samples. If `None`, the variance is estimated from the samples.

**Returns**

_tuple_ : (_np.ndarray_, _np.ndarray_)
: Estimated confidence interval bounds of confidence level $1-\alpha$ for each sample pool.

- The first element of the tuple is the array of lower bounds of the confidence interval for each sample pool.
- The second element of the tuple is the array of upper bounds of the confidence interval for each sample pool.

>[!NOTE]
> If `type` is set as `"normal"` and `variance` is not set, then the method returns an approximation the confidence interval of confidence level $1-\alpha$ deduced from the Normal distribution with
an estimated variance.

#### .confidence_curve()

```python
.confidence_curve(n = None, n_pool = 0, function = lambda x: x, confidence = 0.95,type = "normal", variance = None)
```

The `confidence_curve()` method plots the confidence interval of a given confidence level and type of the estimator of $\mathbb{E}[f(X)]$ along the interval $[1,n]\cap \mathbb{N}$ for a specified sample pool.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`n_pool` : _int_
: Index of the sample pool to work with. Must be an integer between 0 and the first dimension of the `samples` attribute.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`confidence` : _float_
: Confidence level $1-\alpha$ of the confidence interval. Must be a float between 0 and 1.

`type` : _str_
: Type of confidence interval. Must be one of the following: `normal` or `student`.

`variance` : _float_ or `None`
: Exact variance of the samples. If `None`, the variance is estimated from the samples.

**Returns**

No return value.

### Statistical errors

#### .bias_estimator()

```python
.bias_estimator(reference = 0, n = None, function = lambda x: x)
```

The `bias_estimator()` method estimates the bias of the estimator of $\mathbb{E}[f(X)]$ when $\mathbb{E}[f(X)]$ is known.

**Parameters**

`reference` : _float_
: Exact value of $\mathbb{E}[f(X)]$.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Estimated bias of the empirical estimator of $\mathbb{E}[f(X)]$ for each sample pool.

#### .mse_estimator()

```python
.mse_estimator(reference, n = None, function = lambda x: x)
```

The `mse_estimator()` method estimates the mean-squared error estimation of the empirical estimator of $\mathbb{E}[f(X)]$ when $\mathbb{E}[f(X)]$ is known.
The mean-squared error estimation is defined by:

\begin{equation*}
MSE(X) = \mathbb{E}[\left(f(X) - \mathbb{E}[f(X)]\right)^2]
\end{equation*}

**Parameters**

`reference` : _float_
: Exact value of $\mathbb{E}[f(X)]$.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Estimated mean-squared error of the empirical estimator of $\mathbb{E}[f(X)]$ for each sample pool.

#### .rmse_estimator()

```python
.rmse_estimator(reference, n = None, function = lambda x: x)
```

The `rmse_estimator()` method estimates the root mean-squared error estimation of the empirical estimator of $\mathbb{E}[f(X)]$ when $\mathbb{E}[f(X)]$ is known.
The root mean-squared error estimation is defined by:

\begin{equation*}
RMSE(X) = \sqrt{\mathbb{E}[\left(f(X) – \mathbb{E}[f(X)]\right)^2]}
\end{equation*}

**Parameters**

`reference` : _float_
: Exact value of $\mathbb{E}[f(X)]$.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Estimated root mean-squared error of the empirical estimator of $\mathbb{E}[f(X)]$ for each sample pool.

### Descriptive statistics

#### .quantile()

```python
.quantile(q, n=None, function = lambda x: x)
```

The `quantile()` method provides the quantile of order $q$ of the samples.

**Parameters**

`q` : _float_
: Quantile order $q$. Must be a float between 0 and 1.

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Estimated quantile of order $q$ for each sample pool.

#### .min()

```python
.min(n=None, function = lambda x: x)
```

The `min()` method provides the minimum value of each sample pool.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Minimum value of each sample pool.

#### .max()

```python
.max(n=None, function = lambda x: x)
```

The `max()` method provides the maximum value of each sample pool.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Maximum value of each sample pool.

#### .median()

```python
.median(n=None, function = lambda x: x)
```

The `median()` method provides the median value of each sample pool.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Median value of each sample pool.

#### .skewness()

```python
.skewness(n=None, function = lambda x: x)
```

The `skewness()` method provides an estimation of the skewness of each sample pool.
The skewness of a random variable $X$ of expectation $\mu$ and standard deviation $\sigma$ is defined by:

\begin{equation*}
\text{skew}(X) = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]
\end{equation*}

It provides a way to tell if the distribution of the samples is left or right heavy-tailed.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Skewness of each sample pool.


#### .skewness()

```python
.kurtosis(n=None, function = lambda x: x)
```

The `kurtosis()` method provides an estimation of the Pearson kurtosis of each sample pool.
The Pearson kurtosis of a random variable $X$ of expectation $\mu$ and standard deviation $\sigma$ is defined by:

\begin{equation*}
\text{PK}(X) = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]
\end{equation*}

It estimates the tailedness of the distribution of the samples.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

**Returns**

_float_ or _np.ndarray_
: Pearson kurtosis of each sample pool.

### Empirical analysis

#### .histogram()

```python
.histogram(n = None, n_pool = 0, bins = 10, function = lambda x: x, normalized = True, plot = True, distribution = None)
```

The `histogram()` plot the histogram of a specified sample pool. It allows estimating the empirical distribution of the samples.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`n_pool` : _int_
: Index of the sample pool to work with. Must be an integer between 0 and the first dimension of the `samples` attribute.

`bins` : _int_
: Number of bins of the histogram. Must be a strictly positive integer.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`normalized` : _bool_
: Specifies if the histogram should be normalized.

`plot` : _bool_
: Specifies if the histogram should be plotted.

`distribution` : _pystochastic.dist_ or `None`
: Target distribution of the sample pool. If it is specified, the density of the distribution is plotted on the histogram.

**Returns**

_np.ndarray_
: Histogram of the sample pool.

#### .ecdf()

```python
.ecdf(n = None, n_pool = 0, bins = 10, function = lambda x: x, normalized = True, plot = True, distribution = None)
```

The `ecdf()` plot the empirical cumulative distribution function of a specified sample pool.

**Parameters**

`n` : _int_
: Number of considered samples from each sample pool. Must be an integer greater than 2.

`n_pool` : _int_
: Index of the sample pool to work with. Must be an integer between 0 and the first dimension of the `samples` attribute.

`function` : _function_
: Function $f$ to apply to the samples. By default, the `function` argument is set as the identity function.

`plot` : _bool_
: Specifies if the empirical cumulative distribution function should be plotted.

`distribution` : _pystochastic.dist_ or `None`
: Target distribution of the sample pool. If it is specified, the cumulative distribution function of the distribution is plotted on the plot.

**Returns**

_np.ndarray_
: Empirical cumulative distribution function of the sample pool.

## Examples

### Estimating a European Call Payoff

Let us consider an asset $S$, and let us represent the price of $S$ as a random variable $S_t$. We assume that the price of $S$ follows a geometric Brownian motion given by the following stochastic differential equation:
\begin{equation*}
dS_t = 0.05 S_t dt + 0.2 S_t dW_t
\end{equation*}
and $S_0=100$, where $(W_t)_{t\geq 0}$ is a unidimensional standard Brownian motion. We consider the time interval $[0,1]$.

Let $K=100$ be the strike price of a European call option. Its payoff at maturity is given by

\begin{equation*}
\text{Payoff}(K,S) = (S_1 - K)^+ = \max(0,S_1-K)
\end{equation*}

Our goal is to estimate the expected payoff $\mathbb{E}\left[(S_1-K)^+\right]$.

First, we need to generate a large number of trajectories of the previous geometric Brownian motion on $[0,1]$.

```python
import numpy as np
import pystochastic as ps

S = ps.processes.GeometricBrownianMotion(mu = 0.05, volatility = 0.2, initial = 100, T = 1, steps = 365)
prices = S.simulate(10000)
```
Since the payoff of a European call only depends on the asset price at maturity, we extract the final value of the trajectories.

```python
# Extract the final value of each trajectory
final_prices = prices[:,-1,0]
```

Then, we can define the payoff function of this European call.
```python
K = 100

def eu_call_function(x):
    return np.maximum(0, x - K)

```

Finally, we use the `MonteCarlo` class to estimate the expected payoff.
```python
mc = ps.montecarlo.MonteCarlo(final_prices)
payoff = mc.estimate(function = eu_call_function)
print(f"Estimated expected payoff: {payoff.item():.4f} monetary units")
```

The used Monte-Carlo estimator is the empirical mean estimator:
\begin{equation*}
\mathbb{E}\left[(S_1-K)^+\right] \approx \frac{1}{N}\sum_{i=1}^n (S_1^{(i)} - K)^+.
\end{equation*}

