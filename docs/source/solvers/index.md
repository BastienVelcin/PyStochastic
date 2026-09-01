# Stochastic Differential Equations Solvers

PyStochastic provides 3 different solvers for solving stochastic differential equations : Euler-Maruyama, Milstein and Runge-Kutta.

Note that Milstein and Runge-Kutta only support autonomous and unidimensional stochastic differential equations of the form

\begin{equation*}
dX_t = \mu(X_t) dt + \sigma(X_t) dW_t
\end{equation*}

where:
- $(X_t)_{t\geq 0}$ is a unidimensional stochastic process,
- $(W_t)_{t\geq 0}$ is a standard unidimensional Brownian motion,
- $\mu : \mathbb{R} \to \mathbb{R}$ is a continuous mapping, called **drift**,
- $\sigma : \mathbb{R} \to \mathbb{R}$ is a differentiable or continuous mapping, called **diffusion**.

whereas Euler-Maruyama supports any stochastic differential equation of the form

\begin{equation*}
dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
\end{equation*}

where:
- $(X_t)_{t\geq 0}$ is a $n$-dimensional stochastic process,
- $(W_t)_{t\geq 0}$ is a standard $d$-dimensional Brownian motion,
- $\mu : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^n$ is a continuous mapping, called **drift**,
- $\sigma : \mathbb{R}^n\times [0,+\infty) \to \mathbb{R}^{n\times d}$ is a continuous mapping, called **diffusion**.

```{toctree}
:maxdepth: 2
:caption: Solvers
euler_maruyama
milstein
runge_kutta
```
