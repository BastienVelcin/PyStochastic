# Fractional Brownian Motion 

## Import line
You can import the Fractional Brownian Motion class from the `processes` module as follows:
```python
from pystochastic.processes import FractionalBrownianMotion
```
## Description

```python
pystochastic.processes.FractionalBrownianMotion(hurst = 0.5, T = 1, steps = 1000)
```
**Type :** Class

**Multidimensional support :** ❌

Creates an instance of a Fractional Brownian Motion $(B_t)_{t\geq 0}$ on the interval $[0, T]$, with a specified Hurst index $H$.
Let $H\in (0,1)$. A Fractional Brownian Motion of Hurst index $H$ is a Gaussian process that satisfies the following assertions:

- $B_0 = 0 ~~a.s.$ 
- $\forall t \in \mathbb{R}_+, ~~ \mathbb{E}[B_t] = 0 ~~$ 
- $\forall s,t \in \mathbb{R}_+, ~~ \mathbb{E}[B_t B_s] = \frac{1}{2}\left(|t|^{2H} + |s|^{2H} - |t-s|^{2H}\right)$
- $(B_t)_{t\geq 0}$ is continuous process

The Hurst index $H$ is a measure of the correlation between the process increments.
- If $H < \frac{1}{2}$, the increments are negatively correlated.
- If $H = \frac{1}{2}$, the increments are uncorrelated.
- If $H > \frac{1}{2}$, the increments are positively correlated.

>[!NOTE]
> If $H=\frac{1}{2}$, the Fractional Brownian Motion is equivalent to a standard Brownian motion.

### Parameters

`hurst` : _float_
: Hurst index of the Fractional Brownian Motion. Must be strictly between 0 and 1.

`T` : _float_
: Final time of the Fractional Brownian Motion simulation. Must be greater than `0`.

`steps` : _int_
: Number of time steps between `0` and `T` on which the Fractional Brownian Motion is simulated. Must be strictly greater than 0.

### Attributes
The Fractional Brownian Motion class inherits all attributes from the [Processes](<project:/index.md>) class.
It also possesses the attributes deduced from its parameters.

### Methods
The Fractional Brownian Motion class inherits all methods from the [Processes](<project:/index.md>) class.

## Examples

```python
>>> F = FractionalBrownianMotion(hurst=0.7,T=1,steps=1000)
>>> F.simulate(4, plot=True)
array([[[ 1.01256782e-05],
        [-1.54637811e-02],
        [-1.09820019e-02],
        ...,
        [ 4.11235960e-01],
        [ 4.10467180e-01],
        [ 4.01345053e-01]],
       [[-8.83895368e-06],
        [ 6.45146931e-03],
        [ 1.55314352e-02],
        ...,
        [ 7.33364067e-01],
        [ 7.41373400e-01],
        [ 7.39846071e-01]],
       [[-1.99394361e-06],
        [-1.01093609e-03],
        [-4.25493395e-03],
        ...,
        [ 6.43236306e-01],
        [ 6.56436350e-01],
        [ 6.59150772e-01]],
       [[-5.15817732e-06],
        [-7.75335147e-03],
        [-6.48656008e-03],
        ...,
        [ 1.69343110e+00],
        [ 1.67957874e+00],
        [ 1.67707960e+00]]], shape=(4, 1001, 1))
>>> F.quadratic_variation(plot=True)
array([[0.00000000e+00, 2.39441791e-04, 2.59528137e-04, ...,
        6.76039635e-02, 6.76045545e-02, 6.76877677e-02],
       [0.00000000e+00, 4.17355828e-05, 1.24181363e-04, ...,
        5.99056755e-02, 5.99698250e-02, 5.99721577e-02],
       [0.00000000e+00, 1.01796426e-06, 1.15414863e-05, ...,
        5.95434790e-02, 5.97177202e-02, 5.97250883e-02],
       [0.00000000e+00, 6.00344993e-05, 6.16392598e-05, ...,
        6.44473691e-02, 6.46392570e-02, 6.46455027e-02]], shape=(4, 1001))
```

## References

- Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian Motions, Fractional Noises and Applications. SIAM Review, 10(4), 422–437. http://www.jstor.org.bases-doc.univ-lorraine.fr/stable/2027184
- Nourdin, I. (2012). Selected Aspects of Fractional Brownian Motion. Springer Milan. https://doi.org/10.1007/978-88-470-2823-4
- Vojta, T., Warhover, A. (2020). Probability density of fractional Brownian motion and the fractional Langevin equation with absorbing walls. arXiv. https://doi.org/10.48550/ARXIV.2012.03142
