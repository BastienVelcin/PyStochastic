import numpy as np
import pyrandom.crandom
import matplotlib
import plotly.graph_objects as go
from main import is_pos_def


class Brownian:
    def __init__(self, var=None,dim=1,h=0.01,n=1000):
        if var is None:
            var = np.eye(dim)
        self.var = np.array(var)

        if self.var.shape != (dim, dim):
            raise ValueError("The dimension of the covariance matrix must coincide with the specified dimension.")
        if not is_pos_def(self.var):
            raise ValueError("The covariance matrix is not positive-definite.")

        self.dim = dim
        self.path = brownian_motion(self.var,self.dim,h,n)
        self.h = h
        self.n = n
        self.t = np.arange(0, (n + 1) * self.h, self.h)

    def simulate(self):
        self.path = brownian_motion(self.var, self.dim, self.h, self.n)
        return self.path
    def plot(self):
        if self.dim > 3:
            raise ValueError("The path can be plotted only for 1D, 2D and 3D.")
        fig = go.Figure()

        if self.dim == 1:
            fig.add_trace(go.Scatter(x=self.t, y=self.path[:,0],mode="lines", line=dict(width=2)))

        elif self.dim == 2:
            fig.add_trace(go.Scatter(x=self.path[:, 0], y=self.path[:, 1], mode="lines", line=dict(width=2)))

        else:
            fig.add_trace(go.Scatter3d(x=self.path[:, 0],y=self.path[:, 1],z=self.path[:, 2],mode="lines",line=dict(width=2)))

        fig.show()

    def final_position(self):
        return self.path[-1]

    def max(self):
        if self.dim != 1:
            raise ValueError("The maximum value can only be computed in 1D. Please refer to the max_norm function.")
        return np.max(self.path),np.argmax(self.path)

    def min(self):
        if self.dim != 1:
            raise ValueError("The minimum value can only be computed in 1D.")
        return np.min(self.path), np.argmin(self.path)

    def max_norm(self):
        norms = np.sum(self.path**2, axis=1)
        arg_max = np.argmax(norms)
        return self.path[arg_max,:], self.t[arg_max]

    def __repr__(self):
        return (f"Brownian Motion\n------------------------\n "
                f"Dimension : {self.dim}\n "
                f"Time horizon: {self.n*self.h}\n "
                f"Time step: {self.h}\n "
                f"Covariance matrix: \n"
                f"{self.var}")
        '''
        print()
        print("")
        print(f"Dimension: {self.dim}")
        print(f"Time horizon: {self.n}")
        print(f"Time step: {self.h}")
        print(f"Covariance matrix: {self.var}")'''
def brownian_motion(var=np.array(1),d=1, h=0.01, n=1000):
    '''
    Simulate a d-dimensional Brownian motion.

    Parameters
    ----------
    var : ndarray
        Covariance matrix.
    d : int
        Dimension.
    h : float
        Time step.
    n : int
        Number of time steps.

    Returns
    -------
    ndarray
        Simulated Brownian path of shape (n+1,d).
    '''
    L = np.linalg.cholesky(var)

    N = [pyrandom.crandom.normal(0,1,n) for _ in range(d)]
    Z = np.stack(N,axis=0)

    W = np.zeros((n+1,d))
    dW = np.sqrt(h) * L @ Z
    for k in range(n):
        W[k + 1] = W[k] + dW[:, k]
    return W
