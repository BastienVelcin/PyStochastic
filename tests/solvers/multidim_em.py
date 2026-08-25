import numpy as np
from pystochastic.sde import EulerMaruyama

def drift1(x,t):

    A = np.array([
        [2, -0.5, 0],
        [-1, 1.5, 0.3],
        [0.2, -0.4, 1]
    ])

    mu = np.array([1, 2, -1])

    return (mu - x) @ A

def diffusion1(x,t):
    return np.array([[0.5, 0.2, 0], [0, 0.4, 0.1], [0.1, 0, 0.3]])

initial1 = np.ndarray([0,0,0])

def drift2(x,t):
    return np.array([x[0]*(2-x[1]), x[1]*(-1+0.5*x[0])])

def diffusion2(x,t):
    return np.array([[0.3*x[0], 0],[0, 0.2*x[1]]])

initial2 = np.array([2,1])