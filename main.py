# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def default_drift(x, t):
    return 1

def default_diffusion(x, t):
    return 1

def quadrature(f,a,b,n):
    approx = 0
    dt = (b-a)/n
    t = np.linspace(a,b,n)
    for i in range(1,n):
        approx += dt*(f(t[i-1])+f(t[i]))/2
    return approx

mat_quad = np.vectorize(quadrature)

def carre(x):
    return x**2
# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

