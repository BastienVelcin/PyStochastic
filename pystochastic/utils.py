# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def default_drift(x, t=None):
    x = np.atleast_1d(x)
    return np.ones(len(x))

def default_diffusion(x, t=None):
    x = np.atleast_1d(x)
    return np.diag(x)

def _decompose(value):

    """
    Decompose function

    For processes that use Euler-Maruyama method for simulation, we indentify the nature of the drift and diffusion
    to accelerate the simulation depending on the nature of the functions.

    If drift and diffusion are batch compatible, the simulation is vectorized.
    If not, we use multiprocessing with an explicit for loop on every simulations.
    """


    value = np.asarray(value, dtype=float)
    # If the function is already a scalar or a vector, we return it as is.
    if value.ndim <= 1:
        return np.atleast_1d(value), True
    off_diagonal = value - np.diag(np.diag(value))

    # If the function is a diagonal matrix, we return it as is.
    if np.allclose(off_diagonal, 0):
        return np.diag(value), True

    # If the function is not a diagonal matrix, we return it as is and we specify that the function cannot be vectorized.
    return value, False


# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

