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
    to improve the simulation depending on the nature of the functions.

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


def _decompose(value):

    """
    Decompose function

    The decompose function specifies if the argument is a scalar, a vector or a diagonal matrix, or a more general matrix.
    This function allows the Euler-Maruyama algorithm to work efficiently with any of these representations, by choosing automatically
    the right computation strategy depending on the shape of the argument.

    The available computation strategies are :
        - Vectorized : used when the argument is a scalar, vector, or a diagonal matrix (diagonal diffusion, independent noise per dimension).
        - Sequential : used when the argument is a full (dim,dim) non-diagonal matrix (correlated noise across dimensions).
        - Parallel : used when sigma(x,t) returns a full (dim,dim) matrix (non-diagonal diffusion).

    Parameters
    ----------
    value : np.ndarray
        Scalar, vector or matrix to decompose.

    Returns
    -------
    tuple
        (representation, is_diagonal).
        - Scalar or vector : returns the original vector and True (already diagonal).
        - Diagonal (dim,dim) matrix : returns the diagonal of the matrix and True.
        - Non-Diagonal (dim,dim) matrix : returns the matrix itself and False.
    """


    value = np.asarray(value, dtype=float)
    # Checks if the argument is a scalar or a vector.
    if value.ndim <= 1:
        return np.atleast_1d(value), True

    off_diagonal = value - np.diag(np.diag(value))

    # Checks if the argument is a diagonal matrix.
    if np.allclose(off_diagonal, 0):
        return np.diag(value), True

    # If not, we return true
    return value, False


