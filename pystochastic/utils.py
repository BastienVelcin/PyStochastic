# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def default_drift(x, t):
    x = np.atleast_1d(x)
    return np.ones(len(x))

def default_diffusion(x, t):
    x = np.atleast_1d(x)
    return np.eye(len(x))

# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

