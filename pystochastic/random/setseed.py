import numpy as np

_rng = np.random.default_rng()

def seed(n=None):
    global _rng
    _rng = np.random.default_rng(n)


def get_rng():
    return _rng