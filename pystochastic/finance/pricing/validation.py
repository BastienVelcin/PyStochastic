import numpy as np

def _validate_parameters(initial, option_type):

    if not isinstance(initial, (int, np.integer, float, np.floating)) or initial <= 0 or not np.isfinite(initial) or np.isnan(initial):
        raise ValueError("The initial value must be a strictly positive real number.")

    if not option_type in ["call", "put"]:
        raise ValueError("The option type must be either 'call' or 'put'.")