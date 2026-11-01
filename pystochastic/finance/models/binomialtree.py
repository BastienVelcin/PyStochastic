import numpy as np
from plotly import graph_objects as go

class BinomialTree:
    def __init__(self, initial, up, down, T, steps):

        if not isinstance(initial, (int, np.integer, float, np.floating)) or not np.isfinite(initial) or np.isnan(initial):
            raise ValueError("The initial value must be a finite real number.")

        if not isinstance(up, (int, np.integer, float, np.floating)) or not 0 <= up or not np.isfinite(up) or np.isnan(up):
            raise ValueError("The up rate must be a finite real number.")

        if not isinstance(down, (int, np.integer, float, np.floating)) or not 0 <= down or not np.isfinite(down) or np.isnan(down):
            raise ValueError("The down rate must be a finite real number.")

        if not isinstance(T, (int, np.integer, float, np.floating)) or not T > 0 or not np.isfinite(T) or np.isnan(T):
            raise ValueError("The time horizon must be a finite real number.")

        if not isinstance(steps, (int, np.integer)) or not steps > 0 or not np.isfinite(steps) or np.isnan(steps):
            raise ValueError("The number of steps must be an integer.")

        if down > up:
            raise ValueError("The down rate must be greater than the up rate.")

        self.initial = initial
        self.up = up
        self.down = down
        self.T = T
        self.steps = steps
        self.dt = self.T / self.steps
        self.t = np.linspace(0, self.T, self.steps + 1)

    def simulate(self):

        tree = np.zeros((self.steps + 1, self.steps + 1))

        for i in range(self.steps + 1):
            j = np.arange(i + 1)
            tree[i, :i + 1] = (self.initial* (self.up) ** (i - j)* (self.down) ** j)

        return tree