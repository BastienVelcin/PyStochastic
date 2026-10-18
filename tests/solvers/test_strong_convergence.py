from pystochastic.sde import EulerMaruyama, Milstein, RungeKutta
from pystochastic.processes.elementary import Brownian

import numpy as np
import plotly.graph_objects as go


T = 1.0
steps = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800])
n_simulations = 1000

solvers = [EulerMaruyama, Milstein, RungeKutta]

strong_orders = {
    EulerMaruyama: 0.5,
    Milstein: 1.0,
    RungeKutta: 1.0,  # À confirmer selon ton schéma exact
}


def unidim_convergence_gbm(T, steps, n_simulations, solver,
                           dW, W_T):
    mu = 1.0
    sigma = 1.0
    s0 = 1.0

    # Solution exacte au temps T, pour les mêmes trajectoires
    exact_final_S = s0 * np.exp(
        (mu - sigma**2 / 2) * T + sigma * W_T
    )

    solver_instance = solver(
        drift=lambda x, t=None: mu * x,
        diffusion=lambda x, t=None: sigma * x,
        initial=s0,
        T=T,
        steps=steps,
    )

    paths = solver_instance.solve(
        n_simulations=n_simulations,
        plot=False,
        brownian_increments=dW,
    )

    numerical_final_S = paths[:, -1, :]

    return np.mean(np.abs(numerical_final_S - exact_final_S))


# Generate the Brownian increments only once, on the finest grid.
max_steps = int(np.max(steps))

W = Brownian(1, T, max_steps)
W.simulate(n_simulations)

dW_fine = np.asarray(W.increments)

# Adapt to the expected shape: (simulations, steps, dimension).
if dW_fine.shape != (n_simulations, max_steps, 1):
    raise ValueError(
        f"Unexpected Brownian increments shape: {dW_fine.shape}"
    )

# The terminal Brownian value is shared by every discretization.
W_T = dW_fine.sum(axis=1)

# Compute the error separately for each solver.
for solver in solvers:
    errors = []

    for n_steps in steps:
        if max_steps % n_steps != 0:
            raise ValueError(
                "Each step count must divide max_steps."
            )

        factor = max_steps // n_steps

        # Aggregate fine increments into coarse increments.
        dW = dW_fine.reshape(
            n_simulations, n_steps, factor, 1
        ).sum(axis=2)

        error = unidim_convergence_gbm(
            T=T,
            steps=int(n_steps),
            n_simulations=n_simulations,
            solver=solver,
            dW=dW,
            W_T=W_T,
        )

        errors.append(error)

    errors = np.asarray(errors)

    # Reference slope, normalized at the first data point.
    order = strong_orders[solver]
    reference = errors[0] * (steps[0] / steps) ** order

    # Empirical order: error ~ N^(-p).
    slope = np.polyfit(
        np.log(steps),
        np.log(errors),
        1,
    )[0]

    print(
        f"{solver.__name__}: "
        f"estimated strong order = {-slope:.3f}"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=steps,
        y=errors,
        mode="lines+markers",
        name="Strong error",
    ))

    fig.add_trace(go.Scatter(
        x=steps,
        y=reference,
        mode="lines",
        name=f"Reference O(N^-{order})",
    ))

    fig.update_layout(
        title=f"{solver.__name__} strong convergence",
        xaxis_title="Number of steps N",
        yaxis_title="Strong error",
        xaxis_type="log",
        yaxis_type="log",
    )

    fig.show()
