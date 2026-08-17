import numpy as np
import plotly.graph_objects as go

from pystochastic.processes import *
from pystochastic.pyrandom.setseed import seed
from pystochastic.montecarlo.montecarlo import MonteCarloProcess

OU = OrnsteinUhlenbeck()
G = GeometricBrownianMotion()
V = Vasicek()

processes = [OU, G, V]

n_simulations = 1000
n_seed = 42
n_steps = np.array([
    100,
    200,
    400,
    800,
    1600,
    3200,
    6400,
    12800
])

def difference():

    """
    Difference function

    Compute and plot the difference between the simulated values of the process and the exact ones,
    with the same brownian increments.
    """

    for process in processes:
        seed(n_seed)
        solverM = process.simulate(
            n_simulations=n_simulations,
            method="milstein"
        )

        seed(n_seed)
        solverExact = process.simulate(
            n_simulations=n_simulations,
            method="exact"
        )

        mean_diff = np.mean(
            solverM - solverExact,
            axis=0
        )
        print(type(process).__name__)
        print("M:", solverM.shape)
        print("Exact:", solverExact.shape)
        print("t:", process.t.shape)
        print("diff:", mean_diff.shape)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=process.t,
                y=mean_diff[:,0],
                mode="lines",
                name="M - Exact"
            )
        )

        fig.update_layout(
            title=f"{type(process).__name__} — Milstein vs Exact",
            xaxis_title="t",
            yaxis_title="Mean difference"
        )

        fig.show()

def rmse(process):

    """
    RMSE procedure

    Compute the root-mean-square error between the simulated values of the process and the
    exact ones, for different values of the number of steps. The function also compute an
    estimation of the strong convergence order.

    Returns
    -------
    float :
        Estimation of the strong convergence order based on the selected process.
    """

    rmse_vect = np.zeros(n_steps.size)

    for i, steps in enumerate(n_steps):

        steps = int(steps)
        process.n_steps = steps

        # We need to update the time grid at every different value of "n_steps"
        process.t = np.linspace(
            process.t_0,
            process.t_n,
            steps + 1
        )
        process.dt = (process.t[-1] - process.t[0]) / steps

        seed(n_seed)
        solverM = process.simulate(
            n_simulations=n_simulations,
            method="milstein"
        )

        seed(n_seed)
        solverExact = process.simulate(
            n_simulations=n_simulations,
            method="exact"
        )

        rmse_vect[i] = np.sqrt(
            np.mean(
                (solverM[:, -1] - solverExact[:, -1]) ** 2
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_steps,
            y=rmse_vect,
            mode="lines+markers",
            name="RMSE"
        )
    )

    fig.update_layout(
        title=f"{type(process).__name__} — RMSE",
        xaxis_title="N",
        yaxis_title="RMSE"
    )

    fig.show()

    fig.show()
    p = np.polyfit(
        np.log(n_steps),
        np.log(rmse_vect),
        1
    )[0]
    print(f"Estimated strong convergence order with the process {process}: {-p}")
    return -p


def rmse_all():

    """
    RMSE all function

    Compute the root-mean-square error between the simulated values and the exact ones,
    for all implemented processes, and for different values of the number of steps.
    The function also compute an estimation of the strong convergence order.

    Returns
    -------
    float :
        Estimation of the strong convergence.
    """

    all_p = []
    for process in processes:
        all_p.append(rmse(process))
    return all_p


def strong_convergence_order():

    """
    Strong convergence order procedure

    Print the estimation of the strong convergence order.
    """

    print(f"Strong convergence order: {np.mean(rmse_all())}")



def weak_error(
    process,
    n_steps=[100, 500, 1000, 2000, 3000, 4000, 5000],
    n_simulations=100000
):

    n_steps = np.array(n_steps, dtype=int)

    t = process.t_n
    exact_mean = process.mean(t)

    approx_mean = np.zeros(n_steps.size)

    for i, steps in enumerate(n_steps):

        process.n_steps = int(steps)

        process.t = np.linspace(
            process.t_0,
            process.t_n,
            steps + 1
        )

        process.dt = (
            process.t[-1] - process.t[0]
        ) / steps
        mc = MonteCarloProcess(
            process,
            n_simulations=n_simulations,
            method="milstein"
        )

        mean = mc.estimate(
            function=lambda x: x[:, 0],
            t_0=t,
            n=n_simulations
        )

        approx_mean[i] = mean.item()

        print(
            f"N={steps:5d} | "
            f"mean={approx_mean[i]:.8e} | "
            f"error={abs(approx_mean[i] - exact_mean)}"
        )

    weak_err = np.abs(approx_mean - exact_mean)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_steps, y=weak_err, mode="lines+markers", name="Weak error"))
    fig.update_layout(
        title=f"{type(process).__name__} — Weak error (Milstein)",
        xaxis_title="N", yaxis_title="|E[f(X_T)] estime - theorique|",
        xaxis_type="log", yaxis_type="log",
    )
    fig.show()

    p = np.polyfit(np.log(n_steps), np.log(weak_err), 1)[0]
    print(f"Estimated weak order: {-p}")
    return weak_err