import numpy as np
import plotly.graph_objects as go

from pystochastic.processes import *
from pystochastic.pyrandom.setseed import seed

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
        process.dt = (process.t[-1] - process.t[1]) / steps

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
    all_p = []
    for process in processes:
        all_p.append(rmse(process))
    return all_p


def strong_convergence_order():
    print(f"Strong convergence order: {np.mean(rmse_all())}")


n_steps = 1000
n_sim = 100

def approx_mean_diff(process):

    diff = np.zeros(n_steps)
    for i in range(n_sim):

        process.n_steps = n_steps

        process.t = np.linspace(
            process.t_0,
            process.t_n,
            n_steps + 1
        )

        process.dt = (process.t[-1] - process.t[1]) / n_steps
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

        diff[i] = np.mean(
            solverM - solverExact
        )

    return np.mean(diff)