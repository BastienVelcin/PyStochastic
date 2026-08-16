import numpy as np
import plotly.graph_objects as go

from pystochastic.processes import *
from pystochastic.pyrandom.setseed import seed


OU = OrnsteinUhlenbeck()
G = GeometricBrownianMotion()
V = Vasicek()
C = CIR()

processes = [OU, G, V, C]

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
        solverEM = process.simulate(
            n_simulations=n_simulations,
            method="euler-maruyama"
        )

        seed(n_seed)
        solverExact = process.simulate(
            n_simulations=n_simulations,
            method="exact"
        )

        mean_diff = np.mean(
            solverEM - solverExact,
            axis=0
        )
        print(type(process).__name__)
        print("EM:", solverEM.shape)
        print("Exact:", solverExact.shape)
        print("t:", process.t.shape)
        print("diff:", mean_diff.shape)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=process.t,
                y=mean_diff[:,0],
                mode="lines",
                name="EM - Exact"
            )
        )

        fig.update_layout(
            title=f"{type(process).__name__} — Euler-Maruyama vs Exact",
            xaxis_title="t",
            yaxis_title="Mean difference"
        )

        fig.show()

def rmse(process):
    rmse_vect = np.zeros(n_steps.size)

    for i, steps in enumerate(n_steps):

        steps = int(steps)
        process.n_steps = steps

        # Important : mettre à jour la grille temporelle
        process.t = np.linspace(
            process.t_0,
            process.t_n,
            steps + 1
        )

        seed(n_seed)
        solverEM = process.simulate(
            n_simulations=n_simulations,
            method="euler-maruyama"
        )

        seed(n_seed)
        solverExact = process.simulate(
            n_simulations=n_simulations,
            method="exact"
        )

        rmse_vect[i] = np.sqrt(
            np.mean(
                (solverEM[:, -1] - solverExact[:, -1]) ** 2
            )
        )
        print(f"RMSE for {steps} steps")
        print(
            f"N={steps}, "
            f"n_steps={process.n_steps}, "
            f"len(t)={len(process.t)}, "
            f"EM shape={solverEM.shape}, "
            f"Exact shape={solverExact.shape}, "
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
    p = np.polyfit(
            np.log(n_steps),
            np.log(rmse_vect),
            1
        )[0]
    print(f"Estimated order for: {-p}")

def rmse_all():
    for process in processes:
        rmse(process, 1)