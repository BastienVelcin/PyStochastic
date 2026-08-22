import numpy as np
import plotly.graph_objects as go

from pystochastic.processes import *
from pystochastic.random.setseed import seed
from pystochastic.montecarlo.montecarlo import MonteCarloProcess

# Since the exact solution of the CIR don't depend of a Brownian motion,
# but depend of a Non-central Khi Squared law, it don't make sense to
# compute the strong convergence order with it.

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

        # We need to update the time grid at every different value of "n_steps"
        process.t = np.linspace(
            process.t_0,
            process.t_n,
            steps + 1
        )
        process.dt = (process.t[-1]-process.t[0])/steps

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
    print(f"Estimated strong convergence order with the process {process}: {-p}")
    return -p

def rmse_all():
    all_p = []
    for process in processes:
        all_p.append(rmse(process))
    return all_p

def strong_convergence_order():
    print(f"Strong convergence order: {np.mean(rmse_all())}")

def monte_carlo_convergence(
    process,
    n_simulations=np.array([100, 1000, 10_000, 100_000]),
    n_experiments=30,
):
    """
    Test the Monte Carlo convergence rate.

    The theoretical Monte Carlo error is expected to behave as

        RMSE ~ N^(-1/2)

    where N is the number of simulations.
    """

    n_simulations = np.asarray(n_simulations, dtype=int)

    # Exact theoretical expectation
    t = process.t_n
    exact_mean = np.asarray(process.mean(t)).item()

    mc_rmse = np.zeros(n_simulations.size)

    for i, n in enumerate(n_simulations):

        errors = np.zeros(n_experiments)

        for j in range(n_experiments):

            # Different Monte Carlo experiment each time
            seed(None)

            mc = MonteCarloProcess(
                process,
                n_simulations=int(n),
                method="euler-maruyama"
            )

            mean = mc.estimate(
                function=lambda x: x,
                t_0=t,
                n=int(n)
            )

            mean = np.asarray(mean).item()

            errors[j] = mean - exact_mean

        # RMSE over the independent Monte Carlo experiments
        mc_rmse[i] = np.sqrt(
            np.mean(errors ** 2)
        )

        print(
            f"N = {n:7d} | "
            f"MC RMSE = {mc_rmse[i]:.8e}"
        )

    # Estimate convergence order
    p = np.polyfit(
        np.log(n_simulations),
        np.log(mc_rmse),
        1
    )[0]

    estimated_order = -p

    print()
    print(
        f"Estimated Monte Carlo order: "
        f"{estimated_order:.4f}"
    )

    # Theoretical N^(-1/2) reference
    reference = (
        mc_rmse[0]
        * (n_simulations / n_simulations[0]) ** (-0.5)
    )

    # Plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_simulations,
            y=mc_rmse,
            mode="lines+markers",
            name="Monte Carlo RMSE"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=n_simulations,
            y=reference,
            mode="lines",
            name=r"$N^{-1/2}$ reference"
        )
    )

    fig.update_layout(
        title=f"{type(process).__name__} — Monte Carlo convergence",
        xaxis_title="Number of simulations",
        yaxis_title="Monte Carlo RMSE",
        xaxis_type="log",
        yaxis_type="log",
    )

    fig.show()

    return mc_rmse, estimated_order

def weak_error(process, n_steps=(50, 100, 200, 400, 800), n_simulations=50000,
                method="euler-maruyama", function=lambda x: x[:, 0], n_repeats=5,
                base_seed=0):
    """
    Parameters
    ----------
    process : objet de pystochastic.processes (Vasicek, CIR, OrnsteinUhlenbeck, ...)
        Doit exposer .mean(t), la valeur theorique exacte de E[X_t].
    n_steps : sequence d'entiers
        Nombres de pas de discretisation a tester.
    n_simulations : int
        Nombre de trajectoires Monte Carlo par estimation.
    method : str
        Schema approche a tester ("euler-maruyama" ou "milstein").
    function : callable
        Fonction test appliquee au lot entier d'etats finaux, shape
        (n_simulations, dim) -> (n_simulations,). Par defaut : 1ere composante.
    n_repeats : int
        Nombre de repetitions independantes moyennees par valeur de n_steps,
        pour reduire le bruit Monte Carlo de l'estimation d'erreur.
    base_seed : int
        Seed de base ; la repetition k utilise base_seed + k (reproductible).

    Returns
    -------
    n_steps : np.ndarray
    weak_err : np.ndarray
        Erreur faible absolue moyenne (sur n_repeats) pour chaque n_steps.
    order : float
        Ordre de convergence faible estime (pente log-log par moindres carres).
    """
    n_steps = np.asarray(n_steps)
    t = process.t_n
    exact_mean = np.atleast_1d(process.mean(t)).item()

    errors = np.zeros((n_steps.size, n_repeats))

    for i, steps in enumerate(n_steps):
        steps = int(steps)
        process.n_steps = steps
        process.t = np.linspace(process.t_0, process.t_n, steps + 1)
        process.dt = (process.t_n - process.t_0) / steps

        for r in range(n_repeats):
            seed(base_seed + r)
            mc = MonteCarloProcess(process, n_simulations=n_simulations, method=method)
            estimated = mc.estimate(function=function, t_0=t).item()
            errors[i, r] = abs(estimated - exact_mean)

        print(f"n_steps={steps:6d} : erreur moyenne sur {n_repeats} repetitions = "
              f"{errors[i].mean():.6f} (min={errors[i].min():.6f}, max={errors[i].max():.6f})")

    weak_err = errors.mean(axis=1)
    order = -np.polyfit(np.log(n_steps), np.log(weak_err), 1)[0]
    print(f"\nEstimated weak order: {order:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_steps, y=weak_err, mode="lines+markers", name="Weak error"))
    ref = weak_err[0] * (n_steps[0] / n_steps)
    fig.add_trace(go.Scatter(x=n_steps, y=ref, mode="lines", name="Reference O(1/N)",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(
        title=f"{type(process).__name__} -- Weak error ({method}), order~{order:.2f}",
        xaxis_title="N (nombre de pas)",
        yaxis_title="|E[f(X_T)] estime - exact|",
        xaxis_type="log", yaxis_type="log",
        template="plotly_white",
    )
    fig.show()

    return n_steps, weak_err, order