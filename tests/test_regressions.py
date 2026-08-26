"""
Regression tests for the bugs found and fixed during the last full review:
  - dt (and derived quantities) becoming a live property instead of a
    value frozen at construction time.
  - RungeKutta's Milstein-style correction term being silently discarded
    (missing line continuation).
  - BrownianBridge using the wrong closed-form formula.
  - quadratic_variation missing the last increment (off-by-one).
  - hitting_norm_time incorrectly restricted to 1D processes.

Each test is written against the theoretical/expected behaviour, not against
whatever the code currently does -- so a regression on any of these points
will show up as a failing test.
"""

import numpy as np
import pytest

from pystochastic.processes.diffusion.vasicek import Vasicek
from pystochastic.processes.diffusion.cir import CIR
from pystochastic.processes.elementary.brownian import Brownian
from pystochastic.processes.elementary.brownian_bridge import BrownianBridge
from pystochastic.sde.rungekutta import RungeKutta
from pystochastic.random.setseed import seed


# ======================================================================
# dt / derived attributes stay in sync with T, steps
# ======================================================================

class TestDerivedAttributesStayFresh:

    def test_dt_updates_when_steps_changes(self):
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0,T=1, steps=1000)
        assert v.dt == pytest.approx(0.001)
        v.steps = 50
        assert v.dt == pytest.approx(0.02)

    def test_dt_updates_when_horizon_changes(self):
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0,T=1, steps=100)
        assert v.dt == pytest.approx(0.01)
        v.T = 2
        assert v.dt == pytest.approx(0.02)

    def test_t_grid_updates_when_steps_changes(self):
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, T=1, steps=10)
        assert v.t.size == 11
        v.steps = 20
        assert v.t.size == 21
        assert v.t[-1] == pytest.approx(1.0)

    def test_cir_derived_constants_track_dt(self):
        """nu/factor/c depend on dt; changing steps must refresh all three."""
        c = CIR(speed=2, mean=0.05, volatility=0.1, initial=0.03, T=1, steps=1000)
        factor_before = c.factor
        c.steps = 50
        factor_after = c.factor
        assert factor_before != pytest.approx(factor_after)
        # sanity check against the closed-form formula using the *current* dt
        expected_factor = (4*c.speed*np.exp(-c.speed*c.dt)) / (c.volatility**2*(1-np.exp(-c.speed*c.dt)))
        assert factor_after == pytest.approx(expected_factor)


# ======================================================================
# RungeKutta -- correction term must actually be applied (strong order ~1)
# ======================================================================

class TestRungeKuttaConvergence:

    def test_strong_order_close_to_one_on_gbm(self):
        """
        For a state-dependent (multiplicative) diffusion, plain Euler-Maruyama
        has strong order 0.5; the Runge-Kutta/Milstein-style correction should
        bring this up to strong order ~1. If the correction term were dropped
        (as it silently was before the fix), this test would measure ~0.5.
        """
        mu, sigma, x0 = 0.05, 0.3, 1.0

        def drift(x):
            return mu * x

        def diffusion(x):
            return sigma * x

        n_steps_list = [50, 100, 200, 400, 800]
        n_sim = 300
        rmse_vals = []

        for n_steps in n_steps_list:
            seed(42)
            rk = RungeKutta(drift=drift, diffusion=diffusion, initial=x0, T=1, steps=n_steps)
            Y_rk = rk.solve(n_simulations=n_sim, plot=False)

            seed(42)
            W = Brownian(1,  1, n_steps)
            W.simulate(n_sim)
            W_path = np.cumsum(W.increments, axis=1)[:, :, 0]
            t_arr = np.linspace(0, 1, n_steps + 1)[1:]
            Y_exact = x0 * np.exp((mu - 0.5*sigma**2)*t_arr[None, :] + sigma*W_path)

            rmse_vals.append(np.sqrt(np.mean((Y_rk[:, 1:, 0] - Y_exact) ** 2)))

        order = -np.polyfit(np.log(n_steps_list), np.log(rmse_vals), 1)[0]
        assert order == pytest.approx(1.0, abs=0.25)
        # A wide-but-safe upper margin that plain Euler-Maruyama (order ~0.5)
        # would fail, guarding specifically against the correction term being dropped again.
        assert order > 0.75


# ======================================================================
# BrownianBridge -- correct closed-form (pinned at both ends)
# ======================================================================

class TestBrownianBridge:

    def test_endpoints_are_zero(self):
        seed(0)
        bb = BrownianBridge(dim=1, T=1, steps=200)
        Y = bb.simulate(n_simulations=5000)
        assert Y[:, 0, 0].mean() == pytest.approx(0, abs=1e-9)
        assert np.abs(Y[:, -1, 0]).max() < 1e-9

    def test_interior_variance_matches_t_times_one_minus_t(self):
        """Var(B_t) = t(T - t)/T; at t=0.5, T=1 this is 0.25."""
        seed(0)
        bb = BrownianBridge(dim=1,T=1, steps=200)
        Y = bb.simulate(n_simulations=20000)
        mid = Y.shape[1] // 2
        assert Y[:, mid, 0].var() == pytest.approx(0.25, abs=0.02)


# ======================================================================
# quadratic_variation -- must include every increment
# ======================================================================

class TestQuadraticVariation:

    def test_matches_elapsed_time_for_standard_brownian_motion(self):
        """For standard BM, the quadratic variation on [0, t] equals t."""
        seed(0)
        W = Brownian(variance=np.eye(1), T=1, steps=10)
        W.simulate(n_simulations=200000)
        qv = W.quadratic_variation(t=1.0, mean=True, plot=False)
        assert qv == pytest.approx(1.0, abs=0.05)

    def test_no_systematic_bias_with_few_steps(self):
        """
        With only 10 steps, dropping the last increment (the previous bug)
        causes a systematic ~10% underestimation -- far bigger than Monte
        Carlo noise at n_simulations=200000. This guards against that
        specific off-by-one regression.
        """
        seed(1)
        W = Brownian(variance=np.eye(1), T=1, steps=10)
        W.simulate(n_simulations=200000)
        qv = W.quadratic_variation(t=1.0, mean=True, plot=False)
        assert abs(qv - 1.0) < 0.05


# ======================================================================
# hitting_norm_time -- must actually support multi-dimensional processes
# ======================================================================

class TestHittingNormTime:

    def test_runs_on_multidimensional_process(self):
        seed(0)
        W = Brownian(variance=np.eye(2), T=10, steps=1000)
        W.simulate(n_simulations=200)
        result = W.hitting_norm_time(value=1.0)
        assert result.shape == (200,)

    def test_hitting_time_increases_with_higher_threshold(self):
        """On average, it should take longer to reach a norm of 3 than a norm of 1."""
        seed(0)
        W = Brownian(variance=np.eye(2), T=20, steps=2000)
        W.simulate(n_simulations=2000)

        low = W.hitting_norm_time(value=1.0)
        high = W.hitting_norm_time(value=3.0)

        low_found = low[low != None].astype(float)
        high_found = high[high != None].astype(float)

        assert low_found.mean() < high_found.mean()
