import numpy as np
import pytest

from pystochastic.processes import (
    Brownian,
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    Poisson,
    Vasicek,
    CIR,
)


# ======================================================================
# Brownian motion
# ======================================================================

class TestBrownian:
    """Tests for the Brownian motion class."""

    def test_initialization(self):
        W = Brownian(var=1, t_0=0, t_n=1, n_steps=50)

        assert W.dim == 1
        assert W.var.shape == (1, 1)
        assert np.allclose(W.var, np.array([[1.0]]))
        assert W.t.shape == (51,)
        assert W.path is None
        assert W.increments is None
        assert W.n_simulations is None

    def test_multidimensional_initialization(self):
        W = Brownian(var=np.eye(2), t_0=0, t_n=2, n_steps=20)

        assert W.dim == 2
        assert W.var.shape == (2, 2)
        assert W.t.shape == (21,)

    def test_invalid_covariance_matrix(self):
        with pytest.raises(ValueError):
            Brownian(var=np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_invalid_time_interval(self):
        with pytest.raises(ValueError):
            Brownian(t_0=1, t_n=0)

    def test_simulation_shape(self):
        W = Brownian(var=1, t_0=0, t_n=1, n_steps=50)

        path = W.simulate(n_simulations=10)

        assert path.shape == (10, 51, 1)
        assert W.path.shape == (10, 51, 1)
        assert W.increments.shape == (10, 50, 1)
        assert W.n_simulations == 10

    def test_multidimensional_simulation_shape(self):
        W = Brownian(var=np.eye(2), t_0=0, t_n=1, n_steps=50)

        path = W.simulate(n_simulations=8)

        assert path.shape == (8, 51, 2)
        assert W.increments.shape == (8, 50, 2)

    def test_simulation_starts_at_zero(self):
        W = Brownian(t_n=1, n_steps=50)
        path = W.simulate(n_simulations=20)

        assert np.allclose(path[:, 0, :], 0)

    def test_increments_are_consistent_with_path(self):
        W = Brownian(t_n=1, n_steps=50)
        path = W.simulate(n_simulations=10)

        assert np.allclose(np.diff(path, axis=1), W.increments)

    def test_final_position(self):
        W = Brownian(t_n=1, n_steps=50)
        path = W.simulate(n_simulations=10)

        final_position = W.final_position()

        assert final_position.shape == (10, 1)
        assert np.allclose(final_position, path[:, -1, :])

    def test_max_and_min_in_one_dimension(self):
        W = Brownian(t_n=1, n_steps=50)
        path = W.simulate(n_simulations=1)

        maximum, argmax, time_max = W.max()
        minimum, argmin, time_min = W.min()

        expected_max_index = np.argmax(path)
        expected_min_index = np.argmin(path)

        assert np.allclose(maximum, np.max(path))
        assert argmax == expected_max_index
        assert time_max == W.t[expected_max_index]

        assert np.allclose(minimum, np.min(path))
        assert argmin == expected_min_index
        assert time_min == W.t[expected_min_index]

    def test_max_and_min_are_not_available_in_dimension_two(self):
        W = Brownian(var=np.eye(2), t_n=1, n_steps=20)
        W.simulate()

        with pytest.raises(ValueError):
            W.max()

        with pytest.raises(ValueError):
            W.min()

    def test_max_norm(self):
        W = Brownian(var=np.eye(2), t_n=1, n_steps=20)
        path = W.simulate(n_simulations=1)

        position, argmax, time_max = W.max_norm()

        norms = np.sum(path**2, axis=2)
        expected_index = np.argmax(norms)

        assert np.allclose(position, path[0, expected_index, :])
        assert argmax == expected_index
        assert time_max == W.t[expected_index]

    def test_methods_require_simulation(self):
        W = Brownian()

        with pytest.raises(ValueError):
            W.plot()
        with pytest.raises(ValueError):
            W.final_position()
        with pytest.raises(ValueError):
            W.max()
        with pytest.raises(ValueError):
            W.min()
        with pytest.raises(ValueError):
            W.max_norm()


# ======================================================================
# Poisson process
# ======================================================================

class TestPoisson:
    """Tests for the Poisson process class."""

    def test_initialization(self):
        P = Poisson(intensity=2, t_0=0, t_n=1, n_steps=50)

        assert P.intensity == 2
        assert P.dim == 1
        assert P.t.shape == (51,)
        assert P.path is None
        assert P.n_simulations is None

    def test_simulation_shape(self):
        P = Poisson(intensity=2, t_n=1, n_steps=50)
        path = P.simulate(n_simulations=10)

        assert path.shape == (10, 51)
        assert P.n_simulations == 10

    def test_poisson_paths_are_non_decreasing(self):
        P = Poisson(intensity=2, t_n=1, n_steps=50)
        path = P.simulate(n_simulations=100)

        assert np.all(np.diff(path, axis=1) >= 0)

    def test_poisson_paths_are_integer_valued(self):
        P = Poisson(intensity=2, t_n=1, n_steps=50)
        path = P.simulate(n_simulations=100)

        assert np.all(path == np.floor(path))

    def test_poisson_initial_value(self):
        P = Poisson(intensity=2, t_n=1, n_steps=50)
        path = P.simulate(n_simulations=100)

        assert np.all(path[:, 0] == 0)

    def test_poisson_mean(self):
        intensity = 2
        horizon = 2
        P = Poisson(intensity=intensity, t_0=0, t_n=horizon, n_steps=20)

        path = P.simulate(n_simulations=5000)

        assert np.mean(path[:, -1]) == pytest.approx(
            intensity * horizon, abs=0.15
        )


# ======================================================================
# Geometric Brownian motion
# ======================================================================

class TestGeometricBrownianMotion:
    """Tests for the geometric Brownian motion class."""

    def test_initialization(self):
        S = GeometricBrownianMotion(
            mu=0.1, sigma=0.2, S_0=1, t_0=0, t_n=1, n_steps=50
        )

        assert S.dim == 1
        assert S.mu.shape == (1,)
        assert S.sigma.shape == (1, 1)
        assert S.S_0.shape == (1,)
        assert S.t.shape == (51,)
        assert S.path is None

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            GeometricBrownianMotion(
                mu=[0.1, 0.2], sigma=np.eye(2), S_0=[1]
            )

    def test_euler_maruyama_shape(self):
        S = GeometricBrownianMotion(
            mu=0.1, sigma=0.2, S_0=1, t_n=1, n_steps=50
        )
        path = S.simulate(n_simulations=10, method="euler-maruyama")

        assert path.shape == (10, 51, 1)

    def test_exact_shape(self):
        S = GeometricBrownianMotion(
            mu=0.1, sigma=0.2, S_0=1, t_n=1, n_steps=50
        )
        path = S.simulate(n_simulations=10, method="exact")

        assert path.shape == (10, 51, 1)

    def test_invalid_method(self):
        S = GeometricBrownianMotion()

        with pytest.raises(ValueError):
            S.simulate(method="invalid")

    def test_exact_simulation_starts_at_initial_value(self):
        S = GeometricBrownianMotion(
            mu=0.1, sigma=0.2, S_0=3, t_n=1, n_steps=50
        )
        path = S.simulate(n_simulations=20, method="exact")

        assert np.allclose(path[:, 0, 0], 3)

    def test_gbm_remains_positive(self):
        S = GeometricBrownianMotion(
            mu=0.1, sigma=0.2, S_0=1, t_n=1, n_steps=100
        )
        path = S.simulate(n_simulations=100, method="exact")

        assert np.all(path > 0)


# ======================================================================
# Ornstein-Uhlenbeck process
# ======================================================================

class TestOrnsteinUhlenbeck:
    """Tests for the Ornstein-Uhlenbeck process class."""

    def test_initialization(self):
        R = OrnsteinUhlenbeck(
            mean=1, sigma=0.5, theta=2, r_0=0, t_n=1, n_steps=50
        )

        assert R.dim == 1
        assert R.mean.shape == (1,)
        assert R.sigma.shape == (1, 1)
        assert R.theta.shape == (1, 1)
        assert R.r_0.shape == (1,)
        assert R.t.shape == (51,)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(
                mean=[1, 2],
                sigma=np.eye(2),
                theta=np.eye(2),
                r_0=[0],
            )

    def test_euler_maruyama_shape(self):
        R = OrnsteinUhlenbeck(
            mean=1, sigma=0.5, theta=2, r_0=0, t_n=1, n_steps=50
        )
        path = R.simulate(n_simulations=10, method="euler-maruyama")

        assert path.shape == (10, 51, 1)

    def test_exact_shape(self):
        R = OrnsteinUhlenbeck(
            mean=1, sigma=0.5, theta=2, r_0=0, t_n=1, n_steps=50
        )
        path = R.simulate(n_simulations=10, method="exact")

        assert path.shape == (10, 51, 1)

    def test_exact_method_is_only_available_in_one_dimension(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2],
            sigma=np.ones((2, 2)),
            theta=np.ones((2, 2)),
            r_0=[0, 0],
            t_n=1,
            n_steps=20,
        )

        with pytest.raises(ValueError):
            R.simulate(n_simulations=5, method="exact")

    def test_invalid_method(self):
        R = OrnsteinUhlenbeck()

        with pytest.raises(ValueError):
            R.simulate(method="invalid")


# ======================================================================
# Vasicek process
# ======================================================================

class TestVasicek:
    """Tests for the Vasicek process class."""

    def test_initialization(self):
        R = Vasicek(
            reversion_speed=1,
            mean=0.05,
            volatility=0.1,
            r_0=0.03,
            t_n=1,
            n_steps=50,
        )

        assert R.dim == 1
        assert R.t.shape == (51,)

    def test_simulation_shape(self):
        R = Vasicek(
            reversion_speed=1,
            mean=0.05,
            volatility=0.1,
            r_0=0.03,
            t_n=1,
            n_steps=50,
        )

        path = R.simulate(n_simulations=10)

        assert path.shape == (10, 51, 1)


# ======================================================================
# CIR process
# ======================================================================

class TestCIR:
    """Tests for the Cox-Ingersoll-Ross process class."""

    def test_initialization(self):
        R = CIR(
            a=1, b=0.05, sigma=0.1, r_0=0.03, t_n=1, n_steps=50
        )

        assert R.dim == 1
        assert R.t.shape == (51,)

    def test_exact_simulation_shape(self):
        R = CIR(
            a=1, b=0.05, sigma=0.1, r_0=0.03, t_n=1, n_steps=50
        )
        path = R.simulate(n_simulations=10, method="exact")

        assert path.shape == (10, 51, 1)

    def test_euler_maruyama_simulation_shape(self):
        R = CIR(
            a=1, b=0.05, sigma=0.1, r_0=0.03, t_n=1, n_steps=50
        )
        path = R.simulate(n_simulations=10, method="euler-maruyama")

        assert path.shape == (10, 51, 1)

    def test_cir_paths_are_non_negative(self):
        R = CIR(
            a=1, b=0.05, sigma=0.1, r_0=0.03, t_n=1, n_steps=100
        )
        path = R.simulate(n_simulations=100, method="exact")

        assert np.all(path >= 0)

    def test_invalid_method(self):
        R = CIR()

        with pytest.raises(ValueError):
            R.simulate(method="invalid")