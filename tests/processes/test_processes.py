import numpy as np
import pytest

from pystochastic.processes import (
    Brownian, GeometricBrownianMotion, OrnsteinUhlenbeck,
    PoissonProcess, Vasicek, CIR
)


# ======================================================================
# Brownian
# ======================================================================

class TestBrownian:
    def test_scalar_initialization(self):
        W = Brownian(variance=1, T=1, steps=50)
        assert W.dim == 1
        assert W.variance.shape == (1, 1)
        assert W.t.shape == (51,)
        assert W.path is None
        assert W.increments is None

    def test_multidimensional_initialization(self):
        W = Brownian(variance=np.eye(2), T=1, steps=20)
        assert W.dim == 2
        assert W.variance.shape == (2, 2)

    def test_invalid_covariance(self):
        with pytest.raises(ValueError):
            Brownian(variance=np.array([[1., 2.], [2., 1.]]))

    def test_invalid_time(self):
        with pytest.raises(ValueError):
            Brownian(T=0)

    @pytest.mark.parametrize("n_simulations", [1, 5, 20])
    def test_simulation_shape(self, n_simulations):
        W = Brownian(T=1, steps=50)
        path = W.simulate(n_simulations)
        assert path.shape == (n_simulations, 51, 1)
        assert W.increments.shape == (n_simulations, 50, 1)

    def test_multidimensional_shape(self):
        W = Brownian(variance=np.eye(2), T=1, steps=50)
        path = W.simulate(8)
        assert path.shape == (8, 51, 2)
        assert W.increments.shape == (8, 50, 2)

    def test_initial_value_and_increments(self):
        W = Brownian(T=1, steps=50)
        path = W.simulate(10)
        assert np.allclose(path[:, 0, :], 0)
        assert np.allclose(np.diff(path, axis=1), W.increments)

    def test_final_position(self):
        W = Brownian(T=1, steps=50)
        path = W.simulate(10)
        assert np.allclose(W.final_position(), path[:, -1, :])

    def test_methods_require_simulation(self):
        W = Brownian()
        for method in [W.final_position, W.max, W.min, W.max_norm]:
            with pytest.raises(ValueError):
                method()


# ======================================================================
# Poisson
# ======================================================================

class TestPoisson:
    def test_initialization(self):
        P = PoissonProcess(intensity=2, T=1, steps=50)
        assert P.dim == 1
        assert P.t.shape == (51,)
        assert P.path is None

    def test_shape(self):
        P = PoissonProcess(intensity=2, T=1, steps=50)
        path = P.simulate(10)
        assert path.shape == (10, 51)

    def test_paths_are_non_decreasing_and_integer(self):
        P = PoissonProcess(intensity=2, T=1, steps=50)
        path = P.simulate(100)
        assert np.all(np.diff(path, axis=1) >= 0)
        assert np.all(path == np.floor(path))
        assert np.all(path[:, 0] == 0)

    def test_mean(self):
        P = PoissonProcess(intensity=2, T=2, steps=20)
        path = P.simulate(5000)
        assert np.mean(path[:, -1]) == pytest.approx(4, abs=0.15)


# ======================================================================
# Geometric Brownian motion
# ======================================================================

class TestGeometricBrownianMotion:
    def test_scalar_representation(self):
        S = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, steps=20)
        assert S.dim == 1
        assert S.mu.shape == (1,)
        assert S.volatility.shape == (1,)
        assert S.initial.shape == (1,)
        assert S._diagonal

    def test_vector_representation(self):
        S = GeometricBrownianMotion(
            mu=[.1, .2], volatility=[.2, .3], initial=[1, 2], steps=20
        )
        assert S.dim == 2
        assert S.volatility.shape == (2,)
        assert S._diagonal

    def test_diagonal_matrix_representation(self):
        S = GeometricBrownianMotion(
            mu=[.1, .2], volatility=np.diag([.2, .3]), initial=[1, 2], steps=20
        )
        assert S.volatility.shape == (2,)
        assert S._diagonal

    def test_full_matrix_representation(self):
        volatility = np.array([[.2, .1], [.05, .3]])
        S = GeometricBrownianMotion(
            mu=[.1, .2], volatility=volatility, initial=[1, 2], steps=20
        )
        assert S.volatility.shape == (2, 2)
        assert not S._diagonal

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            GeometricBrownianMotion(mu=[.1, .2], volatility=np.eye(2), initial=[1])

    @pytest.mark.parametrize("method", ["euler-maruyama", "milstein"])
    def test_scalar_methods_shape(self, method):
        S = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, steps=30)
        path = S.simulate(10, method)
        assert path.shape == (10, 31, 1)

    def test_exact_scalar_shape_and_positivity(self):
        S = GeometricBrownianMotion(mu=.1, volatility=.2, initial=1, steps=30)
        path = S.simulate(20, "exact")
        assert path.shape == (20, 31, 1)
        assert np.all(path > 0)

    def test_multidimensional_em(self):
        S = GeometricBrownianMotion(
            mu=[.1, .2], volatility=[.2, .3], initial=[1, 2], steps=30
        )
        path = S.simulate(10, "euler-maruyama")
        assert path.shape == (10, 31, 2)

    def test_full_matrix_em(self):
        S = GeometricBrownianMotion(
            mu=[.1, .2],
            volatility=np.array([[.2, .1], [.05, .3]]),
            initial=[1, 2],
            steps=30
        )
        assert S.diffusion(np.array([2., 3.])).shape == (2,)
        path = S.simulate(5, "euler-maruyama")
        assert path.shape == (5, 31, 2)

    def test_drift_and_diagonal_diffusion(self):
        S = GeometricBrownianMotion(
            mu=[.1, .2], volatility=[.2, .3], initial=[1, 2]
        )
        x = np.array([2., 3.])
        assert np.allclose(S.drift(x), [.2, .6])
        assert np.allclose(S.diffusion(x), [.4, .9])

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            GeometricBrownianMotion().simulate(method="invalid")


# ======================================================================
# Ornstein-Uhlenbeck
# ======================================================================

class TestOrnsteinUhlenbeck:
    def test_scalar_representation(self):
        R = OrnsteinUhlenbeck(mean=1, volatility=.5, speed=2, initial=0, steps=20)
        assert R.dim == 1
        assert R.volatility.shape == (1,)
        assert R.speed.shape == (1,)
        assert R._diagonal

    def test_vector_representation(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=[.5, .7], speed=[2, 3], initial=[0, 1]
        )
        assert R.volatility.shape == (2,)
        assert R.speed.shape == (2,)
        assert R._diagonal

    def test_diagonal_matrix_representation(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=np.diag([.5, .7]),
            speed=np.diag([2, 3]), initial=[0, 1]
        )
        assert R.volatility.shape == (2,)
        assert R.speed.shape == (2,)
        assert R._diagonal

    def test_full_matrix_representation(self):
        speed = np.array([[2., .5], [.2, 3.]])
        volatility = np.array([[.5, .1], [.2, .7]])
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=volatility, speed=speed, initial=[0, 1]
        )
        assert not R._diagonal
        assert np.allclose(R.volatility, volatility)
        assert np.allclose(R.speed, speed)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck(
                mean=[1, 2], volatility=np.eye(2),
                speed=np.eye(2), initial=[0]
            )

    @pytest.mark.parametrize("method", ["euler-maruyama", "milstein"])
    def test_scalar_methods_shape(self, method):
        R = OrnsteinUhlenbeck(
            mean=1, volatility=.5, speed=2, initial=0, steps=30
        )
        path = R.simulate(10, method)
        assert path.shape == (10, 31, 1)

    def test_exact_scalar_shape(self):
        R = OrnsteinUhlenbeck(
            mean=1, volatility=.5, speed=2, initial=0, steps=30
        )
        path = R.simulate(10, "exact")
        assert path.shape == (10, 31, 1)

    def test_multidimensional_em(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=[.5, .7], speed=[2, 3],
            initial=[0, 1], steps=30
        )
        path = R.simulate(10, "euler-maruyama")
        assert path.shape == (10, 31, 2)

    def test_full_matrix_em(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2],
            volatility=np.array([[.5, .1], [.2, .7]]),
            speed=np.array([[2., .5], [.2, 3.]]),
            initial=[0, 1], steps=30
        )
        path = R.simulate(5, "euler-maruyama")
        assert path.shape == (5, 31, 2)

    def test_drift_diagonal(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=[.5, .7], speed=[2, 3], initial=[0, 1]
        )
        x = np.array([.5, 1.5])
        assert np.allclose(R.drift(x), [1., 1.5])

    def test_drift_full_matrix(self):
        speed = np.array([[2., .5], [.2, 3.]])
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=np.eye(2), speed=speed, initial=[0, 1]
        )
        x = np.array([.5, 1.5])
        assert np.allclose(R.drift(x), speed @ (R.mean - x))

    def test_exact_is_1d_only(self):
        R = OrnsteinUhlenbeck(
            mean=[1, 2], volatility=np.eye(2), speed=np.eye(2),
            initial=[0, 1], steps=20
        )
        with pytest.raises(ValueError):
            R.simulate(5, "exact")

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeck().simulate(method="invalid")


# ======================================================================
# Vasicek
# ======================================================================

class TestVasicek:
    def test_scalar_representation(self):
        R = Vasicek(
            speed=1, mean=.05, volatility=.1,
            initial=.03, steps=20
        )
        assert R.dim == 1
        assert R.speed.shape == (1,)
        assert R.volatility.shape == (1,)
        assert R._diagonal

    def test_vector_representation(self):
        R = Vasicek(
            speed=[1, 2], mean=[.05, .1],
            volatility=[.1, .2], initial=[.03, .04]
        )
        assert R.speed.shape == (2,)
        assert R.volatility.shape == (2,)
        assert R._diagonal

    def test_diagonal_matrix_representation(self):
        R = Vasicek(
            speed=np.diag([1, 2]), mean=[.05, .1],
            volatility=np.diag([.1, .2]), initial=[.03, .04]
        )
        assert R.speed.shape == (2,)
        assert R.volatility.shape == (2,)
        assert R._diagonal

    def test_full_matrix_representation(self):
        speed = np.array([[1., .2], [.1, 2.]])
        volatility = np.array([[.1, .02], [.03, .2]])
        R = Vasicek(
            speed=speed, mean=[.05, .1],
            volatility=volatility, initial=[.03, .04]
        )
        assert not R._diagonal
        assert np.allclose(R.speed, speed)
        assert np.allclose(R.volatility, volatility)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            Vasicek(
                speed=np.eye(2), mean=[.05, .1],
                volatility=np.eye(2), initial=[.03]
            )

    @pytest.mark.parametrize("method", ["euler-maruyama", "milstein"])
    def test_scalar_methods_shape(self, method):
        R = Vasicek(
            speed=1, mean=.05, volatility=.1,
            initial=.03, steps=30
        )
        path = R.simulate(10, method)
        assert path.shape == (10, 31, 1)

    def test_exact_scalar_shape(self):
        R = Vasicek(
            speed=1, mean=.05, volatility=.1,
            initial=.03, steps=30
        )
        path = R.simulate(10, "exact")
        assert path.shape == (10, 31, 1)

    def test_multidimensional_em(self):
        R = Vasicek(
            speed=[1, 2], mean=[.05, .1],
            volatility=[.1, .2], initial=[.03, .04], steps=30
        )
        path = R.simulate(10, "euler-maruyama")
        assert path.shape == (10, 31, 2)

    def test_full_matrix_em(self):
        R = Vasicek(
            speed=np.array([[1., .2], [.1, 2.]]),
            mean=[.05, .1],
            volatility=np.array([[.1, .02], [.03, .2]]),
            initial=[.03, .04], steps=30
        )
        path = R.simulate(5, "euler-maruyama")
        assert path.shape == (5, 31, 2)

    def test_drift_diagonal(self):
        R = Vasicek(
            speed=[1, 2], mean=[.05, .1],
            volatility=[.1, .2], initial=[.03, .04]
        )
        x = np.array([.03, .08])
        assert np.allclose(R.drift(x), [.02, .04])

    def test_drift_full_matrix(self):
        speed = np.array([[1., .2], [.1, 2.]])
        R = Vasicek(
            speed=speed, mean=[.05, .1],
            volatility=np.eye(2), initial=[.03, .04]
        )
        x = np.array([.03, .08])
        assert np.allclose(R.drift(x), speed @ (R.mean - x))

    def test_exact_is_1d_only(self):
        R = Vasicek(
            speed=[1, 2], mean=[.05, .1],
            volatility=[.1, .2], initial=[.03, .04], steps=20
        )
        with pytest.raises(ValueError):
            R.simulate(5, "exact")

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Vasicek().simulate(method="invalid")


# ======================================================================
# CIR
# ======================================================================

class TestCIR:
    def test_initialization(self):
        R = CIR(speed=1, mean=.05, volatility=.1, initial=.03, steps=50)
        assert R.dim == 1
        assert R.t.shape == (51,)
        assert R.path is None

    def test_invalid_parameters(self):
        for kwargs in [{"speed": 0}, {"mean": -1}, {"volatility": 0}, {"initial": -1}]:
            with pytest.raises(ValueError):
                CIR(**kwargs)

    @pytest.mark.parametrize("method", ["exact", "euler-maruyama", "milstein"])
    def test_simulation_shape(self, method):
        R = CIR(speed=1, mean=.05, volatility=.1, initial=.03, steps=30)
        path = R.simulate(10, method)
        assert path.shape == (10, 31, 1)

    def test_exact_non_negative(self):
        R = CIR(speed=1, mean=.05, volatility=.1, initial=.03, steps=50)
        path = R.simulate(100, "exact")
        assert np.all(path >= 0)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            CIR().simulate(method="invalid")

    def test_mean_and_variance_at_initial_time(self):
        R = CIR(speed=2, mean=.05, volatility=.1, initial=.03, T=1)
        assert R.expectation(0) == pytest.approx(R.initial)
        assert R.variance(0) == pytest.approx(0.)

    def test_time_validation(self):
        R = CIR(T=1)
        with pytest.raises(ValueError):
            R.expectation(-.1)
        with pytest.raises(ValueError):
            R.variance(1.1)