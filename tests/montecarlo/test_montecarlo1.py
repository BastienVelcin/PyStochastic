"""
Tests for pystochastic.montecarlo.
"""

import numpy as np
import pytest

from pystochastic.montecarlo.montecarlo import MonteCarlo, MonteCarloProcess
from pystochastic.random import crandom
from pystochastic.processes.diffusion.vasicek import Vasicek


# ======================================================================
# Helpers
# ======================================================================

def make_normal_samples(
    n_simulations=5,
    n_pool_values=1000,
    dim=1,
    mean=0.0,
    sigma=1.0,
):
    samples = crandom.normal(
        mean,
        sigma,
        n_simulations * n_pool_values * dim,
    )

    return np.asarray(samples).reshape(
        n_simulations,
        n_pool_values,
        dim,
    )


# ======================================================================
# Construction
# ======================================================================

class TestMonteCarloConstruction:

    def test_1d_input(self):
        samples = crandom.normal(0, 1, 1000)

        mc = MonteCarlo(samples)

        assert mc.n_simulations == 1
        assert mc.n_pool_values == 1000
        assert mc.dim == 1
        assert mc.samples.shape == (1, 1000, 1)

    def test_2d_input(self):
        samples = crandom.normal(0, 1, 5 * 200).reshape(5, 200)

        mc = MonteCarlo(samples)

        assert mc.n_simulations == 5
        assert mc.n_pool_values == 200
        assert mc.dim == 1
        assert mc.samples.shape == (5, 200, 1)

    def test_3d_input(self):
        samples = make_normal_samples(
            n_simulations=4,
            n_pool_values=100,
            dim=3,
        )

        mc = MonteCarlo(samples)

        assert mc.n_simulations == 4
        assert mc.n_pool_values == 100
        assert mc.dim == 3
        assert mc.samples.shape == (4, 100, 3)


# ======================================================================
# Estimate
# ======================================================================

class TestEstimate:

    def test_estimate_uses_number_of_samples_per_pool(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[10.0], [20.0], [30.0]],
            ]
        )

        mc = MonteCarlo(samples)

        result = mc.estimate(n=2)

        expected = np.array(
            [
                [1.5],
                [15.0],
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_estimate_default_uses_all_samples(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[10.0], [20.0], [30.0]],
            ]
        )

        mc = MonteCarlo(samples)

        result = mc.estimate()

        expected = np.array(
            [
                [2.0],
                [20.0],
            ]
        )

        np.testing.assert_allclose(result, expected)

    def test_estimate_function(self):
        samples = make_normal_samples(
            n_simulations=3,
            n_pool_values=100,
        )

        mc = MonteCarlo(samples)

        result = mc.estimate(
            n=50,
            function=lambda x: x**2,
        )

        expected = np.mean(
            samples[:, :50] ** 2,
            axis=1,
        )

        np.testing.assert_allclose(result, expected)


# ======================================================================
# Moments
# ======================================================================

class TestMoments:

    def test_variance_is_unbiased(self):
        samples = np.array(
            [
                [[1.0], [2.0], [4.0], [8.0]],
            ]
        )

        mc = MonteCarlo(samples)

        expected = np.var(
            samples[0, :, 0],
            ddof=1,
        )

        result = mc.variance()

        np.testing.assert_allclose(result, expected)

    def test_std_is_square_root_of_variance(self):
        samples = make_normal_samples(
            n_simulations=2,
            n_pool_values=1000,
        )

        mc = MonteCarlo(samples)

        np.testing.assert_allclose(
            mc.std(),
            np.sqrt(mc.variance()),
        )

    def test_second_moment(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
            ]
        )

        mc = MonteCarlo(samples)

        expected = np.mean(
            samples**2,
            axis=1,
        )

        np.testing.assert_allclose(
            mc.moment(order=2),
            expected,
        )

    def test_variance_without_correction(self):
        samples = np.array(
            [
                [[1.0], [2.0], [4.0], [8.0]],
            ]
        )

        mc = MonteCarlo(samples)

        expected = np.var(
            samples[0, :, 0],
            ddof=0,
        )

        result = mc.variance(correction=False)

        np.testing.assert_allclose(
            result,
            expected,
        )

    def test_standard_error(self):
        samples = np.array(
            [
                [[1.0], [2.0], [4.0], [8.0]],
            ]
        )

        mc = MonteCarlo(samples)

        expected = (
            np.std(samples[0, :, 0], ddof=1)
            / np.sqrt(4)
        )

        np.testing.assert_allclose(
            mc.standard_error(),
            expected,
        )


# ======================================================================
# Confidence intervals
# ======================================================================

class TestConfidenceIntervals:

    def test_confidence_interval_contains_estimate(self):
        samples = make_normal_samples(
            n_simulations=5,
            n_pool_values=5000,
        )

        mc = MonteCarlo(samples)

        estimate = mc.estimate()

        lower, upper = mc.confidence_interval(
            confidence=0.95,
            type="student",
        )

        assert np.all(lower <= estimate)
        assert np.all(estimate <= upper)

    def test_higher_confidence_gives_wider_interval(self):
        samples = make_normal_samples(
            n_simulations=1,
            n_pool_values=5000,
        )

        mc = MonteCarlo(samples)

        lower_90, upper_90 = mc.confidence_interval(
            confidence=0.90,
            type="student",
        )

        lower_99, upper_99 = mc.confidence_interval(
            confidence=0.99,
            type="student",
        )

        assert np.all(
            (upper_99 - lower_99)
            > (upper_90 - lower_90)
        )

    def test_normal_and_student_are_close_for_large_n(self):
        samples = make_normal_samples(
            n_simulations=1,
            n_pool_values=10000,
        )

        mc = MonteCarlo(samples)

        normal = mc.confidence_interval(
            confidence=0.95,
            type="normal",
        )

        student = mc.confidence_interval(
            confidence=0.95,
            type="student",
        )

        np.testing.assert_allclose(
            normal,
            student,
            rtol=1e-2,
        )


# ======================================================================
# Quantiles and descriptive statistics
# ======================================================================

class TestDescriptiveStatistics:

    def test_quantile(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
            ]
        )

        mc = MonteCarlo(samples)

        np.testing.assert_allclose(
            mc.quantile(0.5),
            [[2.5]],
        )

    def test_min(self):
        samples = np.array(
            [
                [[3.0], [1.0], [4.0]],
                [[8.0], [2.0], [7.0]],
            ]
        )

        mc = MonteCarlo(samples)

        np.testing.assert_allclose(
            mc.min(),
            [[1.0], [2.0]],
        )

    def test_max(self):
        samples = np.array(
            [
                [[3.0], [1.0], [4.0]],
                [[8.0], [2.0], [7.0]],
            ]
        )

        mc = MonteCarlo(samples)

        np.testing.assert_allclose(
            mc.max(),
            [[4.0], [8.0]],
        )

    def test_median(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
            ]
        )

        mc = MonteCarlo(samples)

        np.testing.assert_allclose(
            mc.median(),
            [[2.5]],
        )


# ======================================================================
# Statistical errors
# ======================================================================

class TestErrors:

    def test_bias_for_exact_reference(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0]],
            ]
        )

        mc = MonteCarlo(samples)

        result = mc.bias(
            reference=2.0,
        )

        np.testing.assert_allclose(
            result,
            [[0.0]],
        )

    def test_rmse(self):
        samples = np.array(
            [
                [[1.0], [2.0], [3.0]],
            ]
        )

        mc = MonteCarlo(samples)

        expected = np.sqrt(
            np.mean(
                (samples[0, :, 0] - 2.0) ** 2
            )
        )

        np.testing.assert_allclose(
            mc.rmse(reference=2.0),
            expected,
        )


# ======================================================================
# Histogram / ECDF / confidence curve
# ======================================================================

class TestPlots:

    def test_histogram_shape(self):
        samples = make_normal_samples(
            n_simulations=4,
            n_pool_values=200,
        )

        mc = MonteCarlo(samples)

        histogram = mc.histogram(
            n=100,
            bins=10,
            plot=False,
        )

        assert histogram.shape == (4, 10)

    def test_ecdf_runs(self):
        samples = make_normal_samples(
            n_simulations=3,
            n_pool_values=100,
        )

        mc = MonteCarlo(samples)

        mc.ecdf(n=100)

    def test_confidence_curve_runs(self):
        samples = make_normal_samples(
            n_simulations=3,
            n_pool_values=100,
        )

        mc = MonteCarlo(samples)

        mc.confidence_curve(
            n=50,
            n_pool=0,
        )


# ======================================================================
# Monte Carlo convergence
# ======================================================================

class TestMonteCarloConvergence:

    def test_rmse_decreases_with_sample_count(self):
        samples = make_normal_samples(
            n_simulations=200,
            n_pool_values=1000,
        )

        mc = MonteCarlo(samples)

        rmse_small = np.sqrt(
            np.mean(
                mc.estimate(n=50) ** 2
            )
        )

        rmse_large = np.sqrt(
            np.mean(
                mc.estimate(n=1000) ** 2
            )
        )

        assert rmse_large < rmse_small

    def test_monte_carlo_order_is_close_to_half(self):
        samples = make_normal_samples(
            n_simulations=500,
            n_pool_values=5000,
        )

        mc = MonteCarlo(samples)

        n_values = np.array(
            [50, 100, 200, 500, 1000, 2000]
        )

        errors = []

        for n in n_values:
            estimate = mc.estimate(n=n)

            errors.append(
                np.sqrt(
                    np.mean(estimate**2)
                )
            )

        errors = np.asarray(errors)

        order = np.polyfit(
            np.log(n_values),
            np.log(errors),
            1,
        )[0]

        assert order == pytest.approx(
            -0.5,
            abs=0.15,
        )


# ======================================================================
# Monte Carlo Process
# ======================================================================

class TestMonteCarloProcess:

    @pytest.fixture
    def vasicek(self):
        return Vasicek(
            speed=2,
            mean=1.5,
            volatility=0.3,
            initial=0,
            t_0=0,
            t_n=5,
            steps=100,
        )

    def test_values_at_shape(self, vasicek):
        mc = MonteCarloProcess(
            vasicek,
            n_simulations=500,
        )

        values = mc.values_at(t_0=5)

        assert values.shape == (500,)

    def test_estimate_matches_theoretical_mean(self, vasicek):
        mc = MonteCarloProcess(
            vasicek,
            n_simulations=5000,
        )

        estimate = mc.estimate(
            t_0=5,
            function=lambda x: x[:, 0],
        )

        theoretical_mean = (
            1.5
            + (0 - 1.5)
            * np.exp(-2 * 5)
        )

        assert estimate.item() == pytest.approx(
            theoretical_mean,
            abs=0.05,
        )

    def test_mean_path(self, vasicek):
        mc = MonteCarloProcess(
            vasicek,
            n_simulations=3000,
        )

        path = mc.mean_path(
            plot_sim=False,
        )

        theoretical_path = (
            1.5
            + (0 - 1.5)
            * np.exp(-2 * vasicek.t)
        )

        assert np.max(
            np.abs(
                path[:, 0]
                - theoretical_path
            )
        ) < 0.1