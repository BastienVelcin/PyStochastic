"""Tests for pystochastic.montecarlo (MonteCarlo, MonteCarloProcess)."""

import numpy as np
import pytest
from scipy import stats

from pystochastic.montecarlo.montecarlo import MonteCarlo, MonteCarloProcess
from pystochastic.random import continuous
from pystochastic.processes.diffusion.vasicek import Vasicek


# ======================================================================
# MonteCarlo -- construction / shapes
# ======================================================================

class TestMonteCarloConstruction:
    def test_1d_input_becomes_single_pool(self):
        samples = continuous.normal(0, 1, 1000)
        mc = MonteCarlo(samples)
        assert mc.n_simulations == 1
        assert mc.n_pool_values == 1000
        assert mc.samples.shape == (1, 1000, 1)

    def test_2d_input_multiple_pools(self):
        samples = continuous.normal(0, 1, 5 * 200).reshape(5, 200)
        mc = MonteCarlo(samples)
        assert mc.n_simulations == 5
        assert mc.n_pool_values == 200


# ======================================================================
# MonteCarlo -- estimators & moments
# ======================================================================

class TestEstimators:

    def test_estimate_default_n(self):
        """
        REGRESSION TEST -- estimate() currently defaults n to
        self.n_simulations (number of pools) instead of self.n_pool_values
        (samples per pool). For a single pool this makes estimate() average
        over essentially one value instead of all available samples.

        Fix in MonteCarlo.estimate:
            if n is None: n = self.n_simulations   -->   n = self.n_pool_values
        """
        np.random.seed(0)
        samples = continuous.normal(0, 1, 20000)
        mc = MonteCarlo(samples)
        assert mc.estimate() == pytest.approx(0, abs=0.05)

    def test_estimate_explicit_n(self):
        np.random.seed(0)
        samples = continuous.normal(5, 2, 20000)
        mc = MonteCarlo(samples)
        assert mc.estimate(n=20000) == pytest.approx(5, abs=0.1)

    def test_variance_matches_gamma_theory(self):
        np.random.seed(0)
        samples = continuous.gamma(3, 2, 20000)  # shape=3, rate=2 -> mean=1.5, var=0.75
        mc = MonteCarlo(samples)
        assert mc.estimate(n=20000) == pytest.approx(1.5, abs=0.05)
        assert mc.variance(n=20000) == pytest.approx(0.75, abs=0.05)

    def test_std_is_sqrt_variance(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 10000)
        mc = MonteCarlo(samples)
        assert mc.std(n=10000) == pytest.approx(np.sqrt(mc.variance(n=10000)))

    def test_standard_error_scales_as_inverse_sqrt_n(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 10000)
        mc = MonteCarlo(samples)
        se_small = mc.standard_error(n=100)
        se_large = mc.standard_error(n=10000)
        assert (se_small / se_large).item() == pytest.approx(10, rel=0.25)

    def test_moment_second_order_matches_variance_plus_mean_squared(self):
        np.random.seed(0)
        samples = continuous.normal(2, 1, 20000)
        mc = MonteCarlo(samples)
        m2 = mc.moment(order=2, n=20000)
        mean = mc.estimate(n=20000)
        var = mc.variance(n=20000, correction=False)
        assert m2.item() == pytest.approx((var + mean ** 2).item(), rel=0.05)


# ======================================================================
# MonteCarlo -- confidence intervals
# ======================================================================

class TestConfidenceInterval:

    def test_true_mean_within_interval(self):
        np.random.seed(0)
        samples = continuous.normal(3, 1, 5000)
        mc = MonteCarlo(samples)
        lo, hi = mc.confidence_interval(n=5000)
        assert lo.item() <= 3 <= hi.item()

    def test_coverage_close_to_confidence_level(self):
        """A 95% CI should contain the true mean ~95% of the time across
        independent repetitions."""
        n_repeats = 200
        covered = 0
        for i in range(n_repeats):
            np.random.seed(i)
            samples = continuous.normal(0, 1, 500)
            mc = MonteCarlo(samples)
            lo, hi = mc.confidence_interval(n=500, confidence=0.95)
            if lo.item() <= 0 <= hi.item():
                covered += 1
        assert covered / n_repeats == pytest.approx(0.95, abs=0.05)

    def test_wider_interval_for_higher_confidence(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 5000)
        mc = MonteCarlo(samples)
        lo90, hi90 = mc.confidence_interval(n=5000, confidence=0.90)
        lo99, hi99 = mc.confidence_interval(n=5000, confidence=0.99)
        assert (hi99 - lo99).item() > (hi90 - lo90).item()


# ======================================================================
# MonteCarlo -- statistical errors
# ======================================================================

class TestStatisticalErrors:

    def test_bias_close_to_zero_for_correct_reference(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 20000)
        mc = MonteCarlo(samples)
        assert mc.bias(reference=0, n=20000).item() == pytest.approx(0, abs=0.05)

    def test_rmse_matches_expected_scale(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 20000)
        mc = MonteCarlo(samples)
        # RMSE of individual N(0,1) draws around the true mean 0 should be close to std=1
        assert mc.rmse(reference=0, n=20000).item() == pytest.approx(1, rel=0.1)


# ======================================================================
# MonteCarlo -- descriptive statistics
# ======================================================================

class TestDescriptiveStatistics:

    def test_quantiles_match_normal_theory(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 50000)
        mc = MonteCarlo(samples)
        assert mc.quantile(0.5, n=50000).item() == pytest.approx(0, abs=0.05)
        assert mc.quantile(0.9, n=50000).item() == pytest.approx(stats.norm.ppf(0.9), abs=0.05)

    def test_min_max_bounds_and_ordering(self):
        np.random.seed(0)
        samples = continuous.uniform(0, 1, 10000)
        mc = MonteCarlo(samples)
        lo, hi, med = mc.min(n=10000).item(), mc.max(n=10000).item(), mc.median(n=10000).item()
        assert 0 <= lo <= med <= hi <= 1

    def test_skewness_of_symmetric_distribution(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 20000)
        mc = MonteCarlo(samples)
        assert mc.skewness(n=20000).item() == pytest.approx(0, abs=0.1)

    def test_kurtosis_normal_reference(self):
        np.random.seed(0)
        samples = continuous.normal(0, 1, 20000)
        mc = MonteCarlo(samples)
        # Raw (unnormalized) 4th standardized moment of N(0,1) is 3
        assert mc.kurtosis(n=20000).item() == pytest.approx(3, abs=0.3)


# ======================================================================
# MonteCarloProcess
# ======================================================================

class TestMonteCarloProcess:

    def test_estimate_matches_theoretical_mean(self):
        np.random.seed(0)
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, t_0=0, t_n=5, steps=100)
        mc = MonteCarloProcess(v, n_simulations=5000)
        est = mc.estimate(t_0=5, function=lambda x: x[:, 0])
        theo = 1.5 + (0 - 1.5) * np.exp(-2 * 5)
        assert est.item() == pytest.approx(theo, abs=0.05)

    def test_values_at_shape(self):
        np.random.seed(0)
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, t_0=0, t_n=5, steps=100)
        mc = MonteCarloProcess(v, n_simulations=500)
        vals = mc.values_at(t_0=5)
        assert vals.shape == (500,)

    def test_mean_path_matches_theory(self):
        np.random.seed(0)
        v = Vasicek(speed=2, mean=1.5, volatility=0.3, initial=0, t_0=0, t_n=5, steps=100)
        mc = MonteCarloProcess(v, n_simulations=3000)
        path = mc.mean_path(plot_sim=False)
        theo_path = 1.5 + (0 - 1.5) * np.exp(-2 * v.t)
        assert np.max(np.abs(path[:, 0] - theo_path)) < 0.1

