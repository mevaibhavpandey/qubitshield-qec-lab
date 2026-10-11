"""
test_analytics.py — Tests for qubitshield.analytics

Tests verify:
- Boundary conditions for theoretical failure probabilities
- Mathematical relationship between encoded and unencoded
- Break-even point at p=0.5
- Gain function behavior
- Wilson confidence interval validity
- Array input compatibility
"""

from __future__ import annotations

import numpy as np
import pytest

from qubitshield.analytics import (
    theoretical_unencoded_failure,
    theoretical_encoded_failure,
    code_gain,
    breakeven_point,
    reliability_curve_data,
    wilson_confidence_interval,
)


# ---------------------------------------------------------------------------
# Unencoded failure probability
# ---------------------------------------------------------------------------

class TestUnencodedFailure:
    def test_zero_probability(self):
        assert theoretical_unencoded_failure(0.0) == pytest.approx(0.0)

    def test_one_probability(self):
        assert theoretical_unencoded_failure(1.0) == pytest.approx(1.0)

    def test_half_probability(self):
        assert theoretical_unencoded_failure(0.5) == pytest.approx(0.5)

    def test_is_identity_function(self):
        for p in [0.1, 0.2, 0.37, 0.8]:
            assert theoretical_unencoded_failure(p) == pytest.approx(p)

    def test_array_input(self):
        p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = theoretical_unencoded_failure(p)
        np.testing.assert_array_almost_equal(result, p)

    def test_invalid_negative_raises(self):
        with pytest.raises(ValueError):
            theoretical_unencoded_failure(-0.1)

    def test_invalid_above_one_raises(self):
        with pytest.raises(ValueError):
            theoretical_unencoded_failure(1.1)


# ---------------------------------------------------------------------------
# Encoded failure probability
# ---------------------------------------------------------------------------

class TestEncodedFailure:
    def test_zero_probability(self):
        assert theoretical_encoded_failure(0.0) == pytest.approx(0.0)

    def test_one_probability(self):
        # At p=1, all qubits flip, always a logical failure
        assert theoretical_encoded_failure(1.0) == pytest.approx(1.0)

    def test_half_probability(self):
        # At p=0.5: 3(0.25)(0.5) + 0.125 = 0.375 + 0.125 = 0.5
        assert theoretical_encoded_failure(0.5) == pytest.approx(0.5)

    def test_formula_at_small_p(self):
        p = 0.1
        expected = 3 * p**2 * (1-p) + p**3
        assert theoretical_encoded_failure(p) == pytest.approx(expected)

    def test_formula_at_medium_p(self):
        p = 0.3
        expected = 3 * p**2 * (1-p) + p**3
        assert theoretical_encoded_failure(p) == pytest.approx(expected)

    def test_encoded_less_than_unencoded_below_half(self):
        for p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.49]:
            enc = theoretical_encoded_failure(p)
            unenc = theoretical_unencoded_failure(p)
            assert enc < unenc, f"At p={p}: encoded={enc} should be < unencoded={unenc}"

    def test_encoded_equal_unencoded_at_half(self):
        enc = theoretical_encoded_failure(0.5)
        unenc = theoretical_unencoded_failure(0.5)
        assert enc == pytest.approx(unenc)

    def test_encoded_greater_than_unencoded_above_half(self):
        for p in [0.51, 0.6, 0.7, 0.9, 0.99]:
            enc = theoretical_encoded_failure(p)
            unenc = theoretical_unencoded_failure(p)
            assert enc > unenc, f"At p={p}: encoded={enc} should be > unencoded={unenc}"

    def test_array_input(self):
        p = np.array([0.0, 0.1, 0.5, 1.0])
        result = theoretical_encoded_failure(p)
        expected = np.array([0.0, 3*0.01*0.9 + 0.001, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            theoretical_encoded_failure(-0.01)

    def test_encoded_quadratic_correction(self):
        """Near p=0, encoded failure ≈ 3p² (quadratic, much less than p)."""
        p = 0.01
        enc = theoretical_encoded_failure(p)
        approx_quadratic = 3 * p**2
        # enc should be very close to 3p² for small p
        assert enc == pytest.approx(approx_quadratic, rel=0.01)


# ---------------------------------------------------------------------------
# Code gain
# ---------------------------------------------------------------------------

class TestCodeGain:
    def test_gain_greater_than_one_below_half(self):
        for p in [0.01, 0.1, 0.2, 0.3, 0.4]:
            gain = code_gain(p)
            assert float(gain) > 1.0, f"Gain should > 1 at p={p}, got {gain}"

    def test_gain_approximately_one_at_half(self):
        gain = float(code_gain(0.5))
        assert gain == pytest.approx(1.0, abs=1e-6)

    def test_gain_less_than_one_above_half(self):
        for p in [0.6, 0.7, 0.8, 0.9]:
            gain = float(code_gain(p))
            assert gain < 1.0, f"Gain should < 1 at p={p}, got {gain}"


# ---------------------------------------------------------------------------
# Break-even point
# ---------------------------------------------------------------------------

class TestBreakevenPoint:
    def test_breakeven_is_half(self):
        assert breakeven_point() == 0.5

    def test_breakeven_has_equal_failure_rates(self):
        p = breakeven_point()
        enc = theoretical_encoded_failure(p)
        unenc = theoretical_unencoded_failure(p)
        assert enc == pytest.approx(unenc)


# ---------------------------------------------------------------------------
# Reliability curve data
# ---------------------------------------------------------------------------

class TestReliabilityCurveData:
    def test_returns_expected_keys(self):
        data = reliability_curve_data(n_points=50)
        for key in ["p", "p_unencoded", "p_encoded", "gain", "breakeven"]:
            assert key in data

    def test_p_values_span_zero_to_one(self):
        data = reliability_curve_data(n_points=50)
        assert data["p"][0] == pytest.approx(0.0)
        assert data["p"][-1] == pytest.approx(1.0)

    def test_arrays_same_length(self):
        data = reliability_curve_data(n_points=100)
        assert len(data["p"]) == len(data["p_unencoded"]) == len(data["p_encoded"]) == 100


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_zero_trials_returns_full_range(self):
        lo, hi = wilson_confidence_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_failures_ci_close_to_one(self):
        lo, hi = wilson_confidence_interval(1000, 1000)
        assert lo > 0.99
        assert hi == pytest.approx(1.0)

    def test_no_failures_ci_close_to_zero(self):
        lo, hi = wilson_confidence_interval(0, 1000)
        assert lo == pytest.approx(0.0)
        assert hi < 0.01

    def test_ci_contains_true_rate(self):
        # 500/1000 trials: CI should contain 0.5
        lo, hi = wilson_confidence_interval(500, 1000)
        assert lo < 0.5 < hi

    def test_ci_in_valid_range(self):
        for n_fail, n_trial in [(0, 100), (50, 100), (100, 100), (10, 10_000)]:
            lo, hi = wilson_confidence_interval(n_fail, n_trial)
            assert 0.0 <= lo <= hi <= 1.0

    def test_ci_width_decreases_with_more_trials(self):
        lo1, hi1 = wilson_confidence_interval(100, 1000)
        lo2, hi2 = wilson_confidence_interval(1000, 10_000)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width1 > width2
