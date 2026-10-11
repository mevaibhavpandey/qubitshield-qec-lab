"""
test_experiments.py — Tests for Monte Carlo simulation in qubitshield.noise

Tests verify:
- Single experiment returns valid probabilities
- n_failures is in valid range
- Reproducibility with same seed
- Monte Carlo results approach theory for large n
- Benchmark result structure
- CSV export contains expected columns
- Invalid inputs raise appropriate errors
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from qubitshield.noise import (
    run_single_experiment,
    run_reliability_benchmark,
    experiment_result_to_dataframe,
    benchmark_result_to_dataframe,
)
from qubitshield.analytics import theoretical_encoded_failure, theoretical_unencoded_failure


# ---------------------------------------------------------------------------
# run_single_experiment
# ---------------------------------------------------------------------------

class TestRunSingleExperiment:
    def test_failure_rate_in_valid_range(self):
        result = run_single_experiment(p=0.1, n_trials=1000, seed=42)
        assert 0.0 <= result.simulated_failure_rate <= 1.0

    def test_n_failures_in_range(self):
        result = run_single_experiment(p=0.1, n_trials=1000, seed=42)
        assert 0 <= result.n_failures <= 1000

    def test_zero_error_no_failures_encoded(self):
        result = run_single_experiment(p=0.0, n_trials=1000, seed=42, encoded=True)
        assert result.n_failures == 0
        assert result.simulated_failure_rate == pytest.approx(0.0)

    def test_zero_error_no_failures_unencoded(self):
        result = run_single_experiment(p=0.0, n_trials=1000, seed=42, encoded=False)
        assert result.n_failures == 0

    def test_full_error_all_failures(self):
        result = run_single_experiment(p=1.0, n_trials=1000, seed=42, encoded=True)
        assert result.n_failures == 1000
        assert result.simulated_failure_rate == pytest.approx(1.0)

    def test_unencoded_full_error_all_failures(self):
        result = run_single_experiment(p=1.0, n_trials=500, seed=7, encoded=False)
        assert result.simulated_failure_rate == pytest.approx(1.0)

    def test_reproducibility_with_same_seed(self):
        r1 = run_single_experiment(p=0.15, n_trials=2000, seed=100)
        r2 = run_single_experiment(p=0.15, n_trials=2000, seed=100)
        assert r1.n_failures == r2.n_failures
        assert r1.simulated_failure_rate == r2.simulated_failure_rate

    def test_different_seeds_may_differ(self):
        r1 = run_single_experiment(p=0.15, n_trials=2000, seed=100)
        r2 = run_single_experiment(p=0.15, n_trials=2000, seed=101)
        # With high probability, different seeds produce different results
        # This is a statistical test — very unlikely to fail
        # (both are within a tight range of theory, but rarely exactly equal)
        # We just verify they both run without error
        assert 0.0 <= r1.simulated_failure_rate <= 1.0
        assert 0.0 <= r2.simulated_failure_rate <= 1.0

    def test_ci_contains_simulated_rate(self):
        result = run_single_experiment(p=0.1, n_trials=5000, seed=42)
        assert result.ci_lower <= result.simulated_failure_rate <= result.ci_upper

    def test_theory_close_to_sim_large_n(self):
        """With 20,000 trials, simulation should be within 1% of theory."""
        result = run_single_experiment(p=0.1, n_trials=20_000, seed=42, encoded=True)
        theory = float(theoretical_encoded_failure(0.1))
        assert abs(result.simulated_failure_rate - theory) < 0.01

    def test_invalid_p_below_zero_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run_single_experiment(p=-0.1, n_trials=100, seed=0)

    def test_invalid_p_above_one_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run_single_experiment(p=1.5, n_trials=100, seed=0)

    def test_invalid_n_trials_raises(self):
        with pytest.raises(ValueError, match="n_trials"):
            run_single_experiment(p=0.1, n_trials=0, seed=0)

    def test_store_trials_captures_per_trial(self):
        result = run_single_experiment(p=0.1, n_trials=100, seed=42, store_trials=True)
        assert result.per_trial_results is not None
        assert len(result.per_trial_results) == 100

    def test_per_trial_keys_present(self):
        result = run_single_experiment(p=0.1, n_trials=10, seed=42, store_trials=True)
        for trial in result.per_trial_results:
            assert "trial" in trial
            assert "logical_failure" in trial

    def test_metadata_stored(self):
        result = run_single_experiment(p=0.2, n_trials=500, seed=99, encoded=True)
        assert result.physical_error_probability == pytest.approx(0.2)
        assert result.n_trials == 500
        assert result.seed == 99
        assert result.encoded is True


# ---------------------------------------------------------------------------
# run_reliability_benchmark
# ---------------------------------------------------------------------------

class TestRunReliabilityBenchmark:
    @pytest.fixture(scope="class")
    @classmethod
    def benchmark(cls):
        return run_reliability_benchmark(n_trials=1000, seed=42, n_p_points=20)

    def test_returns_benchmark_result(self, benchmark):
        from qubitshield.noise import BenchmarkResult
        assert isinstance(benchmark, BenchmarkResult)

    def test_p_values_length(self, benchmark):
        assert len(benchmark.p_values) == 20

    def test_arrays_same_length(self, benchmark):
        n = len(benchmark.p_values)
        assert len(benchmark.simulated_encoded) == n
        assert len(benchmark.simulated_unencoded) == n
        assert len(benchmark.theoretical_encoded) == n
        assert len(benchmark.theoretical_unencoded) == n

    def test_failure_rates_in_valid_range(self, benchmark):
        assert np.all(benchmark.simulated_encoded >= 0.0)
        assert np.all(benchmark.simulated_encoded <= 1.0)
        assert np.all(benchmark.simulated_unencoded >= 0.0)
        assert np.all(benchmark.simulated_unencoded <= 1.0)

    def test_ci_bounds_valid(self, benchmark):
        assert np.all(benchmark.ci_lower_encoded >= 0.0)
        assert np.all(benchmark.ci_upper_encoded <= 1.0)
        assert np.all(benchmark.ci_lower_encoded <= benchmark.ci_upper_encoded)

    def test_elapsed_time_positive(self, benchmark):
        assert benchmark.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_experiment_dataframe_columns(self):
        result = run_single_experiment(p=0.1, n_trials=100, seed=42)
        df = experiment_result_to_dataframe(result)
        expected_cols = [
            "physical_error_probability",
            "n_trials",
            "seed",
            "encoded",
            "n_failures",
            "simulated_failure_rate",
            "theoretical_failure_rate",
            "absolute_error",
            "ci_lower_95pct",
            "ci_upper_95pct",
            "elapsed_seconds",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_benchmark_dataframe_columns(self):
        bench = run_reliability_benchmark(n_trials=100, seed=42, n_p_points=5)
        df = benchmark_result_to_dataframe(bench)
        expected_cols = [
            "physical_error_probability",
            "mode",
            "theoretical_failure_rate",
            "simulated_failure_rate",
            "ci_lower_95pct",
            "ci_upper_95pct",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_benchmark_dataframe_has_correct_rows(self):
        bench = run_reliability_benchmark(n_trials=100, seed=42, n_p_points=5)
        df = benchmark_result_to_dataframe(bench)
        # 5 p values × 2 modes (encoded + unencoded) = 10 rows
        assert len(df) == 10

    def test_benchmark_csv_is_valid(self):
        bench = run_reliability_benchmark(n_trials=100, seed=42, n_p_points=5)
        df = benchmark_result_to_dataframe(bench)
        csv_str = df.to_csv(index=False)
        # Re-parse to verify
        df2 = pd.read_csv(io.StringIO(csv_str))
        assert len(df2) == len(df)
        assert set(df.columns) == set(df2.columns)

    def test_failure_rates_in_csv_are_valid_probabilities(self):
        bench = run_reliability_benchmark(n_trials=100, seed=42, n_p_points=5)
        df = benchmark_result_to_dataframe(bench)
        assert (df["simulated_failure_rate"] >= 0).all()
        assert (df["simulated_failure_rate"] <= 1).all()
