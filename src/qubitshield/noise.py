"""
noise.py — Monte Carlo Noise Simulation for the 3-Qubit Bit-Flip Code

This module implements the Monte Carlo experiment that compares:
    1. An unencoded qubit under independent bit-flip errors.
    2. A qubit protected by the 3-qubit bit-flip repetition code.

Error model
-----------
Each physical data qubit independently experiences a bit-flip (X) error
with probability p. This is the independent identically distributed (i.i.d.)
bit-flip channel model.

The model does NOT include:
    - Errors in encoding gates
    - Errors in ancilla qubit preparation
    - Errors in syndrome measurement
    - Errors in recovery gates
    - Phase-flip errors (Z errors)
    - Depolarising noise
    - Correlated errors

These simplifications correspond to the idealized model described in the
theoretical analysis. A separate circuit-level noise model would be needed
to evaluate the code under realistic conditions.

Reproducibility
---------------
All experiments accept a random seed for exact reproducibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from qubitshield.analytics import (
    theoretical_encoded_failure,
    theoretical_unencoded_failure,
    wilson_confidence_interval,
)
from qubitshield.recovery import monte_carlo_recovery


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Container for a single Monte Carlo experiment result."""
    physical_error_probability: float
    n_trials: int
    seed: int
    encoded: bool
    n_failures: int
    simulated_failure_rate: float
    theoretical_failure_rate: float
    absolute_error: float
    ci_lower: float
    ci_upper: float
    elapsed_seconds: float
    per_trial_results: Optional[List[dict]] = field(default=None, repr=False)


@dataclass
class BenchmarkResult:
    """Container for a full reliability benchmark sweep."""
    p_values: np.ndarray
    n_trials: int
    seed: int
    # Theoretical curves
    theoretical_unencoded: np.ndarray
    theoretical_encoded: np.ndarray
    # Monte Carlo results
    simulated_unencoded: np.ndarray
    simulated_encoded: np.ndarray
    # Confidence intervals
    ci_lower_unencoded: np.ndarray
    ci_upper_unencoded: np.ndarray
    ci_lower_encoded: np.ndarray
    ci_upper_encoded: np.ndarray
    # Metadata
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Single-point experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    p: float,
    n_trials: int,
    seed: int,
    encoded: bool = True,
    store_trials: bool = False,
) -> ExperimentResult:
    """Run a Monte Carlo experiment at a single physical error probability.

    Parameters
    ----------
    p:
        Physical bit-flip probability, in [0, 1].
    n_trials:
        Number of independent Monte Carlo trials.
    seed:
        Random seed for reproducibility.
    encoded:
        If True, simulate the 3-qubit repetition code.
        If False, simulate an unencoded qubit.
    store_trials:
        If True, store per-trial results (uses more memory).

    Returns
    -------
    ExperimentResult
        Simulation outcome with statistics and metadata.

    Raises
    ------
    ValueError
        If p is not in [0, 1] or n_trials < 1.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Physical error probability must be in [0, 1]; got {p}.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be ≥ 1; got {n_trials}.")

    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    n_failures = 0
    per_trial_results: List[dict] = []

    for trial_idx in range(n_trials):
        if encoded:
            # Sample independent errors on 3 data qubits
            error_bits = rng.random(3) < p
            error_pattern = [i for i in range(3) if error_bits[i]]

            # Apply syndrome decoder and check for logical failure
            logical_failure, syndrome, correction = monte_carlo_recovery(error_pattern)
        else:
            # Unencoded qubit: fails if the single qubit experiences a bit flip
            logical_failure = rng.random() < p
            syndrome = None
            correction = None
            error_pattern = [0] if logical_failure else []

        if logical_failure:
            n_failures += 1

        if store_trials:
            per_trial_results.append({
                "trial": trial_idx,
                "error_pattern": error_pattern,
                "syndrome": syndrome,
                "correction": correction,
                "logical_failure": logical_failure,
            })

    elapsed = time.perf_counter() - t_start

    simulated_rate = n_failures / n_trials

    if encoded:
        theoretical_rate = float(theoretical_encoded_failure(p))
    else:
        theoretical_rate = float(theoretical_unencoded_failure(p))

    ci_lower, ci_upper = wilson_confidence_interval(n_failures, n_trials)

    return ExperimentResult(
        physical_error_probability=p,
        n_trials=n_trials,
        seed=seed,
        encoded=encoded,
        n_failures=n_failures,
        simulated_failure_rate=simulated_rate,
        theoretical_failure_rate=theoretical_rate,
        absolute_error=abs(simulated_rate - theoretical_rate),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        elapsed_seconds=elapsed,
        per_trial_results=per_trial_results if store_trials else None,
    )


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------

def run_reliability_benchmark(
    p_values: Optional[np.ndarray] = None,
    n_trials: int = 10_000,
    seed: int = 42,
    n_p_points: int = 50,
) -> BenchmarkResult:
    """Run the full reliability benchmark comparing encoded vs. unencoded.

    Sweeps physical error probability p from 0 to 1, running Monte Carlo
    experiments at each value for both the encoded and unencoded cases.

    Parameters
    ----------
    p_values:
        Array of physical error probabilities to test.
        If None, 50 equally spaced values from 0 to 1 are used.
    n_trials:
        Number of Monte Carlo trials per (p, mode) pair.
    seed:
        Base random seed; each experiment gets a unique derived seed.
    n_p_points:
        Number of p values to test (ignored if p_values is provided).

    Returns
    -------
    BenchmarkResult
        All simulation and theoretical results.
    """
    if p_values is None:
        p_values = np.linspace(0, 1, n_p_points)

    t_start = time.perf_counter()

    n = len(p_values)
    sim_unencoded = np.zeros(n)
    sim_encoded = np.zeros(n)
    ci_lo_unenc = np.zeros(n)
    ci_hi_unenc = np.zeros(n)
    ci_lo_enc = np.zeros(n)
    ci_hi_enc = np.zeros(n)

    for i, p in enumerate(p_values):
        # Encoded
        res_enc = run_single_experiment(
            p=float(p),
            n_trials=n_trials,
            seed=seed + i * 2,
            encoded=True,
        )
        sim_encoded[i] = res_enc.simulated_failure_rate
        ci_lo_enc[i] = res_enc.ci_lower
        ci_hi_enc[i] = res_enc.ci_upper

        # Unencoded
        res_unenc = run_single_experiment(
            p=float(p),
            n_trials=n_trials,
            seed=seed + i * 2 + 1,
            encoded=False,
        )
        sim_unencoded[i] = res_unenc.simulated_failure_rate
        ci_lo_unenc[i] = res_unenc.ci_lower
        ci_hi_unenc[i] = res_unenc.ci_upper

    elapsed = time.perf_counter() - t_start

    return BenchmarkResult(
        p_values=p_values,
        n_trials=n_trials,
        seed=seed,
        theoretical_unencoded=theoretical_unencoded_failure(p_values),
        theoretical_encoded=theoretical_encoded_failure(p_values),
        simulated_unencoded=sim_unencoded,
        simulated_encoded=sim_encoded,
        ci_lower_unencoded=ci_lo_unenc,
        ci_upper_unencoded=ci_hi_unenc,
        ci_lower_encoded=ci_lo_enc,
        ci_upper_encoded=ci_hi_enc,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def experiment_result_to_dataframe(result: ExperimentResult) -> pd.DataFrame:
    """Convert an ExperimentResult to a single-row Pandas DataFrame."""
    return pd.DataFrame([{
        "physical_error_probability": result.physical_error_probability,
        "n_trials": result.n_trials,
        "seed": result.seed,
        "encoded": result.encoded,
        "n_failures": result.n_failures,
        "simulated_failure_rate": result.simulated_failure_rate,
        "theoretical_failure_rate": result.theoretical_failure_rate,
        "absolute_error": result.absolute_error,
        "ci_lower_95pct": result.ci_lower,
        "ci_upper_95pct": result.ci_upper,
        "elapsed_seconds": result.elapsed_seconds,
    }])


def benchmark_result_to_dataframe(result: BenchmarkResult) -> pd.DataFrame:
    """Convert a BenchmarkResult to a Pandas DataFrame suitable for CSV export.

    Each row represents one (p, mode) pair.
    """
    rows = []
    for i, p in enumerate(result.p_values):
        rows.append({
            "physical_error_probability": p,
            "mode": "encoded",
            "theoretical_failure_rate": result.theoretical_encoded[i],
            "simulated_failure_rate": result.simulated_encoded[i],
            "ci_lower_95pct": result.ci_lower_encoded[i],
            "ci_upper_95pct": result.ci_upper_encoded[i],
        })
        rows.append({
            "physical_error_probability": p,
            "mode": "unencoded",
            "theoretical_failure_rate": result.theoretical_unencoded[i],
            "simulated_failure_rate": result.simulated_unencoded[i],
            "ci_lower_95pct": result.ci_lower_unencoded[i],
            "ci_upper_95pct": result.ci_upper_unencoded[i],
        })

    df = pd.DataFrame(rows)
    df["n_trials"] = result.n_trials
    df["seed"] = result.seed
    return df
