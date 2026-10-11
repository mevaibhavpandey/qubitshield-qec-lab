"""
experiments.py — High-Level Experiment Runners for the Syndrome Lab

This module provides functions that run specific deterministic experiments
for the Syndrome Lab section of the dashboard. Each function runs a circuit,
measures the syndrome, applies the decoder, and returns a structured result
for display.

These functions wrap circuits.py, syndrome.py, and recovery.py to provide
a clean interface for the Streamlit UI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from qubitshield.circuits import build_full_qec_circuit, build_syndrome_circuit
from qubitshield.syndrome import decode_syndrome, TWO_ERROR_CASES, SYNDROME_TABLE
from qubitshield.recovery import verify_recovery_statevector


# ---------------------------------------------------------------------------
# Simulator instance (shared, thread-safe in Streamlit via caching)
# ---------------------------------------------------------------------------

def get_simulator() -> AerSimulator:
    """Return the AerSimulator instance for statevector experiments."""
    return AerSimulator(method="statevector")


def get_shot_simulator() -> AerSimulator:
    """Return the AerSimulator instance for shot-based experiments."""
    return AerSimulator()


# ---------------------------------------------------------------------------
# Syndrome Lab experiment
# ---------------------------------------------------------------------------

def run_syndrome_experiment(
    injected_errors: List[int],
    alpha: complex = 1.0,
    beta: complex = 0.0,
    n_shots: int = 1024,
) -> Dict[str, Any]:
    """Run a complete syndrome experiment for the Syndrome Lab.

    Builds the QEC circuit, runs it on the shot-based simulator, extracts
    the syndrome from measurements, applies the decoder, and runs statevector
    verification.

    Parameters
    ----------
    injected_errors:
        List of data qubit indices (0, 1, or 2) to inject X errors on.
    alpha, beta:
        Logical qubit amplitudes (must be normalised).
    n_shots:
        Number of measurement shots.

    Returns
    -------
    dict with comprehensive experiment results for display.
    """
    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(f"|α|² + |β|² = {norm_sq:.6f} ≠ 1.")

    # Build and run the circuit
    qc = build_full_qec_circuit(
        alpha=alpha,
        beta=beta,
        injected_errors=injected_errors,
        measure_data=True,
    )

    simulator = get_shot_simulator()
    t_qc = transpile(qc, simulator)
    job = simulator.run(t_qc, shots=n_shots)
    result = job.result()
    counts = result.get_counts(t_qc)

    # Find the dominant measurement outcome
    most_common = max(counts, key=counts.__getitem__)
    count_val = counts[most_common]

    # Parse syndrome from the ancilla bits
    # The classical register layout is: c_data (3 bits) c_anc (2 bits)
    # In Qiskit counts string (little-endian per register, registers left to right):
    # The string format is "c_data_bits c_anc_bits" where c_anc is the first register
    # appended (rightmost in the composite string after space split).
    parts = most_common.strip().split()

    if len(parts) >= 2:
        # Two registers: first part is c_data, second is c_anc
        # Actually Qiskit orders them as last-registered first in the string
        # We need to determine order based on circuit register order
        anc_str = parts[0]   # c_anc was added to circuit first
        data_str = parts[1] if len(parts) > 1 else "000"
    else:
        # Single string: last 2 bits are ancilla
        combined = parts[0]
        anc_str = combined[-2:]
        data_str = combined[:-2]

    # s01 = bit 0 of ancilla register (rightmost in string)
    # s12 = bit 1 of ancilla register
    s01 = int(anc_str[-1]) if len(anc_str) >= 1 else 0
    s12 = int(anc_str[-2]) if len(anc_str) >= 2 else 0
    syndrome = (s01, s12)

    correction_qubit, correction_description = decode_syndrome(s01, s12)

    # Statevector verification for fidelity
    sv_result = verify_recovery_statevector(
        alpha=alpha,
        beta=beta,
        injected_errors=injected_errors,
        syndrome=syndrome,
    )

    # Build error description
    if not injected_errors:
        error_description = "No error"
    elif len(injected_errors) == 1:
        error_description = f"X on qubit {injected_errors[0]}"
    else:
        error_description = f"X on qubits {', '.join(str(i) for i in injected_errors)}"

    # Expected outcome
    expected_correctable = len(injected_errors) <= 1

    # Determine if this is a known two-error failure case
    is_two_error_failure = len(injected_errors) == 2

    return {
        "injected_errors": injected_errors,
        "error_description": error_description,
        "alpha": alpha,
        "beta": beta,
        "circuit": qc,
        "counts": counts,
        "most_common_outcome": most_common,
        "syndrome": syndrome,
        "correction_qubit": correction_qubit,
        "correction_description": correction_description,
        "fidelity": sv_result["fidelity"],
        "corrected": sv_result["corrected"],
        "expected_correctable": expected_correctable,
        "is_two_error_failure": is_two_error_failure,
        "target_statevector": sv_result["target_statevector"],
        "actual_statevector": sv_result["actual_statevector"],
        "n_shots": n_shots,
        "shot_counts": counts,
    }


# ---------------------------------------------------------------------------
# Full experiment suite (all single-error and selected multi-error cases)
# ---------------------------------------------------------------------------

def run_full_syndrome_suite(
    alpha: complex = 1.0,
    beta: complex = 0.0,
) -> List[Dict[str, Any]]:
    """Run all standard syndrome experiments and return a list of results.

    Covers:
    - No error
    - X₀, X₁, X₂ (all single-qubit errors)
    - X₀X₁, X₀X₂, X₁X₂ (two-qubit error failure cases)

    Parameters
    ----------
    alpha, beta:
        Logical qubit amplitudes.

    Returns
    -------
    List of experiment result dicts.
    """
    error_patterns = [
        [],       # No error
        [0],      # X₀
        [1],      # X₁
        [2],      # X₂
        [0, 1],   # Two-qubit: X₀X₁
        [0, 2],   # Two-qubit: X₀X₂
        [1, 2],   # Two-qubit: X₁X₂
    ]

    results = []
    for pattern in error_patterns:
        try:
            result = run_syndrome_experiment(
                injected_errors=pattern,
                alpha=alpha,
                beta=beta,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "injected_errors": pattern,
                "error": str(e),
                "corrected": None,
            })

    return results


# ---------------------------------------------------------------------------
# Named state experiments
# ---------------------------------------------------------------------------

NAMED_STATE_AMPLITUDES = {
    "|0⟩": (1.0 + 0j, 0.0 + 0j),
    "|1⟩": (0.0 + 0j, 1.0 + 0j),
    "|+⟩": (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    "|−⟩": (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    "|i⟩": (1.0 / np.sqrt(2), 1j / np.sqrt(2)),
    "|−i⟩": (1.0 / np.sqrt(2), -1j / np.sqrt(2)),
}


def run_named_state_experiment(
    state_name: str,
    injected_errors: List[int],
) -> Dict[str, Any]:
    """Run a syndrome experiment for a named logical input state.

    Parameters
    ----------
    state_name:
        Key in NAMED_STATE_AMPLITUDES.
    injected_errors:
        Qubit indices for X error injection.

    Returns
    -------
    dict
        Experiment results.
    """
    if state_name not in NAMED_STATE_AMPLITUDES:
        raise ValueError(
            f"Unknown state '{state_name}'. "
            f"Choose from: {list(NAMED_STATE_AMPLITUDES.keys())}"
        )

    alpha, beta = NAMED_STATE_AMPLITUDES[state_name]
    return run_syndrome_experiment(
        injected_errors=injected_errors,
        alpha=alpha,
        beta=beta,
    )
