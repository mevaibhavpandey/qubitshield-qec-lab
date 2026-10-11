"""
recovery.py — Classical Recovery Operations for the 3-Qubit Bit-Flip Code

After syndrome measurement, the decoder determines which (if any) data qubit
to flip. This module implements:

1. Classical Python recovery (applied to quantum circuits before final
   state measurement, or used in Monte Carlo simulation).
2. Statevector-based recovery verification (verifies that the recovered
   state has the expected fidelity with the original logical state).

Recovery strategy
-----------------
The decoder applies a minimum-weight correction: flip the data qubit
identified by the syndrome. This is optimal for independent identically
distributed bit-flip errors with p < 0.5.

When the code fails
-------------------
The code fails when two or three data qubits experience bit flips. In these
cases, the syndrome matches a different (single) error, and the decoder
applies an incorrect correction that increases the number of flipped qubits
to three — equivalent to a logical X error.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from qubitshield.syndrome import decode_syndrome, SYNDROME_TABLE
from qubitshield.states import encode_state, state_fidelity


# ---------------------------------------------------------------------------
# Classical recovery
# ---------------------------------------------------------------------------

def apply_recovery(
    syndrome: Tuple[int, int],
    data_bits: Optional[List[int]] = None,
) -> Tuple[Optional[int], str]:
    """Determine and optionally apply the recovery operation.

    Given a syndrome tuple (s01, s12), return the qubit to correct.

    Parameters
    ----------
    syndrome:
        (s01, s12) tuple from syndrome measurement.
    data_bits:
        Optional list of 3 classical bits representing the data qubit
        state. If provided, the function flips the appropriate bit in-place.

    Returns
    -------
    (correction_qubit, description):
        correction_qubit: None if no correction, else 0, 1, or 2.
        description: human-readable recovery action.
    """
    correction_qubit, description = decode_syndrome(*syndrome)

    if data_bits is not None and correction_qubit is not None:
        if len(data_bits) != 3:
            raise ValueError(f"data_bits must have length 3, got {len(data_bits)}.")
        data_bits[correction_qubit] ^= 1  # flip the bit

    return correction_qubit, description


def apply_recovery_to_circuit(
    qc: QuantumCircuit,
    syndrome: Tuple[int, int],
    data_qubit_register_name: str = "q",
) -> QuantumCircuit:
    """Append an X gate to the appropriate data qubit based on the syndrome.

    Returns a new circuit with the recovery gate appended.
    The original circuit is not modified.

    Parameters
    ----------
    qc:
        Existing quantum circuit containing the data qubits.
    syndrome:
        (s01, s12) decoded syndrome.
    data_qubit_register_name:
        Name of the data qubit register in the circuit.

    Returns
    -------
    QuantumCircuit
        New circuit with recovery gate appended (or unchanged if no correction).
    """
    correction_qubit, description = decode_syndrome(*syndrome)

    new_qc = qc.copy()

    if correction_qubit is not None:
        # Find the data qubit register
        data_reg = None
        for reg in new_qc.qregs:
            if reg.name == data_qubit_register_name:
                data_reg = reg
                break

        if data_reg is None:
            raise ValueError(
                f"Quantum register '{data_qubit_register_name}' not found in circuit. "
                f"Available registers: {[r.name for r in new_qc.qregs]}"
            )

        new_qc.x(data_reg[correction_qubit])

    return new_qc


# ---------------------------------------------------------------------------
# Statevector-based verification
# ---------------------------------------------------------------------------

def verify_recovery_statevector(
    alpha: complex,
    beta: complex,
    injected_errors: List[int],
    syndrome: Tuple[int, int],
) -> dict:
    """Verify error correction using statevector simulation and state fidelity.

    This function:
    1. Encodes α|0⟩ + β|1⟩ into α|000⟩ + β|111⟩.
    2. Injects X errors on the specified qubits.
    3. Applies the recovery dictated by the syndrome.
    4. Computes state fidelity between the corrected state and the target
       logical codeword.

    Parameters
    ----------
    alpha, beta:
        Input logical state amplitudes (must be normalised).
    injected_errors:
        Qubit indices on which to apply X errors.
    syndrome:
        (s01, s12) to determine the recovery operation.

    Returns
    -------
    dict with keys:
        - 'input_state': (alpha, beta)
        - 'injected_errors': list of error qubit indices
        - 'syndrome': the (s01, s12) syndrome
        - 'correction_qubit': which qubit was corrected (or None)
        - 'correction_description': human-readable correction
        - 'fidelity': state fidelity after correction
        - 'corrected': True if fidelity ≈ 1.0
        - 'target_statevector': the expected 8-element state vector
        - 'actual_statevector': the simulated 8-element state vector
    """
    from qubitshield.circuits import build_statevector_circuit

    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(f"|α|² + |β|² = {norm_sq:.6f} ≠ 1.")

    correction_qubit, correction_description = decode_syndrome(*syndrome)

    # Build and simulate the circuit (encode + errors + recovery)
    qc = build_statevector_circuit(
        alpha=alpha,
        beta=beta,
        injected_errors=injected_errors,
        apply_recovery_gates=correction_qubit,
    )

    # Run statevector simulation
    simulator = AerSimulator(method="statevector")
    qc_sv = qc.copy()
    qc_sv.save_statevector()

    from qiskit import transpile
    t_qc = transpile(qc_sv, simulator)
    job = simulator.run(t_qc)
    result = job.result()
    sv = np.array(result.get_statevector(t_qc))

    # Target state
    target_sv = encode_state(alpha, beta)

    fidelity = state_fidelity(target_sv, sv)
    corrected = fidelity > 0.99

    return {
        "input_state": (alpha, beta),
        "injected_errors": injected_errors,
        "syndrome": syndrome,
        "correction_qubit": correction_qubit,
        "correction_description": correction_description,
        "fidelity": fidelity,
        "corrected": corrected,
        "target_statevector": target_sv,
        "actual_statevector": sv,
    }


# ---------------------------------------------------------------------------
# Monte Carlo single-shot recovery (for noise.py)
# ---------------------------------------------------------------------------

def monte_carlo_recovery(
    error_pattern: List[int],
) -> Tuple[bool, Tuple[int, int], Optional[int]]:
    """Determine whether the code corrects an error pattern in Monte Carlo sim.

    Computes the syndrome analytically for the given error pattern (no quantum
    simulation needed for this fast path), applies the decoder, and checks
    whether the decoded correction matches the actual error.

    Parameters
    ----------
    error_pattern:
        List of qubit indices (0, 1, 2) that experienced bit flips.

    Returns
    -------
    (logical_failure, syndrome, correction_qubit):
        logical_failure: True if the code fails to correct the error.
        syndrome: (s01, s12) computed from the error pattern.
        correction_qubit: the qubit the decoder corrects (or None).
    """
    # Compute syndrome analytically from error pattern
    # s01 = XOR of q0 and q1 errors
    # s12 = XOR of q1 and q2 errors
    e0 = 1 if 0 in error_pattern else 0
    e1 = 1 if 1 in error_pattern else 0
    e2 = 1 if 2 in error_pattern else 0

    s01 = e0 ^ e1
    s12 = e1 ^ e2
    syndrome = (s01, s12)

    correction_qubit, _ = decode_syndrome(s01, s12)

    # Determine the effective error after correction
    corrected_errors = set(error_pattern)
    if correction_qubit is not None:
        if correction_qubit in corrected_errors:
            corrected_errors.discard(correction_qubit)
        else:
            corrected_errors.add(correction_qubit)

    # Logical failure occurs when the total number of flipped qubits is odd
    # (odd total flips = logical X on the encoded state)
    logical_failure = len(corrected_errors) % 2 == 1

    return logical_failure, syndrome, correction_qubit
