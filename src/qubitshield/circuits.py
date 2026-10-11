"""
circuits.py — Qiskit Circuit Construction for the 3-Qubit Bit-Flip Repetition Code

Circuit layout
--------------
Data qubits:    q[0], q[1], q[2]
Ancilla qubits: q[3] (for Z₀Z₁ parity), q[4] (for Z₁Z₂ parity)

Classical registers:
    c_data[3]:   Final data-qubit measurements (used for fidelity checks)
    c_anc[2]:    Ancilla measurements (syndrome bits)

The syndrome extraction uses the stabilizer formalism:
    - CNOT(q[0], q[3]) and CNOT(q[1], q[3])  →  ancilla[3] = parity(q0, q1)
    - CNOT(q[1], q[4]) and CNOT(q[2], q[4])  →  ancilla[4] = parity(q1, q2)

After syndrome measurement, conditional X gates implement the decoder.

Qiskit bit-ordering convention:
    In Qiskit, classical register bit strings are displayed in little-endian:
    the rightmost character corresponds to bit index 0.
    When reading c_anc as a 2-bit string:  "s12 s01" (s01 is rightmost = bit 0).

Implementation note on conditional recovery:
    Qiskit 1.x statevector and AerSimulator support .c_if() for conditional gates
    within a single shot. We use the IfElseOp (if_test context manager) for
    cleaner circuit structure, which is supported in Qiskit 1.4+.
    For the Syndrome Lab experiments, we run without conditional recovery and
    instead apply it classically in Python post-simulation, which is fully
    equivalent under ideal (noise-free) conditions and more transparent for
    educational purposes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate


# ---------------------------------------------------------------------------
# Register layout constants
# ---------------------------------------------------------------------------

N_DATA = 3        # number of data qubits
N_ANCILLA = 2     # number of ancilla qubits
N_TOTAL = N_DATA + N_ANCILLA  # total qubits in full circuit


# ---------------------------------------------------------------------------
# Encoding circuit
# ---------------------------------------------------------------------------

def build_encoding_circuit(
    alpha: complex = 1.0,
    beta: complex = 0.0,
    add_barrier: bool = True,
) -> QuantumCircuit:
    """Build the encoding circuit that maps α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩.

    The encoding uses:
    1. Initialize q[0] to α|0⟩ + β|1⟩ using Qiskit's Initialize instruction.
    2. CNOT(q[0], q[1]) and CNOT(q[0], q[2]) to spread the logical state.

    Parameters
    ----------
    alpha:
        Amplitude of |0⟩ in the input state.
    beta:
        Amplitude of |1⟩ in the input state.
    add_barrier:
        If True, insert a barrier after initialization.

    Returns
    -------
    QuantumCircuit
        3-qubit encoding circuit (no ancilla, no measurement).

    Raises
    ------
    ValueError
        If |α|² + |β|² ≠ 1.
    """
    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(
            f"|α|² + |β|² = {norm_sq:.6f} ≠ 1. "
            "Provide normalised amplitudes for the encoding circuit."
        )

    qr = QuantumRegister(N_DATA, name="q")
    qc = QuantumCircuit(qr, name="Encoding")

    # Initialize qubit 0 to the desired state
    qc.initialize([alpha, beta], 0)

    if add_barrier:
        qc.barrier()

    # Spread to qubits 1 and 2 via CNOT
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])

    return qc


def build_syndrome_circuit(
    injected_errors: Optional[List[int]] = None,
    add_barrier: bool = True,
) -> QuantumCircuit:
    """Build the full syndrome extraction circuit for |000⟩ logical state.

    This circuit:
    1. Encodes |0⟩_L = |000⟩ (no initialization needed since registers start at |0⟩)
    2. Injects X errors on specified data qubits
    3. Extracts the syndrome using ancilla qubits
    4. Measures only the ancilla qubits

    The circuit uses |000⟩ as the logical state to keep the statevector
    representation concrete and verifiable.

    Parameters
    ----------
    injected_errors:
        List of data qubit indices (0, 1, or 2) on which to inject X errors.
        None or empty list means no error.
    add_barrier:
        If True, insert barriers between circuit stages.

    Returns
    -------
    QuantumCircuit
        5-qubit (3 data + 2 ancilla) circuit with 2-bit classical register.
    """
    if injected_errors is None:
        injected_errors = []

    # Validate error indices
    for idx in injected_errors:
        if idx not in (0, 1, 2):
            raise ValueError(
                f"Error qubit index must be 0, 1, or 2; got {idx}."
            )

    qr_data = QuantumRegister(N_DATA, name="q")
    qr_anc = QuantumRegister(N_ANCILLA, name="anc")
    cr_anc = ClassicalRegister(N_ANCILLA, name="c_anc")

    qc = QuantumCircuit(qr_data, qr_anc, cr_anc, name="SyndromeExtraction")

    # Stage 1: Encoding (start from |0⟩, CNOT to make |000⟩ explicit)
    # Since all qubits start at |0⟩, |000⟩ is already prepared.
    # We add a label for clarity but no gates needed for |0⟩_L.

    if add_barrier:
        qc.barrier(label="Encoded |000⟩")

    # Stage 2: Inject errors
    if injected_errors:
        for q_idx in injected_errors:
            qc.x(qr_data[q_idx])
        if add_barrier:
            qc.barrier(label="After error injection")

    # Stage 3: Syndrome extraction
    # Parity check Z₀Z₁: CNOT(data[0] → anc[0]), CNOT(data[1] → anc[0])
    qc.cx(qr_data[0], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[0])

    # Parity check Z₁Z₂: CNOT(data[1] → anc[1]), CNOT(data[2] → anc[1])
    qc.cx(qr_data[1], qr_anc[1])
    qc.cx(qr_data[2], qr_anc[1])

    if add_barrier:
        qc.barrier(label="After syndrome extraction")

    # Stage 4: Measure ancilla only
    qc.measure(qr_anc[0], cr_anc[0])  # s01 in bit 0
    qc.measure(qr_anc[1], cr_anc[1])  # s12 in bit 1

    return qc


def build_full_qec_circuit(
    alpha: complex = 1.0,
    beta: complex = 0.0,
    injected_errors: Optional[List[int]] = None,
    measure_data: bool = True,
    add_barrier: bool = True,
) -> QuantumCircuit:
    """Build the complete QEC circuit: encode → error → syndrome → recover → measure.

    This is the principal circuit used in the Syndrome Lab. The recovery
    is implemented classically in Python after measuring the syndrome
    (see recovery.py), which is fully equivalent under ideal conditions.

    For educational transparency, this circuit does NOT include quantum
    conditional operations. Instead, it:
    1. Encodes the logical qubit.
    2. Injects specified bit-flip errors.
    3. Extracts the syndrome (measuring ancilla).
    4. Optionally measures data qubits.

    The caller then decodes the syndrome and applies the appropriate
    correction classically (see apply_recovery in recovery.py).

    Parameters
    ----------
    alpha, beta:
        Amplitudes for the input logical state.
    injected_errors:
        Qubit indices receiving X gates (error injection).
    measure_data:
        If True, add data-qubit measurements after syndrome extraction.
    add_barrier:
        If True, insert barriers between stages.

    Returns
    -------
    QuantumCircuit
        5-qubit circuit with classical registers.
    """
    if injected_errors is None:
        injected_errors = []

    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(
            f"|α|² + |β|² = {norm_sq:.6f} ≠ 1."
        )

    for idx in injected_errors:
        if idx not in (0, 1, 2):
            raise ValueError(f"Error qubit index must be 0, 1, or 2; got {idx}.")

    qr_data = QuantumRegister(N_DATA, name="q")
    qr_anc = QuantumRegister(N_ANCILLA, name="anc")
    cr_anc = ClassicalRegister(N_ANCILLA, name="c_anc")
    cr_data = ClassicalRegister(N_DATA, name="c_data") if measure_data else None

    regs = [qr_data, qr_anc, cr_anc]
    if cr_data is not None:
        regs.append(cr_data)

    qc = QuantumCircuit(*regs, name="QubitShield_QEC")

    # Stage 1: Encoding
    qc.initialize([alpha, beta], qr_data[0])
    qc.cx(qr_data[0], qr_data[1])
    qc.cx(qr_data[0], qr_data[2])

    if add_barrier:
        qc.barrier(label="Encoded")

    # Stage 2: Error injection
    if injected_errors:
        for q_idx in injected_errors:
            qc.x(qr_data[q_idx])
        if add_barrier:
            qc.barrier(label="Errors injected")

    # Stage 3: Syndrome extraction
    qc.cx(qr_data[0], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[1])
    qc.cx(qr_data[2], qr_anc[1])

    if add_barrier:
        qc.barrier(label="Syndrome extracted")

    # Stage 4: Measure ancilla
    qc.measure(qr_anc[0], cr_anc[0])
    qc.measure(qr_anc[1], cr_anc[1])

    # Stage 5: Measure data (for verification)
    if measure_data and cr_data is not None:
        if add_barrier:
            qc.barrier(label="Measure data")
        qc.measure(qr_data[0], cr_data[0])
        qc.measure(qr_data[1], cr_data[1])
        qc.measure(qr_data[2], cr_data[2])

    return qc


def build_statevector_circuit(
    alpha: complex = 1.0,
    beta: complex = 0.0,
    injected_errors: Optional[List[int]] = None,
    apply_recovery_gates: Optional[int] = None,
) -> QuantumCircuit:
    """Build a circuit for statevector simulation (no measurements).

    This circuit is used to compute the exact quantum state after encoding,
    error injection, and optionally recovery. Since statevector simulation
    does not support mid-circuit measurement + feedback in a single pass,
    we build circuits for each phase separately.

    Parameters
    ----------
    alpha, beta:
        Input logical state amplitudes.
    injected_errors:
        Qubit indices for X error injection.
    apply_recovery_gates:
        If not None, apply X gate to this qubit index (classical correction).

    Returns
    -------
    QuantumCircuit
        3-qubit circuit (data only, no ancilla) suitable for statevector
        simulation. Does not include measurements.
    """
    if injected_errors is None:
        injected_errors = []

    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(f"|α|² + |β|² = {norm_sq:.6f} ≠ 1.")

    qr = QuantumRegister(N_DATA, name="q")
    qc = QuantumCircuit(qr, name="Statevector_QEC")

    # Encoding
    qc.initialize([alpha, beta], qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])

    # Error injection
    for idx in injected_errors:
        qc.x(qr[idx])

    # Recovery (if correction qubit is known)
    if apply_recovery_gates is not None:
        qc.x(qr[apply_recovery_gates])

    return qc


def get_circuit_diagram(qc: QuantumCircuit, output_format: str = "text") -> str:
    """Return a string diagram of the circuit.

    Parameters
    ----------
    qc:
        The Qiskit QuantumCircuit to draw.
    output_format:
        Either 'text' or 'mpl' (matplotlib). For 'mpl', a figure object
        is returned instead. For Streamlit, 'mpl' diagrams are embedded
        directly as images.

    Returns
    -------
    str
        ASCII text diagram of the circuit.
    """
    return qc.draw(output="text", fold=-1).single_string()
