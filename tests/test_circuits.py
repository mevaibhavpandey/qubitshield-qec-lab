"""
test_circuits.py — Tests for qubitshield.circuits

Tests verify:
- Encoding circuit has correct structure
- Syndrome circuit runs and produces expected measurements
- Full QEC circuit is buildable for all error patterns
- Circuit register names are correct
- Statevector circuit produces expected logical states
- Bit-ordering consistency in measurement results
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from qubitshield.circuits import (
    build_encoding_circuit,
    build_syndrome_circuit,
    build_full_qec_circuit,
    build_statevector_circuit,
    N_DATA, N_ANCILLA, N_TOTAL,
)
from qubitshield.states import is_normalized, state_fidelity, encode_state


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMULATOR_SV = AerSimulator(method="statevector")
SIMULATOR_SHOTS = AerSimulator()


def run_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Helper: run a statevector simulation and return the state vector."""
    qc2 = qc.copy()
    qc2.save_statevector()
    tqc = transpile(qc2, SIMULATOR_SV)
    job = SIMULATOR_SV.run(tqc)
    result = job.result()
    return np.array(result.get_statevector(tqc))


def run_shots(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """Helper: run a shot-based simulation and return counts."""
    tqc = transpile(qc, SIMULATOR_SHOTS)
    job = SIMULATOR_SHOTS.run(tqc, shots=shots)
    return job.result().get_counts(tqc)


# ---------------------------------------------------------------------------
# build_encoding_circuit
# ---------------------------------------------------------------------------

class TestBuildEncodingCircuit:
    def test_returns_quantum_circuit(self):
        qc = build_encoding_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_circuit_has_3_qubits(self):
        qc = build_encoding_circuit()
        assert qc.num_qubits == 3

    def test_circuit_no_measurements(self):
        qc = build_encoding_circuit()
        assert qc.num_clbits == 0

    def test_encoding_zero_produces_000(self):
        qc = build_encoding_circuit(alpha=1.0, beta=0.0)
        sv = run_statevector(qc)
        expected = encode_state(1.0, 0.0)
        fidelity = state_fidelity(expected, sv)
        assert fidelity == pytest.approx(1.0, abs=1e-6)

    def test_encoding_one_produces_111(self):
        qc = build_encoding_circuit(alpha=0.0, beta=1.0)
        sv = run_statevector(qc)
        expected = encode_state(0.0, 1.0)
        fidelity = state_fidelity(expected, sv)
        assert fidelity == pytest.approx(1.0, abs=1e-6)

    def test_encoding_plus_produces_plus_codeword(self):
        a = 1.0 / math.sqrt(2)
        qc = build_encoding_circuit(alpha=a, beta=a)
        sv = run_statevector(qc)
        expected = encode_state(a, a)
        fidelity = state_fidelity(expected, sv)
        assert fidelity == pytest.approx(1.0, abs=1e-5)

    def test_encoding_minus_produces_minus_codeword(self):
        a = 1.0 / math.sqrt(2)
        qc = build_encoding_circuit(alpha=a, beta=-a)
        sv = run_statevector(qc)
        expected = encode_state(a, -a)
        fidelity = state_fidelity(expected, sv)
        assert fidelity == pytest.approx(1.0, abs=1e-5)

    def test_invalid_amplitudes_raise(self):
        with pytest.raises(ValueError, match="normalised"):
            build_encoding_circuit(alpha=1.0, beta=1.0)

    def test_result_state_is_normalized(self):
        qc = build_encoding_circuit(alpha=0.6, beta=0.8)
        sv = run_statevector(qc)
        assert is_normalized(sv)


# ---------------------------------------------------------------------------
# build_syndrome_circuit
# ---------------------------------------------------------------------------

class TestBuildSyndromeCircuit:
    def test_returns_quantum_circuit(self):
        qc = build_syndrome_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_circuit_has_5_qubits(self):
        qc = build_syndrome_circuit()
        assert qc.num_qubits == N_TOTAL

    def test_circuit_has_2_classical_bits(self):
        qc = build_syndrome_circuit()
        assert qc.num_clbits == 2

    def test_no_error_syndrome_is_00(self):
        qc = build_syndrome_circuit(injected_errors=[])
        counts = run_shots(qc, shots=1024)
        # All shots should give syndrome "00"
        assert "00" in counts
        assert counts.get("00", 0) == pytest.approx(1024, abs=5)

    def test_x0_syndrome_is_10(self):
        qc = build_syndrome_circuit(injected_errors=[0])
        counts = run_shots(qc, shots=1024)
        # Syndrome should be 10 (s01=1, s12=0):
        # In Qiskit little-endian: bit0=s01, bit1=s12 → string "01" (bit1 first) = s12=0, s01=1
        # The string "01" means: bit1(s12)=0, bit0(s01)=1
        dominant = max(counts, key=counts.__getitem__)
        # Extract syndrome bits
        s01 = int(dominant[-1])  # rightmost = bit 0 = s01
        s12 = int(dominant[-2]) if len(dominant) >= 2 else 0
        assert (s01, s12) == (1, 0), f"Expected syndrome (1,0) for X0, got ({s01},{s12}), counts={counts}"

    def test_x1_syndrome_is_11(self):
        qc = build_syndrome_circuit(injected_errors=[1])
        counts = run_shots(qc, shots=1024)
        dominant = max(counts, key=counts.__getitem__)
        s01 = int(dominant[-1])
        s12 = int(dominant[-2]) if len(dominant) >= 2 else 0
        assert (s01, s12) == (1, 1), f"Expected syndrome (1,1) for X1, got ({s01},{s12})"

    def test_x2_syndrome_is_01(self):
        qc = build_syndrome_circuit(injected_errors=[2])
        counts = run_shots(qc, shots=1024)
        dominant = max(counts, key=counts.__getitem__)
        s01 = int(dominant[-1])
        s12 = int(dominant[-2]) if len(dominant) >= 2 else 0
        assert (s01, s12) == (0, 1), f"Expected syndrome (0,1) for X2, got ({s01},{s12})"

    def test_invalid_error_index_raises(self):
        with pytest.raises(ValueError, match="0, 1, or 2"):
            build_syndrome_circuit(injected_errors=[3])


# ---------------------------------------------------------------------------
# build_full_qec_circuit
# ---------------------------------------------------------------------------

class TestBuildFullQECCircuit:
    def test_returns_quantum_circuit(self):
        qc = build_full_qec_circuit()
        assert isinstance(qc, QuantumCircuit)

    def test_circuit_has_5_qubits(self):
        qc = build_full_qec_circuit()
        assert qc.num_qubits == N_TOTAL

    def test_invalid_amplitudes_raise(self):
        with pytest.raises(ValueError):
            build_full_qec_circuit(alpha=1.0, beta=1.0)

    def test_all_single_error_circuits_build(self):
        for errors in [[], [0], [1], [2]]:
            qc = build_full_qec_circuit(injected_errors=errors)
            assert isinstance(qc, QuantumCircuit)


# ---------------------------------------------------------------------------
# build_statevector_circuit — encoding correctness (no measurement)
# ---------------------------------------------------------------------------

class TestStatevectorCircuit:
    def test_encode_zero_then_no_error_gives_000(self):
        qc = build_statevector_circuit(alpha=1.0, beta=0.0, injected_errors=[])
        sv = run_statevector(qc)
        expected = encode_state(1.0, 0.0)
        assert state_fidelity(expected, sv) == pytest.approx(1.0, abs=1e-6)

    def test_encode_zero_then_x0_gives_100(self):
        qc = build_statevector_circuit(alpha=1.0, beta=0.0, injected_errors=[0])
        sv = run_statevector(qc)
        # After X on qubit 0: |100⟩ = index 1 (binary 001 reversed = 4...)
        # Qiskit: qubit 0 is LSB, so |100⟩ in circuit = index where bit0=1,bit1=0,bit2=0 = index 1
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-6)

    def test_recovery_restores_state(self):
        a = 1.0 / math.sqrt(2)
        # Encode |+⟩, inject X1, apply recovery on qubit 1
        qc = build_statevector_circuit(alpha=a, beta=a, injected_errors=[1], apply_recovery_gates=1)
        sv = run_statevector(qc)
        expected = encode_state(a, a)
        fidelity = state_fidelity(expected, sv)
        assert fidelity == pytest.approx(1.0, abs=1e-5)
