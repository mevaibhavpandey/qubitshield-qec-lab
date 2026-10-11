"""
QubitShield — Interactive Quantum Error-Correction & Reliability Lab

A scientific application for exploring the three-qubit bit-flip repetition code.

Developer: Vaibhav Pandey
Institution: BMS Institute of Technology & Management
"""

__version__ = "1.0.0"
__author__ = "Vaibhav Pandey"
__email__ = "vaibhav@example.com"

from qubitshield.states import (
    bloch_angles_to_statevector,
    is_normalized,
    logical_zero,
    logical_one,
    logical_plus,
    logical_minus,
)
from qubitshield.circuits import (
    build_encoding_circuit,
    build_full_qec_circuit,
    build_syndrome_circuit,
)
from qubitshield.syndrome import decode_syndrome, SYNDROME_TABLE
from qubitshield.recovery import apply_recovery
from qubitshield.analytics import (
    theoretical_unencoded_failure,
    theoretical_encoded_failure,
)

__all__ = [
    "bloch_angles_to_statevector",
    "is_normalized",
    "logical_zero",
    "logical_one",
    "logical_plus",
    "logical_minus",
    "build_encoding_circuit",
    "build_full_qec_circuit",
    "build_syndrome_circuit",
    "decode_syndrome",
    "SYNDROME_TABLE",
    "apply_recovery",
    "theoretical_unencoded_failure",
    "theoretical_encoded_failure",
]
