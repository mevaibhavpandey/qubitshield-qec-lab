"""
syndrome.py — Syndrome Decoding for the 3-Qubit Bit-Flip Repetition Code

The three-qubit bit-flip repetition code uses two parity-check measurements:
    s₀₁ = Z₀ ⊗ Z₁   (parity of qubits 0 and 1)
    s₁₂ = Z₁ ⊗ Z₂   (parity of qubits 1 and 2)

Syndrome (s₀₁, s₁₂) maps to a unique single-qubit error or no error:

    (0, 0) → No error
    (1, 0) → X on qubit 0
    (1, 1) → X on qubit 1
    (0, 1) → X on qubit 2

This mapping assumes at most one error. Two-error cases produce misleading
syndromes (demonstrated explicitly in the Syndrome Explorer).

Bit-ordering note for Qiskit:
    Qiskit uses little-endian bit ordering in measurement results.
    Measurement registers are read right-to-left in the result strings.
    When we measure ancilla[0] and ancilla[1], the result string has
    ancilla[1] in the leftmost position and ancilla[0] in the rightmost.
    All functions here work with (s01, s12) tuples after explicit extraction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Syndrome table
# ---------------------------------------------------------------------------

# Key:   (s01, s12) where s01 = parity(q0,q1), s12 = parity(q1,q2)
# Value: (qubit_to_correct, description)
#   qubit_to_correct is None for no error, otherwise 0, 1, or 2.

SyndromeKey = Tuple[int, int]
SyndromeValue = Tuple[Optional[int], str]

SYNDROME_TABLE: Dict[SyndromeKey, SyndromeValue] = {
    (0, 0): (None, "No error detected"),
    (1, 0): (0,    "Bit-flip on qubit 0 (X₀)"),
    (1, 1): (1,    "Bit-flip on qubit 1 (X₁)"),
    (0, 1): (2,    "Bit-flip on qubit 2 (X₂)"),
}


def decode_syndrome(s01: int, s12: int) -> SyndromeValue:
    """Decode a syndrome pair and return the corrective action.

    Parameters
    ----------
    s01:
        Syndrome bit for the Z₀Z₁ parity check (0 or 1).
    s12:
        Syndrome bit for the Z₁Z₂ parity check (0 or 1).

    Returns
    -------
    (qubit_to_correct, description):
        qubit_to_correct is None when no correction is needed,
        or an integer 0, 1, or 2 indicating which data qubit to flip.

    Raises
    ------
    ValueError
        If s01 or s12 is not 0 or 1.
    """
    if s01 not in (0, 1) or s12 not in (0, 1):
        raise ValueError(
            f"Syndrome bits must be 0 or 1, got s01={s01}, s12={s12}."
        )
    return SYNDROME_TABLE[(s01, s12)]


def parse_measurement_string(result_str: str, num_syndrome_bits: int = 2) -> SyndromeKey:
    """Extract syndrome bits from a Qiskit measurement result string.

    Qiskit returns measurement results as a string in **little-endian** order
    for classical registers within each register, but concatenates registers
    with a space separator, with the rightmost register listed first.

    This function handles the specific circuit layout in QubitShield where
    we use two separate 1-bit classical registers (c_anc[0] and c_anc[1])
    OR one 2-bit register for the ancilla measurements, and a separate
    register for the data qubit measurements.

    The syndrome register layout expected here:
        bits[-2] = ancilla[0] result = s01
        bits[-1] = ancilla[1] result = s12

    Parameters
    ----------
    result_str:
        Measurement result string from Qiskit (e.g., '0 10').
    num_syndrome_bits:
        Number of syndrome ancilla bits (default 2).

    Returns
    -------
    SyndromeKey
        (s01, s12) tuple.
    """
    # Remove spaces for parsing
    parts = result_str.strip().split()
    if len(parts) == 1:
        # Single concatenated register — last two bits are syndrome
        bits = parts[0]
        # Qiskit: string is big-endian for display, but bit 0 is rightmost
        s01 = int(bits[-1])
        s12 = int(bits[-2]) if len(bits) >= 2 else 0
    else:
        # Multiple registers separated by spaces — first segment (rightmost) is ancilla
        # Format: "data_bits ancilla_bits"  (space-separated, ancilla last in string)
        anc_str = parts[-1]  # last space-separated token = last register appended
        s01 = int(anc_str[-1]) if len(anc_str) >= 1 else 0
        s12 = int(anc_str[-2]) if len(anc_str) >= 2 else 0
    return (s01, s12)


def syndrome_string_from_counts(
    counts: dict[str, int]
) -> Tuple[SyndromeKey, str, int]:
    """Return the most probable syndrome and its raw measurement string.

    Parameters
    ----------
    counts:
        Dictionary mapping bit-strings to counts (from Qiskit job.result().get_counts()).

    Returns
    -------
    (syndrome_key, most_probable_string, count):
        syndrome_key: (s01, s12)
        most_probable_string: the raw bit string with the highest count
        count: the count for that string
    """
    if not counts:
        raise ValueError("Measurement counts dictionary is empty.")
    most_probable = max(counts, key=counts.__getitem__)
    syndrome = parse_measurement_string(most_probable)
    return syndrome, most_probable, counts[most_probable]


# ---------------------------------------------------------------------------
# Two-error demonstration
# ---------------------------------------------------------------------------

TWO_ERROR_CASES: Dict[str, dict] = {
    "X₀X₁ (qubits 0 and 1)": {
        "errors": [0, 1],
        "true_syndrome": (0, 1),
        "decoded_as":    (2,    "Bit-flip on qubit 2 (X₂) — WRONG!"),
        "explanation": (
            "Errors on qubits 0 and 1 flip both parity bits: "
            "Z₀Z₁ flips (q0 and q1 both flipped → parity unchanged after two flips = 0), "
            "Z₁Z₂ flips (q1 flipped, q2 not → parity = 1). "
            "Syndrome (0,1) is identical to a single X₂ error. "
            "The decoder applies X₂, leaving three errors total (0,1,2), "
            "which is equivalent to a logical X flip. The code fails."
        ),
    },
    "X₀X₂ (qubits 0 and 2)": {
        "errors": [0, 2],
        "true_syndrome": (1, 1),
        "decoded_as":    (1,    "Bit-flip on qubit 1 (X₁) — WRONG!"),
        "explanation": (
            "Errors on qubits 0 and 2: "
            "Z₀Z₁ parity flips (q0 flipped) → 1. "
            "Z₁Z₂ parity flips (q2 flipped) → 1. "
            "Syndrome (1,1) matches single X₁. "
            "Decoder applies X₁, total errors = {0,1,2} = logical X. Code fails."
        ),
    },
    "X₁X₂ (qubits 1 and 2)": {
        "errors": [1, 2],
        "true_syndrome": (1, 0),
        "decoded_as":    (0,    "Bit-flip on qubit 0 (X₀) — WRONG!"),
        "explanation": (
            "Errors on qubits 1 and 2: "
            "Z₀Z₁ parity flips (q1 flipped) → 1. "
            "Z₁Z₂ parity: both q1 and q2 flipped → parity unchanged = 0. "
            "Syndrome (1,0) matches single X₀. "
            "Decoder applies X₀, total errors = {0,1,2} = logical X. Code fails."
        ),
    },
}
