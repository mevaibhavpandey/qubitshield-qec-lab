"""
states.py — Quantum State Utilities

Provides helper functions for creating, validating, and describing
quantum states used throughout the QubitShield application.

All states are represented as 1-D complex NumPy arrays with shape (2,)
for single-qubit states or shape (2**n,) for n-qubit states.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_normalized(state: np.ndarray, atol: float = 1e-8) -> bool:
    """Return True if *state* has unit norm within absolute tolerance *atol*.

    Parameters
    ----------
    state:
        Complex NumPy array representing the quantum state vector.
    atol:
        Absolute tolerance for the norm check.

    Returns
    -------
    bool
        True if ||state||₂ ≈ 1.
    """
    norm = float(np.linalg.norm(state))
    return abs(norm - 1.0) < atol


def assert_normalized(state: np.ndarray, name: str = "state") -> None:
    """Raise ValueError if *state* is not normalised."""
    if not is_normalized(state):
        norm = float(np.linalg.norm(state))
        raise ValueError(
            f"Quantum {name} is not normalised: ||state||₂ = {norm:.6f}. "
            "Provide a state with unit norm."
        )


# ---------------------------------------------------------------------------
# Single-qubit standard states
# ---------------------------------------------------------------------------

#: |0⟩ computational basis state
ZERO: np.ndarray = np.array([1.0 + 0j, 0.0 + 0j])

#: |1⟩ computational basis state
ONE: np.ndarray = np.array([0.0 + 0j, 1.0 + 0j])

#: |+⟩ = (|0⟩ + |1⟩)/√2
PLUS: np.ndarray = np.array([1.0, 1.0]) / math.sqrt(2)

#: |−⟩ = (|0⟩ − |1⟩)/√2
MINUS: np.ndarray = np.array([1.0, -1.0]) / math.sqrt(2)

#: |i⟩ = (|0⟩ + i|1⟩)/√2  (Y eigenstate)
Y_PLUS: np.ndarray = np.array([1.0, 1j]) / math.sqrt(2)

#: |−i⟩ = (|0⟩ − i|1⟩)/√2  (Y eigenstate)
Y_MINUS: np.ndarray = np.array([1.0, -1j]) / math.sqrt(2)


NAMED_STATES: dict[str, np.ndarray] = {
    "|0⟩": ZERO,
    "|1⟩": ONE,
    "|+⟩": PLUS,
    "|−⟩": MINUS,
    "|i⟩": Y_PLUS,
    "|−i⟩": Y_MINUS,
}


# ---------------------------------------------------------------------------
# Logical codeword states (3-qubit repetition code)
# ---------------------------------------------------------------------------

def logical_zero() -> np.ndarray:
    """Return the logical |0⟩_L = |000⟩ state vector (shape (8,))."""
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0  # |000⟩ = index 0 in binary
    return state


def logical_one() -> np.ndarray:
    """Return the logical |1⟩_L = |111⟩ state vector (shape (8,))."""
    state = np.zeros(8, dtype=complex)
    state[7] = 1.0  # |111⟩ = index 7 in binary
    return state


def logical_plus() -> np.ndarray:
    """Return the logical |+⟩_L = (|000⟩ + |111⟩)/√2 state vector (shape (8,))."""
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0 / math.sqrt(2)
    state[7] = 1.0 / math.sqrt(2)
    return state


def logical_minus() -> np.ndarray:
    """Return the logical |−⟩_L = (|000⟩ − |111⟩)/√2 state vector (shape (8,))."""
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0 / math.sqrt(2)
    state[7] = -1.0 / math.sqrt(2)
    return state


# ---------------------------------------------------------------------------
# State construction helpers
# ---------------------------------------------------------------------------

def bloch_angles_to_statevector(theta: float, phi: float) -> np.ndarray:
    """Convert Bloch-sphere angles to a single-qubit state vector.

    The convention used is the standard physics convention:

        |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

    Parameters
    ----------
    theta:
        Polar angle in radians, 0 ≤ θ ≤ π.
    phi:
        Azimuthal angle in radians, 0 ≤ φ < 2π.

    Returns
    -------
    np.ndarray
        Normalised complex state vector of shape (2,).
    """
    alpha = math.cos(theta / 2)
    beta = cmath_exp(phi) * math.sin(theta / 2)
    state = np.array([alpha, beta], dtype=complex)
    return state


def cmath_exp(phi: float) -> complex:
    """Return e^{iφ} as a complex number."""
    return complex(math.cos(phi), math.sin(phi))


def encode_state(alpha: complex, beta: complex) -> np.ndarray:
    """Encode a logical qubit |ψ⟩ = α|0⟩ + β|1⟩ into the 3-qubit logical codeword.

    The encoding is:
        α|0⟩ + β|1⟩  →  α|000⟩ + β|111⟩

    Parameters
    ----------
    alpha:
        Amplitude of |0⟩.
    beta:
        Amplitude of |1⟩.

    Returns
    -------
    np.ndarray
        8-element complex state vector representing α|000⟩ + β|111⟩.

    Raises
    ------
    ValueError
        If |α|² + |β|² ≠ 1.
    """
    norm_sq = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm_sq - 1.0) > 1e-8:
        raise ValueError(
            f"|α|² + |β|² = {norm_sq:.6f} ≠ 1.  "
            "Provide normalised amplitudes."
        )
    state = np.zeros(8, dtype=complex)
    state[0] = alpha   # |000⟩
    state[7] = beta    # |111⟩
    return state


def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Return the fidelity F = |⟨ψ₁|ψ₂⟩|² between two pure states.

    Parameters
    ----------
    state1, state2:
        Complex state vectors of the same dimension.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    inner = np.dot(np.conj(state1), state2)
    return float(abs(inner) ** 2)


def state_description(state: np.ndarray) -> str:
    """Return a human-readable description of a 2-element state vector."""
    alpha, beta = state[0], state[1]
    parts = []
    if abs(alpha) > 1e-9:
        parts.append(f"({alpha:.4f})|0⟩")
    if abs(beta) > 1e-9:
        parts.append(f"({beta:.4f})|1⟩")
    return " + ".join(parts) if parts else "0 (zero vector)"


def get_named_state(name: str) -> Tuple[np.ndarray, str]:
    """Return the state vector and LaTeX label for a named state.

    Parameters
    ----------
    name:
        One of the keys in NAMED_STATES.

    Returns
    -------
    Tuple[np.ndarray, str]
        (state_vector, latex_label)

    Raises
    ------
    ValueError
        If the name is not recognised.
    """
    if name not in NAMED_STATES:
        raise ValueError(
            f"Unknown state name '{name}'. "
            f"Choose from: {list(NAMED_STATES.keys())}"
        )
    return NAMED_STATES[name], name
