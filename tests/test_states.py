"""
test_states.py — Tests for qubitshield.states

Tests cover:
- State normalization validation
- Known quantum state values
- Logical codeword correctness
- State fidelity calculations
- Bloch sphere conversion
- encode_state function
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from qubitshield.states import (
    is_normalized,
    assert_normalized,
    ZERO, ONE, PLUS, MINUS, Y_PLUS, Y_MINUS,
    logical_zero, logical_one, logical_plus, logical_minus,
    bloch_angles_to_statevector,
    encode_state,
    state_fidelity,
    state_description,
    get_named_state,
    NAMED_STATES,
)


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestIsNormalized:
    def test_zero_state_normalized(self):
        assert is_normalized(ZERO)

    def test_one_state_normalized(self):
        assert is_normalized(ONE)

    def test_plus_state_normalized(self):
        assert is_normalized(PLUS)

    def test_minus_state_normalized(self):
        assert is_normalized(MINUS)

    def test_unnormalized_state_fails(self):
        state = np.array([1.0, 1.0])
        assert not is_normalized(state)

    def test_zero_vector_fails(self):
        state = np.array([0.0, 0.0])
        assert not is_normalized(state)

    def test_near_normalized_passes(self):
        state = np.array([1.0, 1e-10])  # nearly |0⟩
        assert is_normalized(state)

    def test_all_named_states_normalized(self):
        for name, state in NAMED_STATES.items():
            assert is_normalized(state), f"State {name} is not normalized"

    def test_assert_normalized_raises_on_invalid(self):
        with pytest.raises(ValueError, match="not normalised"):
            assert_normalized(np.array([2.0, 0.0]), "test_state")

    def test_assert_normalized_passes_on_valid(self):
        assert_normalized(ZERO)  # should not raise


# ---------------------------------------------------------------------------
# Standard state values
# ---------------------------------------------------------------------------

class TestStandardStates:
    def test_zero_is_correct(self):
        assert ZERO[0] == pytest.approx(1.0)
        assert ZERO[1] == pytest.approx(0.0)

    def test_one_is_correct(self):
        assert ONE[0] == pytest.approx(0.0)
        assert ONE[1] == pytest.approx(1.0)

    def test_plus_is_correct(self):
        expected = 1.0 / math.sqrt(2)
        assert PLUS[0] == pytest.approx(expected)
        assert PLUS[1] == pytest.approx(expected)

    def test_minus_is_correct(self):
        expected = 1.0 / math.sqrt(2)
        assert MINUS[0] == pytest.approx(expected)
        assert MINUS[1] == pytest.approx(-expected)

    def test_zero_and_one_orthogonal(self):
        inner = np.dot(np.conj(ZERO), ONE)
        assert abs(inner) == pytest.approx(0.0)

    def test_plus_and_minus_orthogonal(self):
        inner = np.dot(np.conj(PLUS), MINUS)
        assert abs(inner) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Logical codeword states
# ---------------------------------------------------------------------------

class TestLogicalCodewords:
    def test_logical_zero_shape(self):
        state = logical_zero()
        assert state.shape == (8,)

    def test_logical_one_shape(self):
        state = logical_one()
        assert state.shape == (8,)

    def test_logical_zero_is_000(self):
        state = logical_zero()
        assert state[0] == pytest.approx(1.0)   # |000⟩ = index 0
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_logical_one_is_111(self):
        state = logical_one()
        assert state[7] == pytest.approx(1.0)   # |111⟩ = index 7
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_logical_plus_is_superposition(self):
        state = logical_plus()
        expected_amp = 1.0 / math.sqrt(2)
        assert state[0] == pytest.approx(expected_amp)
        assert state[7] == pytest.approx(expected_amp)
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_logical_minus_is_superposition(self):
        state = logical_minus()
        expected_amp = 1.0 / math.sqrt(2)
        assert state[0] == pytest.approx(expected_amp)
        assert state[7] == pytest.approx(-expected_amp)
        assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_logical_zero_one_orthogonal(self):
        fidelity = state_fidelity(logical_zero(), logical_one())
        assert fidelity == pytest.approx(0.0)

    def test_logical_plus_minus_orthogonal(self):
        fidelity = state_fidelity(logical_plus(), logical_minus())
        assert fidelity == pytest.approx(0.0)

    def test_logical_zero_normalized(self):
        assert is_normalized(logical_zero())

    def test_logical_one_normalized(self):
        assert is_normalized(logical_one())


# ---------------------------------------------------------------------------
# encode_state
# ---------------------------------------------------------------------------

class TestEncodeState:
    def test_encode_zero(self):
        state = encode_state(1.0, 0.0)
        assert state[0] == pytest.approx(1.0)
        assert state[7] == pytest.approx(0.0)

    def test_encode_one(self):
        state = encode_state(0.0, 1.0)
        assert state[0] == pytest.approx(0.0)
        assert state[7] == pytest.approx(1.0)

    def test_encode_plus(self):
        a = 1.0 / math.sqrt(2)
        state = encode_state(a, a)
        assert state[0] == pytest.approx(a)
        assert state[7] == pytest.approx(a)

    def test_encode_minus(self):
        a = 1.0 / math.sqrt(2)
        state = encode_state(a, -a)
        assert state[0] == pytest.approx(a)
        assert state[7] == pytest.approx(-a)

    def test_encode_preserves_norm(self):
        for alpha in [1.0, 0.0, 1/math.sqrt(2)]:
            beta = math.sqrt(1 - alpha**2)
            state = encode_state(alpha, beta)
            assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    def test_encode_raises_on_unnormalized(self):
        with pytest.raises(ValueError, match=r"≠ 1"):
            encode_state(1.0, 1.0)  # norm² = 2 ≠ 1

    def test_encode_matches_logical_zero(self):
        encoded = encode_state(1.0, 0.0)
        np.testing.assert_array_almost_equal(encoded, logical_zero())

    def test_encode_matches_logical_one(self):
        encoded = encode_state(0.0, 1.0)
        np.testing.assert_array_almost_equal(encoded, logical_one())


# ---------------------------------------------------------------------------
# State fidelity
# ---------------------------------------------------------------------------

class TestStateFidelity:
    def test_same_state_fidelity_one(self):
        assert state_fidelity(ZERO, ZERO) == pytest.approx(1.0)
        assert state_fidelity(PLUS, PLUS) == pytest.approx(1.0)

    def test_orthogonal_states_fidelity_zero(self):
        assert state_fidelity(ZERO, ONE) == pytest.approx(0.0)

    def test_fidelity_symmetric(self):
        f12 = state_fidelity(PLUS, ZERO)
        f21 = state_fidelity(ZERO, PLUS)
        assert f12 == pytest.approx(f21)

    def test_fidelity_range(self):
        f = state_fidelity(PLUS, ZERO)
        assert 0.0 <= f <= 1.0

    def test_fidelity_plus_minus(self):
        # ⟨+|−⟩ = 0 so fidelity = 0
        assert state_fidelity(PLUS, MINUS) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bloch sphere conversion
# ---------------------------------------------------------------------------

class TestBlochAngles:
    def test_theta_zero_gives_zero_state(self):
        state = bloch_angles_to_statevector(0.0, 0.0)
        np.testing.assert_array_almost_equal(state, ZERO)

    def test_theta_pi_gives_one_state(self):
        state = bloch_angles_to_statevector(math.pi, 0.0)
        # cos(π/2)|0⟩ + sin(π/2)|1⟩ = |1⟩
        assert abs(state[1]) == pytest.approx(1.0, abs=1e-7)

    def test_theta_half_pi_gives_plus_like(self):
        state = bloch_angles_to_statevector(math.pi / 2, 0.0)
        # cos(π/4)|0⟩ + sin(π/4)|1⟩ = |+⟩
        expected = 1.0 / math.sqrt(2)
        assert abs(state[0]) == pytest.approx(expected)
        assert abs(state[1]) == pytest.approx(expected)

    def test_result_is_normalized(self):
        for theta in [0, math.pi/4, math.pi/2, math.pi]:
            for phi in [0, math.pi/2, math.pi]:
                state = bloch_angles_to_statevector(theta, phi)
                assert is_normalized(state), f"Not normalized for theta={theta}, phi={phi}"
