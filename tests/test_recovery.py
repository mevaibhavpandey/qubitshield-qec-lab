"""
test_recovery.py — Tests for qubitshield.recovery

Tests verify:
- apply_recovery returns correct qubit for each syndrome
- Classical data bit flipping is correct
- monte_carlo_recovery analytical path is consistent with syndrome table
- No error → no logical failure
- Each single-qubit error → correctly corrected (no logical failure)
- Two-qubit errors → logical failure (code fails)
"""

from __future__ import annotations

import pytest

from qubitshield.recovery import apply_recovery, monte_carlo_recovery


# ---------------------------------------------------------------------------
# apply_recovery
# ---------------------------------------------------------------------------

class TestApplyRecovery:
    def test_no_error_syndrome(self):
        correction, desc = apply_recovery((0, 0))
        assert correction is None

    def test_x0_correction(self):
        correction, desc = apply_recovery((1, 0))
        assert correction == 0

    def test_x1_correction(self):
        correction, desc = apply_recovery((1, 1))
        assert correction == 1

    def test_x2_correction(self):
        correction, desc = apply_recovery((0, 1))
        assert correction == 2

    def test_data_bit_flipped_for_x0(self):
        data = [1, 1, 1]  # All qubits flipped
        apply_recovery((1, 0), data_bits=data)
        # Should flip qubit 0 back
        assert data == [0, 1, 1]

    def test_data_bit_flipped_for_x1(self):
        data = [0, 1, 0]  # Qubit 1 is the error
        apply_recovery((1, 1), data_bits=data)
        assert data == [0, 0, 0]  # Corrected

    def test_data_bit_flipped_for_x2(self):
        data = [0, 0, 1]
        apply_recovery((0, 1), data_bits=data)
        assert data == [0, 0, 0]

    def test_no_modification_when_no_error(self):
        data = [0, 0, 0]
        apply_recovery((0, 0), data_bits=data)
        assert data == [0, 0, 0]

    def test_wrong_data_bits_length_raises(self):
        with pytest.raises(ValueError, match="length 3"):
            apply_recovery((1, 0), data_bits=[0, 0])


# ---------------------------------------------------------------------------
# monte_carlo_recovery
# ---------------------------------------------------------------------------

class TestMonteCarloRecovery:
    def test_no_error_no_failure(self):
        failure, syndrome, correction = monte_carlo_recovery([])
        assert not failure
        assert syndrome == (0, 0)
        assert correction is None

    def test_x0_single_error_corrected(self):
        failure, syndrome, correction = monte_carlo_recovery([0])
        assert not failure
        assert syndrome == (1, 0)
        assert correction == 0

    def test_x1_single_error_corrected(self):
        failure, syndrome, correction = monte_carlo_recovery([1])
        assert not failure
        assert syndrome == (1, 1)
        assert correction == 1

    def test_x2_single_error_corrected(self):
        failure, syndrome, correction = monte_carlo_recovery([2])
        assert not failure
        assert syndrome == (0, 1)
        assert correction == 2

    def test_x0x1_two_error_fails(self):
        failure, syndrome, correction = monte_carlo_recovery([0, 1])
        assert failure
        assert syndrome == (0, 1)  # Looks like X2
        assert correction == 2     # Wrong correction

    def test_x0x2_two_error_fails(self):
        failure, syndrome, correction = monte_carlo_recovery([0, 2])
        assert failure
        assert syndrome == (1, 1)  # Looks like X1
        assert correction == 1     # Wrong correction

    def test_x1x2_two_error_fails(self):
        failure, syndrome, correction = monte_carlo_recovery([1, 2])
        assert failure
        assert syndrome == (1, 0)  # Looks like X0
        assert correction == 0     # Wrong correction

    def test_three_error_results_in_failure(self):
        failure, syndrome, correction = monte_carlo_recovery([0, 1, 2])
        # All three qubits flip: syndrome = (0,0), no correction, but logical failure
        # e0=e1=e2=1 → s01=0, s12=0 → no correction applied
        # After no correction: effective errors = {0,1,2} → 3 total flips (odd) = logical failure
        assert failure


# ---------------------------------------------------------------------------
# Consistency: syndrome table vs. monte_carlo_recovery
# ---------------------------------------------------------------------------

class TestSyndromeConsistencyWithRecovery:
    """Cross-check that monte_carlo_recovery and syndrome table agree."""

    def test_single_errors_all_corrected(self):
        for qubit in [0, 1, 2]:
            failure, syndrome, correction = monte_carlo_recovery([qubit])
            assert not failure, (
                f"Single X on qubit {qubit}: expected correction, got logical failure. "
                f"Syndrome={syndrome}, correction={correction}"
            )
            assert correction == qubit, (
                f"Single X on qubit {qubit}: expected correction on qubit {qubit}, "
                f"got {correction}"
            )

    def test_two_errors_all_fail(self):
        for errors in [[0,1], [0,2], [1,2]]:
            failure, syndrome, correction = monte_carlo_recovery(errors)
            assert failure, (
                f"Two-qubit error {errors}: expected logical failure, but code succeeded."
            )
