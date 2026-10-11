"""
test_syndrome.py — Tests for qubitshield.syndrome

Tests cover:
- Syndrome table completeness and correctness
- decode_syndrome function
- All expected single-qubit error syndromes
- Two-error failure case syndrome patterns
- Bit-ordering consistency
- Invalid input handling
"""

from __future__ import annotations

import pytest

from qubitshield.syndrome import (
    decode_syndrome,
    parse_measurement_string,
    syndrome_string_from_counts,
    SYNDROME_TABLE,
    TWO_ERROR_CASES,
)


# ---------------------------------------------------------------------------
# Syndrome table
# ---------------------------------------------------------------------------

class TestSyndromeTable:
    def test_table_has_four_entries(self):
        assert len(SYNDROME_TABLE) == 4

    def test_no_error_syndrome(self):
        correction, desc = SYNDROME_TABLE[(0, 0)]
        assert correction is None

    def test_x0_error_syndrome(self):
        correction, desc = SYNDROME_TABLE[(1, 0)]
        assert correction == 0

    def test_x1_error_syndrome(self):
        correction, desc = SYNDROME_TABLE[(1, 1)]
        assert correction == 1

    def test_x2_error_syndrome(self):
        correction, desc = SYNDROME_TABLE[(0, 1)]
        assert correction == 2

    def test_all_descriptions_are_strings(self):
        for key, (correction, desc) in SYNDROME_TABLE.items():
            assert isinstance(desc, str), f"Description for {key} is not a string"

    def test_correction_values_valid(self):
        valid_corrections = {None, 0, 1, 2}
        for key, (correction, desc) in SYNDROME_TABLE.items():
            assert correction in valid_corrections


# ---------------------------------------------------------------------------
# decode_syndrome function
# ---------------------------------------------------------------------------

class TestDecodeSyndrome:
    def test_no_error(self):
        correction, desc = decode_syndrome(0, 0)
        assert correction is None

    def test_x0_error(self):
        correction, desc = decode_syndrome(1, 0)
        assert correction == 0

    def test_x1_error(self):
        correction, desc = decode_syndrome(1, 1)
        assert correction == 1

    def test_x2_error(self):
        correction, desc = decode_syndrome(0, 1)
        assert correction == 2

    def test_invalid_syndrome_bits_raises(self):
        with pytest.raises(ValueError, match="Syndrome bits must be 0 or 1"):
            decode_syndrome(2, 0)

    def test_invalid_s12_raises(self):
        with pytest.raises(ValueError):
            decode_syndrome(0, -1)

    def test_description_contains_qubit_info(self):
        _, desc = decode_syndrome(1, 0)
        assert "qubit" in desc.lower() or "0" in desc

    def test_no_error_description(self):
        _, desc = decode_syndrome(0, 0)
        assert "no" in desc.lower() or "none" in desc.lower() or "detected" in desc.lower()


# ---------------------------------------------------------------------------
# Syndrome consistency with the error table
# ---------------------------------------------------------------------------

class TestSyndromeConsistency:
    """
    Verify syndrome derivations analytically:
    s01 = e0 XOR e1  (parity of qubit 0 and 1 errors)
    s12 = e1 XOR e2  (parity of qubit 1 and 2 errors)
    """

    def _compute_syndrome(self, errors):
        e0 = 1 if 0 in errors else 0
        e1 = 1 if 1 in errors else 0
        e2 = 1 if 2 in errors else 0
        return (e0 ^ e1, e1 ^ e2)

    def test_no_error_gives_00(self):
        assert self._compute_syndrome([]) == (0, 0)

    def test_x0_gives_10(self):
        assert self._compute_syndrome([0]) == (1, 0)

    def test_x1_gives_11(self):
        assert self._compute_syndrome([1]) == (1, 1)

    def test_x2_gives_01(self):
        assert self._compute_syndrome([2]) == (0, 1)

    def test_x0x1_gives_01(self):
        # Two-qubit error: syndrome mimics X2
        assert self._compute_syndrome([0, 1]) == (0, 1)

    def test_x0x2_gives_11(self):
        # Two-qubit error: syndrome mimics X1
        assert self._compute_syndrome([0, 2]) == (1, 1)

    def test_x1x2_gives_10(self):
        # Two-qubit error: syndrome mimics X0
        assert self._compute_syndrome([1, 2]) == (1, 0)

    def test_syndrome_decoder_correct_for_single_errors(self):
        """Verify that decode_syndrome identifies the right qubit for each single error."""
        for qubit in [0, 1, 2]:
            syndrome = self._compute_syndrome([qubit])
            correction, _ = decode_syndrome(*syndrome)
            assert correction == qubit, (
                f"Expected correction on qubit {qubit} but got {correction} "
                f"for syndrome {syndrome}"
            )

    def test_two_error_syndromes_match_wrong_single_error(self):
        """Verify two-error syndrome matches the WRONG qubit (failure case)."""
        two_error_cases = {
            (0, 1): 2,  # X0X1 mimics X2
            (0, 2): 1,  # X0X2 mimics X1
            (1, 2): 0,  # X1X2 mimics X0
        }
        for errors, wrong_qubit in two_error_cases.items():
            syndrome = self._compute_syndrome(list(errors))
            correction, _ = decode_syndrome(*syndrome)
            assert correction == wrong_qubit, (
                f"X{errors[0]}X{errors[1]} syndrome {syndrome}: "
                f"expected decoder to choose qubit {wrong_qubit}, got {correction}"
            )


# ---------------------------------------------------------------------------
# parse_measurement_string
# ---------------------------------------------------------------------------

class TestParseMeasurementString:
    def test_single_register_no_error(self):
        s01, s12 = parse_measurement_string("00")
        assert (s01, s12) == (0, 0)

    def test_single_register_s01_set(self):
        # bit 0 (rightmost) = s01, bit 1 = s12
        s01, s12 = parse_measurement_string("01")  # rightmost=1 = s01=1, s12=0
        assert s01 == 1
        assert s12 == 0

    def test_single_register_s12_set(self):
        s01, s12 = parse_measurement_string("10")  # rightmost=0, second=1
        assert s01 == 0
        assert s12 == 1

    def test_single_register_both_set(self):
        s01, s12 = parse_measurement_string("11")
        assert s01 == 1
        assert s12 == 1


# ---------------------------------------------------------------------------
# syndrome_string_from_counts
# ---------------------------------------------------------------------------

class TestSyndromeStringFromCounts:
    def test_most_probable_selected(self):
        counts = {"00": 900, "01": 50, "10": 50}
        syndrome, raw, count = syndrome_string_from_counts(counts)
        assert raw == "00"
        assert count == 900

    def test_empty_counts_raises(self):
        with pytest.raises(ValueError, match="empty"):
            syndrome_string_from_counts({})


# ---------------------------------------------------------------------------
# Two-error case documentation
# ---------------------------------------------------------------------------

class TestTwoErrorCases:
    def test_all_three_cases_documented(self):
        assert len(TWO_ERROR_CASES) == 3

    def test_case_descriptions_non_empty(self):
        for key, case in TWO_ERROR_CASES.items():
            assert "explanation" in case
            assert len(case["explanation"]) > 20

    def test_case_errors_have_two_qubits(self):
        for key, case in TWO_ERROR_CASES.items():
            assert len(case["errors"]) == 2
