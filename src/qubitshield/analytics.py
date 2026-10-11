"""
analytics.py — Theoretical Failure Probability Functions

This module implements the theoretical (closed-form) failure probability
formulas for the unencoded qubit and the 3-qubit bit-flip repetition code.

Mathematical derivations
------------------------

**Unencoded qubit:**
    Under an independent bit-flip channel with probability p, the logical
    qubit is flipped with probability p:
        P_unencoded(p) = p

**Three-qubit bit-flip repetition code:**
    The code corrects any single bit-flip error. It fails only when two
    or more qubits experience bit-flip errors. The code fails when:

    - Exactly 2 qubits flip: There are C(3,2)=3 such patterns.
      Each has probability p²(1-p).
    - Exactly 3 qubits flip: There is C(3,3)=1 such pattern.
      It has probability p³.

    Total logical failure probability:
        P_logical(p) = 3p²(1-p) + p³
                     = 3p² - 3p³ + p³
                     = 3p² - 2p³

    This formula assumes:
    - Independent identically distributed bit-flip errors on data qubits.
    - No errors in encoding, syndrome measurement, or recovery gates.
    - Perfect ancilla preparation.

    This is the idealized model. In practice, the encoding circuit, ancilla
    preparation, syndrome measurement, and recovery all introduce additional
    error probability, which would need to be included for a fault-tolerant
    analysis.

**Break-even point:**
    The code outperforms the unencoded qubit when P_logical(p) < P_unencoded(p):
        3p² - 2p³ < p
        3p - 2p² < 1    (dividing by p > 0)
        -2p² + 3p - 1 < 0
        2p² - 3p + 1 > 0
        (2p - 1)(p - 1) > 0

    This is satisfied for 0 < p < 0.5 and p > 1.
    Since p ∈ [0, 1], the code is beneficial for p < 0.5.
    At p = 0.5, P_logical = P_unencoded = 0.5 (code provides no benefit).
    At p > 0.5, the code performs WORSE than the unencoded qubit.
"""

from __future__ import annotations

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Theoretical failure probability functions
# ---------------------------------------------------------------------------

def theoretical_unencoded_failure(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the theoretical logical failure probability for an unencoded qubit.

    Under an independent bit-flip channel with probability p:
        P_unencoded(p) = p

    Parameters
    ----------
    p:
        Physical bit-flip probability, or array of probabilities.
        Must satisfy 0 ≤ p ≤ 1.

    Returns
    -------
    float or np.ndarray
        Logical failure probability.

    Raises
    ------
    ValueError
        If any value in p is outside [0, 1].
    """
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(
            f"Physical error probability must be in [0, 1]; "
            f"got values outside this range."
        )
    return p


def theoretical_encoded_failure(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the theoretical logical failure probability for the 3-qubit repetition code.

    Under independent bit-flip errors on each data qubit with probability p,
    and assuming ideal encoding, syndrome measurement, and recovery:

        P_logical(p) = 3p²(1-p) + p³ = 3p² - 2p³

    Parameters
    ----------
    p:
        Physical bit-flip probability, or array of probabilities.
        Must satisfy 0 ≤ p ≤ 1.

    Returns
    -------
    float or np.ndarray
        Logical failure probability under the repetition code.

    Raises
    ------
    ValueError
        If any value in p is outside [0, 1].
    """
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError(
            f"Physical error probability must be in [0, 1]."
        )
    return 3 * p**2 * (1 - p) + p**3


def code_gain(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return the ratio of unencoded to encoded failure probability.

    A ratio > 1 means the code provides a benefit.
    A ratio = 1 means the code is equivalent (at p = 0.5).
    A ratio < 1 means the code makes things worse (at p > 0.5).

    Parameters
    ----------
    p:
        Physical bit-flip probability.

    Returns
    -------
    float or np.ndarray
        P_unencoded / P_logical. Returns NaN where P_logical = 0.
    """
    p = np.asarray(p, dtype=float)
    p_encoded = theoretical_encoded_failure(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_encoded > 0, p / p_encoded, np.nan)
    return gain


def breakeven_point() -> float:
    """Return the physical error probability at which code benefit vanishes.

    P_logical(p) = P_unencoded(p) has solutions p = 0 and p = 0.5.
    At p > 0.5, the code performs worse.

    Returns
    -------
    float
        The break-even physical error probability (0.5).
    """
    return 0.5


def reliability_curve_data(
    n_points: int = 200,
) -> dict:
    """Return a dictionary of arrays for plotting the reliability curve.

    Parameters
    ----------
    n_points:
        Number of probability values to sample.

    Returns
    -------
    dict with keys:
        - 'p': array of physical error probabilities
        - 'p_unencoded': theoretical unencoded failure probability
        - 'p_encoded': theoretical encoded failure probability (repetition code)
        - 'gain': ratio of unencoded/encoded failure probability
        - 'breakeven': the break-even probability
    """
    p = np.linspace(0, 1, n_points)
    return {
        "p": p,
        "p_unencoded": theoretical_unencoded_failure(p),
        "p_encoded": theoretical_encoded_failure(p),
        "gain": code_gain(p),
        "breakeven": breakeven_point(),
    }


# ---------------------------------------------------------------------------
# Statistical utilities for confidence intervals
# ---------------------------------------------------------------------------

def wilson_confidence_interval(
    n_failures: int,
    n_trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a proportion.

    The Wilson interval is preferred over the normal approximation because it
    performs well at extreme proportions (near 0 or 1) and for small samples.

    Parameters
    ----------
    n_failures:
        Number of observed failures (successes in statistical terms).
    n_trials:
        Total number of trials.
    confidence:
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    (lower, upper):
        Lower and upper bounds of the confidence interval.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical Association.
    """
    import scipy.stats as stats

    if n_trials == 0:
        return (0.0, 1.0)

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    p_hat = n_failures / n_trials
    n = n_trials

    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return (lower, upper)
