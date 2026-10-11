# Limitations — QubitShield

This document catalogues the known limitations of the QubitShield simulation, explains
their origins, and describes what would need to change to address each one.

---

## 1. Narrow Error Model

**What the simulation models:**
Independent, identically distributed bit-flip (X) errors on data qubits with probability p.

**What it does NOT model:**
- Phase-flip (Z) errors
- Depolarizing noise (mixture of X, Z, Y errors)
- Amplitude damping (energy loss / T₁ relaxation)
- Dephasing (T₂ decay)
- Correlated errors (e.g., from cross-talk between adjacent qubits)
- Leakage errors (qubit leaving the computational subspace)
- State preparation and measurement (SPAM) errors

**Why this matters:**
Real quantum processors experience all of the above. The bit-flip channel is a useful
pedagogical abstraction but does not represent the noise experienced by actual qubits.

---

## 2. Gate and Measurement Overhead (Idealized Operations)

**Assumption:** The encoding circuit, CNOT gates, ancilla preparation, syndrome
measurement, and recovery are all performed without error.

**Reality:** Each physical gate introduces error probability ε_gate. In current
superconducting processors, ε_gate ≈ 0.1–1% for two-qubit gates. The encoding
circuit alone requires 2 CNOT gates, and syndrome extraction requires 4 more
(total 6 CNOTs). At ε = 0.1%, this adds approximately 0.6% error per cycle.

**Consequence:** For the repetition code to be beneficial, the physical error rate
must be low enough that the correction benefit outweighs the overhead from
imperfect encoding/decoding operations. The break-even threshold becomes lower
than the ideal p = 0.5 when gate errors are included.

**What would be needed:** A circuit-level noise model using Qiskit Aer's
noise simulation, with parameterized gate error rates. The analysis would show
that the code is only beneficial below a significantly lower physical error rate.

---

## 3. Finite Sample Statistical Uncertainty

**What the simulation does:** Monte Carlo sampling with a finite number of trials n.

**Uncertainty:** The estimated failure rate has statistical error ≈ √(p(1-p)/n).
At p = 0.1 with n = 10,000: uncertainty ≈ 0.003 (0.3 percentage points).

**Mitigation:** Wilson confidence intervals are displayed in the dashboard.
Seeded reproducibility allows exact reproduction of results.

**What would be needed for publication-quality results:** n ≥ 10⁵ trials per
data point, systematic error bars, and comparison across multiple seeds.

---

## 4. Non-Fault-Tolerant Syndrome Extraction

**What the project implements:** Ideal syndrome extraction where the ancilla
CNOT gates are error-free.

**Reality:** An error on the ancilla during syndrome extraction can produce
a wrong syndrome reading without actually corresponding to a data-qubit error.
This is called a "measurement error" and requires flag qubits or repeat
measurements to detect.

**What would be needed:** Implement fault-tolerant syndrome extraction using
flag qubits (Chamberland & Cross, 2018) or repeated measurements.

---

## 5. Classical Post-Processing vs. Quantum Feedback

**What the project does:** The syndrome is measured, decoded in Python, and the
correction gate is applied in a separate circuit stage (equivalent to delayed
classical feedback).

**Real quantum requirement:** True fault-tolerant operation requires real-time
classical feedback: measure ancilla → decode → apply correction — all within the
qubit coherence time. This is challenging because classical processing takes
microseconds while qubit coherence times are in the range of microseconds to
milliseconds.

**What would be needed:** On actual hardware, this requires fast classical control
electronics (FPGAs) and Qiskit Primitives with mid-circuit measurement support.

---

## 6. Simulation Limits

**Exponential scaling:** Statevector simulation requires O(2ⁿ) complex numbers.
The current 5-qubit simulation (3 data + 2 ancilla) uses a 32-element statevector.
A 50-qubit simulation would require 2⁵⁰ ≈ 10¹⁵ elements — infeasible on classical hardware.

**No shot noise from hardware:** Qiskit Aer's statevector simulator is exact
(no hardware shot-to-shot variation, no readout noise, no crosstalk).

---

## 7. What Would Be Needed for Real Hardware Evaluation

To meaningfully evaluate this code on actual quantum hardware:
1. Access to a quantum processor with calibrated error rates (e.g., IBM Quantum)
2. Circuit transpilation to the hardware's native gate set
3. Readout-error mitigation (e.g., M3 mitigation)
4. Repeated execution to build statistics
5. Comparison with hardware characterisation data (T₁, T₂, gate fidelity)
6. Understanding of device-specific noise sources
7. Publication-quality statistical analysis
