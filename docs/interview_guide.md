# Interview Guide — QubitShield

> Practical preparation for quantum computing internship interviews.
> Based on the actual code implemented in this repository.

---

## 30-Second Project Pitch

*"I built QubitShield, an interactive quantum error-correction lab using Python, Qiskit, and Streamlit. The project simulates the three-qubit bit-flip repetition code, which protects a logical qubit against independent bit-flip errors. I implemented the complete pipeline: quantum circuit construction, ancilla-based syndrome extraction, syndrome decoding, and classical recovery. I then compared the theoretical logical failure probability — which follows P = 3p² − 2p³ — with genuine Monte Carlo simulation results. The dashboard lets you inject errors, inspect syndrome measurements, visualise the Qiskit circuit diagrams, run reproducible benchmarks, and export results as CSV. I also wrote a full pytest suite covering normalization, syndrome decoding correctness, and statistical validity."*

---

## Two-Minute Technical Walkthrough

1. **Motivation** (20s): Quantum processors are error-prone. Error correction is essential for useful computation. The three-qubit bit-flip code is the simplest non-trivial correcting code and illustrates the core ideas.

2. **Encoding** (20s): The logical state α|0⟩ + β|1⟩ is spread to α|000⟩ + β|111⟩ using two CNOT gates. This exploits entanglement — it is not classical bit copying.

3. **Syndrome Extraction** (20s): Two ancilla qubits measure the parity checks Z₀Z₁ and Z₁Z₂ without directly measuring and collapsing the logical state. A 4-row syndrome table maps (s₀₁, s₁₂) → correction qubit.

4. **Recovery and Verification** (20s): After syndrome measurement, I apply the decoder classically (equivalent to quantum feedback under ideal conditions) and verify the output state fidelity using Qiskit Aer's statevector simulator.

5. **Benchmark** (20s): Monte Carlo experiments sample random error patterns at each physical error probability p. The simulated logical failure rate is compared against P_logical = 3p² − 2p³. The code outperforms the unencoded qubit for p < 0.5.

6. **Testing** (20s): The test suite validates normalization, syndrome mappings, single-error correction, two-error failures, failure probability formulas, reproducibility, and CSV export validity.

---

## Mathematical Foundations to Know

1. **Qubit state normalization:** |α|² + |β|² = 1 for |ψ⟩ = α|0⟩ + β|1⟩
2. **Pauli operators:** X (bit-flip), Z (phase-flip), Y = iXZ. Know their matrix representations.
3. **Tensor product states:** |q₀q₁q₂⟩ = |q₀⟩ ⊗ |q₁⟩ ⊗ |q₂⟩ is an 8-dimensional vector.
4. **CNOT gate:** Maps |c, t⟩ → |c, c⊕t⟩. Used in encoding.
5. **Stabilizer codes:** A set of operators {g_i} whose +1 eigenspace is the codespace. Errors detectable = those anticommuting with stabilizers.
6. **Syndrome:** Measurement result of stabilizer generators. Bit = 0 for +1 eigenvalue (no error), 1 for −1 eigenvalue (error detected).
7. **Logical failure probability derivation:** Sum of probabilities for all uncorrectable error patterns.
8. **No-cloning theorem:** You cannot clone an unknown quantum state.
9. **State fidelity:** F = |⟨ψ₁|ψ₂⟩|² ∈ [0,1].
10. **Wilson confidence interval:** Preferred CI for proportions at extreme values.

---

## 20 Technical Interview Questions

### Q1: What is the three-qubit repetition code, and what does it correct?
**Answer:** It encodes one logical qubit into three physical qubits as α|000⟩ + β|111⟩. It corrects any single bit-flip (X) error on the data qubits, provided encoding, measurement, and recovery are ideal. It does NOT correct phase-flip (Z) errors.

### Q2: Why can't you just copy a qubit classically?
**Answer:** The no-cloning theorem (Wootters & Zurek, 1982) proves there is no unitary operation that copies an unknown quantum state. The encoding creates an entangled logical state, not three copies.

### Q3: How does syndrome extraction avoid measuring and destroying the logical qubit?
**Answer:** Ancilla qubits are coupled to the data qubits via CNOT gates to accumulate parity information. Measuring the ancilla reveals the syndrome without directly measuring the data qubits in the computational basis.

### Q4: Explain the syndrome mapping for the three-qubit code.
**Answer:** The parity checks are g₁ = Z₀Z₁ and g₂ = Z₁Z₂. For error X_i:
- X₀: anticommutes with g₁ (s₀₁=1), commutes with g₂ (s₁₂=0) → (1,0)
- X₁: anticommutes with both (s₀₁=1, s₁₂=1) → (1,1)  
- X₂: commutes with g₁ (s₀₁=0), anticommutes with g₂ (s₁₂=1) → (0,1)
- No error: (0,0)

### Q5: Why does the code fail for two-qubit errors?
**Answer:** Each two-qubit error pattern produces a syndrome identical to a single error on the third qubit. The decoder applies the wrong correction, creating three total errors — equivalent to a logical X flip.

### Q6: Derive P_logical = 3p² − 2p³.
**Answer:** The code corrects up to one error. Failure occurs for exactly 2 errors [C(3,2)=3 patterns, each with probability p²(1-p)] or 3 errors [1 pattern, probability p³]:
P = 3p²(1-p) + p³ = 3p² - 3p³ + p³ = 3p² - 2p³.

### Q7: At what physical error rate does the code break even?
**Answer:** Set 3p² − 2p³ = p → 2p² − 3p + 1 = 0 → (2p−1)(p−1) = 0. The non-trivial solution is p = 0.5. For p < 0.5, the code helps; for p > 0.5, it makes things worse.

### Q8: Why does the code make things worse for p > 0.5?
**Answer:** When individual qubit flip probability exceeds 50%, the majority of physical qubits are in the "wrong" state. The majority-vote decoder then "corrects" toward the erroneous majority, amplifying the failure rate.

### Q9: What is the Hamming distance of this code?
**Answer:** The minimum Hamming distance is d = 3. This means the code can detect up to d−1 = 2 errors and correct up to ⌊(d−1)/2⌋ = 1 error.

### Q10: How does your implementation verify that error correction worked?
**Answer:** I use Qiskit Aer's statevector simulator to compute the exact quantum state after encoding, error injection, and recovery. I then compute the state fidelity F = |⟨ψ_target|ψ_actual⟩|² and check it is ≥ 0.99.

### Q11: Why do you implement recovery classically rather than using quantum conditional gates?
**Answer:** Under ideal (noise-free) conditions, classical post-processing of the syndrome followed by a delayed correction is completely equivalent to real-time quantum feedback. This approach is transparent and easier to verify. For true fault-tolerant quantum computation, real-time feedback would be essential, but adds circuit complexity and requires hardware support.

### Q12: What is the difference between a statevector simulation and a shot-based simulation?
**Answer:** Statevector simulation computes the exact quantum state as a complex vector — exponential in qubit count, but precise. Shot-based simulation mimics individual measurements by sampling from the probability distribution, just as a real quantum computer would. For syndrome verification, I use shots to simulate measurement outcomes and statevector to verify the exact output state fidelity.

### Q13: Explain the bit-ordering issue in Qiskit measurement results.
**Answer:** Qiskit uses little-endian ordering within classical registers: in a result string, the rightmost character corresponds to classical bit 0. When reading a 2-bit syndrome string, the rightmost bit is s₀₁ (ancilla[0]) and the second-from-right is s₁₂ (ancilla[1]). Failing to account for this would produce wrong syndrome decoding. I explicitly document and test for this in test_circuits.py.

### Q14: What statistical method do you use for confidence intervals?
**Answer:** I use the Wilson score confidence interval, which is preferred over the normal approximation for proportions near 0 or 1 and for small sample sizes. It is documented in Wilson (1927) and implemented in analytics.py using scipy.stats.norm.ppf.

### Q15: What is the logical X operator for this code?
**Answer:** The logical X operator is X̄ = X₀X₁X₂ (flip all three physical qubits). It maps |0_L⟩ = |000⟩ to |111⟩ = |1_L⟩, which is a logical bit flip.

### Q16: Does this code protect against phase-flip errors? Why not?
**Answer:** No. The codewords |000⟩ and |111⟩ are +1 eigenstates of Z₀, Z₁, Z₂. A Z error on any qubit commutes with all stabilizers g₁ = Z₀Z₁ and g₂ = Z₁Z₂, so it produces no syndrome and goes undetected. The Shor 9-qubit code or CSS codes are needed to correct both X and Z errors.

### Q17: What would a fault-tolerant implementation require beyond this project?
**Answer:** Fault-tolerant computation requires: (1) error correction of the encoding circuit itself using ancilla verification, (2) measurement-error-corrected syndrome extraction, (3) a transversal or fault-tolerant gate set for logical operations, (4) magic state distillation for non-Clifford gates (e.g., T), and (5) achieving physical error rates below the fault-tolerance threshold (approximately 1% for surface codes).

### Q18: How does the Monte Carlo simulation relate to the theoretical formula?
**Answer:** The Monte Carlo simulation independently samples random error patterns and applies the syndrome decoder, counting logical failures. With sufficient trials (n → ∞), the simulated failure rate converges to the theoretical value by the law of large numbers. The convergence rate is governed by the standard error ≈ √(p(1-p)/n). I verify this agreement for various p values and display confidence intervals quantifying the remaining statistical uncertainty.

### Q19: What is the code rate of the three-qubit repetition code?
**Answer:** Code rate R = k/n where k = number of logical qubits encoded and n = number of physical qubits used. For the three-qubit code: R = 1/3. This is a significant overhead — one logical qubit requires three physical qubits plus two ancilla for syndrome extraction.

### Q20: How could you extend this project to correct both X and Z errors?
**Answer:** The Shor 9-qubit code concatenates two codes: the bit-flip repetition code (for X errors) and the phase-flip repetition code (for Z errors). Alternatively, the 5-qubit perfect code achieves [[5,1,3]] parameters and corrects arbitrary single-qubit errors with higher efficiency. Modern quantum computing research focuses on topological codes like the surface code, which has a high fault-tolerance threshold (~1%) and local operations on a 2D grid.

---

## Likely Design Questions

**"Why did you use the repetition code and not a more powerful code?"**
The repetition code is pedagogically ideal: it illustrates all core QEC concepts (encoding, stabilizers, syndrome extraction, recovery, failure modes) with minimal circuit complexity. More powerful codes would obscure these concepts without adding educational value for a portfolio project.

**"Why Streamlit instead of a web framework?"**
Streamlit enables rapid scientific dashboard development in pure Python. It is widely used in data science and quantum computing communities for interactive demonstrations. A React/Next.js frontend would be better for production, but would add significant frontend development overhead irrelevant to the quantum computing content.

**"How did you validate the syndrome decoding?"**
Three independent validation paths: (1) analytical derivation of syndromes from XOR of error indicator bits, (2) Qiskit shot-based simulation measuring actual ancilla qubits, (3) statevector fidelity verification after recovery. All three must agree.

---

## What This Project Does NOT Prove

- That the repetition code is useful for practical quantum computation (it corrects only one error type with 3× overhead)
- That quantum error correction will work on current NISQ hardware
- That simulated results match physical quantum hardware behavior
- That the code is fault-tolerant (encoding circuit errors are not corrected)
- Results from real quantum hardware (this uses classical simulation throughout)

---

## Credible Future Extensions

1. **Phase-flip code:** Implement the three-qubit phase-flip code and combine with the bit-flip code for a six-qubit code.
2. **Shor 9-qubit code:** Implement a code correcting arbitrary single-qubit errors.
3. **Circuit-level noise:** Add Qiskit Aer noise models (depolarizing, T1/T2 relaxation) to encoding, CNOT, measurement operations and evaluate code performance.
4. **Fault-tolerant circuits:** Implement flag qubits or Steane-style ancilla verification to make syndrome measurement itself fault-tolerant.
5. **Surface code introduction:** Implement a small (3×3) surface code and compare with the repetition code.
6. **Threshold estimation:** For a circuit-level noise model, numerically estimate the fault-tolerance threshold.
7. **Real hardware execution:** Submit circuits to IBM Quantum devices via the Qiskit Runtime API and compare results with simulation.
