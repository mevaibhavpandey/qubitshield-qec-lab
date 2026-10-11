# experimental_methodology.md

# Experimental Methodology — QubitShield

This document explains the error model, experimental parameters, random sampling
procedure, reproducibility guarantees, and the distinction between theoretical
predictions and simulation results.

---

## Error Model

### Physical Channel
Each data qubit is modelled as independently experiencing a bit-flip error with
probability p. This is the **i.i.d. bit-flip channel**:

$$\mathcal{E}(\rho) = (1-p)\rho + p \cdot X\rho X^\dagger$$

For a pure state |ψ⟩, this corresponds to applying X with probability p and
leaving the state unchanged with probability 1-p.

### Assumptions
1. Errors occur independently across the three data qubits
2. Encoding (Initialize + CNOT gates) is performed without errors
3. Syndrome measurement (CNOT + ancilla measurement) is performed without errors  
4. Recovery gates are performed without errors
5. The initial state is perfectly prepared

These assumptions define the **idealized** model. The theoretical formula
P_logical = 3p² - 2p³ is derived under exactly these assumptions.

---

## Experiment Types

### Type 1: Deterministic Syndrome Lab Experiments
- **Purpose:** Verify the syndrome-to-correction mapping for specific error patterns
- **Method:** Build a Qiskit circuit, inject specified X errors, extract syndrome via ancilla measurement, verify state fidelity via statevector simulation
- **Parameters:** Error pattern, input state (α, β), number of shots for measurement histogram
- **Validation:** State fidelity F = |⟨ψ_target|ψ_actual⟩|² must be ≥ 0.99 for correctable errors, < 0.99 for failure cases

### Type 2: Monte Carlo Reliability Benchmark
- **Purpose:** Compare theoretical and simulated logical failure probability across p values
- **Method:** Random sampling of error patterns, analytical syndrome computation, classical recovery, logical failure determination
- **Parameters:** Physical error probability p, number of trials n, random seed s
- **Output:** Simulated failure rate, 95% Wilson confidence interval, theoretical prediction, absolute difference

---

## Random Sampling Procedure

```python
rng = numpy.random.default_rng(seed)

# For each trial, sample 3 independent Bernoulli(p) events:
error_bits = rng.random(3) < p
# error_bits[i] is True iff qubit i experiences a bit-flip

# Convert to error pattern list
error_pattern = [i for i in range(3) if error_bits[i]]

# Apply analytical syndrome decoder
logical_failure, syndrome, correction = monte_carlo_recovery(error_pattern)
```

The analytical Monte Carlo path (in `recovery.py:monte_carlo_recovery`) computes the
syndrome from the error pattern using XOR logic rather than circuit simulation.
This is orders of magnitude faster and produces identical results to circuit simulation
under the ideal model.

---

## Reproducibility

Every experiment is seeded. The seed determines the complete sequence of random numbers
for all trials. Provided the same seed, n_trials, and p value, an experiment produces
identical results on any platform running NumPy's PCG64 generator.

### Verification of reproducibility
```python
r1 = run_single_experiment(p=0.15, n_trials=2000, seed=100)
r2 = run_single_experiment(p=0.15, n_trials=2000, seed=100)
assert r1.n_failures == r2.n_failures  # always True
```

### CSV export
Each exported CSV contains the seed and n_trials columns, allowing any result to be
reproduced by re-running with the same parameters.

---

## Statistical Methodology

### Estimating Failure Rate
With n_failures observed in n_trials, the estimated failure rate is:
$$\hat{p} = n_{\text{failures}} / n_{\text{trials}}$$

### Confidence Intervals
We use the Wilson score confidence interval (Wilson, 1927) at 95% confidence level.
For a 95% CI with z = 1.96:

$$\hat{p}_W = \frac{\hat{p} + z^2/(2n)}{1 + z^2/n}$$

$$\text{CI} = \hat{p}_W \pm \frac{z\sqrt{\hat{p}(1-\hat{p})/n + z^2/(4n^2)}}{1 + z^2/n}$$

The Wilson CI is preferred over the normal approximation because it:
- Remains valid near p = 0 and p = 1
- Provides good coverage for small samples (n < 100)
- Is always within [0, 1]

### Convergence
By the law of large numbers, $\hat{p} \to P_{\text{logical}}$ as n → ∞.
The standard error is approximately $\sqrt{P(1-P)/n}$.

At p = 0.1 (P_logical ≈ 0.028) with n = 10,000: σ ≈ 0.0016.
At p = 0.3 (P_logical ≈ 0.216) with n = 10,000: σ ≈ 0.0041.

---

## Distinction: Theoretical vs. Simulated Results

| Aspect | Theoretical | Simulated |
|--------|------------|-----------|
| Source | Closed-form formula | Random sampling |
| Accuracy | Exact (under model assumptions) | Statistical (±σ) |
| Computation | O(1) | O(n × n_p) |
| Assumptions | Stated explicitly in analytics.py | Same idealized model |
| Usefulness | Reference curve | Validation of model |

The simulation's primary purpose is to **validate** that the Monte Carlo procedure
correctly implements the idealized model described by the theoretical formula. If the
theoretical and simulated curves agree within statistical uncertainty, both the theory
and the simulation are internally consistent.

The simulation does NOT independently validate that the theoretical model describes
real quantum hardware behavior.
