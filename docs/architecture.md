# Architecture — QubitShield

## Module Dependency Map

```
app.py  (Streamlit UI)
  ├── qubitshield.experiments    ← Syndrome Lab runners
  │     ├── qubitshield.circuits       ← Qiskit circuit builders
  │     ├── qubitshield.syndrome       ← Syndrome table & decoder
  │     └── qubitshield.recovery       ← Classical recovery, statevector verification
  ├── qubitshield.noise          ← Monte Carlo simulation engine
  │     ├── qubitshield.recovery       ← monte_carlo_recovery (fast path)
  │     └── qubitshield.analytics      ← Theoretical failure formulas
  ├── qubitshield.visualization  ← Plotly/Matplotlib figures
  │     └── qubitshield.analytics
  └── qubitshield.states         ← State vectors, fidelity
```

## Module Responsibilities

### `states.py`
- Defines standard single-qubit states (|0⟩, |1⟩, |+⟩, |−⟩, ...)
- Defines logical codeword states (logical_zero, logical_one, logical_plus, logical_minus)
- State normalization validation
- State fidelity: F = |⟨ψ₁|ψ₂⟩|²
- encode_state(α, β): produces 8-element logical state vector
- Bloch sphere → state vector conversion

### `circuits.py`
- build_encoding_circuit(): 3-qubit encoding (no measurement)
- build_syndrome_circuit(): 5-qubit syndrome extraction (measure ancilla only)
- build_full_qec_circuit(): Full pipeline (encode + errors + syndrome + data measurement)
- build_statevector_circuit(): 3-qubit circuit for fidelity verification (no measurement)
- All circuits use named registers for clarity: q (data), anc (ancilla), c_anc, c_data

### `syndrome.py`
- SYNDROME_TABLE: maps (s₀₁, s₁₂) → (correction_qubit, description)
- decode_syndrome(s01, s12): returns correction qubit and description
- parse_measurement_string(): extracts syndrome from Qiskit measurement strings (handles bit ordering)
- TWO_ERROR_CASES: documents two-qubit error failure cases with explanations

### `recovery.py`
- apply_recovery(): determines and applies classical correction
- apply_recovery_to_circuit(): appends X gate to Qiskit circuit
- verify_recovery_statevector(): full statevector verification with fidelity
- monte_carlo_recovery(): fast analytical path for Monte Carlo simulation

### `analytics.py`
- theoretical_unencoded_failure(p): P(p) = p
- theoretical_encoded_failure(p): P(p) = 3p² − 2p³
- code_gain(p): ratio of unencoded to encoded failure probability
- breakeven_point(): returns 0.5
- reliability_curve_data(): arrays for plotting
- wilson_confidence_interval(): 95% CI for proportions

### `noise.py`
- ExperimentResult dataclass: stores all experiment metadata
- BenchmarkResult dataclass: stores full benchmark sweep data
- run_single_experiment(): Monte Carlo at one p value
- run_reliability_benchmark(): full sweep over p values
- experiment_result_to_dataframe(): export single result as DataFrame
- benchmark_result_to_dataframe(): export benchmark as DataFrame (for CSV)

### `experiments.py`
- run_syndrome_experiment(): full syndrome lab pipeline
- run_full_syndrome_suite(): all 7 standard error patterns
- NAMED_STATE_AMPLITUDES: named state (α, β) pairs
- run_named_state_experiment(): convenience wrapper

### `visualization.py`
- plot_reliability_curve(): theory + simulation on one figure
- plot_error_rate_comparison(): grouped bar chart at specific p
- plot_syndrome_table(): Plotly table of syndrome decoding
- plot_code_gain(): gain ratio vs. p
- plot_theoretical_curves(): theory-only overlay plot
- plot_fidelity_gauge(): Plotly gauge for state fidelity
- render_circuit_diagram(): Matplotlib figure from Qiskit circuit

## Data Flow for Syndrome Lab

```
User selects: state name + error pattern + n_shots
    ↓
experiments.run_syndrome_experiment()
    ├── circuits.build_full_qec_circuit(alpha, beta, errors)
    ├── AerSimulator.run(circuit, shots=n_shots)
    ├── Parse measurement counts → extract syndrome (s01, s12)
    ├── syndrome.decode_syndrome(s01, s12) → correction_qubit
    └── recovery.verify_recovery_statevector(alpha, beta, errors, syndrome)
            ├── circuits.build_statevector_circuit(alpha, beta, errors, recovery_qubit)
            ├── AerSimulator(method="statevector").run(circuit)
            └── states.state_fidelity(target, actual) → fidelity
    ↓
Return: {circuit, counts, syndrome, correction, fidelity, corrected, ...}
    ↓
app.py renders: circuit diagram + gauge + histogram + explanation
```

## Data Flow for Reliability Benchmark

```
User sets: n_trials, n_p_points, seed
    ↓
noise.run_reliability_benchmark()
    ↓ (for each p value)
    ├── noise.run_single_experiment(p, n_trials, seed, encoded=True)
    │       ├── rng.random(3) < p → error pattern
    │       ├── recovery.monte_carlo_recovery(error_pattern)
    │       │       └── syndrome.decode_syndrome(s01, s12)
    │       └── Count logical failures, compute CI
    └── noise.run_single_experiment(p, n_trials, seed, encoded=False)
            └── rng.random() < p → logical failure
    ↓
BenchmarkResult (theoretical arrays + simulated arrays + CI arrays)
    ↓
app.py renders: reliability curve + comparison bar + export button
```
