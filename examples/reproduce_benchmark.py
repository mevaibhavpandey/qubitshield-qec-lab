"""
reproduce_benchmark.py — Standalone Benchmark Reproduction Script

Reproduces the QubitShield reliability benchmark from the command line.
Results can be compared with the Streamlit dashboard to verify consistency.

Usage:
    python examples/reproduce_benchmark.py
    python examples/reproduce_benchmark.py --n-trials 10000 --n-points 50 --seed 42
    python examples/reproduce_benchmark.py --output results.csv

The script:
1. Runs a full reliability benchmark sweep (p from 0 to 1)
2. Prints a summary table to the console
3. Optionally saves results to CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to the Python path so we can import qubitshield without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from qubitshield.noise import run_reliability_benchmark, benchmark_result_to_dataframe
from qubitshield.analytics import theoretical_encoded_failure, theoretical_unencoded_failure


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the QubitShield reliability benchmark."
    )
    parser.add_argument(
        "--n-trials", type=int, default=10_000,
        help="Number of Monte Carlo trials per p value (default: 10000)"
    )
    parser.add_argument(
        "--n-points", type=int, default=50,
        help="Number of physical error probability values to test (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save CSV results (e.g., results.csv)"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("QubitShield - Reliability Benchmark Reproduction")
    print("=" * 65)
    print(f"Parameters:")
    print(f"  Trials per p value:  {args.n_trials:,}")
    print(f"  Number of p values:  {args.n_points}")
    print(f"  Random seed:         {args.seed}")
    print()
    print("Running Monte Carlo simulation...")

    bench = run_reliability_benchmark(
        n_trials=args.n_trials,
        seed=args.seed,
        n_p_points=args.n_points,
    )

    print(f"Done. Elapsed: {bench.elapsed_seconds:.2f}s")
    print(f"Total trials: {2 * len(bench.p_values) * bench.n_trials:,}")
    print()

    # Print summary table
    print(f"{'p':>6} | {'P_enc (theory)':>14} | {'P_enc (sim)':>11} | "
          f"{'P_unenc (theory)':>16} | {'P_unenc (sim)':>13} | {'Code helps?':>11}")
    print("-" * 80)

    # Show 15 representative points
    indices = np.linspace(0, len(bench.p_values) - 1, 15, dtype=int)
    for i in indices:
        p = bench.p_values[i]
        t_enc = bench.theoretical_encoded[i]
        s_enc = bench.simulated_encoded[i]
        t_unenc = bench.theoretical_unencoded[i]
        s_unenc = bench.simulated_unencoded[i]
        helps = "Yes" if t_enc < t_unenc else ("Equal" if t_enc == t_unenc else "No")
        print(f"{p:6.3f} | {t_enc:14.6f} | {s_enc:11.6f} | "
              f"{t_unenc:16.6f} | {s_unenc:13.6f} | {helps:>11}")

    print()
    print("Summary:")
    print(f"  Break-even physical error rate: p = 0.500")
    print(f"  At p=0.1: theory says code reduces failure by {(0.1 - theoretical_encoded_failure(0.1)) / 0.1 * 100:.1f}%")
    print(f"  At p=0.3: theory says code reduces failure by {(0.3 - theoretical_encoded_failure(0.3)) / 0.3 * 100:.1f}%")
    print(f"  At p=0.5: theory says code has {0:.1f}% improvement (break-even)")

    if args.output:
        df = benchmark_result_to_dataframe(bench)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nTo save results: add --output results.csv")

    print()
    print("To reproduce this exact run:")
    print(f"  python examples/reproduce_benchmark.py "
          f"--n-trials {args.n_trials} --n-points {args.n_points} --seed {args.seed}")


if __name__ == "__main__":
    main()
