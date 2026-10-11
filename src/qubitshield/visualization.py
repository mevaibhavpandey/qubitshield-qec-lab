"""
visualization.py — Plotting Functions for QubitShield

All plotting functions return Plotly figures (go.Figure) unless noted.
The consistent visual style uses a dark scientific theme optimised for
the Streamlit dashboard.

Color palette:
    - Encoded (code): #4C9BE8 (steel blue)
    - Unencoded: #E8884C (warm orange)
    - Theory: dashed lines
    - Simulation: solid lines with markers
    - Error / failure: #E84C4C (red)
    - Success: #4CE87A (green)
    - Neutral: #94A3B8 (slate)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from qiskit import QuantumCircuit

from qubitshield.analytics import theoretical_unencoded_failure, theoretical_encoded_failure
from qubitshield.noise import BenchmarkResult


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

THEME = dict(
    bg="#0F172A",          # Dark navy background
    panel="#1E293B",       # Slightly lighter panel
    text="#E2E8F0",        # Light text
    subtext="#94A3B8",     # Subdued text
    grid="#334155",        # Grid lines
    encoded="#4C9BE8",     # Steel blue (encoded)
    unencoded="#E8884C",   # Warm orange (unencoded)
    theory_encoded="#93C5FD",    # Lighter blue for theory
    theory_unencoded="#FCD34D",  # Gold for unencoded theory
    success="#4ADE80",     # Green
    failure="#F87171",     # Red
    highlight="#A78BFA",   # Purple accent
)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["panel"],
    font=dict(color=THEME["text"], family="Inter, sans-serif", size=13),
    xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    legend=dict(
        bgcolor=THEME["panel"],
        bordercolor=THEME["grid"],
        borderwidth=1,
    ),
)


def apply_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Apply the QubitShield dark theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=THEME["text"])),
        height=height,
        **PLOTLY_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------

def plot_reliability_curve(
    benchmark: BenchmarkResult,
    show_confidence_intervals: bool = True,
    show_theory: bool = True,
) -> go.Figure:
    """Plot the reliability benchmark: logical failure probability vs. p.

    Shows theoretical curves and Monte Carlo simulation results for both
    the encoded and unencoded cases.

    Parameters
    ----------
    benchmark:
        BenchmarkResult from noise.run_reliability_benchmark().
    show_confidence_intervals:
        If True, add shaded confidence interval bands.
    show_theory:
        If True, overlay the theoretical prediction curves.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    p = benchmark.p_values

    fig = go.Figure()

    # Theoretical curves (dashed)
    if show_theory:
        fig.add_trace(go.Scatter(
            x=p,
            y=benchmark.theoretical_encoded,
            mode="lines",
            name="Theory: Encoded (repetition code)",
            line=dict(color=THEME["encoded"], dash="dash", width=2),
            hovertemplate="p=%{x:.3f}<br>P_logical=%{y:.4f}<extra>Theory (Encoded)</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=p,
            y=benchmark.theoretical_unencoded,
            mode="lines",
            name="Theory: Unencoded qubit",
            line=dict(color=THEME["unencoded"], dash="dash", width=2),
            hovertemplate="p=%{x:.3f}<br>P_unencoded=%{y:.4f}<extra>Theory (Unencoded)</extra>",
        ))

    # Simulation results
    fig.add_trace(go.Scatter(
        x=p,
        y=benchmark.simulated_encoded,
        mode="lines+markers",
        name="Simulation: Encoded",
        line=dict(color=THEME["encoded"], width=2),
        marker=dict(size=4),
        hovertemplate="p=%{x:.3f}<br>Simulated P_logical=%{y:.4f}<extra>Simulation (Encoded)</extra>",
    ))

    if show_confidence_intervals:
        # CI for encoded
        fig.add_trace(go.Scatter(
            x=np.concatenate([p, p[::-1]]),
            y=np.concatenate([benchmark.ci_upper_encoded, benchmark.ci_lower_encoded[::-1]]),
            fill="toself",
            fillcolor=f"rgba(76,155,232,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            name="95% CI (Encoded)",
        ))

    fig.add_trace(go.Scatter(
        x=p,
        y=benchmark.simulated_unencoded,
        mode="lines+markers",
        name="Simulation: Unencoded",
        line=dict(color=THEME["unencoded"], width=2),
        marker=dict(size=4),
        hovertemplate="p=%{x:.3f}<br>Simulated P_unencoded=%{y:.4f}<extra>Simulation (Unencoded)</extra>",
    ))

    if show_confidence_intervals:
        # CI for unencoded
        fig.add_trace(go.Scatter(
            x=np.concatenate([p, p[::-1]]),
            y=np.concatenate([benchmark.ci_upper_unencoded, benchmark.ci_lower_unencoded[::-1]]),
            fill="toself",
            fillcolor=f"rgba(232,136,76,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Break-even line
    fig.add_vline(
        x=0.5,
        line=dict(color=THEME["subtext"], dash="dot", width=1),
        annotation_text="Break-even p=0.5",
        annotation_position="top right",
        annotation_font_color=THEME["subtext"],
    )

    fig.update_xaxes(
        title_text="Physical bit-flip probability  p",
        range=[0, 1],
        tickformat=".1f",
    )
    fig.update_yaxes(
        title_text="Logical failure probability",
        range=[0, 1.02],
        tickformat=".2f",
    )

    apply_theme(
        fig,
        title=f"Reliability Benchmark  (N={benchmark.n_trials:,} trials per point)",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Error rate comparison bar chart
# ---------------------------------------------------------------------------

def plot_error_rate_comparison(
    p: float,
    sim_encoded: float,
    sim_unencoded: float,
    theory_encoded: float,
    theory_unencoded: float,
    ci_enc: tuple[float, float] = (0, 0),
    ci_unenc: tuple[float, float] = (0, 0),
) -> go.Figure:
    """Plot a grouped bar chart comparing error rates at a specific p value."""
    categories = ["Theory", "Simulation"]
    encoded_vals = [theory_encoded, sim_encoded]
    unencoded_vals = [theory_unencoded, sim_unencoded]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Encoded (repetition code)",
        x=categories,
        y=encoded_vals,
        marker_color=THEME["encoded"],
        error_y=dict(
            type="data",
            array=[0, ci_enc[1] - sim_encoded],
            arrayminus=[0, sim_encoded - ci_enc[0]],
            visible=True,
            color=THEME["text"],
        ),
        hovertemplate="%{x}<br>Failure Rate: %{y:.4f}<extra>Encoded</extra>",
    ))

    fig.add_trace(go.Bar(
        name="Unencoded qubit",
        x=categories,
        y=unencoded_vals,
        marker_color=THEME["unencoded"],
        error_y=dict(
            type="data",
            array=[0, ci_unenc[1] - sim_unencoded],
            arrayminus=[0, sim_unencoded - ci_unenc[0]],
            visible=True,
            color=THEME["text"],
        ),
        hovertemplate="%{x}<br>Failure Rate: %{y:.4f}<extra>Unencoded</extra>",
    ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Result type")
    fig.update_yaxes(title_text="Logical failure probability")

    apply_theme(fig, title=f"Error Rate Comparison at p = {p:.3f}", height=400)
    return fig


# ---------------------------------------------------------------------------
# Syndrome table heatmap
# ---------------------------------------------------------------------------

def plot_syndrome_table() -> go.Figure:
    """Return a Plotly heatmap of the syndrome decoding table."""
    syndromes = ["(0,0)", "(1,0)", "(1,1)", "(0,1)"]
    corrected_qubits = ["None", "Qubit 0", "Qubit 1", "Qubit 2"]
    descriptions = [
        "No error",
        "Bit-flip on qubit 0",
        "Bit-flip on qubit 1",
        "Bit-flip on qubit 2",
    ]
    colors = [
        THEME["success"],
        THEME["failure"],
        THEME["failure"],
        THEME["failure"],
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Syndrome (s₀₁, s₁₂)", "Error Identified", "Recovery Action"],
            fill_color=THEME["panel"],
            align="center",
            font=dict(color=THEME["text"], size=13),
            line_color=THEME["grid"],
        ),
        cells=dict(
            values=[
                syndromes,
                descriptions,
                ["No correction", "Apply X₀", "Apply X₁", "Apply X₂"],
            ],
            fill_color=[THEME["bg"]] * 3,
            align="center",
            font=dict(color=THEME["text"], size=12),
            line_color=THEME["grid"],
        ),
    ))

    apply_theme(fig, title="Syndrome Decoding Table", height=250)
    return fig


# ---------------------------------------------------------------------------
# Code gain plot
# ---------------------------------------------------------------------------

def plot_code_gain() -> go.Figure:
    """Plot the code gain ratio P_unencoded / P_logical vs. p."""
    from qubitshield.analytics import code_gain

    p = np.linspace(0.001, 0.999, 500)
    gain = code_gain(p)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p,
        y=gain,
        mode="lines",
        name="Code gain  P_unencoded / P_logical",
        line=dict(color=THEME["highlight"], width=2.5),
        hovertemplate="p=%{x:.3f}<br>Gain=%{y:.2f}×<extra>Code Gain</extra>",
    ))

    # Break-even line
    fig.add_hline(y=1.0, line=dict(color=THEME["subtext"], dash="dot", width=1))
    fig.add_vline(x=0.5, line=dict(color=THEME["subtext"], dash="dot", width=1))

    fig.update_xaxes(title_text="Physical bit-flip probability  p", range=[0, 1])
    fig.update_yaxes(title_text="Gain ratio  (>1 means code helps)", type="log")

    apply_theme(fig, title="Code Gain vs. Physical Error Probability", height=400)
    return fig


# ---------------------------------------------------------------------------
# Fidelity display (for Syndrome Lab)
# ---------------------------------------------------------------------------

def plot_fidelity_gauge(fidelity: float) -> go.Figure:
    """Display state fidelity as a gauge chart."""
    color = THEME["success"] if fidelity > 0.99 else (
        THEME["highlight"] if fidelity > 0.5 else THEME["failure"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fidelity * 100,
        number=dict(suffix="%", font=dict(color=color, size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=THEME["text"]),
            bar=dict(color=color),
            bgcolor=THEME["panel"],
            bordercolor=THEME["grid"],
            steps=[
                dict(range=[0, 50], color=THEME["bg"]),
                dict(range=[50, 99], color=THEME["bg"]),
                dict(range=[99, 100], color=THEME["bg"]),
            ],
            threshold=dict(
                line=dict(color=THEME["success"], width=3),
                thickness=0.75,
                value=99,
            ),
        ),
        title=dict(text="State Fidelity After Recovery", font=dict(color=THEME["text"])),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    apply_theme(fig, height=250)
    return fig


# ---------------------------------------------------------------------------
# Circuit diagram rendering
# ---------------------------------------------------------------------------

def render_circuit_diagram(
    qc: QuantumCircuit,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Render a Qiskit circuit as a Matplotlib figure.

    Returns a Matplotlib figure object for embedding in Streamlit via
    st.pyplot().

    Parameters
    ----------
    qc:
        The Qiskit QuantumCircuit to render.
    figsize:
        Figure size in inches (width, height).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = qc.draw(
        output="mpl",
        style={
            "backgroundcolor": "#1E293B",
            "textcolor": "#E2E8F0",
            "gatetextcolor": "#0F172A",
            "subtextcolor": "#94A3B8",
            "linecolor": "#94A3B8",
            "creglinecolor": "#4C9BE8",
            "gatefacecolor": "#4C9BE8",
            "controlcolor": "#A78BFA",
            "measurecolor": "#4ADE80",
            "latexdrawerstyle": False,
            "usepiformat": False,
        },
        fold=60,
        scale=0.9,
    )

    if fig is not None:
        fig.set_size_inches(*figsize)
        fig.patch.set_facecolor("#1E293B")

    return fig


# ---------------------------------------------------------------------------
# Theoretical curves (standalone, for Theory Explorer)
# ---------------------------------------------------------------------------

def plot_theoretical_curves() -> go.Figure:
    """Plot only the theoretical failure probability curves."""
    p = np.linspace(0, 1, 300)
    p_enc = theoretical_encoded_failure(p)
    p_unenc = theoretical_unencoded_failure(p)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p, y=p_enc,
        mode="lines",
        name="Encoded: P = 3p²(1−p) + p³",
        line=dict(color=THEME["encoded"], width=2.5),
        hovertemplate="p=%{x:.3f}<br>P_logical=%{y:.4f}<extra>Repetition Code</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=p, y=p_unenc,
        mode="lines",
        name="Unencoded: P = p",
        line=dict(color=THEME["unencoded"], width=2.5, dash="dash"),
        hovertemplate="p=%{x:.3f}<br>P_unencoded=%{y:.4f}<extra>Unencoded Qubit</extra>",
    ))

    fig.add_vline(x=0.5, line=dict(color=THEME["subtext"], dash="dot", width=1.5),
                  annotation_text="Break-even p=0.5", annotation_position="top right",
                  annotation_font_color=THEME["subtext"])

    # Shaded benefit region
    benefit_p = p[p <= 0.5]
    fig.add_trace(go.Scatter(
        x=np.concatenate([benefit_p, benefit_p[::-1]]),
        y=np.concatenate([
            theoretical_encoded_failure(benefit_p),
            theoretical_unencoded_failure(benefit_p)[::-1],
        ]),
        fill="toself",
        fillcolor="rgba(74,222,128,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name="Region where code helps (p < 0.5)",
        hoverinfo="skip",
    ))

    fig.update_xaxes(title_text="Physical bit-flip probability  p", range=[0, 1])
    fig.update_yaxes(title_text="Logical failure probability", range=[0, 1.02])

    apply_theme(
        fig,
        title="Theoretical Logical Failure Probability vs. Physical Error Rate",
        height=450,
    )
    return fig
