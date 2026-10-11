"""
app.py — QubitShield Streamlit Dashboard

QubitShield — Interactive Quantum Error-Correction & Reliability Lab
Developer: Vaibhav Pandey | BMS Institute of Technology & Management

This is the main entry point for the Streamlit dashboard. It orchestrates
six main sections:
    1. Overview
    2. Theory Explorer
    3. Syndrome Lab
    4. Reliability Benchmark
    5. Results & Interpretation
    6. About & Limitations

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QubitShield — Quantum Error-Correction Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/mevaibhavpandey/qubitshield-qec-lab",
        "Report a bug": "https://github.com/mevaibhavpandey/qubitshield-qec-lab/issues",
        "About": "QubitShield — Quantum Error-Correction & Reliability Lab by Vaibhav Pandey",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection — Premium dark scientific theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Crimson+Pro:ital,wght@0,400;1,400&display=swap');

/* ── Root colour tokens ── */
:root {
    --bg:          #0F172A;
    --surface:     #1E293B;
    --surface2:    #263346;
    --border:      #334155;
    --text:        #E2E8F0;
    --subtext:     #94A3B8;
    --accent:      #4C9BE8;
    --accent2:     #A78BFA;
    --success:     #4ADE80;
    --warning:     #FBBF24;
    --failure:     #F87171;
    --encoded:     #4C9BE8;
    --unencoded:   #E8884C;
}

/* ── Global overrides ── */
.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
.main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1 { font-size: 2.2rem !important; font-weight: 700 !important;
     background: linear-gradient(135deg, #4C9BE8, #A78BFA);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     margin-bottom: 0.25rem !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important;
     color: var(--text) !important; border-bottom: 1px solid var(--border);
     padding-bottom: 0.4rem; margin-top: 1.5rem !important; }
h3 { font-size: 1.15rem !important; font-weight: 500 !important;
     color: var(--accent) !important; }
p, li, label { color: var(--text); }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--accent); }
[data-testid="stMetricLabel"] { color: var(--subtext) !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    border-bottom: none;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--subtext) !important;
    background: transparent;
    border-radius: 6px;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0F172A !important;
}

/* ── Code blocks ── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--surface2) !important;
    color: #93C5FD !important;
    border-radius: 6px;
    border: 1px solid var(--border);
}

/* ── Math / LaTeX ── */
.katex { color: var(--text) !important; font-size: 1.05em !important; }

/* ── Info / warning boxes ── */
[data-testid="stInfo"] { background: rgba(76,155,232,0.1); border: 1px solid rgba(76,155,232,0.3); border-radius: 8px; }
[data-testid="stWarning"] { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); border-radius: 8px; }
[data-testid="stSuccess"] { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); border-radius: 8px; }
[data-testid="stError"] { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); border-radius: 8px; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #0F172A !important;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.5rem;
    transition: opacity 0.2s, transform 0.15s;
}
[data-testid="stButton"] > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Selectbox / sliders ── */
[data-baseweb="select"] { background: var(--surface2) !important; border-color: var(--border) !important; border-radius: 8px !important; }
[data-testid="stSlider"] [data-baseweb="slider"] .rc-slider-track { background: var(--accent); }

/* ── Expander ── */
[data-testid="stExpander"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; }

/* ── Custom badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 2px;
}
.badge-success { background: rgba(74,222,128,0.15); color: #4ADE80; border: 1px solid rgba(74,222,128,0.3); }
.badge-failure { background: rgba(248,113,113,0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.3); }
.badge-info    { background: rgba(76,155,232,0.15);  color: #4C9BE8; border: 1px solid rgba(76,155,232,0.3); }
.badge-warn    { background: rgba(251,191,36,0.15);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.3); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(76,155,232,0.12) 0%, rgba(167,139,250,0.12) 100%);
    border: 1px solid rgba(76,155,232,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.4rem !important; }

/* ── Section card ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Deferred imports (after page config, before any st.* calls that render)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _import_qubitshield():
    """Import heavy dependencies once and cache the modules."""
    try:
        import qubitshield
        from qubitshield import analytics, circuits, experiments, noise, visualization, syndrome
        return qubitshield, analytics, circuits, experiments, noise, visualization, syndrome, None
    except ImportError as e:
        return None, None, None, None, None, None, None, str(e)


qs_pkg, analytics, circuits, experiments_mod, noise_mod, viz, syndrome_mod, import_error = _import_qubitshield()

if import_error:
    st.error(f"❌ Failed to import QubitShield: {import_error}")
    st.info("Run `pip install -e .` in the project root, then restart the app.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem'>🛡️</div>
        <div style='font-size:1.4rem; font-weight:700; background:linear-gradient(135deg,#4C9BE8,#A78BFA);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            QubitShield
        </div>
        <div style='color:#94A3B8; font-size:0.82rem; margin-top:2px;'>
            Quantum Error-Correction Lab
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📐 Theory Explorer",
            "🔬 Syndrome Lab",
            "📊 Reliability Benchmark",
            "📋 Results & Interpretation",
            "ℹ️ About & Limitations",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.78rem; text-align:center;'>
        <b>Developer:</b> Vaibhav Pandey<br>
        BMS Institute of Technology<br>
        <span style='color:#4C9BE8'>Python · Qiskit · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def success_badge(text: str) -> str:
    return f'<span class="badge badge-success">✓ {text}</span>'


def failure_badge(text: str) -> str:
    return f'<span class="badge badge-failure">✗ {text}</span>'


def info_badge(text: str) -> str:
    return f'<span class="badge badge-info">{text}</span>'


def warn_badge(text: str) -> str:
    return f'<span class="badge badge-warn">⚠ {text}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
        <h1>🛡️ QubitShield</h1>
        <p style='font-size:1.1rem; color:#94A3B8; margin-top:0;'>
            Interactive Quantum Error-Correction &amp; Reliability Lab
        </p>
        <p style='color:#CBD5E1; max-width:800px;'>
            An interactive scientific application that investigates how a 
            <b>three-qubit bit-flip repetition code</b> protects an encoded logical qubit 
            against independent bit-flip errors, and quantifies the improvement over an 
            unprotected qubit using quantum circuit simulation and Monte Carlo analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='section-card'>
            <h3>🎯 Core Research Question</h3>
            <p style='color:#CBD5E1; font-size:0.9rem;'>
                How effectively does a 3-qubit repetition code protect a logical qubit against 
                independent bit-flip errors, and how does its logical failure probability compare 
                with that of an unencoded qubit?
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='section-card'>
            <h3>⚛️ Quantum Framework</h3>
            <p style='color:#CBD5E1; font-size:0.9rem;'>
                Built on <b>Qiskit 1.4</b> + <b>Qiskit Aer</b> for circuit simulation.
                Implements genuine stabilizer syndrome extraction, decoder logic, and 
                statevector fidelity verification — not static pre-computed results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='section-card'>
            <h3>📈 What You'll Find</h3>
            <p style='color:#CBD5E1; font-size:0.9rem;'>
                Theoretical predictions vs. actual Monte Carlo simulations.
                Syndrome decoding demonstrations. Circuit diagrams from the real 
                Qiskit circuits. Honest discussion of assumptions and limitations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## The Repetition Code in Brief")
    col_eq1, col_eq2 = st.columns([1, 1])
    with col_eq1:
        st.markdown("""
        **Encoding:**  A logical qubit $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$ 
        is encoded as three physical qubits:

        $$|\\psi_L\\rangle = \\alpha|000\\rangle + \\beta|111\\rangle$$

        The logical information is spread across three qubits using two CNOT gates.
        This preserves the quantum superposition — it is not a classical copy.
        """)

        st.markdown("""
        **Syndrome extraction:** Two ancilla qubits measure parity checks 
        $Z_0 Z_1$ and $Z_1 Z_2$ without destroying the logical information:

        | Syndrome $(s_{01}, s_{12})$ | Error detected |
        |:---:|:---|
        | $(0,0)$ | No error |
        | $(1,0)$ | Bit-flip on qubit 0 |
        | $(1,1)$ | Bit-flip on qubit 1 |
        | $(0,1)$ | Bit-flip on qubit 2 |
        """)

    with col_eq2:
        st.markdown("""
        **Logical failure probability:** Under independent bit-flip errors with 
        probability $p$ on each data qubit, and ideal correction operations:

        $$P_{\\text{logical}} = 3p^2(1-p) + p^3 = 3p^2 - 2p^3$$

        vs. the unencoded qubit:

        $$P_{\\text{unencoded}} = p$$

        **The code helps when $p < 0.5$.** At $p = 0.5$, both have equal failure 
        rates. At $p > 0.5$, the code makes things worse.
        """)

        # Quick theoretical curve
        p_demo = np.linspace(0, 1, 200)
        import plotly.graph_objects as go
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=p_demo, y=analytics.theoretical_encoded_failure(p_demo),
            name="Repetition Code", line=dict(color="#4C9BE8", width=2.5),
        ))
        fig_demo.add_trace(go.Scatter(
            x=p_demo, y=analytics.theoretical_unencoded_failure(p_demo),
            name="Unencoded", line=dict(color="#E8884C", width=2.5, dash="dash"),
        ))
        fig_demo.update_layout(
            height=220, margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="#1E293B", plot_bgcolor="#263346",
            font=dict(color="#E2E8F0", size=11),
            legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1, font=dict(size=10)),
            xaxis=dict(title="Physical error p", gridcolor="#334155"),
            yaxis=dict(title="Failure prob.", gridcolor="#334155"),
        )
        fig_demo.add_vline(x=0.5, line=dict(color="#94A3B8", dash="dot"))
        st.plotly_chart(fig_demo, use_container_width=True)

    st.markdown("## Key Limitations of this Simulation")
    st.warning(
        "**This simulation uses an idealized error model.** The theoretical formula and "
        "Monte Carlo experiments model only independent data-qubit bit-flip errors. They do "
        "**not** include errors in encoding gates, ancilla preparation, syndrome measurement, "
        "or recovery gates. Actual fault-tolerant quantum computation requires each of these "
        "operations to be itself error-corrected, introducing significant overhead. "
        "Simulated results are from classical software simulation, not physical quantum hardware."
    )

    st.markdown("## Project Navigation")
    nav_cols = st.columns(5)
    nav_items = [
        ("📐", "Theory", "Mathematical foundations of the repetition code"),
        ("🔬", "Syndrome Lab", "Interactive fault injection & syndrome decoding"),
        ("📊", "Benchmark", "Monte Carlo reliability comparison"),
        ("📋", "Results", "Interpretation of experimental outcomes"),
        ("ℹ️", "About", "Architecture, assumptions & limitations"),
    ]
    for col, (icon, name, desc) in zip(nav_cols, nav_items):
        with col:
            st.markdown(f"""
            <div style='text-align:center; padding:0.8rem; background:#1E293B; 
                        border:1px solid #334155; border-radius:10px;'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='color:#4C9BE8; font-weight:600; font-size:0.9rem'>{name}</div>
                <div style='color:#94A3B8; font-size:0.75rem; margin-top:4px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Theory Explorer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📐 Theory Explorer":
    st.title("📐 Theory Explorer")
    st.markdown("*Mathematical foundations of the three-qubit bit-flip repetition code*")

    theory_tab = st.tabs([
        "Encoding", "Stabilizers & Syndromes", "Recovery Logic",
        "Failure Probability", "Theoretical Curves",
    ])

    with theory_tab[0]:
        st.markdown("## Encoding: From Logical to Physical Qubit")
        st.markdown("""
        A single logical qubit carries the state:
        $$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle, \\quad |\\alpha|^2 + |\\beta|^2 = 1$$

        The three-qubit bit-flip repetition code encodes this into three physical qubits:
        $$|\\psi_L\\rangle = \\alpha|000\\rangle + \\beta|111\\rangle$$

        **This is not classical copying.** The no-cloning theorem prohibits copying an 
        unknown quantum state. Instead, we create a *logical codeword* where the 
        quantum superposition of $\\alpha$ and $\\beta$ is preserved in the entangled 
        three-qubit state.

        **Circuit implementation:**
        1. Prepare qubit 0 in state $\\alpha|0\\rangle + \\beta|1\\rangle$
        2. Apply CNOT$(q_0 \\to q_1)$: spreads the superposition to qubit 1
        3. Apply CNOT$(q_0 \\to q_2)$: spreads the superposition to qubit 2

        The CNOT gate maps $|c, t\\rangle \\to |c, c \\oplus t\\rangle$, so:

        $$|\\psi\\rangle \\otimes |00\\rangle 
          \\xrightarrow{\\text{CNOT}_{01}} 
          \\alpha|000\\rangle + \\beta|110\\rangle
          \\xrightarrow{\\text{CNOT}_{02}}
          \\alpha|000\\rangle + \\beta|111\\rangle$$
        """)

        st.info("""
        **Preservation of superposition:** After encoding, measurements in the standard basis 
        $(\\{|0\\rangle, |1\\rangle\\}$) would collapse the state to either $|000\\rangle$ or 
        $|111\\rangle$ with probabilities $|\\alpha|^2$ and $|\\beta|^2$ respectively. The 
        syndrome extraction uses ancilla measurements that avoid directly measuring the data 
        qubits in this basis, preserving the logical coherence.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Codeword states:**")
            st.markdown("""
            | Logical state | Physical codeword |
            |:---:|:---:|
            | $|0\\rangle_L$ | $|000\\rangle$ |
            | $|1\\rangle_L$ | $|111\\rangle$ |
            | $|+\\rangle_L$ | $(|000\\rangle + |111\\rangle)/\\sqrt{2}$ |
            | $|-\\rangle_L$ | $(|000\\rangle - |111\\rangle)/\\sqrt{2}$ |
            """)

        with col2:
            st.markdown("**What the code corrects:**")
            st.markdown("""
            ✅ Single bit-flip (X) error on any one of the three data qubits  
            ❌ Two or three simultaneous bit-flip errors  
            ❌ Phase-flip (Z) errors  
            ❌ Depolarising noise  
            ❌ Errors in the encoding/decoding circuit itself  
            """)

    with theory_tab[1]:
        st.markdown("## Stabilizer Formalism & Syndrome Extraction")
        st.markdown("""
        The repetition code is a **stabilizer code**. The codespace is the +1 
        eigenspace of the stabilizer generators:
        
        $$g_1 = Z_0 Z_1 \\qquad g_2 = Z_1 Z_2$$

        Both codewords $|000\\rangle$ and $|111\\rangle$ are +1 eigenstates of 
        $g_1$ and $g_2$, so any state in the codespace satisfies:
        
        $$g_1|\\psi_L\\rangle = |\\psi_L\\rangle \\quad \\text{and} \\quad g_2|\\psi_L\\rangle = |\\psi_L\\rangle$$

        A bit-flip error on qubit $i$ (applying $X_i$) anti-commutes with the 
        stabilizers that involve qubit $i$, changing their eigenvalue from $+1$ 
        to $-1$. This change is the **syndrome**.

        **Syndrome extraction circuit** (no direct data-qubit measurement):
        1. Prepare two ancilla qubits in $|0\\rangle$
        2. For $Z_0 Z_1$: apply CNOT$(q_0 \\to \\text{anc}_0)$ then CNOT$(q_1 \\to \\text{anc}_0)$
        3. For $Z_1 Z_2$: apply CNOT$(q_1 \\to \\text{anc}_1)$ then CNOT$(q_2 \\to \\text{anc}_1)$
        4. Measure ancilla qubits → syndrome bits $s_{01}$, $s_{12}$
        """)

        st.plotly_chart(viz.plot_syndrome_table(), use_container_width=True)

        st.markdown("""
        **Why does this work?** The CNOT gates with ancilla qubits compute the 
        parity of the data qubit pair onto the ancilla without measuring the 
        data qubits directly. A measurement of ancilla in state $|0\\rangle$ means 
        parity is even (+1 stabilizer eigenvalue, no error); $|1\\rangle$ means 
        parity is odd (−1 eigenvalue, error detected).
        
        **Qiskit implementation note:** Qiskit uses little-endian bit ordering in 
        measurement result strings. In the result string "10", bit 0 (rightmost) 
        corresponds to the first classical register bit. The QubitShield code 
        explicitly handles this convention.
        """)

    with theory_tab[2]:
        st.markdown("## Recovery Logic")
        st.markdown("""
        Given the measured syndrome $(s_{01}, s_{12})$, the minimum-weight decoder 
        applies a Pauli $X$ correction to the qubit most likely to have experienced 
        an error.

        **Derivation of the syndrome table:**

        | Error | Effect on $Z_0Z_1$ | Effect on $Z_1Z_2$ | $s_{01}$ | $s_{12}$ |
        |:---:|:---:|:---:|:---:|:---:|
        | None | Commutes | Commutes | 0 | 0 |
        | $X_0$ | Anti-commutes | Commutes | 1 | 0 |
        | $X_1$ | Anti-commutes | Anti-commutes | 1 | 1 |
        | $X_2$ | Commutes | Anti-commutes | 0 | 1 |

        **Why the decoder fails for two-qubit errors:**
        
        Consider $X_0 X_1$ (both qubits 0 and 1 flip):
        - $s_{01}$: $Z_0Z_1$ anti-commutes with $X_0$ and anti-commutes with $X_1$.
          Two anti-commutations → net commutation → $s_{01} = 0$
        - $s_{12}$: $Z_1Z_2$ anti-commutes with $X_1$ → $s_{12} = 1$
        
        Syndrome $(0,1)$ is identical to the syndrome for a single $X_2$ error!
        The decoder applies $X_2$, giving total errors $\\{0, 1, 2\\}$ — equivalent 
        to a logical $X$ flip. **The code fails.**
        """)

        st.warning("""
        **Two-error failure modes:** All three two-qubit error patterns produce syndromes 
        that mimic a single error on the *other* qubit. This is an inherent limitation 
        of the three-qubit code: it can correct any single error but cannot distinguish 
        two-error patterns from single errors. This is explored interactively in the 
        Syndrome Lab.
        """)

    with theory_tab[3]:
        st.markdown("## Logical Failure Probability")
        st.markdown(r"""
        **Model assumptions:**
        - Each data qubit independently experiences a bit-flip (X) error with probability $p$.
        - Encoding, syndrome measurement, and recovery are performed perfectly (no gate errors).
        - Error patterns are independent across qubits and shots.

        **Unencoded qubit:**
        $$P_{\text{unencoded}}(p) = p$$

        **Three-qubit repetition code — failure probability derivation:**
        
        The code *fails* when 2 or 3 data qubits experience errors:
        
        $$P_{\text{logical}}(p) = \binom{3}{2} p^2(1-p) + \binom{3}{3} p^3$$
        $$= 3p^2(1-p) + p^3 = 3p^2 - 3p^3 + p^3 = 3p^2 - 2p^3$$

        **Break-even condition** ($P_{\text{logical}} = P_{\text{unencoded}}$):
        $$3p^2 - 2p^3 = p$$
        $$p(3p - 2p^2 - 1) = 0$$
        $$p(2p-1)(p-1) = 0$$

        Solutions: $p = 0$, $p = 0.5$, $p = 1$.
        
        For $0 < p < 0.5$: the code *helps* ($P_{\text{logical}} < P_{\text{unencoded}}$).
        
        For $p = 0.5$: both fail equally.
        
        For $0.5 < p < 1$: the code *hurts* ($P_{\text{logical}} > P_{\text{unencoded}}$).
        """)

    with theory_tab[4]:
        st.markdown("## Theoretical Failure Probability Curves")
        st.plotly_chart(viz.plot_theoretical_curves(), use_container_width=True)
        st.plotly_chart(viz.plot_code_gain(), use_container_width=True)
        st.markdown("""
        **Reading the gain plot:** A gain > 1 means the code reduces logical failure rate.
        The gain diverges as $p \\to 0$ because $P_{\\text{logical}} \\to 3p^2$ while 
        $P_{\\text{unencoded}} \\to p$, giving gain $\\approx 1/(3p) \\to \\infty$.
        In practice, gate noise places a lower bound on achievable physical error rates.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Syndrome Lab
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬 Syndrome Lab":
    st.title("🔬 Syndrome Lab")
    st.markdown("*Inject errors, extract syndromes, verify recovery*")

    # Controls
    st.markdown("### Experiment Configuration")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])

    with col_ctrl1:
        state_name = st.selectbox(
            "Logical input state",
            list(experiments_mod.NAMED_STATE_AMPLITUDES.keys()),
            index=0,
            help="The quantum state to encode and protect.",
        )
        alpha, beta = experiments_mod.NAMED_STATE_AMPLITUDES[state_name]

    with col_ctrl2:
        error_options = {
            "No error": [],
            "X on qubit 0": [0],
            "X on qubit 1": [1],
            "X on qubit 2": [2],
            "X on qubits 0 and 1 (two-error)": [0, 1],
            "X on qubits 0 and 2 (two-error)": [0, 2],
            "X on qubits 1 and 2 (two-error)": [1, 2],
        }
        error_choice = st.selectbox(
            "Error injection pattern",
            list(error_options.keys()),
            index=0,
            help="Select which qubit(s) to inject bit-flip (X) errors.",
        )
        injected_errors = error_options[error_choice]

    with col_ctrl3:
        n_shots = st.slider(
            "Measurement shots",
            min_value=256,
            max_value=8192,
            value=1024,
            step=256,
            help="Number of circuit measurement shots for the histogram.",
        )

    run_button = st.button("▶ Run Syndrome Experiment", use_container_width=True)

    if run_button or "syndrome_result" not in st.session_state:
        if run_button or "syndrome_result" not in st.session_state:
            with st.spinner("Running quantum circuit simulation…"):
                try:
                    result = experiments_mod.run_syndrome_experiment(
                        injected_errors=injected_errors,
                        alpha=complex(alpha),
                        beta=complex(beta),
                        n_shots=n_shots,
                    )
                    st.session_state.syndrome_result = result
                    st.session_state.syndrome_error = None
                except Exception as e:
                    st.session_state.syndrome_result = None
                    st.session_state.syndrome_error = str(e)

    if st.session_state.get("syndrome_error"):
        st.error(f"Experiment failed: {st.session_state.syndrome_error}")
        st.stop()

    result = st.session_state.get("syndrome_result")
    if result is None:
        st.info("Configure the experiment above and click **Run Syndrome Experiment**.")
        st.stop()

    # Results display
    st.divider()
    st.markdown("### Experiment Results")

    metric_cols = st.columns(4)
    s01, s12 = result["syndrome"]
    correction = result["correction_qubit"]
    fidelity = result["fidelity"]
    corrected = result["corrected"]

    with metric_cols[0]:
        st.metric("Error Injected", result["error_description"])
    with metric_cols[1]:
        st.metric("Syndrome (s₀₁, s₁₂)", f"({s01}, {s12})")
    with metric_cols[2]:
        st.metric("Recovery Action", result["correction_description"])
    with metric_cols[3]:
        st.metric(
            "State Fidelity",
            f"{fidelity:.4f}",
            delta="✓ Corrected" if corrected else "✗ Failed",
            delta_color="normal" if corrected else "inverse",
        )

    # Status banner
    if corrected:
        st.success(
            f"✅ **Error Corrected.** The decoded syndrome `{result['syndrome']}` "
            f"correctly identified `{result['correction_description']}`. "
            f"State fidelity after recovery: **{fidelity:.4f}**"
        )
    else:
        st.error(
            f"❌ **Correction Failed.** The decoder applied `{result['correction_description']}` "
            f"but the error pattern `{result['injected_errors']}` cannot be uniquely corrected. "
            f"State fidelity after recovery: **{fidelity:.4f}** "
            f"(logical information is corrupted)."
        )

    # Circuit diagram
    st.markdown("### Quantum Circuit")
    with st.spinner("Rendering circuit diagram…"):
        try:
            fig_circuit = viz.render_circuit_diagram(result["circuit"])
            if fig_circuit is not None:
                st.pyplot(fig_circuit, clear_figure=True)
            else:
                st.text(result["circuit"].draw(output="text", fold=80))
        except Exception as e:
            # Fallback to text
            st.code(str(result["circuit"].draw(output="text", fold=80)), language="text")

    # Fidelity gauge + measurement counts
    col_gauge, col_counts = st.columns([1, 1])

    with col_gauge:
        st.markdown("### State Fidelity Gauge")
        st.plotly_chart(viz.plot_fidelity_gauge(fidelity), use_container_width=True)

    with col_counts:
        st.markdown("### Measurement Outcome Histogram")
        counts = result["shot_counts"]
        sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))

        import plotly.graph_objects as go
        fig_hist = go.Figure(go.Bar(
            x=list(sorted_counts.keys()),
            y=list(sorted_counts.values()),
            marker_color="#4C9BE8",
            hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.update_layout(
            height=250, margin=dict(l=40, r=20, t=20, b=60),
            paper_bgcolor="#1E293B", plot_bgcolor="#263346",
            font=dict(color="#E2E8F0", size=11),
            xaxis=dict(title="Measurement outcome (c_anc | c_data)", gridcolor="#334155"),
            yaxis=dict(title="Count", gridcolor="#334155"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Mathematical explanation
    st.markdown("### Mathematical Explanation")
    is_two_err = result["is_two_error_failure"]

    if not injected_errors:
        st.markdown("""
        **No error case:** The encoded state $\\alpha|000\\rangle + \\beta|111\\rangle$ is 
        undisturbed. Both parity checks $Z_0Z_1$ and $Z_1Z_2$ return $+1$ eigenvalue,
        so the syndrome is $(0,0)$. No correction is applied, and the logical state 
        is perfectly preserved.
        """)
    elif len(injected_errors) == 1:
        q = injected_errors[0]
        st.markdown(f"""
        **Single error $X_{q}$:** A bit flip on qubit {q} changes the state from 
        $\\alpha|000\\rangle + \\beta|111\\rangle$ to a state with qubit {q} flipped.
        
        The parity checks reveal this:
        - $s_{{01}}$: Z₀Z₁ anti-commutes with $X_{q}$ for $q \\in \\{{0,1\\}}$ → syndrome bit flips
        - $s_{{12}}$: Z₁Z₂ anti-commutes with $X_{q}$ for $q \\in \\{{1,2\\}}$ → syndrome bit flips
        
        Resulting syndrome: **{result['syndrome']}** → decoder applies $X_{correction}$ to qubit 
        **{result['correction_qubit']}** → restores the original logical state.
        """)
    else:
        from qubitshield.syndrome import TWO_ERROR_CASES
        err_key = f"X₀X₁ (qubits 0 and 1)" if injected_errors == [0,1] else \
                  f"X₀X₂ (qubits 0 and 2)" if injected_errors == [0,2] else \
                  f"X₁X₂ (qubits 1 and 2)"
        case_info = TWO_ERROR_CASES.get(err_key, {})
        explanation = case_info.get("explanation", "Two-qubit error exceeds code correction capacity.")
        st.markdown(f"""
        **Two-qubit error failure case:** {explanation}
        
        The three-qubit repetition code can only correct **one** bit-flip error.
        When two qubits experience errors, the syndrome is indistinguishable from 
        a single error on the third qubit. The decoder applies the wrong correction,
        resulting in a logical failure.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Reliability Benchmark
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Reliability Benchmark":
    st.title("📊 Reliability Benchmark")
    st.markdown("*Monte Carlo comparison: encoded vs. unencoded qubit across all error rates*")

    st.info(
        "This benchmark runs genuine Monte Carlo simulations sampling random error patterns "
        "at each physical error probability $p$, applies the syndrome decoder, and measures "
        "the logical failure rate. The theoretical curves are computed from the closed-form "
        "formula $P_{\\text{logical}} = 3p^2 - 2p^3$."
    )

    # Benchmark controls
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        n_trials = st.number_input(
            "Trials per p value",
            min_value=100,
            max_value=100_000,
            value=5_000,
            step=500,
            help="More trials → smaller statistical uncertainty.",
        )
    with col_b2:
        n_p_points = st.slider(
            "Number of p values",
            min_value=10,
            max_value=100,
            value=40,
            help="Points on the x-axis of the reliability curve.",
        )
    with col_b3:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
            help="Set the same seed to reproduce a run exactly.",
        )
    with col_b4:
        show_ci = st.checkbox("Show 95% confidence intervals", value=True)

    run_bench = st.button("▶ Run Full Reliability Benchmark", use_container_width=True)

    if run_bench or "benchmark_result" not in st.session_state:
        with st.spinner(f"Running {2 * n_p_points * n_trials:,} total Monte Carlo trials…"):
            try:
                bench_result = noise_mod.run_reliability_benchmark(
                    n_trials=int(n_trials),
                    seed=int(seed),
                    n_p_points=int(n_p_points),
                )
                st.session_state.benchmark_result = bench_result
                st.session_state.benchmark_error = None
                st.session_state.benchmark_params = dict(
                    n_trials=n_trials, n_p_points=n_p_points, seed=seed,
                )
            except Exception as e:
                st.session_state.benchmark_result = None
                st.session_state.benchmark_error = str(e) + "\n" + traceback.format_exc()

    if st.session_state.get("benchmark_error"):
        st.error(f"Benchmark failed: {st.session_state.benchmark_error}")
        st.stop()

    bench = st.session_state.get("benchmark_result")
    if bench is None:
        st.info("Click **Run Full Reliability Benchmark** to start.")
        st.stop()

    # Results
    st.success(
        f"✅ Benchmark complete. {2 * len(bench.p_values) * bench.n_trials:,} total trials. "
        f"Elapsed: {bench.elapsed_seconds:.1f}s. Seed: {bench.seed}."
    )

    st.plotly_chart(
        viz.plot_reliability_curve(bench, show_confidence_intervals=show_ci),
        use_container_width=True,
    )

    # Point-specific comparison
    st.markdown("### Detailed Comparison at a Specific p Value")
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        selected_p = st.slider(
            "Select p value for detailed view",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
        )

    # Find nearest benchmark point
    idx = int(np.argmin(np.abs(bench.p_values - selected_p)))
    p_actual = float(bench.p_values[idx])

    with col_p2:
        enc_sim = float(bench.simulated_encoded[idx])
        unenc_sim = float(bench.simulated_unencoded[idx])
        enc_theory = float(bench.theoretical_encoded[idx])
        unenc_theory = float(bench.theoretical_unencoded[idx])

        comparison_cols = st.columns(4)
        with comparison_cols[0]:
            st.metric("p (actual)", f"{p_actual:.3f}")
        with comparison_cols[1]:
            st.metric(
                "Encoded failure rate",
                f"{enc_sim:.4f}",
                delta=f"Theory: {enc_theory:.4f}",
                delta_color="off",
            )
        with comparison_cols[2]:
            st.metric(
                "Unencoded failure rate",
                f"{unenc_sim:.4f}",
                delta=f"Theory: {unenc_theory:.4f}",
                delta_color="off",
            )
        with comparison_cols[3]:
            if enc_sim > 0:
                gain = unenc_sim / enc_sim
                st.metric("Observed gain", f"{gain:.2f}×",
                          delta="Code helps" if gain > 1 else "Code hurts")
            else:
                st.metric("Observed gain", "∞ (no failures)")

    fig_compare = viz.plot_error_rate_comparison(
        p=p_actual,
        sim_encoded=enc_sim,
        sim_unencoded=unenc_sim,
        theory_encoded=enc_theory,
        theory_unencoded=unenc_theory,
        ci_enc=(float(bench.ci_lower_encoded[idx]), float(bench.ci_upper_encoded[idx])),
        ci_unenc=(float(bench.ci_lower_unencoded[idx]), float(bench.ci_upper_unencoded[idx])),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Export
    st.markdown("### Export Results")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        df_export = noise_mod.benchmark_result_to_dataframe(bench)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇ Download benchmark results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"qubitshield_benchmark_seed{bench.seed}_n{bench.n_trials}.csv",
            mime="text/csv",
        )
    with col_exp2:
        # Reproduce command
        st.markdown("**Reproduce this run:**")
        st.code(
            f"python examples/reproduce_benchmark.py "
            f"--n-trials {bench.n_trials} "
            f"--n-points {len(bench.p_values)} "
            f"--seed {bench.seed}",
            language="bash",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Results & Interpretation
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📋 Results & Interpretation":
    st.title("📋 Results & Interpretation")

    bench = st.session_state.get("benchmark_result")
    if bench is None:
        st.warning(
            "No benchmark results available. Please run the **Reliability Benchmark** first."
        )
        st.stop()

    params = st.session_state.get("benchmark_params", {})
    n_trials = params.get("n_trials", bench.n_trials)
    seed = params.get("seed", bench.seed)

    st.markdown(f"""
    ### Experiment Summary
    
    The benchmark ran **{bench.n_trials:,} Monte Carlo trials** at each of **{len(bench.p_values)}** 
    physical error probability values, using seed **{bench.seed}**. A total of 
    **{2 * len(bench.p_values) * bench.n_trials:,}** random trials were performed. 
    Elapsed simulation time: **{bench.elapsed_seconds:.2f} seconds**.
    """)

    # Interpretation at key p values
    st.markdown("### Automatic Interpretation")

    key_ps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    for p_key in key_ps:
        idx = int(np.argmin(np.abs(bench.p_values - p_key)))
        p_v = float(bench.p_values[idx])
        enc_t = float(bench.theoretical_encoded[idx])
        unenc_t = float(bench.theoretical_unencoded[idx])
        enc_s = float(bench.simulated_encoded[idx])
        unenc_s = float(bench.simulated_unencoded[idx])
        ci_lo = float(bench.ci_lower_encoded[idx])
        ci_hi = float(bench.ci_upper_encoded[idx])
        abs_diff = abs(enc_s - enc_t)

        benefit = enc_t < unenc_t

        with st.expander(
            f"p = {p_v:.2f}  →  {'✅ Code helps' if benefit else '❌ Code hurts'}  "
            f"(P_encoded={enc_t:.4f} vs P_unencoded={unenc_t:.4f})"
        ):
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
                **Theoretical predictions:**
                - $P_{{\\text{{encoded}}}}({p_v:.2f}) = {enc_t:.6f}$
                - $P_{{\\text{{unencoded}}}}({p_v:.2f}) = {unenc_t:.6f}$
                - Reduction factor: {"%.2f×" % (unenc_t / enc_t) if enc_t > 0 else "∞"}
                
                **Monte Carlo simulation:**
                - Encoded failure rate: {enc_s:.6f}
                - 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]
                - |Sim − Theory|: {abs_diff:.6f}
                """)
            with col_r2:
                if benefit:
                    st.success(
                        f"At p={p_v:.2f}, the repetition code **reduces** logical failure "
                        f"probability from {unenc_t:.4f} to {enc_t:.4f} — a "
                        f"{'%.1f' % ((unenc_t - enc_t)/unenc_t * 100)}% reduction. "
                        f"The simulation differs from theory by {abs_diff:.4f}, consistent with "
                        f"statistical sampling fluctuations ({bench.n_trials:,} trials)."
                    )
                else:
                    st.error(
                        f"At p={p_v:.2f}, the repetition code **increases** logical failure "
                        f"probability from {unenc_t:.4f} to {enc_t:.4f}. When the physical "
                        f"error rate exceeds 50%, the code makes errors more likely by trying "
                        f"to correct what is actually the majority (correct) value."
                    )

    st.markdown("### Theory vs. Simulation Agreement")
    st.markdown("""
    The simulation uses genuinely random Monte Carlo sampling — it does not look up 
    pre-computed values. Deviations from the theoretical curve arise from finite 
    sampling (statistical noise). The 95% Wilson confidence intervals quantify this uncertainty.
    
    For **$n$ trials at error rate $p$**, the standard error of the simulated failure rate is 
    approximately $\\sqrt{p(1-p)/n}$. At $p=0.1$ with $n=5000$, this is approximately $0.0042$.
    """)

    # Overall summary table
    st.markdown("### Full Results Table (10 representative points)")
    sample_indices = np.linspace(0, len(bench.p_values)-1, 10, dtype=int)
    df_summary = pd.DataFrame({
        "p": [f"{bench.p_values[i]:.3f}" for i in sample_indices],
        "P_encoded (theory)": [f"{bench.theoretical_encoded[i]:.6f}" for i in sample_indices],
        "P_encoded (sim)": [f"{bench.simulated_encoded[i]:.6f}" for i in sample_indices],
        "P_unencoded (theory)": [f"{bench.theoretical_unencoded[i]:.6f}" for i in sample_indices],
        "P_unencoded (sim)": [f"{bench.simulated_unencoded[i]:.6f}" for i in sample_indices],
        "|Sim − Theory|": [f"{abs(bench.simulated_encoded[i]-bench.theoretical_encoded[i]):.6f}" for i in sample_indices],
        "Code helps?": ["✓ Yes" if bench.theoretical_encoded[i] < bench.theoretical_unencoded[i] else "✗ No" for i in sample_indices],
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: About & Limitations
# ─────────────────────────────────────────────────────────────────────────────

elif page == "ℹ️ About & Limitations":
    st.title("ℹ️ About & Limitations")

    about_tabs = st.tabs(["About", "Architecture", "Limitations", "References"])

    with about_tabs[0]:
        st.markdown("""
        ## About QubitShield

        **QubitShield** is a scientific portfolio project by **Vaibhav Pandey**, 
        B.E. Computer Science and Engineering student at BMS Institute of Technology 
        & Management (CGPA: 8.86/10).

        The project demonstrates practical understanding of:
        - Quantum error correction: stabilizer codes, syndrome extraction, recovery
        - Quantum circuit design and simulation with Qiskit
        - Statistical comparison of theoretical predictions with Monte Carlo simulation
        - Scientific software architecture: modular Python, type hints, test coverage

        **Technologies used:**
        - **Python 3.13** — primary language
        - **Qiskit 1.4.x** — quantum circuit construction and gate-level simulation
        - **Qiskit Aer 0.15.x** — statevector and shot-based simulation backend
        - **Streamlit 1.50.x** — interactive dashboard framework
        - **Plotly 6.x** — interactive scientific charts
        - **NumPy / Pandas** — numerical and data processing
        - **SciPy** — statistical confidence intervals
        - **Pytest** — automated test suite

        **Repository:** [github.com/mevaibhavpandey/qubitshield-qec-lab](https://github.com/mevaibhavpandey/qubitshield-qec-lab)
        """)

        st.info(
            "**Simulation disclaimer:** All results in this application are produced by "
            "classical software simulation using Qiskit Aer. No results were obtained from "
            "physical quantum hardware. Qiskit Aer is a high-performance simulator that "
            "replicates the mathematical behavior of quantum circuits, but it does not "
            "capture all physical noise sources present in real quantum devices."
        )

    with about_tabs[1]:
        st.markdown("""
        ## Software Architecture

        ```
        qubitshield-qec-lab/
        ├── app.py                    # Streamlit dashboard (this file)
        ├── src/qubitshield/
        │   ├── states.py             # State vectors, normalization, fidelity
        │   ├── circuits.py           # Qiskit circuit construction
        │   ├── syndrome.py           # Syndrome table and decoder
        │   ├── recovery.py           # Recovery operations, statevector verification
        │   ├── noise.py              # Monte Carlo experiments, BenchmarkResult
        │   ├── experiments.py        # High-level Syndrome Lab runners
        │   ├── analytics.py          # Theoretical formulas, confidence intervals
        │   └── visualization.py     # Plotly/Matplotlib charts
        ├── tests/                    # Pytest test suite
        ├── docs/                     # Mathematical and architectural documentation
        └── examples/reproduce_benchmark.py
        ```

        **Data flow:**
        1. `circuits.py` builds Qiskit circuits from parameters
        2. `experiments.py` runs them via `qiskit_aer.AerSimulator`
        3. `syndrome.py` decodes measurement results
        4. `recovery.py` verifies correction via statevector simulation
        5. `noise.py` runs Monte Carlo sweeps using analytical syndrome logic
        6. `analytics.py` provides theoretical reference values
        7. `visualization.py` converts results to Plotly figures
        8. `app.py` assembles everything into the Streamlit interface
        """)

    with about_tabs[2]:
        st.markdown("""
        ## Known Limitations

        ### Error model limitations
        - **Idealized independent bit-flip model only.** Real quantum devices experience 
          depolarizing noise, phase-flip errors, amplitude damping, crosstalk, and 
          correlated errors.
        - **No circuit-level noise.** Encoding, CNOT gates, ancilla preparation, and 
          measurement all introduce additional errors in real systems. A fault-tolerant 
          design would need to account for these.
        - **Phase errors not corrected.** The 3-qubit bit-flip code corrects X errors 
          but not Z errors. A complete error-correcting code requires addressing both 
          (e.g., the Shor 9-qubit code or surface codes).

        ### Simulation limitations
        - **Classical simulation.** Qiskit Aer computes exact quantum states, which 
          becomes exponentially expensive for larger circuits. Real hardware adds noise 
          that classical simulation cannot perfectly model.
        - **No mid-circuit feedback in statevector mode.** The recovery operation is 
          applied classically in Python after syndrome measurement, which is equivalent 
          under ideal conditions but differs architecturally from true real-time feedback.

        ### What this project does NOT demonstrate
        - Fault-tolerant quantum error correction (where the correction operations 
          are themselves protected)
        - Threshold theorem results
        - Results from physical quantum hardware
        - Performance under realistic circuit-level noise models
        - Scalable codes (surface codes, LDPC codes)
        """)

    with about_tabs[3]:
        st.markdown("""
        ## References & Learning Resources

        **Official Documentation:**
        - [Qiskit Documentation](https://docs.quantum.ibm.com/)
        - [Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/)
        - [Qiskit Textbook — Quantum Error Correction](https://learn.qiskit.org/course/quantum-hardware/error-correction-repetition-code)

        **Textbooks:**
        - Nielsen & Chuang, *Quantum Computation and Quantum Information* (Cambridge, 2000)
          — Chapter 10 covers quantum error correction
        - Preskill, *Lecture Notes on Quantum Computation* (Caltech) 
          — Available free at [theory.caltech.edu/~preskill/ph229/](http://theory.caltech.edu/~preskill/ph229/)

        **Research Papers:**
        - Shor, P. W. (1995). Scheme for reducing decoherence in quantum computer memory. 
          *Physical Review A*, 52(4), R2493.
        - Calderbank, A. R., & Shor, P. W. (1996). Good quantum error-correcting codes exist. 
          *Physical Review A*, 54(2), 1098.

        **Statistical methodology:**
        - Wilson, E. B. (1927). Probable inference, the law of succession, and statistical 
          inference. *Journal of the American Statistical Association*, 22(158), 209–212.
          (Wilson score confidence interval, used for failure rate estimation)
        """)
