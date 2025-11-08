# chsh_mdpi.py
# Complete CHSH experiment on |Ψ-> with repetition, uncertainty, and MDPI-style plots.
# Qiskit Aer simulator; swap in a hardware backend if desired.

from __future__ import annotations
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------- Configuration ----------------
SHOTS = 1024          # Shots per setting per run (assignment uses 1024)
N_RUNS = 5            # Number of independent repeats for uncertainty (Task 2.1 suggests ~5)
BASE_SEED = None      # Set to an int for reproducibility; None = nondeterministic seeds
SAVE_FIGS = True      # Save figures to disk (PDF + PNG)
SHOW_FIGS = False     # Show figures interactively

# ---------------- State preparation & measurement ----------------

def prepare_psi_minus(qc: QuantumCircuit):
    """Prepare |Ψ-> = (|10> - |01>)/√2 on qubits 0,1."""
    qc.x(1)       # |01>
    qc.h(0)
    qc.cx(0, 1)   # Bell
    qc.z(0)       # Map to Ψ- up to global phase
    return qc

def rotate_to_basis(qc: QuantumCircuit, qubit: int, basis: str):
    """
    Basis choices:
      'Z' : measure Z (no rotation)
      'X' : H† Z H = X  ⇒ apply H before Z-measure
      'W' : U† Z U = (Z + X)/√2 with U = R_y(-π/4)
      'V' : U† Z U = (Z - X)/√2 with U = R_y(+π/4)
    """
    from qiskit.circuit.library import HGate, RYGate
    if basis == "Z":
        return
    elif basis == "X":
        qc.append(HGate(), [qubit])
    elif basis == "W":
        qc.append(RYGate(-np.pi/4), [qubit])
    elif basis == "V":
        qc.append(RYGate(+np.pi/4), [qubit])
    else:
        raise ValueError(f"Unknown basis '{basis}'. Use one of: Z, X, W, V.")

def make_measure_circuit(a_basis: str, b_basis: str) -> QuantumCircuit:
    """
    Build the full circuit:
      - Prepare |Ψ->
      - Rotate qubit 0 to a_basis, qubit 1 to b_basis
      - Measure in Z, store as classical bits [a b] (left bit=a, right bit=b)
    """
    qc = QuantumCircuit(2, 2, name=f"{a_basis}⊗{b_basis}")
    prepare_psi_minus(qc)
    qc.barrier()
    rotate_to_basis(qc, 0, a_basis)
    rotate_to_basis(qc, 1, b_basis)
    qc.barrier()
    # Measure so that readout string is "ab"
    qc.measure(0, 1)  # a -> c1 (left char)
    qc.measure(1, 0)  # b -> c0 (right char)
    return qc

def counts_to_probs(counts: Dict[str, int], shots: int) -> Dict[Tuple[int, int], float]:
    """Convert {'ab': count} to p(a,b) with a,b∈{0,1}; ensure all outcomes present."""
    p = defaultdict(float)
    for key, c in counts.items():
        s = key.replace(" ", "")
        if len(s) == 2 and all(ch in "01" for ch in s):
            a, b = int(s[0]), int(s[1])
            p[(a, b)] += c / shots
    for a in (0, 1):
        for b in (0, 1):
            p[(a, b)] += 0.0
    return dict(p)

def expected_value_from_probs(pab: Dict[Tuple[int, int], float]) -> float:
    """E = Σ v(a)v(b) p(a,b) with mapping 0→+1, 1→−1."""
    v = {0: +1, 1: -1}
    return sum(v[a] * v[b] * p for (a, b), p in pab.items())

# ---------------- Single-run execution ----------------

def run_setting(a_basis: str, b_basis: str, shots: int, backend, seed: int | None):
    circ = make_measure_circuit(a_basis, b_basis)
    tcirc = transpile(circ, backend)
    job = backend.run(tcirc, shots=shots, seed_simulator=seed)
    result = job.result()
    counts = result.get_counts(tcirc)
    pab = counts_to_probs(counts, shots)
    E = expected_value_from_probs(pab)
    return {"counts": counts, "p": pab, "E": E}

def run_all(shots: int, seed: int | None = None):
    """
    Run the four CHSH settings and compute S = E(Z,W)+E(Z,V)+E(X,W)-E(X,V).
    Returns a dict keyed by setting tags and 'S', '|S|'.
    """
    backend = AerSimulator()
    labels = [("Z", "W", "ZW"), ("Z", "V", "ZV"), ("X", "W", "XW"), ("X", "V", "XV")]
    out: Dict[str, dict] = {}
    for a, b, tag in labels:
        out[tag] = run_setting(a, b, shots=shots, backend=backend, seed=seed)
    S = out["ZW"]["E"] + out["ZV"]["E"] + out["XW"]["E"] - out["XV"]["E"]
    out["S"] = float(S)
    out["|S|"] = float(abs(S))
    return out

# ---------------- Repetition & statistics (Task 2.1) ----------------

@dataclass
class Summary:
    shots: int
    n_runs: int
    s_abs_values: List[float]
    mean_abs_s: float
    std_abs_s: float
    E_matrix: np.ndarray   # shape (n_runs, 4) in order [ZW, ZV, XW, XV]
    E_mean: np.ndarray     # shape (4,)
    E_std: np.ndarray      # shape (4,)

def repeat_and_summarize(n_runs: int = N_RUNS, shots: int = SHOTS, base_seed: int | None = BASE_SEED) -> Tuple[Summary, List[dict]]:
    tags = ["ZW", "ZV", "XW", "XV"]
    per_run_results: List[dict] = []
    E_runs = np.zeros((n_runs, 4), dtype=float)
    s_abs = []

    for r in range(n_runs):
        seed = None if base_seed is None else (base_seed + r)
        res = run_all(shots=shots, seed=seed)
        per_run_results.append(res)
        s_abs.append(res["|S|"])
        E_runs[r, :] = [res[t]["E"] for t in tags]

    s_abs = np.array(s_abs, dtype=float)
    E_mean = E_runs.mean(axis=0)
    E_std = E_runs.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(4)

    summary = Summary(
        shots=shots,
        n_runs=n_runs,
        s_abs_values=list(s_abs),
        mean_abs_s=float(s_abs.mean()),
        std_abs_s=float(s_abs.std(ddof=1)) if n_runs > 1 else 0.0,
        E_matrix=E_runs,
        E_mean=E_mean,
        E_std=E_std,
    )
    return summary, per_run_results

# ---------------- Plotting (MDPI-friendly single-panel figures) ----------------

def _apply_mdpi_rc():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # Serif font, modest sizes; figure width ~12 cm
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (4.72, 3.54),  # ~12cm x 9cm
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt

def plot_E_bars(summary: Summary, save_prefix: str = "fig_E_bars"):
    """
    Bar plot of mean E with 1σ error bars across runs.
    Saves PDF and PNG if SAVE_FIGS is True.
    """
    plt = _apply_mdpi_rc()
    import matplotlib.pyplot as plt

    tags = ["ZW", "ZV", "XW", "XV"]
    x = np.arange(len(tags))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, summary.E_mean, yerr=summary.E_std, capsize=4)
    ax.axhline( 1/np.sqrt(2), linestyle="--", linewidth=1)
    ax.axhline(-1/np.sqrt(2), linestyle="--", linewidth=1)
    ax.set_xticks(x, tags)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Correlation  E")
    ax.set_title("CHSH Correlators (mean ± 1σ over runs)")
    ax.text(0.98, 0.05, r"$\pm 1/\sqrt{2}$ ref.", transform=ax.transAxes, ha="right", va="bottom")
    if SAVE_FIGS:
        fig.savefig(f"{save_prefix}.pdf")
        fig.savefig(f"{save_prefix}.png")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)

def plot_S_hist(summary: Summary, save_prefix: str = "fig_S_hist"):
    """
    Histogram of |S| across runs with classical (2) and Tsirelson (2√2) reference lines.
    Saves PDF and PNG if SAVE_FIGS is True.
    """
    plt = _apply_mdpi_rc()
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(summary.s_abs_values, bins=min(10, max(3, summary.n_runs // 1)))
    ax.axvline(2.0, linestyle="--", linewidth=1)
    ax.axvline(2*np.sqrt(2), linestyle="--", linewidth=1)
    ax.set_xlabel(r"$|S|$")
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $|S|$ across runs")
    ax.text(0.02, 0.84, "Classical bound = 2", transform=ax.transAxes, va="top")
    ax.text(0.02, 0.74, r"Tsirelson bound = $2\sqrt{2}$", transform=ax.transAxes, va="top")
    if SAVE_FIGS:
        fig.savefig(f"{save_prefix}.pdf")
        fig.savefig(f"{save_prefix}.png")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)

# ---------------- Pretty printing ----------------

def print_run(res: dict):
    print("\nRun results:")
    for tag in ["ZW", "ZV", "XW", "XV"]:
        print(f"  {tag}: E = {res[tag]['E']:+.6f}  counts = {res[tag]['counts']}")
    print(f"  S   = {res['S']:+.6f}")
    print(f"  |S| = {res['|S|']:.6f}")

def print_summary(summary: Summary):
    tags = ["ZW", "ZV", "XW", "XV"]
    print("\n====== Summary over runs ======")
    for i, tag in enumerate(tags):
        print(f"E_{tag}: mean = {summary.E_mean[i]:+.6f} , std = {summary.E_std[i]:.6f}")
    print(f"\n|S| values: {', '.join(f'{v:.6f}' for v in summary.s_abs_values)}")
    print(f"mean(|S|)  = {summary.mean_abs_s:.6f}")
    print(f"std(|S|)   = {summary.std_abs_s:.6f}")
    print("Reference: ideal E = ±1/√2 ≈ ±0.7071, ideal |S| = 2√2 ≈ 2.8284")

# ---------------- Main ----------------

if __name__ == "__main__":
    # One illustrative run (mirrors the assignment’s Task 1 output)
    single = run_all(shots=SHOTS)
    print_run(single)

    # Task 2.1: repeat n_runs times and compute mean/std of |S|
    summary, runs = repeat_and_summarize(n_runs=N_RUNS, shots=SHOTS, base_seed=BASE_SEED)
    print_summary(summary)

    # MDPI-style figures
    plot_E_bars(summary, save_prefix="fig_E_bars_chsh")
    plot_S_hist(summary, save_prefix="fig_S_hist_chsh")

    print("\nFigures saved as PDF (vector) and PNG (300 dpi):")
    print("  - fig_E_bars_chsh.pdf  /  fig_E_bars_chsh.png")
    print("  - fig_S_hist_chsh.pdf  /  fig_S_hist_chsh.png")