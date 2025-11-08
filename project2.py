# chsh_lab_exact.py
# CHSH on |Ψ-> with lab-exact circuits/plots (no Ry). Saves ALL outputs to p2_results.
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import HGate, SGate, TGate, TdgGate

# ---------------- Config ----------------
SHOTS = 1024
N_RUNS = 5
OUTDIR = "p2_results"  # everything goes here

os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Lab-exact state prep & measurements ----------------
def prepare_psi_minus_lab(qc: QuantumCircuit):
    """
    Lab slide exact |Ψ-> prep from |00>:
      X on both qubits, H on q0, CX(0->1).  (No Z.)
    """
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def apply_bottom_measurement(qc: QuantumCircuit, basis: str, qb: int):
    """Bottom (Alice): Z or X only. X = H, Z = no rotation."""
    if basis == "X":
        qc.append(HGate(), [qb])
    elif basis == "Z":
        pass
    else:
        raise ValueError("Bottom must be 'X' or 'Z'.")

def apply_top_measurement(qc: QuantumCircuit, basis: str, qt: int):
    """
    Top (Bob): W or V only.
      W = S–H–T–H
      V = S–H–T†–H
    """
    if basis == "W":
        qc.append(SGate(), [qt]); qc.append(HGate(), [qt]); qc.append(TGate(), [qt]); qc.append(HGate(), [qt])
    elif basis == "V":
        qc.append(SGate(), [qt]); qc.append(HGate(), [qt]); qc.append(TdgGate(), [qt]); qc.append(HGate(), [qt])
    else:
        raise ValueError("Top must be 'W' or 'V'.")

def make_measure_circuit(bottom_basis: str, top_basis: str) -> QuantumCircuit:
    """
    Build circuit in the lab convention:
      top wire (q0) = W/V  ; bottom wire (q1) = X/Z
      classical mapping: top → c1, bottom → c0  (so readout strings are 'ab')
    """
    qc = QuantumCircuit(2, 2, name=f"{bottom_basis}⊗{top_basis}")
    # |Ψ-> prep (slide exact)
    prepare_psi_minus_lab(qc)
    qc.barrier()
    # rotate to requested measurement bases
    apply_top_measurement(qc,   top_basis,    0)  # q0 (top)
    apply_bottom_measurement(qc, bottom_basis, 1)  # q1 (bottom)
    qc.barrier()
    # measure: top→c1, bottom→c0  (bitstring "ab")
    qc.measure(0, 0)  # a (top) -> c1
    qc.measure(1, 1)  # b (bottom) -> c0
    return qc

# ---------------- Stats helpers ----------------
def counts_to_probs(counts: Dict[str, int], shots: int) -> Dict[Tuple[int, int], float]:
    """Convert {'ab': count} to p(a,b) with a,b∈{0,1}."""
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
    """E = Σ v(a)v(b) p(a,b) with 0→+1, 1→−1."""
    v = {0: +1, 1: -1}
    return sum(v[a] * v[b] * p for (a, b), p in pab.items())

# ---------------- Run & summarize ----------------
@dataclass
class RunResult:
    counts: Dict[str, int]
    probs: Dict[Tuple[int, int], float]
    E: float

def run_setting(bottom_basis: str, top_basis: str, shots: int, backend) -> RunResult:
    circ = make_measure_circuit(bottom_basis, top_basis)
    tc = transpile(circ, backend)
    job = backend.run(tc, shots=shots)
    res = job.result()
    counts = res.get_counts(tc)
    pab = counts_to_probs(counts, shots)
    E = expected_value_from_probs(pab)
    return RunResult(counts=counts, probs=pab, E=E)

def run_all(shots: int = SHOTS, n_runs: int = N_RUNS):
    """
    Combos in lab order: (bottom, top)
      X⊗W, X⊗V, Z⊗W, Z⊗V
    """
    backend = AerSimulator()
    combos = [("X", "W"), ("X", "V"), ("Z", "W"), ("Z", "V")]
    per_run = []
    E_by_run = []
    for r in range(n_runs):
        row = {}
        for b, t in combos:
            rr = run_setting(b, t, shots, backend)
            row[f"E_{b}{t}"] = rr.E
            per_run.append(((b, t), rr))
        S = row["E_ZW"] + row["E_ZV"] + row["E_XW"] - row["E_XV"]
        row["S"] = S
        E_by_run.append(row)
    return combos, per_run, E_by_run

# ---------------- Saving utilities ----------------
def save_circuit_images():
    backend = AerSimulator()
    for b, t in [("X","W"),("X","V"),("Z","W"),("Z","V")]:
        circ = make_measure_circuit(b, t)
        path_png = os.path.join(OUTDIR, f"circ_{b}{t}.png")
        path_pdf = os.path.join(OUTDIR, f"circ_{b}{t}.pdf")
        fig = circ.draw(output="mpl")
        fig.savefig(path_png, bbox_inches="tight", dpi=300)
        fig.savefig(path_pdf, bbox_inches="tight")
        plt.close(fig)

def save_probability_plots(all_runs: List[Tuple[Tuple[str,str], RunResult]]):
    """
    For each measurement type, show bar plots of outcome probabilities across the N runs.
    """
    order = ["00","01","10","11"]
    for b, t in [("X","W"),("X","V"),("Z","W"),("Z","V")]:
        probs_per_run = []
        for run_idx in range(N_RUNS):
            # pick the run_idx-th record for this (b,t)
            matches = [rr for (cmb, rr) in all_runs if cmb==(b,t)]
            rr = matches[run_idx]
            # map back to strings for plotting
            dd = {"00":0.0,"01":0.0,"10":0.0,"11":0.0}
            for (a,b2), p in rr.probs.items():
                dd[f"{a}{b2}"] = p
            probs_per_run.append([dd[k] for k in order])

        x = np.arange(len(order))
        width = 0.12
        fig, ax = plt.subplots(figsize=(8,4.5))
        for i in range(N_RUNS):
            ax.bar(x + i*width - (N_RUNS-1)*width/2, probs_per_run[i], width, label=f"Run {i+1}")
        ax.set_xticks(x, order)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability")
        ax.set_title(f"Outcome probabilities for {b}⊗{t}")
        ax.legend(ncols=min(N_RUNS,5), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"prob_{b}{t}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(OUTDIR, f"prob_{b}{t}.pdf"), bbox_inches="tight")
        plt.close(fig)

def save_tables(E_by_run: List[Dict[str,float]], all_runs: List[Tuple[Tuple[str,str], RunResult]]):
    # Table: E and S per run (CSV + TeX)
    import csv
    cols = ["E_ZW","E_ZV","E_XW","E_XV","S"]
    with open(os.path.join(OUTDIR, "table_E_S_by_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"]+cols)
        for i,row in enumerate(E_by_run):
            w.writerow([i] + [row[c] for c in cols])
    with open(os.path.join(OUTDIR, "table_E_S_by_run.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\\toprule\nRun & $E_{ZW}$ & $E_{ZV}$ & $E_{XW}$ & $E_{XV}$ & $S$\\\\\\midrule\n")
        for i,row in enumerate(E_by_run):
            f.write(f"{i+1} & {row['E_ZW']:+.6f} & {row['E_ZV']:+.6f} & {row['E_XW']:+.6f} & {row['E_XV']:+.6f} & {row['S']:+.6f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    # Table: joint probabilities for the FIRST run of each measurement (CSV + TeX)
    # (bitstrings listed as 00,01,10,11)
    first_probs = {("X","W"):None,("X","V"):None,("Z","W"):None,("Z","V"):None}
    for (cmb, rr) in all_runs:
        if first_probs[cmb] is None:
            first_probs[cmb] = rr
    order = ["00","01","10","11"]
    import csv
    with open(os.path.join(OUTDIR, "table_joint_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring","X⊗W","X⊗V","Z⊗W","Z⊗V"])
        for bs in order:
            def getp(b,t):
                dd = { (0,0):0,(0,1):0,(1,0):0,(1,1):0}
                for (a,b2),p in first_probs[(b,t)].probs.items():
                    dd[(a,b2)] = p
                a,b2 = int(bs[0]), int(bs[1])
                return dd[(a,b2)]
            w.writerow([bs, getp("X","W"), getp("X","V"), getp("Z","W"), getp("Z","V")])

    with open(os.path.join(OUTDIR, "table_joint_probs.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\\toprule\nBitstring & $X\\otimes W$ & $X\\otimes V$ & $Z\\otimes W$ & $Z\\otimes V$\\\\\\midrule\n")
        for bs in order:
            def getp(b,t):
                dd = { (0,0):0,(0,1):0,(1,0):0,(1,1):0}
                for (a,b2),p in first_probs[(b,t)].probs.items():
                    dd[(a,b2)] = p
                a,b2 = int(bs[0]), int(bs[1])
                    # value exists after dict fill
                return dd[(a,b2)]
            f.write(f"{bs} & {getp('X','W'):.6f} & {getp('X','V'):.6f} & {getp('Z','W'):.6f} & {getp('Z','V'):.6f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

# ---------------- Main ----------------
if __name__ == "__main__":
    # Save lab-exact circuit schematics first (so you can compare visually)
    save_circuit_images()

    combos, per_run, E_by_run = run_all(SHOTS, N_RUNS)
    save_probability_plots(per_run)
    save_tables(E_by_run, per_run)

    # Quick textual summary
    absS = [abs(row["S"]) for row in E_by_run]
    print("\n====== Summary over runs (lab rule: top=W/V, bottom=X/Z) ======")
    for k in ("E_ZW","E_ZV","E_XW","E_XV"):
        arr = [row[k] for row in E_by_run]
        print(f"{k}: mean = {np.mean(arr):+.6f} , std = {np.std(arr, ddof=1):.6f}")
    print(f"\n|S| values: " + ", ".join(f"{v:.6f}" for v in absS))
    print(f"mean(|S|)  = {np.mean(absS):.6f}")
    print(f"std(|S|)   = {np.std(absS, ddof=1):.6f}")
    print(f"\nSaved files in {OUTDIR}/:")
    print("  - circ_XW/XV/ZW/ZV.(pdf|png)")
    print("  - prob_XW/XV/ZW/ZV.(pdf|png)")
    print("  - table_E_S_by_run.(csv|tex), table_joint_probs.(csv|tex)")