# ==========================================================
# IQI & QML Project – Complete Qiskit Implementation
# Author: (Your name)
# Faculty of Materials Engineering and Technical Physics
# Poznań University of Technology
#
# Tasks included:
#   Task 1 – Z-type projection measurement – reading of the qubit state
#   Task 2 – Operation and reading of the result of quantum gate X
#   Task 3 – SUPERPOSITION OF STATES – Operation and reading of H gate
#   Task 4 – State tomography of one qubit – Measurement in X, Y, Z base
#   Task 5 – Extra visualisations for 3 quantum circuits
# ==========================================================

import os, math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit_aer import Aer
except Exception:
    from qiskit import Aer
from qiskit.visualization import (
    plot_state_city,
    plot_state_hinton,
    plot_state_qsphere,
    plot_bloch_multivector,
)

# ----------------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------------
np.random.seed(7)
SHOTS = 2048
OUTDIR = "iqi_qml_final_output"
os.makedirs(OUTDIR, exist_ok=True)
backend = Aer.get_backend("qasm_simulator")

# ----------------------------------------------------------
# PLOTTING HELPERS (for counts & probabilities)
# ----------------------------------------------------------
def _annotate(ax):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(round(h))}",
                        (p.get_x() + p.get_width()/2., h),
                        ha="center", va="bottom", fontsize=8)

def plot_counts_and_probs(runs, title, stem):
    keys = sorted(set().union(*[set(d.keys()) for d in runs])) or ["0000"]
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(runs))
    labels = ["First execution", "Second execution", "Third execution"]

    # Plot counts
    fig = plt.figure(figsize=(7.5,4.5)); ax = plt.gca()
    for i, d in enumerate(runs):
        ax.bar(x+i*width, [d.get(k,0) for k in keys], width=width, label=labels[i])
    ax.set_title("Distribution (counts)")
    ax.set_xticks(x + width*(len(runs)-1)/2); ax.set_xticklabels(keys, rotation=60)
    ax.set_ylabel("Count"); ax.legend(); _annotate(ax)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{stem}_counts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot probabilities
    fig = plt.figure(figsize=(7.5,4.5)); ax = plt.gca()
    for i, d in enumerate(runs):
        probs = [d.get(k,0)/SHOTS for k in keys]
        ax.bar(x+i*width, probs, width=width, label=labels[i])
        for xi, yi in zip(x+i*width, probs):
            ax.text(xi, yi + 0.015, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Probability"); ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width*(len(runs)-1)/2); ax.set_xticklabels(keys, rotation=60)
    ax.set_ylabel("Quasi-probability"); ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{stem}_probs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# EXECUTION HELPERS
# ----------------------------------------------------------
def execute_three_times(qc):
    runs=[]
    for _ in range(3):
        job = backend.run(qc, shots=SHOTS)
        runs.append(dict(job.result().get_counts()))
    return runs

# ==========================================================
# TASK 1 – Z-type projection measurement – reading of the qubit state
# ==========================================================
q=QuantumRegister(4,'q'); c=ClassicalRegister(4,'c')
qc1=QuantumCircuit(q,c,name="Task1_Z_projection")
qc1.measure(q[0],c[0])
runs=execute_three_times(qc1)
plot_counts_and_probs(runs, "Task 1 – Z-type projection measurement – reading of the qubit state", "task1_z_projection")

# ==========================================================
# TASK 2 – Operation and reading of the result of quantum gate X
# ==========================================================
q=QuantumRegister(4,'q'); c=ClassicalRegister(4,'c')
qc2=QuantumCircuit(q,c,name="Task2_X_gate")
qc2.x(q[0])
qc2.measure(q[0],c[0])
runs=execute_three_times(qc2)
plot_counts_and_probs(runs, "Task 2 – Operation and reading of the result of quantum gate X", "task2_x_gate")

# ==========================================================
# TASK 3 – SUPERPOSITION OF STATES
# Operation and reading of the result of Hadamard quantum gate (H gate)
# ==========================================================
q=QuantumRegister(4,'q'); c=ClassicalRegister(4,'c')
qc3=QuantumCircuit(q,c,name="Task3_H_gate")
qc3.h(q[0])
qc3.measure(q[0],c[0])
runs=execute_three_times(qc3)
plot_counts_and_probs(runs, "Task 3 – SUPERPOSITION OF STATES – Hadamard quantum gate (H gate)", "task3_h_gate")

# ==========================================================
# TASK 4 – State tomography of one qubit
# Measurement in the X, Y, Z base
# ==========================================================
def prep_state_ry_p(qc,q0):
    qc.ry(np.pi/2,q0)
    qc.p(np.pi/2,q0)   # same as S gate

def measure_in_basis(basis):
    def f(qc,q0):
        if basis=="X": qc.h(q0)
        elif basis=="Y": qc.sdg(q0); qc.h(q0)
        elif basis=="Z": pass
    return f

for basis in ["X","Y","Z"]:
    q=QuantumRegister(4,'q'); c=ClassicalRegister(4,'c')
    qc=QuantumCircuit(q,c,name=f"Task4_{basis}_basis")
    prep_state_ry_p(qc,q[0])
    measure_in_basis(basis)(qc,q[0])
    qc.measure(q[0],c[0])
    runs=execute_three_times(qc)
    plot_counts_and_probs(runs, f"Task 4 – Measurement in the {basis} base", f"task4_{basis}_basis")

# ==========================================================
# TASK 5 – Extra visualisations for 3 quantum circuits
# (|0⟩, |1⟩, |+⟩, and prepared RY–P state)
# ==========================================================
sv_backend = Aer.get_backend("statevector_simulator")

def statevector_of(qc):
    job=sv_backend.run(qc)
    return job.result().get_statevector(qc)

def prep_only(name):
    q=QuantumRegister(1,'q'); qc=QuantumCircuit(q,name=name)
    if name=="|0>": pass
    elif name=="|1>": qc.x(0)
    elif name=="|+>": qc.h(0)
    elif name=="RY-P": qc.ry(np.pi/2,0); qc.p(np.pi/2,0)
    return qc

def save(fig,stem):
    fig.savefig(os.path.join(OUTDIR,stem),dpi=150,bbox_inches="tight"); plt.close(fig)

preps=["|0>","|1>","|+>","RY-P"]
for p in preps:
    qc=prep_only(p)
    psi=statevector_of(qc)
    tag=p.replace("|","").replace(">","").replace("+","plus")

    save(plot_state_city(psi,title=f"Task 5 – State city – {p}"),f"task5_city_{tag}.png")
    save(plot_state_hinton(psi,title=f"Task 5 – State hinton – {p}"),f"task5_hinton_{tag}.png")
    save(plot_state_qsphere(psi),f"task5_qsphere_{tag}.png")
    save(plot_bloch_multivector(psi,title=f"Task 5 – Bloch multivector – {p}"),f"task5_bloch_{tag}.png")

print(f"All tasks complete. Results saved to: {os.path.abspath(OUTDIR)}")