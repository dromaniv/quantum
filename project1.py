# IQI & QML – full tasks (Qiskit)
# Tasks:
# 1) Z-type projection measurement – reading of the qubit state
# 2) Operation and reading of the result of quantum gate X
# 3) SUPERPOSITION OF STATES – Operation and reading of the result of H gate
# 4) State tomography of one qubit – Measurement in the X, Y, Z base
# 5) Extra visualisations for 3 quantum circuits

import os, numpy as np, matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit_aer import Aer
except Exception:
    from qiskit import Aer
from qiskit.visualization import (
    plot_state_city, plot_state_hinton, plot_state_qsphere, plot_bloch_multivector
)

# ---------- setup ----------
np.random.seed(7)
SHOTS = 2048
OUT = "p1_results"
os.makedirs(OUT, exist_ok=True)
qasm_backend = Aer.get_backend("qasm_simulator")
sv_backend = Aer.get_backend("statevector_simulator")

# ---------- tiny helpers ----------
def run3(qc):
    out = []
    for _ in range(3):
        res = qasm_backend.run(qc, shots=SHOTS).result()
        out.append(dict(res.get_counts()))
    return out

def plot_counts_probs(runs, title, stem):
    keys = sorted(set().union(*map(dict.keys, runs))) or ["0000"]
    x = np.arange(len(keys)); width = 0.8/len(runs)
    labels = ["First execution","Second execution","Third execution"]

    # counts
    fig = plt.figure(figsize=(7.5,4.5)); ax = fig.gca()
    for i,d in enumerate(runs):
        ax.bar(x+i*width, [d.get(k,0) for k in keys], width=width, label=labels[i])
    ax.set_title("Distribution (counts)"); ax.set_ylabel("Count"); ax.legend()
    ax.set_xticks(x+width); ax.set_xticklabels(keys, rotation=60)
    for p in ax.patches:
        h = p.get_height()
        if h: ax.annotate(str(int(h)), (p.get_x()+p.get_width()/2, h), ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, y=1.02); plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"{stem}_counts.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # probability
    fig = plt.figure(figsize=(7.5,4.5)); ax = fig.gca()
    for i,d in enumerate(runs):
        probs = [d.get(k,0)/SHOTS for k in keys]
        ax.bar(x+i*width, probs, width=width, label=labels[i])
        for xi, yi in zip(x+i*width, probs):
            ax.text(xi, yi+0.015, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Probability"); ax.set_ylim(0,1.05); ax.set_ylabel("Quasi-probability"); ax.legend()
    ax.set_xticks(x+width); ax.set_xticklabels(keys, rotation=60)
    fig.suptitle(title, y=1.02); plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"{stem}_probs.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Task 1 ----------
q = QuantumRegister(4,'q'); c = ClassicalRegister(4,'c')
t1 = QuantumCircuit(q,c, name="Task 1 – Z-type projection")
t1.measure(q[0], c[0])
plot_counts_probs(run3(t1),
    "Task 1 – Z-type projection measurement – reading of the qubit state",
    "task1_z_projection"
)

# ---------- Task 2 ----------
q = QuantumRegister(4,'q'); c = ClassicalRegister(4,'c')
t2 = QuantumCircuit(q,c, name="Task 2 – X gate")
t2.x(q[0]); t2.measure(q[0], c[0])
plot_counts_probs(run3(t2),
    "Task 2 – Operation and reading of the result of quantum gate X",
    "task2_x_gate"
)

# ---------- Task 3 ----------
q = QuantumRegister(4,'q'); c = ClassicalRegister(4,'c')
t3 = QuantumCircuit(q,c, name="Task 3 – H gate")
t3.h(q[0]); t3.measure(q[0], c[0])
plot_counts_probs(run3(t3),
    "Task 3 – SUPERPOSITION OF STATES – Operation and reading of the result of Hadamard quantum gate (H gate)",
    "task3_h_gate"
)

# ---------- Task 4 (state tomography: measure same state in X/Y/Z) ----------
def prep_ry_p(qc, q0):  # RY(pi/2) then P(pi/2)
    qc.ry(np.pi/2, q0); qc.p(np.pi/2, q0)

def add_basis_change(qc, q0, basis):
    if basis == "X": qc.h(q0)
    if basis == "Y": qc.sdg(q0); qc.h(q0)

for basis in ["X","Y","Z"]:
    q = QuantumRegister(4,'q'); c = ClassicalRegister(4,'c')
    t4 = QuantumCircuit(q,c, name=f"Task 4 – {basis} basis")
    prep_ry_p(t4, q[0])
    add_basis_change(t4, q[0], basis)
    t4.measure(q[0], c[0])
    plot_counts_probs(run3(t4),
        f"Task 4 – State tomography of one qubit – Measurement in the {basis} base",
        f"task4_{basis}_basis"
    )

# ---------- Task 5 (extra visualisations for 3 circuits) ----------
def statevector(qc):  # no measurements in these circuits
    return sv_backend.run(qc).result().get_statevector(qc)

def prep_single(name):
    qc = QuantumCircuit(1, name=name)
    if name == "|0>": pass
    elif name == "|1>": qc.x(0)
    elif name == "|+>": qc.h(0)
    return qc

for name in ["|0>", "|1>", "|+>"]:
    psi = statevector(prep_single(name))
    tag = name.replace("|","").replace(">","").replace("+","plus")
    save(plot_state_city(psi, title=f"Task 5 – State city – {name}"),   f"task5_city_{tag}.png")
    save(plot_state_hinton(psi, title=f"Task 5 – State hinton – {name}"), f"task5_hinton_{tag}.png")

    # qsphere doesn't support title parameter, so we add it manually
    fig_qsphere = plot_state_qsphere(psi)
    fig_qsphere.suptitle(f"Task 5 – Qsphere – {name}", y=0.98)
    save(fig_qsphere, f"task5_qsphere_{tag}.png")

    save(plot_bloch_multivector(psi, title=f"Task 5 – Bloch multivector – {name}"), f"task5_bloch_{tag}.png")

print(f"done. results in: {os.path.abspath(OUT)}")