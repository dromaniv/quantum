# Qiskit 1.x solution for IQI & QML Lab 2 (tasks 1–7)
# Bell states + XX/YY/XZ measurements with plots

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_distribution
import matplotlib.pyplot as plt
from copy import deepcopy
import os


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def bell_state(label: str) -> QuantumCircuit:
    """
    Build:
      'phi_plus'  = (|00> + |11>)/√2
      'phi_minus' = (|00> - |11>)/√2
      'psi_plus'  = (|01> + |10>)/√2
      'psi_minus' = (|01> - |10>)/√2
    Gates are applied before the entangling CX to match the baseline circuits.
    """
    qc = QuantumCircuit(2, 2, name=label)

    # Build circuits to match the baseline diagrams exactly:
    #   phi_plus  : H(q0); CX(q0->q1)
    #   phi_minus : X(q0); H(q0); CX(q0->q1)
    #   psi_plus  : H(q0); X(q1); CX(q0->q1)
    #   psi_minus : X(q0); X(q1); H(q0); CX(q0->q1)
    if label == "phi_plus":
        qc.h(0)
        qc.cx(0, 1)
    elif label == "phi_minus":
        qc.x(0)
        qc.h(0)
        qc.cx(0, 1)
    elif label == "psi_plus":
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
    elif label == "psi_minus":
        qc.x(0)
        qc.x(1)
        qc.h(0)
        qc.cx(0, 1)
    else:
        raise ValueError("label must be one of: phi_plus, phi_minus, psi_plus, psi_minus")

    return qc


def add_measurement(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    """
    Append basis-change ops and Z-basis measurements:
      'ZZ' : computational basis on both
      'XX' : H on both, then measure
      'YY' : Sdg then H on both, then measure
      'XZ' : H on qubit 0 only; qubit 1 in Z
    """
    c = deepcopy(qc)

    # Add a visual barrier to separate state preparation from measurement basis gates
    c.barrier()

    if basis == "ZZ":
        pass
    elif basis == "XX":
        c.h(0); c.h(1)
    elif basis == "YY":
        c.sdg(0); c.h(0)
        c.sdg(1); c.h(1)
    elif basis == "XZ":
        c.h(0)
    else:
        raise ValueError("basis must be one of: ZZ, XX, YY, XZ")

    # Map qubits → same-index classical bits
    c.measure(0, 0)
    c.measure(1, 1)
    c.name = f"{qc.name}_{basis}"
    return c


def run_and_plot(circuits, shots=1024, title_prefix=""):
    """
    Transpile for AerSimulator, run, and plot distributions.
    Returns list of (counts, fig) for convenience.
    """
    sim = AerSimulator(seed_simulator=123)
    tcircs = transpile(circuits, sim)
    job = sim.run(tcircs, shots=shots)
    result = job.result()

    plot_dir = "l2_results"
    os.makedirs(plot_dir, exist_ok=True)
    outputs = []
    for i, c in enumerate(circuits):
        counts = result.get_counts(i)  # counts dict
        fig = plot_distribution(counts, title=f"{title_prefix}{c.name} (shots={shots})")
        path = os.path.join(plot_dir, f"{c.name}.png")
        fig.savefig(path)
        print(f"Saved plot to {path}")
        plt.close(fig)
        outputs.append((counts, fig))
    return outputs


def draw_and_show(qc: QuantumCircuit, title=None):
    """Draw circuit with MPL drawer, save, and show."""
    fig = qc.draw(output="mpl")

    plot_dir = "l2_results"
    os.makedirs(plot_dir, exist_ok=True)

    filename = title if title else qc.name
    path = os.path.join(plot_dir, f"{filename}.png")
    fig.savefig(path)
    print(f"Saved circuit plot to {path}")
    plt.close(fig)
    return fig


# ------------------------------------------------------------
# Tasks 1–4: build + plot the four Bell-state circuits in ZZ
# ------------------------------------------------------------
bells = {
    "task1_phi_plus":  bell_state("phi_plus"),
    "task2_phi_minus": bell_state("phi_minus"),
    "task3_psi_plus":  bell_state("psi_plus"),
    "task4_psi_minus": bell_state("psi_minus"),
}

# Plot circuits (as required in the homework)
for name, qc in bells.items():
    draw_and_show(add_measurement(qc, "ZZ"), title=f"{name} - ZZ measurement")

# Run and plot ZZ distributions (tasks 1–4 results)
zz_circuits = [add_measurement(qc, "ZZ") for qc in bells.values()]
zz_results = run_and_plot(zz_circuits, shots=1024, title_prefix="Result: ")


# ------------------------------------------------------------
# Tasks 5–7: same Bell states, other measurement bases
#   Task 5: XX on circuits from tasks 1–4
#   Task 6: YY on circuits from tasks 1–4
#   Task 7: XZ on circuits from tasks 1–4
# ------------------------------------------------------------

# Prepare all requested measurement settings
xx_circuits = [add_measurement(qc, "XX") for qc in bells.values()]
yy_circuits = [add_measurement(qc, "YY") for qc in bells.values()]
xz_circuits = [add_measurement(qc, "XZ") for qc in bells.values()]

# Plot circuits (optional; comment out if you only want results)
for c in xx_circuits + yy_circuits + xz_circuits:
    draw_and_show(c, title=f"{c.name} - circuit")

# Run and plot results
xx_results = run_and_plot(xx_circuits, shots=1024, title_prefix="Result: ")
yy_results = run_and_plot(yy_circuits, shots=1024, title_prefix="Result: ")
xz_results = run_and_plot(xz_circuits, shots=1024, title_prefix="Result: ")


# ------------------------------------------------------------
# (Optional) quick expectation helper for 2-qubit settings
# E = Σ_{b in {00,01,10,11}} s(b)*P(b),
# where s(b) = +1 for 00,11 and −1 for 01,10 (correlation).
# ------------------------------------------------------------
def expectation_from_counts(counts):
    shots = sum(counts.values())
    p = {k: v/shots for k, v in counts.items()}
    return (p.get('00',0)+p.get('11',0)) - (p.get('01',0)+p.get('10',0))

# Example: compute <XX> for Φ+
# (rerun after the above execution so xx_circuits results exist)
counts_xx_phi_plus = xx_results[0][0]   # (counts, fig) for first XX circuit (Φ+)
print("⟨XX⟩ for Φ+ ≈", expectation_from_counts(counts_xx_phi_plus))