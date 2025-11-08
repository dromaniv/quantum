# lab3.py - BB84 protocol simulation

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile

import matplotlib.pyplot as plt
import pandas as pd

# Create results folder if it doesn't exist
import os

os.makedirs("l3_results", exist_ok=True)

# -----------------------------------------------------------
# 1. Build the BB84 quantum circuit (Alice + Bob)
# -----------------------------------------------------------

nx = 4
q = QuantumRegister(nx, "q")
c = ClassicalRegister(nx, "c")
circuit = QuantumCircuit(q, c)

# Initialize qubits to |0>
circuit.reset([q[0], q[1], q[2], q[3]])

# --- Alice's random bits xA and yA ---
circuit.h(q[1])
circuit.measure(q[1], c[1])

circuit.h(q[2])
circuit.measure(q[2], c[2])

circuit.barrier(q[0], q[1], q[2], q[3])

# --- Alice's encoding ---
circuit.cx(q[1], q[0])
circuit.ch(q[2], q[0])

circuit.barrier(q[0], q[1], q[2], q[3])

# --- Bob's random basis yB ---
circuit.h(q[3])
circuit.measure(q[3], c[3])

circuit.barrier(q[0], q[1], q[2], q[3])

# --- Bob's decoding and measurement xB ---
circuit.ch(q[3], q[0])
circuit.measure(q[0], c[0])

circuit.barrier(q[0], q[1], q[2], q[3])

# --------- NEW: plot and save circuit diagram ----------
# This will create bb84_circuit.png in the current folder
fig = circuit.draw(output="mpl")
plt.tight_layout()
plt.savefig("l3_results/bb84_circuit.png", dpi=300)
plt.close()
# -----------------------------------------------------------


# -----------------------------------------------------------
# 2. Helper functions: running and sifting
# -----------------------------------------------------------

def run_experiment_bits_result(circ, backend, executions: int, shots: int = 1):
    """
    Run the given circuit 'executions' times with 'shots' shots each
    (slides use shots=1 in a loop), and collect [xA, yA, yB, xB].
    Bitstring format: { 'yB yA xA xB' : 1 }.
    """
    bits = []

    for _ in range(executions):
        compiled = transpile(circ, backend)
        job = backend.run(compiled, shots=shots)
        result = job.result().get_counts(circ)

        key = list(result.keys())[0]

        # 'yB yA xA xB'
        yB = int(key[0])
        yA = int(key[1])
        xA = int(key[2])
        xB = int(key[3])

        bits.append([xA, yA, yB, xB])

    return bits


def sift_key(bits):
    """
    Key sifting as in the slides: keep only positions where yA == yB.
    Returns (length, keyA, keyB).
    """
    keyA = []
    keyB = []

    for xA, yA, yB, xB in bits:
        if yA == yB:
            keyA.append(xA)
            keyB.append(xB)

    return len(keyA), keyA, keyB


# -----------------------------------------------------------
# 3. Main program: single-shot demo + table + logging
# -----------------------------------------------------------

if __name__ == "__main__":
    backend = Aer.get_backend("qasm_simulator")

    # -------------------------------
    # Single test run (like slide 16)
    # -------------------------------
    compiled_circuit = transpile(circuit, backend)
    job_single = backend.run(compiled_circuit, shots=1)
    result_single = job_single.result()
    wyniki = result_single.get_counts(circuit)

    print("Single run result (format {'yB yA xA xB': 1}):")
    print(wyniki)

    key = list(wyniki.keys())[0]
    yB = int(key[0])
    yA = int(key[1])
    xA = int(key[2])
    xB = int(key[3])
    print("Parsed bits -> [xA, yA, yB, xB] =", [xA, yA, yB, xB])
    print()

    # -------------------------------
    # 10-sample demo (like slide 17)
    # -------------------------------
    sample = 10
    bits = []

    print(f"Generating {sample} raw results:")
    for kk in range(sample):
        compiled = transpile(circuit, backend)
        job = backend.run(compiled, shots=1)
        res = job.result()
        wynik = res.get_counts(circuit)

        key = list(wynik.keys())[0]
        yB = int(key[0])
        yA = int(key[1])
        xA = int(key[2])
        xB = int(key[3])

        print(wynik, "->", [xA, yA, yB, xB])
        bits.append([xA, yA, yB, xB])

    print("All bits:", bits)
    print()

    # -------------------------------
    # Key sifting demo (slide 18)
    # -------------------------------
    sift_len, keyA, keyB = sift_key(bits)

    print("Key sifting (keep positions where yA == yB):")
    for xA, yA, yB, xB in bits:
        if yA == yB:
            print(f"yA={yA}, yB={yB} -> xA={xA}, xB={xB}")

    print("kluczA =", keyA)
    print("kluczB =", keyB)
    print("kluczA == kluczB ?", keyA == keyB)
    print()

    # -------------------------------
    # Final experiment table (slide 19)
    # -------------------------------
    samples = [16, 32, 64, 128, 256]
    tests = 3
    shots = 1

    results_rows = []

    for n in samples:
        row = [n]
        for t in range(tests):
            bits_n = run_experiment_bits_result(circuit, backend, executions=n, shots=shots)
            sift_len_n, _, _ = sift_key(bits_n)
            row.append(sift_len_n)
        results_rows.append(row)

    df = pd.DataFrame(results_rows, columns=["Sample = n", "Test 1", "Test 2", "Test 3"])

    print("Final table: length of sifted key vs n")
    print(df.to_string(index=False))
    print()

    # --------- NEW: save & print report-friendly formats ----------

    # 1) Save as CSV (you can import this in Excel / LibreOffice)
    df.to_csv("l3_results/bb84_results.csv", index=False)
    print("Saved results to bb84_results.csv")

    # 2) Print LaTeX table (paste directly into your report)
    print("\nLaTeX table (for your report):\n")
    print(df.to_latex(index=False))

    # 3) Optional: simple Markdown table (if you write the report in Markdown)
    try:
        print("\nMarkdown table:\n")
        print(df.to_markdown(index=False))
    except AttributeError:
        # to_markdown may require tabulate; ignore if not available
        pass