# BB84 (prepare-and-measure) simulation in modern Qiskit
# Tested with: qiskit>=2.x, qiskit-aer>=0.17
# Install (if needed):
#   pip install "qiskit[visualization]" qiskit-aer

from __future__ import annotations

from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator


def build_bb84_round_circuit() -> QuantumCircuit:
    """
    One BB84 'round' circuit that:
      - uses qubit q1 (after H+measure) as Alice's random bit x_A
      - uses qubit q2 (after H+measure) as Alice's random basis y_A
      - encodes on data qubit q0 via CX (for x_A) and CH (for y_A)
      - uses qubit q3 (after H+measure) as Bob's random basis y_B
      - decodes on q0 via CH (controlled by y_B), measures x_B
    Classical bits (c[0..3]) store [x_B, x_A, y_A, y_B] respectively.
    Keys returned by the backend will list bits as 'yB yA xA xB' (c3 c2 c1 c0).
    """
    q = QuantumRegister(4, "q")  # [q0=data, q1=Alice bit, q2=Alice basis, q3=Bob basis]
    c = ClassicalRegister(4, "c")  # [c0=x_B, c1=x_A, c2=y_A, c3=y_B]
    qc = QuantumCircuit(q, c, name="bb84_round")

    # Ensure a clean start
    qc.reset(q)

    # --- Alice's randomness ---
    # x_A (random bit)
    qc.h(q[1])
    qc.measure(q[1], c[1])

    # y_A (random basis: 0->Z, 1->X)
    qc.h(q[2])
    qc.measure(q[2], c[2])

    qc.barrier()

    # --- Alice's encoding on the data qubit q0 ---
    # Encode bit (flip if x_A==1)
    qc.cx(q[1], q[0])
    # Encode basis (apply H if y_A==1)
    qc.ch(q[2], q[0])

    qc.barrier()

    # --- Bob's randomness y_B and decoding ---
    qc.h(q[3])               # y_B random basis selector
    qc.measure(q[3], c[3])   # store y_B
    qc.barrier()

    # Apply H to data iff y_B==1 (controlled by q3)
    qc.ch(q[3], q[0])

    # Bob measures data -> x_B
    qc.measure(q[0], c[0])

    qc.barrier()
    return qc


def run_bb84(sample: int = 10, seed: int | None = None) -> Tuple[List[List[int]], Dict[str, int], QuantumCircuit]:
    """
    Execute 'sample' independent BB84 rounds by running the same circuit multiple shots.
    Returns:
      bit_quads: list of [xA, yA, yB, xB] per shot
      counts: raw counts from the simulator (bitstring 'yB yA xA xB')
      qc: the compiled circuit (for inspection/plotting)
    """
    backend = AerSimulator()
    qc = build_bb84_round_circuit()
    tqc = transpile(qc, backend)

    # Run all rounds in one job (shots = sample)
    job = backend.run(tqc, shots=sample, seed_simulator=seed)
    result = job.result()
    counts = result.get_counts(tqc)

    # Parse counts strings: 'yB yA xA xB' (c3 c2 c1 c0)
    bit_quads: List[List[int]] = []
    for bitstring, reps in counts.items():
        yB = int(bitstring[0])
        yA = int(bitstring[1])
        xA = int(bitstring[2])
        xB = int(bitstring[3])
        for _ in range(reps):
            bit_quads.append([xA, yA, yB, xB])

    return bit_quads, counts, qc


def sift_key(bit_quads: List[List[int]]) -> Tuple[List[int], List[int]]:
    """
    Keep bits where y_A == y_B. Returns (keyA, keyB).
    """
    keyA, keyB = [], []
    for xA, yA, yB, xB in bit_quads:
        if yA == yB:
            keyA.append(xA)
            keyB.append(xB)
    return keyA, keyB


def bb84_trial(sample: int, seed: int | None = None) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Run a single trial of 'sample' rounds and return (keyA, keyB, bit_quads).
    """
    bits, counts, _ = run_bb84(sample=sample, seed=seed)
    keyA, keyB = sift_key(bits)
    # Sanity check (no eavesdropper in this ideal simulation)
    assert keyA == keyB, "Keys should match in a noiseless simulation."
    return keyA, keyB, bits


def experiment_table(samples=(16, 32, 64, 128, 256), trials: int = 3, seed: int | None = None) -> Dict[int, List[int]]:
    """
    For each sample size n, run 'trials' independent tests and record the sifted-key length.
    Returns a dict: { n: [len_trial1, len_trial2, ...] }
    """
    out: Dict[int, List[int]] = {}
    base_seed = seed
    for idx, n in enumerate(samples):
        lengths = []
        for t in range(trials):
            # Optionally vary seed for reproducibility without removing randomness entirely
            trial_seed = None if base_seed is None else (base_seed + 1000 * idx + t)
            keyA, keyB, _ = bb84_trial(sample=n, seed=trial_seed)
            lengths.append(len(keyA))
        out[n] = lengths
    return out


if __name__ == "__main__":
    # Demo run (mirrors the slide’s printed output style)
    sample = 10
    bits, counts, qc = run_bb84(sample=sample, seed=None)
    print("Raw counts (format {'yB yA xA xB': count}):", counts)

    for row in bits:
        xA, yA, yB, xB = row
        print(f"[xA,yA,yB,xB] = {row}")

    keyA, keyB = sift_key(bits)
    print("kluczA =", keyA)
    print("kluczB =", keyB)
    print("kluczA == kluczB ?", keyA == keyB)

    # Fill the table for your report (Test 1..3 lengths per n)
    table = experiment_table(samples=(16, 32, 64, 128, 256), trials=3, seed=None)
    print("\nTable: sifted key lengths per n (trials columns):")
    for n, lens in table.items():
        print(f"n={n}: {lens}")