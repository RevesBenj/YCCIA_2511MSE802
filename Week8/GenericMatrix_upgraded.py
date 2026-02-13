# -------------------------------------------------------
# Generic Matrix
# Author: Benjelyn Reves Patiag
# Description:
A small, **pure NumPy** "quantum-by-matrices" toolkit.

What you can do here:
- Build common gates (I, X, H)
- Build tensor products (Kronecker products)
- Build basis states |...>
- Apply gates to statevectors
- Build Deutsch / Deutsch–Jozsa oracle matrix U_f from a truth table or expression
- Simulate Deutsch (n=1) and Deutsch–Jozsa (any n) using only matrices

Important convention (to match the original oracle builder in this file):
- Total qubits = n input qubits + 1 ancilla qubit y
- Basis index for |x, y> is: index = (x << 1) | y
  -> ancilla y is the **least significant** qubit (rightmost bit)
  -> input register x occupies the higher bits.

So for n=2:
|x,y> ordering is |00,0>,|00,1>,|01,0>,|01,1>,|10,0>,|10,1>,|11,0>,|11,1>
indices:            0     1     2     3     4     5     6     7
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
========================================




# =========================
# 1) Basic single-qubit gates
# =========================

def I2(dtype=np.complex128) -> np.ndarray:
    """1-qubit Identity gate (2x2)."""
    return np.eye(2, dtype=dtype)

def X(dtype=np.complex128) -> np.ndarray:
    """Pauli-X (bit flip) gate (2x2)."""
    return np.array([[0, 1],
                     [1, 0]], dtype=dtype)

def H(dtype=np.complex128) -> np.ndarray:
    """Hadamard gate (2x2). Makes superposition."""
    return (1 / math.sqrt(2)) * np.array([[1,  1],
                                          [1, -1]], dtype=dtype)


# =========================
# 2) Tensor product utilities
# =========================

def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tensor product (Kronecker product) of two matrices."""
    return np.kron(A, B)

def kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    """
    Tensor product of a list of matrices.
    Example: H ⊗ H ⊗ I
    """
    if not mats:
        raise ValueError("kron_all needs at least 1 matrix")
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def gate_on_qubits(total_qubits: int,
                   targets: Sequence[int],
                   single_qubit_gate: np.ndarray,
                   dtype=np.complex128) -> np.ndarray:
    """
    Build a big 2^N x 2^N matrix that applies a given 2x2 gate on selected qubits.

    Qubit indexing:
      - qubit 0 is the **least significant** (rightmost) bit in |...>
      - qubit (total_qubits-1) is the most significant (leftmost)

    Example:
      total_qubits=3, targets=[2,1] with H => H on qubit2 and qubit1, identity on qubit0.
    """
    mats: List[np.ndarray] = []
    for q in reversed(range(total_qubits)):  # build from MSB to LSB
        mats.append(single_qubit_gate.astype(dtype) if q in targets else I2(dtype))
    return kron_all(mats)


# =========================
# 3) Statevector helpers
# =========================

def dim_from_qubits(total_qubits: int) -> int:
    return 1 << total_qubits

def ket(index: int, total_qubits: int, dtype=np.complex128) -> np.ndarray:
    """Computational basis ket |index> as a column vector (size 2^N)."""
    dim = dim_from_qubits(total_qubits)
    if index < 0 or index >= dim:
        raise ValueError(f"index must be in [0,{dim-1}]")
    v = np.zeros(dim, dtype=dtype)
    v[index] = 1
    return v

def ket_x_y(x: int, y: int, n_inputs: int, dtype=np.complex128) -> np.ndarray:
    """Return |x,y> as statevector, with index = (x<<1)|y."""
    total = n_inputs + 1
    idx = (x << 1) | (y & 1)
    return ket(idx, total, dtype=dtype)

def apply(U: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply a unitary (or any matrix) to a statevector."""
    return U @ state

def normalize(state: np.ndarray) -> np.ndarray:
    """Normalize a statevector (useful if you build something manually)."""
    nrm = np.linalg.norm(state)
    if nrm == 0:
        raise ValueError("Cannot normalize zero vector")
    return state / nrm

def probs(state: np.ndarray) -> np.ndarray:
    """Measurement probabilities for each basis state."""
    return np.abs(state) ** 2

def is_unitary(U: np.ndarray, atol: float = 1e-9) -> bool:
    """Check U†U = I."""
    I = np.eye(U.shape[0], dtype=U.dtype)
    return np.allclose(U.conj().T @ U, I, atol=atol)

def pretty_state(index: int, total_qubits: int) -> str:
    """Format |0101> style."""
    return f"|{index:0{total_qubits}b}>"

def measure_register_probs(state: np.ndarray, n_inputs: int) -> np.ndarray:
    """
    For total qubits = n_inputs + 1 (ancilla is last/LSB),
    return probabilities of measuring ONLY the input register (size 2^n_inputs).

    We sum over ancilla y=0 and y=1.
    """
    total = n_inputs + 1
    p_full = probs(state)
    out = np.zeros(1 << n_inputs, dtype=float)
    for x in range(1 << n_inputs):
        out[x] = p_full[(x << 1) | 0] + p_full[(x << 1) | 1]
    return out


# =========================
# 4) Oracle (Uf) builder (kept + upgraded)
# =========================

def bits_of_x(x: int, n: int) -> Dict[str, int]:
    """
    Return dict like {'x0':LSB, 'x1':..., 'x{n-1}':MSB}.
    We use x0 as the least-significant bit (rightmost).
    """
    return {f"x{i}": (x >> i) & 1 for i in range(n)}

def parse_truth_table(tt: str, n: int) -> List[int]:
    """
    Parse truth table string into list of 0/1 of length 2^n.
    Accept formats like: "0 1 1 0" or "0110" or "0,1,1,0".
    Order: x=0..2^n-1 (binary order).
    """
    cleaned = tt.replace(",", " ").strip()
    parts = cleaned.split()

    if len(parts) == 1 and set(parts[0]).issubset({"0", "1"}):
        values = [int(ch) for ch in parts[0]]
    else:
        values = [int(p) for p in parts]

    N = 1 << n
    if len(values) != N:
        raise ValueError(f"Need exactly {N} values for n={n} (for x=0..{N-1}).")
    if any(v not in (0, 1) for v in values):
        raise ValueError("Truth table values must be only 0 or 1.")
    return values

def build_fx_from_expression(expr: str, n: int) -> List[int]:
    """
    Build truth table by evaluating an expression for each x.

    Allowed variables:
      - x  (integer 0..2^n-1)
      - x0..x(n-1) (bits)
    Allowed helpers: bin, int, abs

    Example expressions:
      x0 ^ x1
      x0 & x1
      bin(x).count('1') % 2
    """
    N = 1 << n
    fx: List[int] = []

    safe_builtins = {"bin": bin, "int": int, "abs": abs}

    for x in range(N):
        env = {"x": x, **bits_of_x(x, n)}
        val = eval(expr, {"__builtins__": safe_builtins}, env)  # user-controlled expr
        val = int(val) & 1  # force 0/1
        fx.append(val)

    return fx

def build_uf_matrix(
    n: int,
    f: Union[Callable[[int], int], Iterable[int]],
    dtype=np.complex128
) -> np.ndarray:
    """
    Build U_f for n input qubits + 1 ancilla qubit.

    Definition:
        U_f |x, y> = |x, y XOR f(x)>

    Returns:
        Uf of shape (2^(n+1), 2^(n+1))
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    N = 1 << n
    dim = 1 << (n + 1)

    # Normalize f into list fx[x] in {0,1}
    if callable(f):
        fx = [int(f(x)) & 1 for x in range(N)]
    else:
        fx = [int(v) & 1 for v in f]

    if len(fx) != N:
        raise ValueError(f"Truth table must have length 2^n = {N}, got {len(fx)}")
    if any(v not in (0, 1) for v in fx):
        raise ValueError("f(x) outputs must be 0 or 1 only")

    Uf = np.zeros((dim, dim), dtype=dtype)

    # Column is input |x,y>, row is output |x, y XOR f(x)>
    for x in range(N):
        for y in (0, 1):
            col = (x << 1) | y
            row = (x << 1) | (y ^ fx[x])
            Uf[row, col] = 1

    return Uf

def basis_mapping(Uf: np.ndarray) -> List[Tuple[int, int]]:
    """Return mapping (col -> row) for a permutation-type Uf."""
    mapping: List[Tuple[int, int]] = []
    for col in range(Uf.shape[1]):
        row = int(np.argmax(np.abs(Uf[:, col])))
        mapping.append((col, row))
    return mapping

def print_mapping(Uf: np.ndarray, n_inputs: int) -> None:
    """Print basis mapping in |...> format."""
    total = n_inputs + 1
    for col, row in basis_mapping(Uf):
        print(f"{pretty_state(col, total)} -> {pretty_state(row, total)}")


# =========================
# 5) Matrix-only simulations
# =========================

def simulate_deutsch(fx: Union[Iterable[int], Callable[[int], int]],
                     dtype=np.complex128) -> Tuple[np.ndarray, str]:
    """
    Deutsch algorithm (n=1) simulation using matrices only.

    Input:
      fx: truth table of length 2 (f(0), f(1)) OR a callable f(x)->0/1

    Returns:
      (final_state, verdict)
      verdict: "constant" or "balanced" based on measuring the top qubit.
    """
    n = 1
    Uf = build_uf_matrix(n, fx, dtype=dtype)

    # total qubits = 2 (x + ancilla y)
    total = 2

    # Start |0,1>
    state = ket_x_y(0, 1, n_inputs=n, dtype=dtype)

    # Apply H on both qubits: H ⊗ H
    HH = kron_all([H(dtype), H(dtype)])  # MSB then LSB, but both H so OK
    state = apply(HH, state)

    # Apply oracle
    state = apply(Uf, state)

    # Apply H on input qubit only (x is the MSB here)
    Hx = gate_on_qubits(total_qubits=2, targets=[1], single_qubit_gate=H(dtype), dtype=dtype)
    # Explanation: with total=2, qubit indexing LSB=0 (ancilla), MSB=1 (input)
    state = apply(Hx, state)

    # Measure input register only (1 bit)
    p_in = measure_register_probs(state, n_inputs=1)
    # If p(|0>) ~ 1 => constant, else balanced
    verdict = "constant" if np.isclose(p_in[0], 1.0, atol=1e-9) else "balanced"
    return state, verdict

def simulate_deutsch_jozsa(n: int,
                           fx: Union[Iterable[int], Callable[[int], int]],
                           dtype=np.complex128) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Deutsch–Jozsa algorithm simulation (general n), using only matrices.

    Promise problem:
      - f is either constant or balanced (half 0s, half 1s)

    Circuit:
      |0...0> |1>
        -> H^(⊗(n+1))
        -> U_f
        -> H^(⊗n) on input register
        -> measure input register

    Decision:
      - If measured input is all-zeros with probability 1 => constant
      - Else => balanced

    Returns:
      final_state, verdict, input_register_probs
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    Uf = build_uf_matrix(n, fx, dtype=dtype)
    total = n + 1

    # start |0...0, 1>
    # index = (x<<1)|y with x=0, y=1 -> index = 1
    state = ket(index=1, total_qubits=total, dtype=dtype)

    # H on all qubits
    H_all = gate_on_qubits(total_qubits=total, targets=list(range(total)), single_qubit_gate=H(dtype), dtype=dtype)
    state = apply(H_all, state)

    # Oracle
    state = apply(Uf, state)

    # H on input register only (qubits 1..n are input? careful)
    # With our convention: ancilla is qubit 0 (LSB). Inputs occupy qubits 1..n (n qubits).
    input_targets = list(range(1, n + 1))
    H_inputs = gate_on_qubits(total_qubits=total, targets=input_targets, single_qubit_gate=H(dtype), dtype=dtype)
    state = apply(H_inputs, state)

    # Measure only inputs
    p_inputs = measure_register_probs(state, n_inputs=n)
    verdict = "constant" if np.isclose(p_inputs[0], 1.0, atol=1e-9) else "balanced"
    return state, verdict, p_inputs


# =========================
# 6) Optional: adapters (only if installed)
# =========================

def uf_to_qiskit_gate(Uf: np.ndarray, label: str = "Uf"):
    """
    Convert Uf matrix into a Qiskit gate.
    Requires qiskit installed. If not installed, raises ImportError.
    """
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import UnitaryGate
    return UnitaryGate(Operator(Uf), label=label)

def uf_to_cirq_gate(Uf: np.ndarray):
    """
    Convert Uf matrix into a Cirq gate.
    Requires cirq installed. If not installed, raises ImportError.
    """
    import cirq
    return cirq.MatrixGate(Uf)


# =========================
# 7) CLI (interactive) - still available
# =========================

def main() -> None:
    print("=== GenericMatrix (Oracle + Gates + Simulation) ===")
    print("We will build U_f and (optionally) simulate Deutsch–Jozsa.\n")

    # --- Get n ---
    while True:
        try:
            n = int(input("Enter n (number of input qubits, e.g. 1,2,3): ").strip())
            if n < 1:
                raise ValueError
            break
        except Exception:
            print("Please enter a valid integer n >= 1.\n")

    N = 1 << n
    print(f"\nDefine f(x) for x=0..{N-1} (total {N} inputs).")
    method = input("Choose method: 1) Truth table  2) Expression  : ").strip()

    if method == "1":
        print("\nTruth table order is x=0..(2^n-1). For n=2 => 00,01,10,11.")
        tt = input(f"Enter {N} outputs (example: 0110 or 0 1 1 0): ").strip()
        fx = parse_truth_table(tt, n)
    else:
        print("\nExpression tips:")
        print(" - x0 is rightmost bit (LSB), x1 next, etc.")
        print(" - use ^ XOR, & AND, | OR")
        expr = input("Enter expression for f(x) (example: x0 ^ x1): ").strip()
        fx = build_fx_from_expression(expr, n)

    Uf = build_uf_matrix(n, fx)
    print("\n=== U_f built ===")
    print(f"U_f size: {Uf.shape[0]} x {Uf.shape[1]}")
    print("Unitary:", is_unitary(Uf))

    if input("Print basis mapping? (y/n): ").strip().lower() == "y":
        print_mapping(Uf, n_inputs=n)

    if input("\nSimulate Deutsch–Jozsa with this f? (y/n): ").strip().lower() == "y":
        _, verdict, p_inputs = simulate_deutsch_jozsa(n, fx)
        print("\n=== Deutsch–Jozsa result ===")
        print("Verdict:", verdict)
        print("Input register probabilities (index 0..2^n-1):")
        for i, p in enumerate(p_inputs):
            if p > 1e-6:
                print(f"  {i:>2} ({i:0{n}b}) : {p:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
