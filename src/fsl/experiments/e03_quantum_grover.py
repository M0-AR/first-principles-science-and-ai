"""E03: Quantum computing from scratch — statevector simulator + Grover.

A pure-numpy n-qubit statevector simulator. We verify the textbook
quadratic speedup: Grover finds the marked item after ~(pi/4)sqrt(N)
oracle calls with probability approaching 1.
"""
from __future__ import annotations

import numpy as np

from fsl.common import ExperimentResult, run_experiment


def apply_hadamard(psi: np.ndarray, q: int) -> np.ndarray:
    h = (np.array([1, 1, 1, -1]) / np.sqrt(2)).reshape(2, 2)
    shaped = psi.reshape(-1, 2, 2**q)
    return np.einsum("abc,db->adc", shaped, h).reshape(-1)


def oracle(psi: np.ndarray, marked: int) -> np.ndarray:
    out = psi.copy()
    out[marked] *= -1
    return out


def diffusion(psi: np.ndarray) -> np.ndarray:
    return 2 * psi.mean() - psi


def grover(n: int, marked: int, iterations: int) -> float:
    N = 2**n
    psi = np.zeros(N, dtype=complex)
    psi[0] = 1.0
    for q in range(n):
        psi = apply_hadamard(psi, q)
    for _ in range(iterations):
        psi = diffusion(oracle(psi, marked))
    return float(abs(psi[marked]) ** 2)


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        rows = []
        prev = 0.0
        for n in (2, 3, 4, 5, 6):
            N = 2**n
            k_max = int(np.floor(np.pi / 4 * np.sqrt(N))) + 2
            probs = [grover(n, 1, k) for k in range(0, k_max + 1)]
            best_k = int(np.argmax(probs))
            best_p = probs[best_k]
            rows.append({"n": n, "N": N, "best_k": best_k, "p_success": best_p,
                         "classical_queries": N, "speedup": N / max(best_k, 1)})
            if n >= 3:
                res.check(f"Grover n={n} amplifies the marked state", best_p > 0.9,
                          f"p={best_p:.5f} at k={best_k}")
            prev = best_p
        res.check("n=6 Grover success > 0.99", rows[-1]["p_success"] > 0.99,
                  f"p={rows[-1]['p_success']:.6f}")
        res.check("iterations scale like sqrt(N)",
                  abs(rows[-1]["best_k"] - np.pi / 4 * np.sqrt(rows[-1]["N"])) <= 2,
                  f"k={rows[-1]['best_k']} vs pi/4*sqrt(N)={np.pi/4*np.sqrt(rows[-1]['N']):.2f}")
        res.check("quantum speedup is quadratic", 6 <= rows[-1]["speedup"] <= 11,
                  f"N/k={rows[-1]['speedup']:.2f}")
        res.metrics.update({"rows": rows})
    return run_experiment("e03_quantum_grover", body)
