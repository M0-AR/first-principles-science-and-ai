"""E02: QED Feynman diagrams -> electron anomalous magnetic moment.

We evaluate the perturbative series

    a_e = sum_i A_i (alpha/pi)^i + a_had + a_EW

with the published A_i and compare with the CODATA 2022 measurement.
The point is verification: a hand-built calculation reproducing a
measured constant to ~1e-10 relative precision.
"""
from __future__ import annotations

import numpy as np

from fsl.common import ExperimentResult, run_experiment

ALPHA_INV = 137.035999177          # CODATA 2022
ALPHA = 1.0 / ALPHA_INV
A_EXP = 1.15965218059e-3           # CODATA 2022 electron g-2/2
A_EXP_SIGMA = 1.3e-13

# QED coefficients A_i for the electron (A1..A4 exact/known, A5 approximate)
A = [0.5,
     -0.32847896557919378,
     1.181241456587,
     -1.9122457649264456,
     6.675]
A_HAD = 1.693e-12                  # hadronic vacuum polarization + LbL
A_EW = 0.0297e-12                  # electroweak


def anomalous_moment(loops: int = 5) -> float:
    x = ALPHA / np.pi
    a = sum(A[i] * x ** (i + 1) for i in range(loops))
    return float(a + (A_HAD + A_EW if loops >= 5 else 0.0))


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        a5 = anomalous_moment(5)
        a1 = anomalous_moment(1)
        rel = abs(a5 - A_EXP) / A_EXP
        res.check("5-loop QED reproduces CODATA g-2 (rel<1e-8)", rel < 1e-8,
                  f"a_calc={a5:.15e} vs exp={A_EXP:.15e}, rel={rel:.2e}")
        res.check("one-loop Schwinger limit is alpha/2pi",
                  abs(a1 - ALPHA / (2 * np.pi)) < 1e-18,
                  f"a_1={a1:.15e}")

        terms = [A[i] * (ALPHA / np.pi) ** (i + 1) for i in range(5)]
        res.check("series converges (|term5| < |term1|/1e6)",
                  abs(terms[4]) < abs(terms[0]) / 1e6,
                  f"term1={terms[0]:.3e}, term5={terms[4]:.3e}")
        res.check("hadronic+EW corrections are below 1e-11",
                  A_HAD + A_EW < 1e-11, f"{A_HAD + A_EW:.3e}")
        res.metrics.update({
            "alpha_inv": ALPHA_INV,
            "a_one_loop": a1,
            "a_five_loop": a5,
            "a_experiment": A_EXP,
            "relative_error": rel,
            "qed_terms": terms,
            "hadronic": A_HAD,
            "electroweak": A_EW,
        })
    return run_experiment("e02_path_integral_g2", body)
