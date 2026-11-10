"""E22: Double descent — the interpolation threshold on real data.

Random Fourier features + ridge regression on UCI handwritten digits
with 20% label noise. Sweep the number of features through the
interpolation threshold and verify the test error peak predicted by
modern double-descent theory.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from fsl.common import FIGURES, ExperimentResult, run_experiment


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        rng = np.random.default_rng(0)
        flip = rng.random(len(ytr)) < 0.2
        ytr = ytr.copy()
        ytr[flip] = rng.integers(0, 10, size=flip.sum())

        widths = [20, 40, 80, 160, 320, 500, 700, 1000, 1400, 2000, 3000]
        errs = []
        for m in widths:
            W = rng.normal(scale=1 / np.sqrt(X.shape[1]), size=(X.shape[1], m))
            b = rng.uniform(0, 2 * np.pi, size=m)
            Ftr = np.cos(Xtr @ W + b)
            Fte = np.cos(Xte @ W + b)
            Y = np.eye(10)[ytr]
            A = Ftr.T @ Ftr + 1e-8 * np.eye(m)
            beta = np.linalg.solve(A, Ftr.T @ Y)
            pred = np.argmax(Fte @ beta, axis=1)
            errs.append(float(np.mean(pred != yte)))

        n = len(ytr)
        peak_idx = int(np.argmax(errs))
        peak_m = widths[peak_idx]
        tail = float(np.mean(errs[-2:]))
        res.check("test error peaks near the interpolation threshold",
                  0.4 * n < peak_m < 2.5 * n,
                  f"peak at m={peak_m}, n_train={n}")
        res.check("double descent is pronounced (peak > 1.5x tail)",
                  errs[peak_idx] > 1.5 * tail,
                  f"peak={errs[peak_idx]:.3f}, tail={tail:.3f}")
        res.check("error falls again in the overparameterized regime",
                  errs[-1] < errs[peak_idx] * 0.75,
                  f"err(m=3000)={errs[-1]:.3f} vs peak={errs[peak_idx]:.3f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.semilogx(widths, errs, "o-")
        ax.axvline(n, ls="--", c="gray", label=f"n_train={n}")
        ax.set_xlabel("random Fourier features m")
        ax.set_ylabel("test error")
        ax.set_title("double descent on noisy digits")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / "e22_double_descent.png", dpi=140)
        plt.close(fig)

        res.metrics.update({"widths": widths, "test_errors": errs,
                            "n_train": n, "peak_m": peak_m,
                            "peak_error": errs[peak_idx], "tail_error": tail})
    return run_experiment("e22_double_descent", body)
