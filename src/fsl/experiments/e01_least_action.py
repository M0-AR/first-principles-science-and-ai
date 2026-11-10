"""E01: Feynman's least action -> gradient descent on a path functional.

We discretize S[x] = ∫ (m/2 xdot^2 - V(x)) dt, solve the discrete
Euler-Lagrange equations exactly, and recover the same solution with
plain gradient descent from a random initial path. Verification:
  * free particle -> exact straight line,
  * harmonic oscillator -> sin(w t)/sin(w) up to O(dt^2),
  * gradient descent converges to the exact discrete minimizer.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from fsl.common import ExperimentResult, FIGURES, run_experiment


def build(n: int, dt: float, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Interior-point Hessian and RHS for the discrete action."""
    m = n - 1
    A = np.zeros((m, m))
    for i in range(m):
        A[i, i] = 2 / dt - w**2 * dt
        if i > 0:
            A[i, i - 1] = -1 / dt
        if i < m - 1:
            A[i, i + 1] = -1 / dt
    b = np.zeros(m)
    b[-1] += 1 / dt  # x[n] = 1
    return A, b


def action(x: np.ndarray, dt: float, w: float) -> float:
    kin = 0.5 * ((x[1:] - x[:-1]) / dt) ** 2
    pot = 0.5 * w**2 * x[1:-1] ** 2
    return float(dt * (kin.sum() - pot.sum()))


def solve_exact(n: int, dt: float, w: float) -> np.ndarray:
    A, b = build(n, dt, w)
    x = np.zeros(n + 1)
    x[1:-1] = solve(A, b)
    x[-1] = 1.0
    return x


def gradient_descent(n: int, dt: float, w: float, steps: int = 20000,
                     lr: float | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lr = lr or 0.45 * dt
    x = rng.normal(scale=0.1, size=n + 1)
    x[0], x[-1] = 0.0, 1.0
    for _ in range(steps):
        g = np.zeros_like(x)
        g[1:-1] = (2 * x[1:-1] - x[:-2] - x[2:]) / dt - w**2 * x[1:-1] * dt
        x[1:-1] -= lr * g[1:-1]
    return x


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        n, t = 64, np.linspace(0, 1, 65)
        dt = 1.0 / n

        for kind, w in (("free", 0.0), ("harmonic", np.pi / 2)):
            x_star = solve_exact(n, dt, w)
            x_gd = gradient_descent(n, dt, w)
            exact = t if w == 0 else np.sin(w * t) / np.sin(w)
            err_discrete = float(np.max(np.abs(x_star - exact)))
            err_gd = float(np.max(np.abs(x_gd - x_star)))
            if kind == "free":
                res.check("free-particle discrete minimum is a straight line",
                          err_discrete < 1e-12, f"max|dx|={err_discrete:.2e}")
                res.check("gradient descent reaches the straight line",
                          err_gd < 1e-3, f"max|GD-exact|={err_gd:.2e}")
                res.check("free-particle action is 1/2", abs(action(x_star, dt, w) - 0.5) < 1e-12,
                          f"S={action(x_star, dt, w):.12f}")
            else:
                res.check("oscillator discrete minimum matches sin(wt)/sin(w)",
                          err_discrete < 5e-3, f"max|dx|={err_discrete:.2e}")
                res.check("gradient descent matches the oscillator minimum",
                          err_gd < 1e-3, f"max|GD-exact|={err_gd:.2e}")
                res.check("oscillator action is stationary",
                          abs(action(x_gd, dt, w) - action(x_star, dt, w)) < 1e-6,
                          f"dS={abs(action(x_gd, dt, w) - action(x_star, dt, w)):.2e}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for k, (name, w) in enumerate((("free", 0.0), ("harmonic", np.pi / 2))):
            x_star = solve_exact(n, dt, w)
            x_gd = gradient_descent(n, dt, w)
            ax[k].plot(t, x_gd, lw=3, label="gradient descent")
            ax[k].plot(t, x_star, "--", label="exact discrete minimum")
            ax[k].set_title(name); ax[k].legend()
        fig.tight_layout(); fig.savefig(FIGURES / "e01_least_action.png", dpi=140)
        plt.close(fig)

        res.metrics.update({"grid_points": n, "dt": dt,
                            "gd_steps": 20000, "learning_rate": 0.45 * dt})
    return run_experiment("e01_least_action", body)
