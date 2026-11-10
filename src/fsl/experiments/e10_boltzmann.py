"""E10: Boltzmann entropy and statistical mechanics from scratch.

Part A: enumerate all 2^16 states of a 4x4 periodic Ising model, compute
the exact density of states Omega(E), and check that Metropolis sampling
reproduces P(E) proportional to Omega(E) exp(-E/T). This is the literal
content of S = k_B ln Omega.

Part B: Langevin molecular dynamics of a 2D gas; verify equipartition and
the Maxwell-Boltzmann speed distribution.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from fsl.common import ExperimentResult, run_experiment


def ising_dos(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    states = ((np.arange(2 ** (n * n))[:, None] >> np.arange(n * n)) & 1) * 2 - 1
    grid = states.reshape(-1, n, n)
    e = -(grid * np.roll(grid, 1, axis=1)).sum(axis=(1, 2))
    e = e - (grid * np.roll(grid, 1, axis=2)).sum(axis=(1, 2))
    energies, counts = np.unique(e, return_counts=True)
    return energies, counts


def metropolis(energies: np.ndarray, temp: float, steps: int = 300000,
               seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 4
    grid = np.ones((n, n), dtype=int)
    e = -2 * n * n

    def local(g, i, j):
        return -g[i, j] * (g[(i + 1) % n, j] + g[(i - 1) % n, j]
                           + g[i, (j + 1) % n] + g[i, (j - 1) % n])

    hist = []
    for step in range(steps):
        i, j = rng.integers(0, n, 2)
        de = -2 * local(grid, i, j)
        if de <= 0 or rng.random() < np.exp(-de / temp):
            grid[i, j] *= -1
            e += de
        if step > steps // 10 and step % 20 == 0:
            hist.append(e)
    return np.array(hist)


def lj_gas(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n, box, temp, gamma, dt, steps = 600, 24.0, 1.0, 1.0, 0.01, 6000
    x = rng.uniform(0, box, size=(n, 2))
    v = rng.normal(scale=np.sqrt(temp), size=(n, 2))
    for _ in range(steps):
        v += -gamma * v * dt + np.sqrt(2 * gamma * temp * dt) * rng.normal(size=(n, 2))
        x += v * dt
        x %= box
    speed2 = np.sum(v**2, axis=1)
    ke = 0.5 * speed2
    return {"equipartition": float(np.mean(ke)), "speed2": speed2, "temp": temp}


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        energies, counts = ising_dos()
        temp = 2.5
        p_exact = counts * np.exp(-energies / temp)
        p_exact /= p_exact.sum()
        hist = metropolis(energies, temp)
        emp, _ = np.histogram(hist, bins=np.append(energies - 0.5, energies[-1] + 0.5))
        p_emp = emp / emp.sum()
        tv = 0.5 * np.abs(p_emp - p_exact).sum()
        res.check("Metropolis reproduces P(E) ~ Omega(E)exp(-E/T)", tv < 0.06,
                  f"total-variation distance={tv:.4f}")

        s_ordered = float(np.log(counts[energies == energies.min()][0]))
        mean_e = float((energies * p_exact).sum())
        typical = int(np.argmin(np.abs(energies - mean_e)))
        s_eq = float(np.log(counts[typical]))
        res.check("entropy of the typical equilibrium macrostate is larger",
                  s_eq > s_ordered + 2,
                  f"S(E={energies[typical]})={s_eq:.2f} k_B vs S_ordered={s_ordered:.2f} k_B")

        gas = lj_gas()
        res.check("equipartition <KE> = kT (2D)", abs(gas["equipartition"] - gas["temp"]) / gas["temp"] < 0.05,
                  f"<KE>={gas['equipartition']:.3f} vs kT={gas['temp']}")
        sigma2 = gas["temp"]  # per component
        speeds = np.sqrt(gas["speed2"])
        ks = stats.kstest(speeds, lambda s: 1 - np.exp(-s**2 / (2 * sigma2)))
        res.check("speeds follow Maxwell-Boltzmann (Rayleigh)", ks.pvalue > 0.01,
                  f"KS p-value={ks.pvalue:.3f}")
        res.metrics.update({"ising_tv": float(tv), "S_eq": s_eq,
                            "S_ordered": s_ordered,
                            "equipartition": gas["equipartition"],
                            "rayleigh_ks_p": float(ks.pvalue)})
    return run_experiment("e10_boltzmann", body)
