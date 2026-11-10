"""E14: Riemann zeros and the GUE hidden pattern.

Using Odlyzko's published table of the first 100,000 nontrivial zeros of
the Riemann zeta function, we unfold the zeros and test the Montgomery-
Odlyzko conjecture: the local statistics match the Gaussian Unitary
Ensemble (GUE) of random Hermitian matrices, not a Poisson process.

Checks:
  * first 10 zeros reproduce the published values,
  * unfolded mean spacing is 1,
  * KS distance to GUE is far smaller than to Poisson,
  * Montgomery pair correlation 1-(sin(pi r)/(pi r))^2 is matched.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import erf

from fsl.common import DATA, ExperimentResult, run_experiment

ZEROS = DATA / "odlyzko_zeros1.txt"

KNOWN10 = np.array([
    14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
])


def gue_cdf(s: np.ndarray) -> np.ndarray:
    # P(S<=s) = erf(2s/sqrt(pi)) - (4s/pi) exp(-4s^2/pi)
    s = np.asarray(s, dtype=float)
    return erf(2 * s / np.sqrt(np.pi)) - (4 * s / np.pi) * np.exp(-4 * s**2 / np.pi)


def poisson_cdf(s: np.ndarray) -> np.ndarray:
    return 1 - np.exp(-s)


def unfolding(z: np.ndarray) -> np.ndarray:
    # smooth Riemann-von Mangoldt counting function
    t = z / (2 * np.pi)
    return (z / (2 * np.pi)) * (np.log(t) - 1) + 7 / 8


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        z = np.loadtxt(ZEROS)
        n_used = 20000
        z = z[:n_used]
        res.check("first 10 zeros match published values",
                  float(np.max(np.abs(z[:10] - KNOWN10))) < 2e-9,
                  f"max diff={np.max(np.abs(z[:10] - KNOWN10)):.2e}")

        u = unfolding(z)
        gap = np.diff(u)
        gap = gap[(gap > 0) & (gap < 10)]
        raw_mean = float(gap.mean())
        s = gap / gap.mean()
        res.check("unfolded mean spacing is 1", abs(raw_mean - 1) < 0.05,
                  f"mean={raw_mean:.4f}, N={s.size}")

        ks_gue = stats.kstest(s, gue_cdf)
        ks_poi = stats.kstest(s, poisson_cdf)
        res.check("spacings reject Poisson", ks_poi.statistic > 0.1,
                  f"KS(poisson)={ks_poi.statistic:.4f}")
        res.check("spacings agree with the GUE Wigner surmise", ks_gue.statistic < 0.03,
                  f"KS(GUE)={ks_gue.statistic:.4f}")
        res.check("GUE fit beats Poisson by >10x",
                  ks_poi.statistic / max(ks_gue.statistic, 1e-12) > 10,
                  f"ratio={ks_poi.statistic / ks_gue.statistic:.1f}")

        # pair correlation with 2500 zeros
        m = 2500
        zz = z[:m]
        uu = unfolding(zz)
        diffs = np.abs(uu[:, None] - uu[None, :])
        iu = np.triu_indices(m, k=1)
        r = diffs[iu]
        rs = np.linspace(0.05, 3.0, 30)
        emp = np.array([np.sum((r > a - 0.1) & (r < a + 0.1)) / (m * 0.2) for a in rs])
        mont = 1 - (np.sin(np.pi * rs) / (np.pi * rs)) ** 2
        err = float(np.mean(np.abs(emp - mont)))
        res.check("Montgomery pair correlation matches GUE prediction", err < 0.08,
                  f"mean|empirical - Montgomery|={err:.3f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
        ax[0].hist(s, bins=80, density=True, alpha=0.75, label="zeta zeros")
        grid = np.linspace(0.01, 3.5, 300)
        p_gue = (32 / np.pi**2) * grid**2 * np.exp(-4 * grid**2 / np.pi)
        ax[0].plot(grid, p_gue, lw=2, label="GUE Wigner surmise")
        ax[0].plot(grid, np.exp(-grid), "--", label="Poisson")
        ax[0].set_xlabel("unfolded spacing s"); ax[0].set_ylabel("density")
        ax[0].legend(); ax[0].set_title("nearest-neighbor spacings")
        ax[1].plot(rs, mont, lw=2, label="Montgomery")
        ax[1].plot(rs, emp, "o", ms=3, label="zeta zeros")
        ax[1].set_xlabel("r"); ax[1].set_ylabel("pair correlation")
        ax[1].legend(); ax[1].set_title("pair correlation")
        fig.tight_layout(); fig.savefig(DATA.parent / "results" / "figures" / "e14_riemann_gue.png", dpi=140)
        plt.close(fig)

        res.metrics.update({
            "zeros_used": int(n_used),
            "ks_gue": float(ks_gue.statistic),
            "ks_poisson": float(ks_poi.statistic),
            "ks_ratio": float(ks_poi.statistic / ks_gue.statistic),
            "pair_correlation_error": err,
            "first_zero": float(z[0]),
        })
    return run_experiment("e14_riemann_gue", body)
