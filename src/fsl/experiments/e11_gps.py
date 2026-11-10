"""E11: GPS — trilateration from a real satellite constellation.

We parse live TLE elements for the operational GPS constellation from
CelesTrak, propagate each satellite with Kepler + J2 secular rates, and
build a realistic pseudorange navigation problem for a receiver in
Damascus. A Gauss-Newton solver must recover the receiver position and
clock bias.

Verification:
  * orbital radii and periods match the published GPS constellation,
  * relativistic frequency offset is ~38.5 microseconds/day,
  * the position solution is recovered to metres,
  * the clock bias is recovered to nanoseconds.
"""
from __future__ import annotations

import numpy as np
import requests

from fsl.common import DATA, ExperimentResult, run_experiment

MU = 398600.4418  # km^3/s^2
RE = 6371.0
C = 299792.458  # km/s
J2 = 1.08263e-3
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
CACHE = DATA / "gps_tle.txt"


def fetch_tle() -> str:
    if CACHE.exists():
        return CACHE.read_text()
    txt = requests.get(TLE_URL, timeout=60, headers={"User-Agent": "fsl-lab"}).text
    CACHE.write_text(txt)
    return txt


def parse_tle(txt: str) -> list[dict]:
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    sats = []
    for i in range(0, len(lines) - 2, 3):
        if not lines[i + 1].startswith("1 ") or not lines[i + 2].startswith("2 "):
            continue
        l2 = lines[i + 2]
        sats.append({
            "name": lines[i],
            "inc": np.deg2rad(float(l2[8:16])),
            "raan": np.deg2rad(float(l2[17:25])),
            "ecc": float("0." + l2[26:33].strip()),
            "argp": np.deg2rad(float(l2[34:42])),
            "M": np.deg2rad(float(l2[43:51])),
            "n_rev_day": float(l2[52:63]),
        })
    return sats


def propagate(sat: dict, dt: float) -> np.ndarray:
    n = sat["n_rev_day"] * 2 * np.pi / 86400.0
    a = (MU / n**2) ** (1 / 3)
    p = a * (1 - sat["ecc"] ** 2)
    raan_dot = -1.5 * J2 * n * (RE / p) ** 2 * np.cos(sat["inc"])
    argp_dot = 0.75 * J2 * n * (RE / p) ** 2 * (5 * np.cos(sat["inc"]) ** 2 - 1)
    M = sat["M"] + n * dt
    O = sat["raan"] + raan_dot * dt
    w = sat["argp"] + argp_dot * dt
    E = M.copy()
    for _ in range(20):
        E = E - (E - sat["ecc"] * np.sin(E) - M) / (1 - sat["ecc"] * np.cos(E))
    xp = a * (np.cos(E) - sat["ecc"])
    yp = a * np.sqrt(1 - sat["ecc"] ** 2) * np.sin(E)
    cO, sO = np.cos(O), np.sin(O)
    ci, si = np.cos(sat["inc"]), np.sin(sat["inc"])
    cw, sw = np.cos(w), np.sin(w)
    R = np.array([[cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci, sO*si],
                  [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
                  [sw*si, cw*si, ci]])
    return R @ np.array([xp, yp, 0.0]), a


def llh_to_ecef(lat: float, lon: float, h: float = 0.0) -> np.ndarray:
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    N = RE / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + h) * np.sin(lat)])


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        sats = parse_tle(fetch_tle())
        res.check("parsed a full GPS constellation", len(sats) > 24, f"{len(sats)} satellites")

        dt = 3600.0
        pos, radii = [], []
        for s in sats:
            r, a = propagate(s, dt)
            pos.append(r); radii.append(a)
        pos = np.array(pos); radii = np.array(radii)
        period_min = 2 * np.pi * np.sqrt(radii.mean() ** 3 / MU) / 60
        res.check("semi-major axis is GPS-like (26,560 km +- 400)", abs(radii.mean() - 26560) < 400,
                  f"a_mean={radii.mean():.0f} km")
        res.check("orbital period is ~11h58m", abs(period_min - 717.9) < 3,
                  f"T={period_min:.1f} min")

        # relativistic frequency offset: GM/c^2 (1/RE - 1/r) - v^2/(2c^2)
        v = np.sqrt(MU / radii)
        grav = (MU / C**2) * (1 / RE - 1 / radii)
        kin = v**2 / (2 * C**2)
        offset_us_day = float(np.mean(grav - kin) * 86400 * 1e6)
        res.check("relativistic clock offset is ~38.5 us/day", 35 < offset_us_day < 42,
                  f"offset={offset_us_day:.2f} us/day")

        # navigation solution for a receiver in Damascus
        lat, lon = np.deg2rad(33.5138), np.deg2rad(36.2765)
        truth = llh_to_ecef(lat, lon)
        clk_bias = 8.1e-5 * C  # km (time-equivalent range error)
        enu_z = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        elev = (pos - truth) @ enu_z / np.linalg.norm(pos - truth, axis=1)
        idx = np.where(elev > 0.17)[0]  # all satellites above ~10 degrees
        n_used = len(idx)
        rng = np.random.default_rng(0)
        ranges = np.linalg.norm(pos[idx] - truth, axis=1) + clk_bias + rng.normal(scale=1e-3, size=n_used)

        x = np.zeros(4)
        for _ in range(50):
            d = np.linalg.norm(pos[idx] - x[:3], axis=1)
            H = np.column_stack([-(pos[idx] - x[:3]) / d[:, None], np.ones(n_used)])
            r = ranges - (d + x[3])
            dx, *_ = np.linalg.lstsq(H, r, rcond=None)
            x += dx
            if np.linalg.norm(dx[:3]) < 1e-9:
                break
        pos_err = float(np.linalg.norm(x[:3] - truth) * 1000)
        clk_err = float(abs(x[3] - clk_bias) / C * 1e9)
        gdop = float(np.sqrt(np.trace(np.linalg.inv(H.T @ H))))
        res.check("receiver position recovered to metres", pos_err < 5.0,
                  f"3D error={pos_err:.2f} m (n={n_used}, GDOP={gdop:.1f})")
        res.check("receiver clock bias recovered to nanoseconds", clk_err < 5.0,
                  f"clock error={clk_err:.2f} ns (truth {clk_bias / C * 1e9:.0f} ns)")

        res.metrics.update({"n_satellites": len(sats), "mean_radius_km": float(radii.mean()),
                            "period_min": float(period_min),
                            "relativistic_us_per_day": offset_us_day,
                            "position_error_m": pos_err, "clock_error_ns": clk_err,
                            "clock_bias_ns": float(clk_bias / C * 1e9)})
    return run_experiment("e11_gps", body)
