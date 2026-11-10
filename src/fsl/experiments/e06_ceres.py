"""E06: Gauss, Ceres, and least squares (1801).

We rebuild the historical inverse problem with real celestial mechanics:
JPL Horizons gives the true heliocentric states of Ceres and the Earth
for Piazzi's 41-day 1801 observing window. We form geocentric directions,
add 2-arcsecond Piazzi-level noise, and fit a 6-parameter Keplerian orbit
from scratch (multi-start Levenberg-Marquardt).

Verification:
  * observation residual stays at the noise level (< 3 arcsec),
  * recovered a, e, |i| match JPL's published elements,
  * the orbit predicts Ceres' position 11 months later (the December
    1801 rediscovery) to better than 5 arcminutes.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

from fsl.common import DATA, ExperimentResult, run_experiment

HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
MU = 0.0002959122082855911  # AU^3 / day^2
OBL = np.deg2rad(23.4392911111)
CACHE = DATA / "ceres_horizons.json"
NOISE_ARCSEC = 2.0

JPL = {"a": 2.769289292, "e": 0.076874650, "i_deg": 10.591277671}


def horizons_vectors(cmd: str, start: str, stop: str) -> np.ndarray:
    import requests
    params = {
        "format": "json", "COMMAND": f"'{cmd}'", "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
        "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": "'1 d'",
        "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'J2000'", "VEC_TABLE": "2",
        "OUT_UNITS": "'AU-D'",
    }
    txt = requests.get(HORIZONS, params=params, timeout=90).json()["result"]
    body = txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    out, i = [], 0
    while i < len(body):
        line = body[i].strip()
        if "= A.D." in line:
            jd = float(line.split("=")[0])
            parts = body[i + 1].replace("=", " ").split()
            out.append((jd, float(parts[1]), float(parts[3]), float(parts[5])))
            i += 2
        else:
            i += 1
    return np.array(out)


def fetch() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    data = {
        "obs_ceres": horizons_vectors("1;", "1801-01-01", "1801-02-11").tolist(),
        "obs_earth": horizons_vectors("399", "1801-01-01", "1801-02-11").tolist(),
        "pred_ceres": horizons_vectors("1;", "1801-12-30", "1802-01-02").tolist(),
        "pred_earth": horizons_vectors("399", "1801-12-30", "1802-01-02").tolist(),
    }
    CACHE.write_text(json.dumps(data))
    return data


def kepler_positions(el: np.ndarray, t: np.ndarray, t0: float) -> np.ndarray:
    a, e, i, O, w, M0 = el
    n = np.sqrt(MU / a**3)
    M = M0 + n * (t - t0)
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1 - e**2) * np.sin(E)
    co, so = np.cos(w), np.sin(w)
    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(O), np.sin(O)
    R = np.array([[cO*co - sO*so*ci, -cO*so - sO*co*ci, sO*si],
                  [sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
                  [so*si, co*si, ci]])
    return (R @ np.vstack([xp, yp, np.zeros_like(xp)])).T


def radec(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eq = np.column_stack([d[:, 0], d[:, 1]*np.cos(OBL) - d[:, 2]*np.sin(OBL),
                          d[:, 1]*np.sin(OBL) + d[:, 2]*np.cos(OBL)])
    rho = np.linalg.norm(d, axis=1)
    return np.arctan2(eq[:, 1], eq[:, 0]), np.arcsin(eq[:, 2] / rho)


def fit_orbit(t, Re, ra_o, dec_o) -> least_squares:
    t0 = t[0]

    def resid(el):
        ra_m, dec_m = radec(kepler_positions(el, t, t0) - Re)
        dra = (ra_m - ra_o + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([dra * np.cos(dec_m), dec_m - dec_o]) * 206264.806

    best = None
    for a0 in (2.4, 2.77, 3.1):
        for i0 in (0, 10, 20):
            for m0 in np.deg2rad(np.arange(0, 360, 30)):
                guess = [a0, 0.1, np.deg2rad(i0), 0.0, 0.0, m0]
                r = least_squares(resid, guess, method="lm", max_nfev=4000)
                if best is None or r.cost < best.cost:
                    best = r
    return best


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        d = fetch()
        C = np.array(d["obs_ceres"]); E = np.array(d["obs_earth"])
        t = C[:, 0]; Re = E[:, 1:]
        ra, dec = radec(C[:, 1:] - Re)

        rng = np.random.default_rng(1)
        ra_o = ra + rng.normal(scale=np.deg2rad(NOISE_ARCSEC / 3600) / np.cos(dec))
        dec_o = dec + rng.normal(scale=np.deg2rad(NOISE_ARCSEC / 3600))

        best = fit_orbit(t, Re, ra_o, dec_o)
        el = best.x
        rms = float(np.sqrt(np.mean(best.fun**2)))
        res.check("observations fitted at the 2-arcsec noise level", rms < 3.0,
                  f"RMS residual = {rms:.2f} arcsec over {len(t)} nights")

        a_ok = abs(el[0] - JPL["a"]) / JPL["a"] < 0.02
        res.check("semi-major axis recovered within 2%", a_ok,
                  f"a = {el[0]:.4f} AU vs JPL {JPL['a']:.4f}")
        res.check("eccentricity recovered within 0.02", abs(el[1] - JPL["e"]) < 0.02,
                  f"e = {el[1]:.4f} vs JPL {JPL['e']:.4f}")
        res.check("inclination recovered within 1.5 deg",
                  abs(abs(np.rad2deg(el[2])) - JPL["i_deg"]) < 1.5,
                  f"|i| = {abs(np.rad2deg(el[2])):.2f} deg vs JPL {JPL['i_deg']:.2f}")

        Cf = np.array(d["pred_ceres"]); Ef = np.array(d["pred_earth"])
        ra_p, dec_p = radec(kepler_positions(el, Cf[:, 0], t[0]) - Ef[:, 1:])
        ra_t, dec_t = radec(Cf[:, 1:] - Ef[:, 1:])
        cos_sep = (np.sin(dec_p) * np.sin(dec_t)
                   + np.cos(dec_p) * np.cos(dec_t) * np.cos(ra_p - ra_t))
        sep_arcmin = float(np.max(np.rad2deg(np.arccos(np.clip(cos_sep, -1, 1))) * 60))
        res.check("11-month position prediction better than 5 arcmin",
                  sep_arcmin < 5.0,
                  f"separation = {sep_arcmin:.2f} arcmin on 1801-12-31")

        res.metrics.update({
            "n_observations": int(len(t)),
            "noise_arcsec": NOISE_ARCSEC,
            "fit_rms_arcsec": rms,
            "elements": {"a": float(el[0]), "e": float(el[1]),
                         "i_deg": float(np.rad2deg(el[2])),
                         "Omega_deg": float(np.rad2deg(el[3]) % 360),
                         "omega_deg": float(np.rad2deg(el[4]) % 360),
                         "M0_deg": float(np.rad2deg(el[5]) % 360)},
            "jpl_elements": JPL,
            "rediscovery_error_arcmin": sep_arcmin,
        })
    return run_experiment("e06_ceres", body)
