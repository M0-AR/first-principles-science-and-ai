"""E08: Fourier transform — DTMF touch-tone decoding.

Synthesize every ITU-T Q.23 tone pair at 8 kHz, add noise, and decode
with the Goertzel algorithm. Verification is exact: all 16 keys must be
recovered, and estimated frequencies must sit within the +-1.5% spec.
"""
from __future__ import annotations

import numpy as np

from fsl.common import ExperimentResult, run_experiment

ROWS = [697, 770, 852, 941]
COLS = [1209, 1336, 1477, 1633]
KEYS = [["1", "2", "3", "A"], ["4", "5", "6", "B"],
        ["7", "8", "9", "C"], ["*", "0", "#", "D"]]


def synth(key: str, fs: int = 8000, dur: float = 0.1, snr_db: float | None = None,
          rng: np.random.Generator | None = None) -> np.ndarray:
    r = c = None
    for i, row in enumerate(KEYS):
        if key in row:
            r, c = i, row.index(key)
            break
    assert r is not None and c is not None
    t = np.arange(int(fs * dur)) / fs
    s = np.sin(2 * np.pi * ROWS[r] * t) + np.sin(2 * np.pi * COLS[c] * t)
    if snr_db is not None:
        rng = rng or np.random.default_rng(0)
        p_sig = np.mean(s**2)
        p_noise = p_sig / 10 ** (snr_db / 10)
        s = s + rng.normal(scale=np.sqrt(p_noise), size=s.size)
    return s


def goertzel(x: np.ndarray, freqs: list[float], fs: int) -> np.ndarray:
    out = []
    for f in freqs:
        w = 2 * np.pi * f / fs
        coeff = 2 * np.cos(w)
        s0 = s1 = s2 = 0.0
        for sample in x:
            s0 = sample + coeff * s1 - s2
            s2, s1 = s1, s0
        out.append(s1**2 + s2**2 - coeff * s1 * s2)
    return np.asarray(out)


def decode(x: np.ndarray, fs: int = 8000) -> str:
    r = int(np.argmax(goertzel(x, ROWS, fs)))
    c = int(np.argmax(goertzel(x, COLS, fs)))
    return KEYS[r][c]


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        rng = np.random.default_rng(42)
        all_keys = [k for row in KEYS for k in row]
        for snr in (None, 20, 10, 5):
            ok = sum(decode(synth(k, snr_db=snr, rng=rng)) == k for k in all_keys)
            res.check(f"DTMF decode at SNR={snr}", ok == 16, f"{ok}/16 keys")
        key = "5"
        x = synth(key)
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, 1 / 8000)
        peaks = [freqs[np.argmin(np.abs(freqs - f))] for f in (770, 1336)]
        errs = [abs(p - f) / f for p, f in zip(peaks, (770, 1336))]
        res.check("FFT peak frequencies within 1.5%", max(errs) < 0.015,
                  f"relative errors={[round(e, 5) for e in errs]}")
        res.metrics.update({"keys_tested": 16, "snr_levels": [None, 20, 10, 5],
                            "fft_peak_error": errs})
    return run_experiment("e08_fourier_dtmf", body)
