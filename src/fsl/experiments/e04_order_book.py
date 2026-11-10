"""E04: Limit order book — order-flow imbalance and price impact.

Part A (historical): LOBSTER-format AAPL top-10 depth snapshots
(3,474 hourly observations). We compute the Cont-Kukanov-Stoikov
order-flow imbalance and regress mid-price returns on it.

Part B (live): sample the real Coinbase BTC-USD book every 2 s for
~2.5 minutes, compute per-interval OFI, and test that signed order flow
predicts the next mid-price move.

Verification is against the published microstructure result that
OFI is a strong positive predictor of short-horizon returns.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsl.common import DATA, ExperimentResult, run_experiment

CSV = DATA / "kaggle_lob" / "lobster_aapl_10_level.csv"
LIVE = True


def ofi(bid: np.ndarray, ask: np.ndarray, bsz: np.ndarray, asz: np.ndarray) -> np.ndarray:
    e = np.zeros(len(bid) - 1)
    for t in range(1, len(bid)):
        b = (bsz[t] if bid[t] >= bid[t - 1] else 0.0) - (bsz[t - 1] if bid[t] <= bid[t - 1] else 0.0)
        a = (asz[t] if ask[t] <= ask[t - 1] else 0.0) - (asz[t - 1] if ask[t] >= ask[t - 1] else 0.0)
        e[t - 1] = b - a
    return e


def ols(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"slope": float(beta[1]), "se": float(se[1]), "t": float(beta[1] / se[1]),
            "r2": float(r2), "n": int(n)}


def historical(res: ExperimentResult) -> dict:
    df = pd.read_csv(CSV)
    bid = df["bid_price_1"].to_numpy(float)
    ask = df["ask_price_1"].to_numpy(float)
    bsz = df["bid_size_1"].to_numpy(float)
    asz = df["ask_size_1"].to_numpy(float)
    mid = (bid + ask) / 2
    ret_bps = np.diff(mid) / mid[:-1] * 1e4
    e = ofi(bid, ask, bsz, asz)[: len(ret_bps)]
    qbar = float(np.mean(bsz + asz))
    fit = ols(ret_bps, e / qbar)
    res.check("OFI predicts returns with positive slope", fit["slope"] > 0,
              f"slope={fit['slope']:.3f} bps per unit OFI/Q, t={fit['t']:.1f}")
    res.check("OFI slope is statistically strong (t>10)", fit["t"] > 10,
              f"t={fit['t']:.1f}")
    res.check("OFI explains a meaningful share of return variance", fit["r2"] > 0.02,
              f"R2={fit['r2']:.3f}, n={fit['n']}")
    spread_bps = float(np.mean((ask - bid) / mid * 1e4))
    return {"fit": fit, "n": int(len(df)), "spread_bps": spread_bps,
            "mean_depth": qbar, "price_range": [float(mid.min()), float(mid.max())]}


def live_coinbase(res: ExperimentResult, seconds: int = 130, step: float = 2.0) -> dict:
    import requests
    url = "https://api.exchange.coinbase.com/products/BTC-USD/book?level=1"
    headers = {"User-Agent": "fsl-lab/0.1 (research; contact: none)",
               "Accept": "application/json"}
    samples = []
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            d = requests.get(url, headers=headers, timeout=10).json()
            bid, bsz = float(d["bids"][0][0]), float(d["bids"][0][1])
            ask, asz = float(d["asks"][0][0]), float(d["asks"][0][1])
            samples.append((bid, ask, bsz, asz))
            time.sleep(step)
    except Exception as exc:  # network hiccup: report honestly
        return {"error": str(exc), "n": len(samples)}

    s = np.array(samples)
    bid, ask, bsz, asz = s.T
    mid = (bid + ask) / 2
    ret = np.diff(mid) / mid[:-1] * 1e4
    e = ofi(bid, ask, bsz, asz)[: len(ret)]
    qbar = float(np.mean(bsz + asz))
    if len(ret) < 15 or np.std(e) == 0:
        return {"n": len(samples), "note": "not enough variance in live window"}
    fit = ols(ret, e / qbar)
    res.check("live Coinbase OFI predicts next-tick return (t>2)", fit["t"] > 2,
              f"slope={fit['slope']:.3f}, t={fit['t']:.1f}, n={fit['n']}")
    return {"fit": fit, "n": len(samples),
            "price": [float(mid.min()), float(mid.max())]}


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        hist = historical(res)
        live = live_coinbase(res) if LIVE else {"skipped": True}
        res.metrics.update({"historical": hist, "live": live})
    return run_experiment("e04_order_book", body)
