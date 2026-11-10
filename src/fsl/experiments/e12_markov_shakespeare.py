"""E12: Markov chains and the entropy of English.

Fit order-k character Markov chains with Witten-Bell interpolation to
Tiny Shakespeare and evaluate held-out cross-entropy. Prediction
improves with context length up to the point where the 1M-character
corpus runs out of statistics — an empirical look at the vocabulary
curse the asymptotic theory ignores.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from fsl.common import DATA, ExperimentResult, run_experiment

TEXT = DATA / "tinyshakespeare.txt"
V = 128
ALPHA = 0.3


def build_counts(train: np.ndarray, order: int) -> list[dict]:
    counts: list[dict] = [defaultdict(lambda: np.zeros(V)) for _ in range(order + 1)]
    for i in range(len(train)):
        for o in range(min(order, i) + 1):
            counts[o][tuple(train[i - o:i])][train[i]] += 1
    return counts


def make_prob(counts: list[dict]):
    def p(ctx: tuple, ch: int) -> float:
        if len(ctx) == 0:
            c = counts[0][()]
            return (c[ch] + ALPHA) / (c.sum() + ALPHA * V)
        if ctx in counts[len(ctx)]:
            c = counts[len(ctx)][ctx]
            n = c.sum()
            seen = int((c > 0).sum())  # Witten-Bell: backoff mass ~ novel continuations
            return (c[ch] + seen * p(ctx[1:], ch)) / (n + seen)
        return p(ctx[1:], ch)
    return p


def fit_eval(order: int, train: np.ndarray, test: np.ndarray) -> float:
    p = make_prob(build_counts(train, order))
    total = 0.0
    for i in range(order, len(test)):
        total -= np.log2(p(tuple(test[i - order:i]), test[i]))
    return float(total / (len(test) - order))


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        raw = np.frombuffer(TEXT.read_bytes(), dtype=np.uint8)
        raw = raw[raw < 128]
        split = int(0.9 * len(raw))
        train, test = raw[:split], raw[split:]

        scores = {k: fit_eval(k, train, test) for k in range(1, 7)}
        best_k = min(scores, key=scores.get)
        res.check("context helps: best order beats the bigram by >20%",
                  scores[1] / scores[best_k] > 1.2,
                  f"{scores[best_k]:.3f} vs {scores[1]:.3f} bits/char (order {best_k})")
        res.check("best context length is 3-5 characters",
                  3 <= best_k <= 5, f"argmin at order {best_k}")
        res.check("beyond the data limit, longer context overfits", scores[6] > scores[best_k],
                  f"order6={scores[6]:.3f} vs best={scores[best_k]:.3f}")

        res.metrics.update({"bits_per_char": {str(k): v for k, v in scores.items()},
                            "best_order": best_k,
                            "corpus_bytes": int(len(raw)),
                            "vocabulary": int(len(np.unique(raw)))})
    return run_experiment("e12_markov_shakespeare", body)
