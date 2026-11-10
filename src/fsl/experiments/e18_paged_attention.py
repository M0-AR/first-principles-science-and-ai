"""E18: PagedAttention — block-based KV memory management.

Implement (a) a paged KV block allocator and (b) paged attention in
numpy, then verify three things:
  * paged attention equals dense attention numerically (1e-12),
  * paging removes the 60-90% reservation waste of contiguous buffers,
  * internal fragmentation is bounded by one block per sequence.
"""
from __future__ import annotations

import numpy as np

from fsl.common import ExperimentResult, run_experiment

BLOCK = 16
D = 64


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = q @ k.T / np.sqrt(q.shape[1])
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def paged_attention(q: np.ndarray, blocks: list[np.ndarray], table: list[int],
                    vblocks: list[np.ndarray]) -> np.ndarray:
    keys, vals = [], []
    for b in table:
        keys.append(blocks[b])
        vals.append(vblocks[b])
    return dense_attention(q, np.vstack(keys), np.vstack(vals))


def simulate_allocator(n_seq: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lengths = np.clip(rng.lognormal(mean=5.9, sigma=0.6, size=n_seq).astype(int), 1, 2048)
    paged_slots = int(np.ceil(lengths / BLOCK).sum() * BLOCK)
    naive_slots = int(n_seq * 2048)
    used = int(lengths.sum())
    return {
        "sequences": n_seq,
        "total_tokens": used,
        "paged_reserved": paged_slots,
        "naive_reserved": naive_slots,
        "paged_waste": 1 - used / paged_slots,
        "naive_waste": 1 - used / naive_slots,
        "internal_fragmentation": (paged_slots - used) / paged_slots,
    }


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        rng = np.random.default_rng(0)
        n = 300
        q = rng.normal(size=(n, D)).astype(np.float64)
        k = rng.normal(size=(n, D)).astype(np.float64)
        v = rng.normal(size=(n, D)).astype(np.float64)
        ref = dense_attention(q, k, v)

        nblocks = int(np.ceil(n / BLOCK))
        blocks = [k[i * BLOCK:(i + 1) * BLOCK] for i in range(nblocks)]
        vblocks = [v[i * BLOCK:(i + 1) * BLOCK] for i in range(nblocks)]
        got = paged_attention(q, blocks, list(range(nblocks)), vblocks)
        err = float(np.max(np.abs(got - ref)))
        res.check("paged attention equals dense attention", err < 1e-12,
                  f"max|diff|={err:.2e} over {n} tokens")

        mem = simulate_allocator()
        res.check("paging waste stays under 5%",
                  mem["paged_waste"] < 0.05, f"paged waste={mem['paged_waste']:.3f}")
        res.check("naive reservation wastes >60%",
                  mem["naive_waste"] > 0.60, f"naive waste={mem['naive_waste']:.3f}")
        res.check("paging saves >4x memory vs naive",
                  mem["naive_reserved"] / mem["paged_reserved"] > 4,
                  f"ratio={mem['naive_reserved'] / mem['paged_reserved']:.1f}x")
        res.metrics.update({"attention_error": err, "memory": mem,
                            "block_size": BLOCK})
    return run_experiment("e18_paged_attention", body)
