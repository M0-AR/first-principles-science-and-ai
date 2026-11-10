"""E05: DeepSeek KV-cache compression — arithmetic from the real config.

We load the published config.json of DeepSeek-V4.1-Flash and derive the
KV-cache footprint per token from first principles:

  entries/token = sum over KV-source layers of 1/compress_ratio
  bytes/entry   = (head_dim + rope_dim) * fp4_bytes
                + index_head_dim * fp4_bytes + index scale
  bytes/token   = entries/token * bytes/entry

The reported public figure is 890 bytes/token; we verify it exactly and
compare against dense-attention caches at the same context length.
"""
from __future__ import annotations

import json

from fsl.common import DATA, ExperimentResult, run_experiment

CONFIG = DATA / "deepseek_config.json"


def compute(cfg: dict) -> dict:
    t = cfg["text_config"]
    layers = t["num_hidden_layers"]
    ratios = t["compress_ratios"][:layers]
    sources = t["kv_source_layer_ids"]

    entries = 0.0
    for layer in sources:
        r = ratios[layer]
        entries += 1.0 / (r if r > 0 else 1.0)

    main_dims = t["head_dim"] + t["qk_rope_head_dim"]
    fp4 = 0.5  # bytes per parameter in 4-bit
    main_bytes = main_dims * fp4
    index_bytes = t["index_head_dim"] * fp4 + 4.0
    per_entry = main_bytes + index_bytes
    per_token = entries * per_entry

    heads = t["num_attention_heads"]
    dense_mha = layers * heads * main_dims * 2 * 2.0  # K and V, bf16
    dense_gqa = layers * 1 * main_dims * 2 * 2.0

    return {
        "layers": layers,
        "kv_source_layers": sources,
        "ratios_at_sources": [ratios[i] for i in sources],
        "entries_per_token": entries,
        "main_dims": main_dims,
        "main_bytes": main_bytes,
        "index_bytes": index_bytes,
        "bytes_per_entry": per_entry,
        "bytes_per_token": per_token,
        "dense_mha_bytes_per_token": dense_mha,
        "dense_gqa1_bytes_per_token": dense_gqa,
        "compression_vs_mha": dense_mha / per_token,
        "compression_vs_gqa1": dense_gqa / per_token,
        "context_1m_gb": per_token * 1_000_000 / 1e9,
    }


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        cfg = json.loads(CONFIG.read_text())
        m = compute(cfg)

        res.check("2.5 KV entries per token from source layers",
                  abs(m["entries_per_token"] - 2.5) < 1e-12,
                  f"entries/token={m['entries_per_token']}")
        res.check("356 bytes per KV entry (288 main + 68 index)",
                  abs(m["bytes_per_entry"] - 356) < 1e-9,
                  f"per-entry={m['bytes_per_entry']} B "
                  f"(main {m['main_bytes']} B + index {m['index_bytes']} B)")
        res.check("890 bytes per token reproduces the published figure",
                  abs(m["bytes_per_token"] - 890) < 1e-9,
                  f"bytes/token={m['bytes_per_token']}")
        res.check("1M-token context fits in under 1 GB", m["context_1m_gb"] < 1.0,
                  f"{m['context_1m_gb']:.3f} GB")
        res.check("compression vs dense MHA exceeds 1000x",
                  m["compression_vs_mha"] > 1000,
                  f"{m['compression_vs_mha']:.0f}x vs MHA, "
                  f"{m['compression_vs_gqa1']:.0f}x vs GQA-1")
        res.metrics.update(m)
    return run_experiment("e05_kv_cache", body)
