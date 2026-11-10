"""Fetch public model configs and exact released weight sizes.

Run from the repository root:
    python scripts/fetch_model_zoo.py

The script writes data/model_zoo.json. It requires network access and is not
run by the test suite.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "model_zoo.json"

MODELS: list[dict[str, Any]] = [
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "source": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "license": "llama3.1",
        "params_b": 8.03,
        "notes": "official repo gated; public mirror used for config and weight metadata",
    },
    {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "source": "NousResearch/Meta-Llama-3.1-70B-Instruct",
        "license": "llama3.1",
        "params_b": 70.55,
        "notes": "official repo gated; public mirror used for config and weight metadata",
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "source": "mistralai/Mistral-7B-Instruct-v0.3",
        "license": "apache-2.0",
        "params_b": 7.25,
        "notes": "",
    },
    {
        "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "source": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "license": "apache-2.0",
        "params_b": 46.7,
        "notes": "sparse mixture of experts",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "source": "Qwen/Qwen2.5-7B-Instruct",
        "license": "apache-2.0",
        "params_b": 7.62,
        "notes": "",
    },
    {
        "id": "Qwen/Qwen2.5-14B-Instruct",
        "source": "Qwen/Qwen2.5-14B-Instruct",
        "license": "apache-2.0",
        "params_b": 14.7,
        "notes": "",
    },
    {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "source": "Qwen/Qwen2.5-32B-Instruct",
        "license": "apache-2.0",
        "params_b": 32.5,
        "notes": "",
    },
    {
        "id": "Qwen/Qwen3-8B",
        "source": "Qwen/Qwen3-8B",
        "license": "apache-2.0",
        "params_b": 8.19,
        "notes": "",
    },
    {
        "id": "Qwen/Qwen3-30B-A3B",
        "source": "Qwen/Qwen3-30B-A3B",
        "license": "apache-2.0",
        "params_b": 30.5,
        "notes": "sparse mixture of experts",
    },
    {
        "id": "google/gemma-2-9b-it",
        "source": "unsloth/gemma-2-9b-it",
        "license": "gemma",
        "params_b": 9.24,
        "notes": "official repo gated; public mirror used; alternating sliding/full attention",
    },
    {
        "id": "google/gemma-2-27b-it",
        "source": "unsloth/gemma-2-27b-it",
        "license": "gemma",
        "params_b": 27.23,
        "notes": "official repo gated; public mirror used; alternating sliding/full attention",
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "source": "microsoft/Phi-3.5-mini-instruct",
        "license": "mit",
        "params_b": 3.82,
        "notes": "",
    },
    {
        "id": "microsoft/Phi-4-mini-instruct",
        "source": "microsoft/Phi-4-mini-instruct",
        "license": "mit",
        "params_b": 3.8,
        "notes": "",
    },
    {
        "id": "01-ai/Yi-1.5-34B-Chat",
        "source": "01-ai/Yi-1.5-34B-Chat",
        "license": "apache-2.0",
        "params_b": 34.4,
        "notes": "",
    },
    {
        "id": "tiiuae/Falcon3-10B-Instruct",
        "source": "tiiuae/Falcon3-10B-Instruct",
        "license": "apache-2.0",
        "params_b": 10.3,
        "notes": "",
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "source": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "license": "apache-2.0",
        "params_b": 1.1,
        "notes": "",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "source": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "license": "apache-2.0",
        "params_b": 1.71,
        "notes": "",
    },
    {
        "id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "source": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "license": "mit",
        "params_b": 15.7,
        "notes": "multi-head latent attention",
    },
    {
        "id": "deepseek-ai/DeepSeek-V3",
        "source": "deepseek-ai/DeepSeek-V3",
        "license": "mit",
        "params_b": 671.0,
        "notes": "multi-head latent attention; released weights are FP8",
    },
    {
        "id": "ibm-granite/granite-3.1-8b-instruct",
        "source": "ibm-granite/granite-3.1-8b-instruct",
        "license": "apache-2.0",
        "params_b": 8.17,
        "notes": "",
    },
]


def fetch_bytes(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "kvbudget-zoo/1.0 (+public research)"},
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}") from last


def fetch_json(url: str) -> tuple[bytes, dict[str, Any]]:
    raw = fetch_bytes(url)
    return raw, json.loads(raw)


def safetensors_bytes(api: dict[str, Any]) -> tuple[int, list[str]]:
    files = [
        s for s in api.get("siblings", [])
        if str(s.get("rfilename", "")).endswith(".safetensors")
    ]
    sharded = [
        s for s in files
        if re.search(r"model-\d+-of-\d+\.safetensors$", str(s["rfilename"]))
    ]
    selected = sharded or files
    names = [str(s["rfilename"]) for s in selected]
    total = sum(int(s.get("size") or 0) for s in selected)
    return total, names


def normalize(
    raw: dict[str, Any],
    entry: dict[str, Any],
    weight_bytes: int,
    weight_files: list[str],
    config_hash: str,
    api_license: str,
) -> dict[str, Any]:
    cfg = raw.get("text_config", raw)
    layers = int(cfg.get("num_hidden_layers") or cfg["n_layer"])
    hidden = int(cfg.get("hidden_size") or cfg["n_embd"])
    heads = int(cfg.get("num_attention_heads") or cfg["n_head"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    kv_lora = cfg.get("kv_lora_rank")
    rope = cfg.get("qk_rope_head_dim")
    head_dim = int(
        cfg.get("head_dim")
        or cfg.get("qk_nope_head_dim")
        or hidden // heads
    )
    max_context = int(
        cfg.get("max_position_embeddings")
        or cfg.get("n_positions")
        or cfg.get("seq_length")
        or 2048
    )

    sliding = cfg.get("sliding_window")
    if sliding is not None:
        sliding = int(sliding)
        if sliding >= max_context:
            sliding = None

    if kv_lora and rope:
        attention_type = "mla"
    elif kv_heads == 1:
        attention_type = "mqa"
    elif kv_heads < heads:
        attention_type = "gqa"
    else:
        attention_type = "mha"

    layer_types: list[str] | None = None
    if cfg.get("model_type") == "gemma2" and sliding:
        layer_types = [
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(layers)
        ]

    params_b = entry.get("params_b")
    ratio = weight_bytes / (params_b * 1e9) if params_b and weight_bytes else 0.0
    if ratio and ratio < 1.3:
        weight_dtype = "fp8"
    elif ratio and ratio < 2.4:
        weight_dtype = "bfloat16"
    elif ratio and ratio < 4.4:
        weight_dtype = "float32"
    else:
        weight_dtype = "unknown"

    note = entry.get("notes", "")
    if len(weight_files) > 1:
        shard_note = f"{len(weight_files)} safetensors shards"
        note = f"{note}; {shard_note}" if note else shard_note

    return {
        "id": entry["id"],
        "source_repo": entry["source"],
        "family": str(cfg.get("model_type", "")),
        "license": api_license or entry.get("license", ""),
        "params_b": params_b,
        "layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_context": max_context,
        "attention_type": attention_type,
        "sliding_window": sliding,
        "layer_types": layer_types,
        "kv_lora_rank": kv_lora,
        "qk_rope_head_dim": rope,
        "weight_bytes": weight_bytes,
        "weight_dtype": weight_dtype,
        "weight_files": weight_files,
        "config_url": f"https://huggingface.co/{entry['source']}/raw/main/config.json",
        "weight_url": f"https://huggingface.co/{entry['source']}",
        "config_sha256": config_hash,
        "notes": note,
    }


def main() -> int:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, entry in enumerate(MODELS, 1):
        repo = entry["source"]
        print(f"[{index}/{len(MODELS)}] {entry['id']} <- {repo}", file=sys.stderr)
        try:
            raw, cfg = fetch_json(
                f"https://huggingface.co/{repo}/raw/main/config.json"
            )
            _, api = fetch_json(
                f"https://huggingface.co/api/models/{repo}?blobs=true"
            )
            weight_bytes, weight_files = safetensors_bytes(api)
            if weight_bytes <= 0:
                raise RuntimeError("no safetensors sizes returned")
            card = api.get("cardData") or {}
            api_license = card.get("license") or api.get("license") or ""
            if isinstance(api_license, list):
                api_license = ", ".join(str(x) for x in api_license)
            records.append(
                normalize(
                    cfg,
                    entry,
                    weight_bytes,
                    weight_files,
                    hashlib.sha256(raw).hexdigest()[:16],
                    str(api_license),
                )
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"{entry['id']}: {exc}")
            print(f"  ERROR: {exc}", file=sys.stderr)
        time.sleep(0.25)

    if errors:
        print("\nFailures:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)

    records.sort(key=lambda r: r["id"].lower())
    payload = {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%d"),
        "method": (
            "configs from Hugging Face raw config.json; weight_bytes is the sum of "
            "released .safetensors shard sizes from the model API; params_b is the "
            "published parameter count and is informational only"
        ),
        "models": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} with {len(records)} models", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
