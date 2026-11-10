"""Quality-risk and speed priors for KV-cache plans.

The numbers are deliberately ranges collapsed to conservative point estimates.
Every table entry carries a source so the compiler can explain itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVFormat:
    name: str
    bytes_per_element: float
    risk: float
    speed: float
    source: str
    note: str = ""


@dataclass(frozen=True)
class EvictionPolicy:
    name: str
    keep_ratio: float
    risk: float
    speed: float
    source: str
    note: str = ""
    compaction: bool = True


FORMATS: dict[str, KVFormat] = {
    "bf16": KVFormat(
        "bf16", 2.0, 0.00, 1.00,
        "reference; vLLM quantized-KV docs",
        "uncompressed baseline",
    ),
    "fp8": KVFormat(
        "fp8", 1.0, 0.05, 1.05,
        "https://vllm.ai/blog/2026-04-22-fp8-kvcache",
        "e4m3; Hopper needs two-level FP32 accumulation; head_dim=256 can regress prefill",
    ),
    "int8": KVFormat(
        "int8", 1.0, 0.02, 0.98,
        "KIVI https://arxiv.org/abs/2402.02750; LMDeploy KV quant docs",
        "symmetric per-token; per-channel for keys",
    ),
    "int4": KVFormat(
        "int4", 0.5, 0.05, 0.95,
        "KIVI https://arxiv.org/abs/2402.02750",
        "group-wise quantization strongly recommended",
    ),
    "int2": KVFormat(
        "int2", 0.25, 0.42, 0.90,
        "KIVI https://arxiv.org/abs/2402.02750; lab measurement (OPT-125m)",
        "measured 56.7% top-1 retrieval and 22% output error; experimental only",
    ),
    "nvfp4": KVFormat(
        "nvfp4", 0.5, 0.08, 0.97,
        "NVIDIA Efficient AI 2026; vLLM quantized-KV docs",
        "experimental; calibration required",
    ),
}

EVICTIONS: dict[str, EvictionPolicy] = {
    "none": EvictionPolicy(
        "none", 1.0, 0.00, 1.00,
        "reference",
        "keep every token",
    ),
    "triattention": EvictionPolicy(
        "triattention", 0.25, 0.20, 1.15,
        "https://arxiv.org/html/2604.04921",
        "matches full attention on AIME25 32K; reported 10.7x KV reduction",
    ),
    "random": EvictionPolicy(
        "random", 0.25, 0.15, 1.30,
        "https://arxiv.org/abs/2609.03430",
        "random eviction is competitive with selection and avoids scoring cost",
    ),
    "snapkv": EvictionPolicy(
        "snapkv", 0.25, 0.50, 1.10,
        "NVIDIA Efficient AI 2026; SnapKV",
        "AIME25 20.0 vs 40.8 full; strongest throughput in the 2026 comparison",
    ),
    "rkv": EvictionPolicy(
        "rkv", 0.25, 0.57, 1.05,
        "NVIDIA Efficient AI 2026; R-KV",
        "AIME25 17.5 vs 40.8 full; paged-attention integration is nontrivial",
    ),
    "streamingllm": EvictionPolicy(
        "streamingllm", 0.25, 0.40, 1.20,
        "StreamingLLM https://arxiv.org/abs/2309.17453",
        "attention sinks; weak on retrieval and recall",
    ),
    "h2o": EvictionPolicy(
        "h2o", 0.25, 0.50, 1.10,
        "H2O https://arxiv.org/abs/2306.14048",
        "heavy hitters; workload-sensitive",
    ),
}

WORKLOAD_PENALTY: dict[str, float] = {
    "chat": 0.00,
    "retrieval": 0.05,
    "summarization": 0.10,
    "reasoning": 0.05,
    "code": 0.10,
}

BLOCK_SPEED: dict[int, float] = {16: 1.00, 32: 1.03, 64: 1.05, 128: 1.06}


def format_speed(name: str, head_dim: int = 128) -> float:
    fmt = FORMATS[name]
    if name == "fp8" and head_dim >= 256:
        return 0.90
    return fmt.speed


def block_speed(block_size: int) -> float:
    return BLOCK_SPEED.get(block_size, 1.0)


def plan_risk(
    format_name: str,
    eviction_name: str = "none",
    workload: str = "chat",
    head_dim: int = 128,
) -> float:
    if format_name not in FORMATS:
        raise KeyError(f"unknown KV format {format_name!r}")
    if eviction_name not in EVICTIONS:
        raise KeyError(f"unknown eviction policy {eviction_name!r}")
    if workload not in WORKLOAD_PENALTY:
        raise KeyError(f"unknown workload {workload!r}")

    fmt = FORMATS[format_name]
    ev = EVICTIONS[eviction_name]
    penalty = WORKLOAD_PENALTY[workload]

    fmt_risk = fmt.risk
    if format_name in {"int4", "int2", "nvfp4"}:
        fmt_risk += 0.5 * penalty
    if format_name == "fp8" and head_dim >= 256:
        fmt_risk += 0.02

    ev_risk = ev.risk + (penalty if eviction_name != "none" else 0.0)
    combined = 1.0 - (1.0 - min(0.95, fmt_risk)) * (1.0 - min(0.95, ev_risk))
    return round(min(0.95, combined), 4)


def evidence_table() -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for fmt in FORMATS.values():
        rows.append({
            "kind": "kv_format",
            "name": fmt.name,
            "bytes_per_element": fmt.bytes_per_element,
            "risk": fmt.risk,
            "speed": fmt.speed,
            "source": fmt.source,
            "note": fmt.note,
        })
    for ev in EVICTIONS.values():
        rows.append({
            "kind": "eviction",
            "name": ev.name,
            "keep_ratio": ev.keep_ratio,
            "risk": ev.risk,
            "speed": ev.speed,
            "source": ev.source,
            "note": ev.note,
        })
    return rows
