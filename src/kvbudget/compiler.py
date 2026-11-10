"""Plan enumeration and Pareto selection for KV-cache budgets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import HardwareSpec, ModelSpec
from .quality import (
    EVICTIONS,
    FORMATS,
    block_speed,
    format_speed,
    plan_risk,
)


@dataclass(frozen=True)
class CompileRequest:
    model: ModelSpec
    hardware: HardwareSpec
    context: int
    concurrency: int = 1
    max_quality_risk: float = 0.10
    allow_eviction: bool = False
    block_sizes: tuple[int, ...] = (16, 32, 64)
    formats: tuple[str, ...] = ("bf16", "fp8", "int8", "int4", "int2")
    evictions: tuple[str, ...] | None = None
    tp: int = 1
    weight_dtype: str | None = None
    keep_ratio: float | None = None
    workload: str = "chat"

    def __post_init__(self) -> None:
        if self.context < 1:
            raise ValueError("context must be positive")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.tp < 1:
            raise ValueError("tp must be positive")
        if not 0.0 <= self.max_quality_risk <= 1.0:
            raise ValueError("max_quality_risk must be in [0, 1]")
        if not self.block_sizes:
            raise ValueError("block_sizes must not be empty")
        if not self.formats:
            raise ValueError("formats must not be empty")
        if any(b < 1 for b in self.block_sizes):
            raise ValueError("block sizes must be positive")
        unknown = [f for f in self.formats if f not in FORMATS]
        if unknown:
            raise ValueError(f"unknown KV formats: {unknown}")
        if self.keep_ratio is not None and not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

    def resolved_evictions(self) -> tuple[str, ...]:
        if self.evictions is not None:
            unknown = [e for e in self.evictions if e not in EVICTIONS]
            if unknown:
                raise ValueError(f"unknown eviction policies: {unknown}")
            return self.evictions
        if self.allow_eviction:
            return ("none", "triattention", "random", "snapkv", "rkv")
        return ("none",)


@dataclass(frozen=True)
class Plan:
    model_id: str
    tp: int
    kv_format: str
    eviction: str
    block_size: int
    context: int
    concurrency: int
    kv_bytes_per_token: float
    kv_bytes_per_sequence: float
    kv_pool_bytes: int
    weights_per_gpu: int
    reserved_bytes: int
    total_memory_bytes: int
    budget_bytes: int
    max_context: int
    max_concurrency: int
    waste_fraction: float
    quality_risk: float
    speed: float
    feasible: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def name(self) -> str:
        if self.eviction == "none":
            return f"{self.kv_format}/b{self.block_size}"
        return f"{self.kv_format}/{self.eviction}/b{self.block_size}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "tp": self.tp,
            "kv_format": self.kv_format,
            "eviction": self.eviction,
            "block_size": self.block_size,
            "context": self.context,
            "concurrency": self.concurrency,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_bytes_per_sequence": self.kv_bytes_per_sequence,
            "kv_pool_bytes": self.kv_pool_bytes,
            "weights_per_gpu": self.weights_per_gpu,
            "reserved_bytes": self.reserved_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "budget_bytes": self.budget_bytes,
            "max_context": self.max_context,
            "max_concurrency": self.max_concurrency,
            "waste_fraction": self.waste_fraction,
            "quality_risk": self.quality_risk,
            "speed": self.speed,
            "feasible": self.feasible,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CompileResult:
    request: CompileRequest
    plans: tuple[Plan, ...]
    frontier: tuple[Plan, ...]
    recommended: dict[str, Plan]
    feasible: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.request.model.id,
            "hardware": {
                "name": self.request.hardware.name,
                "total_memory_bytes": self.request.hardware.total_memory_bytes,
                "gpu_memory_utilization": self.request.hardware.gpu_memory_utilization,
                "reserved_bytes": self.request.hardware.reserved(),
                "budget_bytes": self.request.hardware.budget(),
            },
            "context": self.request.context,
            "concurrency": self.request.concurrency,
            "tp": self.request.tp,
            "workload": self.request.workload,
            "max_quality_risk": self.request.max_quality_risk,
            "feasible": self.feasible,
            "notes": list(self.notes),
            "recommended": {k: v.to_dict() for k, v in self.recommended.items()},
            "frontier": [p.to_dict() for p in self.frontier],
            "plans": [p.to_dict() for p in self.plans],
        }


def _dominates(a: Plan, b: Plan) -> bool:
    no_worse = (
        a.total_memory_bytes <= b.total_memory_bytes
        and a.quality_risk <= b.quality_risk
        and a.speed >= b.speed
    )
    better = (
        a.total_memory_bytes < b.total_memory_bytes
        or a.quality_risk < b.quality_risk
        or a.speed > b.speed
    )
    return no_worse and better


def pareto(plans: list[Plan]) -> list[Plan]:
    feasible = [p for p in plans if p.feasible]
    frontier = [p for p in feasible if not any(_dominates(q, p) for q in feasible if q is not p)]
    return sorted(frontier, key=lambda p: (p.total_memory_bytes, p.quality_risk, -p.speed))


def compile_plans(request: CompileRequest) -> CompileResult:
    model = request.model
    hardware = request.hardware
    tp = request.tp
    weight_dtype = request.weight_dtype or model.weight_dtype

    weights_total = model.weight_bytes_for(weight_dtype)
    weights_per_gpu = int(round(weights_total / tp)) if weights_total else 0
    reserved = hardware.reserved()
    budget = hardware.budget()
    available = budget - weights_per_gpu - reserved

    notes: list[str] = []
    if model.weight_bytes and weight_dtype != model.weight_dtype:
        notes.append(
            f"weights converted from {model.weight_dtype} to {weight_dtype}; "
            "verify the checkpoint actually supports that dtype"
        )
    if model.is_mla and tp > 1:
        notes.append(
            "MLA keeps one latent KV per token; tensor parallelism replicates it. "
            "Decode context parallelism is required to shard tokens."
        )
    elif tp > model.num_kv_heads:
        notes.append(
            f"TP={tp} exceeds {model.num_kv_heads} KV heads: per-GPU KV stops shrinking "
            "because every rank needs at least one KV head."
        )
    if model.is_hybrid:
        notes.append(
            "hybrid attention: sliding-window layers cache at most "
            f"{model.sliding_window} tokens each"
        )
    if available <= 0:
        notes.append(
            "weights plus reserved memory exceed the GPU budget; no KV plan can fit"
        )
    if tp > model.layers:
        notes.append(f"TP={tp} exceeds layer count {model.layers}; this is not a valid deployment")

    plans: list[Plan] = []
    for format_name in request.formats:
        fmt = FORMATS[format_name]
        for eviction_name in request.resolved_evictions():
            ev = EVICTIONS[eviction_name]
            keep = request.keep_ratio if request.keep_ratio is not None else ev.keep_ratio
            for block_size in request.block_sizes:
                per_seq_alloc = model.allocated_bytes(
                    request.context, fmt.bytes_per_element, block_size, tp, keep
                )
                per_seq_kv = model.kv_bytes(
                    request.context, fmt.bytes_per_element, tp, keep
                )
                max_context = model.max_context_for(
                    available, fmt.bytes_per_element, block_size, tp, keep
                )
                max_concurrency = model.concurrency(
                    available, request.context, fmt.bytes_per_element, block_size, tp, keep
                )
                pool = per_seq_alloc * request.concurrency
                total = weights_per_gpu + reserved + pool

                reasons: list[str] = []
                if request.context > model.max_context:
                    reasons.append(
                        f"context {request.context} exceeds model maximum {model.max_context}"
                    )
                elif available <= 0:
                    reasons.append("KV budget is negative after weights and reserved memory")
                elif max_context < request.context:
                    reasons.append(
                        f"KV budget supports at most {max_context} tokens at this plan"
                    )
                elif max_concurrency < request.concurrency:
                    reasons.append(
                        f"KV budget supports at most {max_concurrency} sequences at this context"
                    )

                waste = 0.0
                if per_seq_alloc > 0:
                    waste = max(0.0, 1.0 - per_seq_kv / per_seq_alloc)

                plan_notes: list[str] = []
                if format_name == "fp8" and model.head_dim >= 256:
                    plan_notes.append(
                        "FP8 with head_dim>=256 may regress prefill; benchmark before deploy"
                    )
                if eviction_name != "none":
                    plan_notes.append(
                        "eviction memory assumes a compacting cache; paged non-compacting "
                        "reclaim is much lower (NVIDIA 2026)"
                    )
                if block_size >= 64:
                    plan_notes.append(
                        "larger blocks reduce table overhead but increase padding waste"
                    )
                if model.attention_type == "mla":
                    plan_notes.append("MLA latent KV is replicated across TP ranks")

                plans.append(
                    Plan(
                        model_id=model.id,
                        tp=tp,
                        kv_format=format_name,
                        eviction=eviction_name,
                        block_size=block_size,
                        context=request.context,
                        concurrency=request.concurrency,
                        kv_bytes_per_token=per_seq_kv / request.context,
                        kv_bytes_per_sequence=per_seq_kv,
                        kv_pool_bytes=pool,
                        weights_per_gpu=weights_per_gpu,
                        reserved_bytes=reserved,
                        total_memory_bytes=total,
                        budget_bytes=budget,
                        max_context=max_context,
                        max_concurrency=max_concurrency,
                        waste_fraction=waste,
                        quality_risk=plan_risk(
                            format_name, eviction_name, request.workload, model.head_dim
                        ),
                        speed=format_speed(format_name, model.head_dim)
                        * ev.speed
                        * block_speed(block_size),
                        feasible=not reasons,
                        reasons=tuple(reasons),
                        notes=tuple(plan_notes),
                    )
                )

    plans.sort(key=lambda p: (not p.feasible, p.total_memory_bytes, p.quality_risk, -p.speed))

    feasible = [p for p in plans if p.feasible]
    safe = [p for p in feasible if p.quality_risk <= request.max_quality_risk]
    frontier_source = safe if safe else feasible
    frontier = pareto(frontier_source)
    recommended: dict[str, Plan] = {}
    if safe:
        recommended["safest"] = min(
            safe, key=lambda p: (p.quality_risk, p.total_memory_bytes, -p.speed)
        )
        recommended["balanced"] = min(
            safe, key=lambda p: (p.total_memory_bytes, p.quality_risk, -p.speed)
        )
        recommended["max_context"] = max(
            safe, key=lambda p: (p.max_context, -p.quality_risk, -p.total_memory_bytes)
        )
        recommended["max_concurrency"] = max(
            safe, key=lambda p: (p.max_concurrency, -p.quality_risk, -p.total_memory_bytes)
        )
    elif feasible:
        recommended["safest"] = min(
            feasible, key=lambda p: (p.quality_risk, p.total_memory_bytes, -p.speed)
        )
        recommended["balanced"] = min(
            feasible, key=lambda p: (p.total_memory_bytes, p.quality_risk, -p.speed)
        )
        recommended["max_context"] = max(
            feasible, key=lambda p: (p.max_context, -p.quality_risk, -p.total_memory_bytes)
        )
        recommended["max_concurrency"] = max(
            feasible, key=lambda p: (p.max_concurrency, -p.quality_risk, -p.total_memory_bytes)
        )
        notes.append(
            "no feasible plan meets max_quality_risk; recommendations are best-effort"
        )
    else:
        notes.append(
            "no feasible plan: reduce context or concurrency, raise GPU memory utilization, "
            "add tensor parallelism, or allow quantization/eviction"
        )

    return CompileResult(
        request=request,
        plans=tuple(plans),
        frontier=tuple(frontier),
        recommended=recommended,
        feasible=bool(safe),
        notes=tuple(notes),
    )
