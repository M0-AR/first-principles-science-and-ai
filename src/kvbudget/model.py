"""Model and hardware specifications for the KV-cache budget compiler."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

GiB = 2**30
MiB = 2**20
KiB = 2**10

AttentionType = Literal["mha", "gqa", "mqa", "mla"]

DTYPE_BYTES: dict[str, float] = {
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "float32": 4.0,
    "fp32": 4.0,
    "fp8": 1.0,
    "float8_e4m3fn": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "nf4": 0.5,
    "int2": 0.25,
}


def dtype_bytes(name: str) -> float:
    try:
        return DTYPE_BYTES[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(DTYPE_BYTES)}") from exc


@dataclass(frozen=True)
class ModelSpec:
    id: str
    layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_context: int
    attention_type: AttentionType = "gqa"
    sliding_window: int | None = None
    layer_types: tuple[str, ...] | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    weight_bytes: int = 0
    weight_dtype: str = "bfloat16"
    params_b: float | None = None
    config_url: str = ""
    weight_url: str = ""
    license: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelSpec":
        layer_types = raw.get("layer_types")
        return cls(
            id=str(raw["id"]),
            layers=int(raw["layers"]),
            hidden_size=int(raw["hidden_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw["num_kv_heads"]),
            head_dim=int(raw["head_dim"]),
            max_context=int(raw["max_context"]),
            attention_type=raw.get("attention_type", "gqa"),
            sliding_window=raw.get("sliding_window"),
            layer_types=tuple(layer_types) if layer_types else None,
            kv_lora_rank=raw.get("kv_lora_rank"),
            qk_rope_head_dim=raw.get("qk_rope_head_dim"),
            weight_bytes=int(raw.get("weight_bytes", 0)),
            weight_dtype=str(raw.get("weight_dtype", "bfloat16")),
            params_b=raw.get("params_b"),
            config_url=str(raw.get("config_url", "")),
            weight_url=str(raw.get("weight_url", "")),
            license=str(raw.get("license", "")),
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_context": self.max_context,
            "attention_type": self.attention_type,
            "sliding_window": self.sliding_window,
            "layer_types": list(self.layer_types) if self.layer_types else None,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "weight_bytes": self.weight_bytes,
            "weight_dtype": self.weight_dtype,
            "params_b": self.params_b,
            "config_url": self.config_url,
            "weight_url": self.weight_url,
            "license": self.license,
            "notes": self.notes,
        }

    @property
    def is_mla(self) -> bool:
        return self.attention_type == "mla"

    @property
    def is_hybrid(self) -> bool:
        return bool(self.layer_types and self.sliding_window)

    def kv_heads_per_gpu(self, tp: int = 1) -> int:
        if self.is_mla:
            return 1
        return max(1, self.num_kv_heads // max(1, tp))

    def replication_factor(self, tp: int = 1) -> int:
        if self.is_mla:
            return max(1, tp)
        return max(1, tp // max(1, self.num_kv_heads))

    def elements_per_token_per_layer(self, tp: int = 1) -> float:
        if self.is_mla:
            if not self.kv_lora_rank or not self.qk_rope_head_dim:
                raise ValueError(
                    f"{self.id}: MLA requires kv_lora_rank and qk_rope_head_dim"
                )
            return float(self.kv_lora_rank + self.qk_rope_head_dim)
        return float(2 * self.kv_heads_per_gpu(tp) * self.head_dim)

    def _layer_token_counts(self, seq_len: int) -> tuple[int, int, int]:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if not self.layer_types or not self.sliding_window:
            return self.layers, 0, seq_len
        window = min(seq_len, self.sliding_window)
        local = sum(1 for kind in self.layer_types if kind == "sliding_attention")
        full = self.layers - local
        return full, local, window

    def kv_bytes(
        self,
        seq_len: int,
        format_bytes: float,
        tp: int = 1,
        keep_ratio: float = 1.0,
    ) -> float:
        full, local, window = self._layer_token_counts(seq_len)
        per_layer = self.elements_per_token_per_layer(tp) * format_bytes
        tokens = (full * seq_len + local * window) * keep_ratio
        return per_layer * tokens

    def allocated_bytes(
        self,
        seq_len: int,
        format_bytes: float,
        block_size: int,
        tp: int = 1,
        keep_ratio: float = 1.0,
    ) -> int:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        full, local, window = self._layer_token_counts(seq_len)
        per_layer = self.elements_per_token_per_layer(tp) * format_bytes

        def rounded(tokens: float) -> int:
            if tokens <= 0:
                return 0
            return math.ceil(tokens / block_size) * block_size

        full_tokens = rounded(seq_len * keep_ratio)
        local_tokens = rounded(window * keep_ratio)
        return int(round(per_layer * (full * full_tokens + local * local_tokens)))

    def kv_bytes_per_token(
        self,
        seq_len: int,
        format_bytes: float,
        tp: int = 1,
        keep_ratio: float = 1.0,
    ) -> float:
        if seq_len <= 0:
            return 0.0
        return self.kv_bytes(seq_len, format_bytes, tp, keep_ratio) / seq_len

    def max_context_for(
        self,
        available_bytes: float,
        format_bytes: float,
        block_size: int,
        tp: int = 1,
        keep_ratio: float = 1.0,
    ) -> int:
        if available_bytes <= 0:
            return 0
        high = self.max_context
        if self.allocated_bytes(high, format_bytes, block_size, tp, keep_ratio) <= available_bytes:
            return high
        low = 0
        while low < high:
            mid = (low + high + 1) // 2
            if self.allocated_bytes(mid, format_bytes, block_size, tp, keep_ratio) <= available_bytes:
                low = mid
            else:
                high = mid - 1
        return low

    def concurrency(
        self,
        available_bytes: float,
        seq_len: int,
        format_bytes: float,
        block_size: int,
        tp: int = 1,
        keep_ratio: float = 1.0,
    ) -> int:
        per_seq = self.allocated_bytes(seq_len, format_bytes, block_size, tp, keep_ratio)
        if per_seq <= 0 or available_bytes <= 0:
            return 0
        return int(available_bytes // per_seq)

    def weight_bytes_for(self, target_dtype: str) -> int:
        if not self.weight_bytes:
            return 0
        if target_dtype == self.weight_dtype:
            return self.weight_bytes
        target = dtype_bytes(target_dtype)
        if self.params_b:
            return int(round(self.params_b * 1e9 * target))
        source = dtype_bytes(self.weight_dtype)
        return int(round(self.weight_bytes * target / source))

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layers": self.layers,
            "kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "attention_type": self.attention_type,
            "max_context": self.max_context,
            "weight_gib": self.weight_bytes / GiB if self.weight_bytes else None,
            "params_b": self.params_b,
            "hybrid": self.is_hybrid,
        }


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    total_memory_bytes: int
    gpu_memory_utilization: float = 0.92
    reserved_bytes: int | None = None
    num_gpus: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def reserved(self) -> int:
        if self.reserved_bytes is not None:
            return self.reserved_bytes
        return int(1.0 * GiB + 0.03 * self.total_memory_bytes)

    def budget(self) -> int:
        return int(self.total_memory_bytes * self.gpu_memory_utilization)

    def available_kv(self, weights_per_gpu: int) -> int:
        return self.budget() - weights_per_gpu - self.reserved()

    @classmethod
    def from_gb(
        cls,
        gb: float,
        name: str | None = None,
        utilization: float = 0.92,
        reserved_gb: float | None = None,
        num_gpus: int = 1,
    ) -> "HardwareSpec":
        label = f"{gb:g} GB GPU" if num_gpus == 1 else f"{num_gpus}x {gb:g} GB GPU"
        return cls(
            name=name or label,
            total_memory_bytes=int(gb * 1e9),
            gpu_memory_utilization=utilization,
            reserved_bytes=int(reserved_gb * 1e9) if reserved_gb is not None else None,
            num_gpus=num_gpus,
        )
