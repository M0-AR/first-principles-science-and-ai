#!/usr/bin/env python3
"""Validate KV formulas, measure a real cache, and stress-test quantization.

Run from the repository root:
    python scripts/kvbudget_bench.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvbudget import (  # noqa: E402
    CompileRequest,
    HardwareSpec,
    ModelSpec,
    compile_plans,
    find_model,
    human_bytes,
    load_zoo,
)

OUT = ROOT / "results" / "kvbudget.json"


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def cache_pairs(past: Any) -> Iterable[tuple[Any, Any]]:
    if hasattr(past, "layers"):
        for layer in past.layers:
            yield layer.keys, layer.values
        return
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        yield from zip(past.key_cache, past.value_cache)
        return
    for layer in past:
        yield layer[0], layer[1]


def cache_bytes(past: Any) -> int:
    total = 0
    for key, value in cache_pairs(past):
        if key is not None:
            total += key.numel() * key.element_size()
        if value is not None:
            total += value.numel() * value.element_size()
    return total


def quantize(x: Any, fmt: str, torch: Any) -> Any:
    if fmt == "bf16":
        return x
    if fmt == "fp8":
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    bits = {"int8": 8, "int4": 4, "int2": 2}[fmt]
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale


def run_opt_benchmark(model_spec: ModelSpec) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m", local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    seq_results: dict[str, dict[str, int]] = {}
    for seq_len in (128, 512, 1024, 2048):
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, seq_len), device=device
        )
        with torch.no_grad():
            output = model(input_ids, use_cache=True)
        measured = cache_bytes(output.past_key_values)
        formula = int(model_spec.kv_bytes(seq_len, torch.finfo(dtype).bits / 8))
        seq_results[str(seq_len)] = {"formula": formula, "measured": measured}
        del output
        if device == "cuda":
            torch.cuda.empty_cache()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 512), device=device)
    with torch.no_grad():
        output = model(input_ids, use_cache=True)
    pairs = [(k[0], v[0]) for k, v in cache_pairs(output.past_key_values)]

    quantization: dict[str, dict[str, float]] = {}
    positions = torch.arange(0, 512, 8, device=device)
    std = pairs[0][0].float().std().item()
    for fmt in ("bf16", "fp8", "int8", "int4", "int2"):
        hits = 0
        total = 0
        errors: list[float] = []
        for key, value in pairs:
            kf = key.float()
            vf = value.float()
            kq = quantize(kf, fmt, torch)
            vq = quantize(vf, fmt, torch)
            heads, _seq_len, head_dim = kf.shape
            for head in range(heads):
                q = kf[head, positions]
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                k_norm = kq[head] / kq[head].norm(dim=-1, keepdim=True).clamp_min(1e-8)
                scores = q @ k_norm.transpose(0, 1)
                pred = scores.argmax(dim=-1)
                hits += int((pred == positions).sum().item())
                total += int(positions.numel())

                weights = torch.softmax(scores, dim=-1)
                out = weights @ vq[head]
                ref_k_norm = kf[head] / kf[head].norm(dim=-1, keepdim=True).clamp_min(1e-8)
                ref_scores = q @ ref_k_norm.transpose(0, 1)
                ref_weights = torch.softmax(ref_scores, dim=-1)
                ref = ref_weights @ vf[head]
                rel = (out - ref).norm(dim=-1) / (ref.norm(dim=-1) + 1e-8)
                errors.append(float(rel.mean().item()))
        quantization[fmt] = {
            "top1": hits / total if total else 0.0,
            "relative_error": sum(errors) / len(errors) if errors else 0.0,
        }

    del model, output, pairs
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "cache": seq_results,
        "quantization": quantization,
        "key_std": std,
    }


def synthetic_check(torch: Any, model_spec: ModelSpec) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA unavailable"}
    dtype = torch.bfloat16
    seq_len = 1024
    key = torch.empty(
        (model_spec.layers, model_spec.num_kv_heads, seq_len, model_spec.head_dim),
        dtype=dtype,
        device="cuda",
    )
    value = torch.empty_like(key)
    measured = key.numel() * key.element_size() + value.numel() * value.element_size()
    formula = int(model_spec.kv_bytes(seq_len, 2.0))
    del key, value
    torch.cuda.empty_cache()
    return {
        "skipped": False,
        "formula": formula,
        "measured": measured,
        "matches": measured == formula,
    }


def main() -> int:
    zoo = load_zoo()
    llama8 = find_model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    llama70 = find_model(zoo, "meta-llama/Llama-3.1-70B-Instruct")
    qwen7 = find_model(zoo, "Qwen/Qwen2.5-7B-Instruct")
    gemma9 = find_model(zoo, "google/gemma-2-9b-it")
    deepseek3 = find_model(zoo, "deepseek-ai/DeepSeek-V3")
    deepseek2 = find_model(zoo, "deepseek-ai/DeepSeek-V2-Lite-Chat")

    checks: list[dict[str, Any]] = []
    checks.append(check(
        "Llama-3.1-8B is 128 KiB/token BF16",
        llama8.kv_bytes_per_token(131072, 2.0) == 131072,
        human_bytes(llama8.kv_bytes_per_token(131072, 2.0)),
    ))
    checks.append(check(
        "Qwen2.5-7B is 56 KiB/token BF16",
        qwen7.kv_bytes_per_token(32768, 2.0) == 57344,
        human_bytes(qwen7.kv_bytes_per_token(32768, 2.0)),
    ))
    checks.append(check(
        "Llama-3.1-70B is 320 KiB/token BF16",
        llama70.kv_bytes_per_token(131072, 2.0) == 327680,
        human_bytes(llama70.kv_bytes_per_token(131072, 2.0)),
    ))
    checks.append(check(
        "TP=8 shards Llama-70B KV to 40 KiB/token/GPU",
        llama70.kv_bytes_per_token(131072, 2.0, tp=8) == 40960,
        human_bytes(llama70.kv_bytes_per_token(131072, 2.0, tp=8)),
    ))
    checks.append(check(
        "TP=16 cannot shrink Llama-70B below one KV head",
        llama70.kv_bytes_per_token(131072, 2.0, tp=16) == 40960,
        human_bytes(llama70.kv_bytes_per_token(131072, 2.0, tp=16)),
    ))
    checks.append(check(
        "DeepSeek-V3 MLA is 70,272 B/token BF16",
        deepseek3.kv_bytes_per_token(163840, 2.0) == 70272,
        human_bytes(deepseek3.kv_bytes_per_token(163840, 2.0)),
    ))
    checks.append(check(
        "Gemma-2-9B hybrid averages 252 KiB/token at 8K",
        gemma9.kv_bytes_per_token(8192, 2.0) == 258048,
        human_bytes(gemma9.kv_bytes_per_token(8192, 2.0)),
    ))

    metrics: dict[str, Any] = {}
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        synthetic = synthetic_check(torch, llama8)
        metrics["synthetic"] = synthetic
        if not synthetic.get("skipped"):
            checks.append(check(
                "synthetic BF16 K+V tensor matches formula",
                synthetic["matches"],
                f"{synthetic['measured']} vs {synthetic['formula']} bytes",
            ))
    else:
        metrics["synthetic"] = {"skipped": True, "reason": "torch unavailable"}

    try:
        opt_spec = ModelSpec(
            id="facebook/opt-125m",
            layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_kv_heads=12,
            head_dim=64,
            max_context=2048,
        )
        opt = run_opt_benchmark(opt_spec)
        metrics["opt125m"] = opt
        for seq_len, row in opt["cache"].items():
            checks.append(check(
                f"OPT-125m cache at {seq_len} tokens matches formula",
                abs(row["measured"] - row["formula"]) <= 1,
                f"{row['measured']} vs {row['formula']} bytes",
            ))
        q = opt["quantization"]
        checks.append(check(
            "OPT-125m BF16 retrieval is 100%",
            q["bf16"]["top1"] >= 0.999,
            f"{q['bf16']['top1']:.4f}",
        ))
        checks.append(check(
            "FP8 cache keeps >=99.5% top-1 retrieval",
            q["fp8"]["top1"] >= 0.995,
            f"{q['fp8']['top1']:.4f}",
        ))
        checks.append(check(
            "INT8 cache keeps >=99.5% top-1 retrieval",
            q["int8"]["top1"] >= 0.995,
            f"{q['int8']['top1']:.4f}",
        ))
        checks.append(check(
            "INT4 cache keeps >=95% top-1 retrieval",
            q["int4"]["top1"] >= 0.95,
            f"{q['int4']['top1']:.4f}",
        ))
        checks.append(check(
            "INT2 is materially worse than INT4",
            q["int2"]["top1"] < q["int4"]["top1"] and q["int2"]["top1"] < 0.90,
            f"top-1 {q['int2']['top1']:.4f}, relative error "
            f"{q['int2']['relative_error']:.3f}",
        ))
    except Exception as exc:  # noqa: BLE001 - benchmark should report, not crash
        metrics["opt125m"] = {"skipped": True, "reason": str(exc)}
        checks.append(check("OPT-125m real-cache benchmark", False, str(exc)))

    llm24 = compile_plans(CompileRequest(
        model=llama8,
        hardware=HardwareSpec.from_gb(24.0),
        context=32768,
        concurrency=1,
    ))
    metrics["compiler_llama8_24gb"] = {
        "feasible": llm24.feasible,
        "balanced": llm24.recommended["balanced"].to_dict(),
    }
    checks.append(check(
        "compiler finds a plan for Llama-8B 32K on 24 GB",
        llm24.feasible,
        llm24.recommended["balanced"].name,
    ))
    checks.append(check(
        "compiler plan fits the hardware budget",
        llm24.recommended["balanced"].total_memory_bytes
        <= llm24.request.hardware.budget(),
        human_bytes(llm24.recommended["balanced"].total_memory_bytes),
    ))

    ds16 = compile_plans(CompileRequest(
        model=deepseek3,
        hardware=HardwareSpec.from_gb(80.0, num_gpus=16),
        context=32768,
        concurrency=1,
        tp=16,
        formats=("bf16",),
        block_sizes=(16,),
    ))
    metrics["compiler_deepseek_v3_16xh100"] = {
        "feasible": ds16.feasible,
        "plan": ds16.recommended["balanced"].to_dict(),
    }
    checks.append(check(
        "compiler plans DeepSeek-V3 32K on 16x80 GB",
        ds16.feasible and ds16.recommended["balanced"].kv_bytes_per_token == 70272,
        ds16.recommended["balanced"].name,
    ))

    ds2 = compile_plans(CompileRequest(
        model=deepseek2,
        hardware=HardwareSpec.from_gb(24.0, num_gpus=2),
        context=32768,
        concurrency=1,
        tp=2,
        formats=("bf16", "fp8", "int8"),
    ))
    metrics["compiler_deepseek_v2_lite_24gb"] = {
        "feasible": ds2.feasible,
        "balanced": ds2.recommended["balanced"].to_dict(),
    }
    checks.append(check(
        "compiler handles MLA on a 24 GB GPU",
        ds2.feasible,
        ds2.recommended["balanced"].name,
    ))

    ok = all(c["passed"] for c in checks)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ok": ok,
        "checks": checks,
        "metrics": metrics,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    for item in checks:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"[{mark}] {item['name']}: {item['detail']}")
    print(f"\nwrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
