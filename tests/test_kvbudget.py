"""Deterministic tests for the KV-cache budget compiler."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvbudget import (  # noqa: E402
    CompileRequest,
    HardwareSpec,
    ZooError,
    compile_plans,
    find_model,
    load_zoo,
)
from kvbudget.quality import FORMATS, plan_risk  # noqa: E402


@pytest.fixture(scope="module")
def zoo():
    return load_zoo()


def model(zoo, name):
    return find_model(zoo, name)


def test_zoo_has_twenty_public_models(zoo):
    assert len(zoo) == 20
    for entry in zoo:
        assert entry.id
        assert entry.layers > 0
        assert entry.head_dim > 0
        assert entry.max_context > 0
        assert entry.config_url.startswith("https://huggingface.co/")
        assert entry.weight_bytes > 0
        assert entry.weight_dtype in {"bfloat16", "fp8", "float32"}


def test_known_llama_8b_anchor(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    assert m.kv_bytes_per_token(131072, 2.0) == 131072
    assert m.kv_bytes_per_token(131072, 1.0) == 65536


def test_known_llama_70b_anchor(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-70B-Instruct")
    assert m.kv_bytes_per_token(131072, 2.0) == 327680
    assert m.kv_bytes_per_token(131072, 2.0, tp=8) == 40960
    assert m.kv_bytes_per_token(131072, 2.0, tp=16) == 40960
    assert m.replication_factor(16) == 2


def test_known_qwen_7b_anchor(zoo):
    m = model(zoo, "Qwen/Qwen2.5-7B-Instruct")
    assert m.kv_bytes_per_token(32768, 2.0) == 57344


def test_deepseek_v3_mla_anchor(zoo):
    m = model(zoo, "deepseek-ai/DeepSeek-V3")
    assert m.attention_type == "mla"
    assert m.kv_bytes_per_token(163840, 2.0) == 70272
    assert m.kv_bytes_per_token(163840, 2.0, tp=8) == 70272
    assert m.replication_factor(8) == 8
    assert m.kv_bytes(1_000_000, 2.0) == 70_272_000_000


def test_gemma_hybrid_window(zoo):
    m = model(zoo, "google/gemma-2-9b-it")
    assert m.is_hybrid
    assert m.sliding_window == 4096
    assert m.layer_types is not None
    assert m.layer_types[0] == "sliding_attention"
    assert m.layer_types[1] == "full_attention"

    at_window = m.kv_bytes(4096, 2.0)
    assert at_window == 42 * 8192 * 4096
    at_max = m.kv_bytes(8192, 2.0)
    assert at_max == 21 * 8192 * 8192 + 21 * 8192 * 4096
    assert m.kv_bytes_per_token(8192, 2.0) == 258048


def test_paged_rounding_waste(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    exact = m.kv_bytes(1000, 2.0)
    allocated = m.allocated_bytes(1000, 2.0, 16)
    assert exact == 131_072_000
    assert allocated == 132_120_576
    assert 1.0 - exact / allocated == pytest.approx(0.0079365, rel=1e-5)


def test_tp_replication_never_shrinks_below_one_head(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-70B-Instruct")
    per_gpu = [m.kv_bytes_per_token(32768, 2.0, tp=tp) for tp in (1, 2, 4, 8, 16)]
    assert per_gpu == [327680, 163840, 81920, 40960, 40960]
    assert m.kv_heads_per_gpu(16) == 1


def test_quality_risk_ordering():
    bf16 = plan_risk("bf16", "none", "chat")
    int8 = plan_risk("int8", "none", "chat")
    fp8 = plan_risk("fp8", "none", "chat")
    int4 = plan_risk("int4", "none", "chat")
    int2 = plan_risk("int2", "none", "chat")
    assert bf16 == 0.0
    assert bf16 < int8 < fp8 <= int4 < int2
    assert plan_risk("bf16", "snapkv", "chat") > plan_risk("bf16", "none", "chat")
    assert plan_risk("int4", "none", "code") > plan_risk("int4", "none", "chat")


def test_compiler_24gb_llama_32k(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    hw = HardwareSpec.from_gb(24.0)
    req = CompileRequest(model=m, hardware=hw, context=32768, concurrency=1)
    result = compile_plans(req)
    assert result.feasible
    assert "safest" in result.recommended
    assert result.recommended["safest"].quality_risk <= req.max_quality_risk
    balanced = result.recommended["balanced"]
    assert balanced.kv_bytes_per_token < 65536
    assert balanced.total_memory_bytes <= hw.budget()
    assert result.frontier


def test_compiler_128k_needs_low_precision(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    hw = HardwareSpec.from_gb(24.0)
    req = CompileRequest(
        model=m,
        hardware=hw,
        context=131072,
        concurrency=1,
        max_quality_risk=0.10,
    )
    result = compile_plans(req)
    assert not result.feasible
    assert all(plan.quality_risk > 0.10 for plan in result.recommended.values())
    assert any(
        plan.feasible and plan.kv_format == "int2" for plan in result.plans
    )


def test_compiler_tp_shards_kv(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-70B-Instruct")
    hw = HardwareSpec.from_gb(24.0)
    req = CompileRequest(
        model=m,
        hardware=hw,
        context=32768,
        concurrency=1,
        tp=8,
        formats=("bf16",),
        block_sizes=(16,),
    )
    result = compile_plans(req)
    assert result.feasible
    plan = result.recommended["balanced"]
    assert plan.tp == 8
    assert plan.kv_bytes_per_token == 40960
    assert plan.max_concurrency >= 1


def test_compiler_mla_replicates_across_tp(zoo):
    m = model(zoo, "deepseek-ai/DeepSeek-V2-Lite-Chat")
    hw = HardwareSpec.from_gb(80.0)
    req = CompileRequest(
        model=m,
        hardware=hw,
        context=32768,
        concurrency=1,
        tp=8,
        formats=("bf16",),
        block_sizes=(16,),
    )
    result = compile_plans(req)
    plan = result.recommended["balanced"]
    assert plan.kv_bytes_per_token == 31104
    assert any("replicates" in note for note in result.notes)


def test_request_validation(zoo):
    m = model(zoo, "meta-llama/Llama-3.1-8B-Instruct")
    hw = HardwareSpec.from_gb(24.0)
    with pytest.raises(ValueError):
        CompileRequest(model=m, hardware=hw, context=0)
    with pytest.raises(ValueError):
        CompileRequest(model=m, hardware=hw, context=1024, formats=("nope",))
    with pytest.raises(ValueError):
        CompileRequest(model=m, hardware=hw, context=1024, keep_ratio=0.0)


def test_json_round_trip(zoo):
    m = model(zoo, "Qwen/Qwen2.5-7B-Instruct")
    hw = HardwareSpec.from_gb(24.0)
    req = CompileRequest(model=m, hardware=hw, context=32768, concurrency=4)
    result = compile_plans(req)
    payload = result.to_dict()
    text = json.dumps(payload)
    assert "recommended" in payload
    assert "frontier" in payload
    assert json.loads(text)["model"] == m.id


def test_zoo_lookup_errors(zoo):
    with pytest.raises(ZooError):
        find_model(zoo, "not-a-real-model")
    with pytest.raises(ZooError):
        find_model(zoo, "Qwen/Qwen")
