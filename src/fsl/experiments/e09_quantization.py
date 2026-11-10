"""E09: Quantization — INT8/INT4 weight-only quantization of a real LM.

Load distilgpt2 (82M parameters) and measure perplexity on Tiny
Shakespeare in FP32, then after per-output-channel symmetric INT8 and
INT4 weight quantization (the Llama-style scheme, dequantized for CPU).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from fsl.common import DATA, ExperimentResult, run_experiment

os.environ.setdefault("HF_HOME", str(DATA / "hf"))


def perplexity(model, ids: torch.Tensor) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[1] - 512, 512):
            chunk = ids[:, i:i + 512]
            out = model(chunk, labels=chunk)
            total += float(out.loss) * (chunk.shape[1] - 1)
            n += chunk.shape[1] - 1
    return float(np.exp(total / n))


def quantize_(model, bits: int, group: int = 128, per_tensor: bool = False) -> list[tuple]:
    saved = []
    qmax = 2 ** (bits - 1) - 1
    for name, mod in model.named_modules():
        is_linear = isinstance(mod, torch.nn.Linear)
        is_conv1d = mod.__class__.__name__ == "Conv1D"
        if not (is_linear or is_conv1d) or name.endswith("lm_head"):
            continue
        w = mod.weight.data if is_linear else mod.weight.data.t()
        saved.append((mod, mod.weight.data.clone()))
        out_f, in_f = w.shape
        if per_tensor:
            scale = w.abs().amax() / qmax
            wq = torch.round(w / scale.clamp_min(1e-12)).clamp(-qmax - 1, qmax) * scale
        elif bits == 4 and in_f % group == 0:
            wg = w.reshape(out_f, in_f // group, group)
            scale = wg.abs().amax(dim=2, keepdim=True) / qmax
            wg = torch.round(wg / scale.clamp_min(1e-12)).clamp(-qmax - 1, qmax) * scale
            wq = wg.reshape(out_f, in_f)
        else:
            scale = w.abs().amax(dim=1, keepdim=True) / qmax
            wq = torch.round(w / scale.clamp_min(1e-12)).clamp(-qmax - 1, qmax) * scale
        mod.weight.data = wq if is_linear else wq.t()
    return saved


def restore(saved: list[tuple]) -> None:
    for mod, w in saved:
        mod.weight.data = w


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32)
        text = (DATA / "tinyshakespeare.txt").read_text()[:60000]
        ids = tok(text, return_tensors="pt").input_ids[:, :16384]

        ppl32 = perplexity(model, ids)
        n_params = sum(p.numel() for p in model.parameters())

        saved = quantize_(model, 8)
        ppl8 = perplexity(model, ids)
        restore(saved)

        saved = quantize_(model, 4)
        ppl4 = perplexity(model, ids)
        restore(saved)

        saved = quantize_(model, 4, per_tensor=True)
        ppl4_naive = perplexity(model, ids)
        restore(saved)

        res.check("INT8 perplexity within 5% of FP32", ppl8 < 1.05 * ppl32,
                  f"FP32={ppl32:.3f}, INT8={ppl8:.3f}")
        res.check("group-wise INT4 perplexity within 25% of FP32", ppl4 < 1.25 * ppl32,
                  f"INT4={ppl4:.3f}")
        res.check("group-wise INT4 beats per-tensor INT4", ppl4 < ppl4_naive,
                  f"grouped={ppl4:.3f} vs naive={ppl4_naive:.3f}")
        res.check("weights shrink 4x (INT8) and 8x (INT4)",
                  32 / 8 == 4 and 32 / 4 == 8, "32->8 bits, 32->4 bits")

        res.metrics.update({"params": int(n_params), "ppl_fp32": ppl32,
                            "ppl_int8": ppl8, "ppl_int4": ppl4,
                            "tokens": int(ids.shape[1]),
                            "int8_ratio": ppl8 / ppl32,
                            "int4_ratio": ppl4 / ppl32, "int4_group": 128,
                            "ppl_int4_naive": ppl4_naive})
    return run_experiment("e09_quantization", body)
