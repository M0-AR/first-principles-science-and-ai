"""Human-readable output for the KV-budget compiler."""
from __future__ import annotations

from .compiler import CompileResult, Plan
from .model import ModelSpec


def human_bytes(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value < 1024:
        return f"{sign}{value:.0f} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024.0
        if value < 1024 or unit == "TiB":
            return f"{sign}{value:.2f} {unit}"
    return f"{sign}{value:.2f} TiB"


def human_count(value: float) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def model_table(models: list[ModelSpec]) -> str:
    headers = ["model", "params", "layers", "kv", "head_dim", "max ctx", "weights"]
    rows: list[list[str]] = []
    for m in models:
        kv = "MLA" if m.attention_type == "mla" else str(m.num_kv_heads)
        rows.append([
            m.id,
            f"{m.params_b:.1f}B" if m.params_b else "?",
            str(m.layers),
            kv,
            str(m.head_dim),
            human_count(m.max_context),
            human_bytes(m.weight_bytes) if m.weight_bytes else "?",
        ])
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def plans_table(plans: list[Plan], top: int | None = None) -> str:
    if top is not None:
        plans = plans[:top]
    headers = ["plan", "KV/token", "KV/seq", "max ctx", "seqs", "risk", "speed", "GPU total"]
    rows: list[list[str]] = []
    for p in plans:
        mark = "" if p.feasible else "! "
        rows.append([
            mark + p.name,
            human_bytes(p.kv_bytes_per_token),
            human_bytes(p.kv_bytes_per_sequence),
            human_count(p.max_context),
            str(p.max_concurrency),
            f"{p.quality_risk:.3f}",
            f"{p.speed:.2f}x",
            f"{human_bytes(p.total_memory_bytes)} / {human_bytes(p.budget_bytes)}",
        ])
    if not rows:
        return "(no plans)"
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def summary_lines(result: CompileResult) -> list[str]:
    model = result.request.model
    hw = result.request.hardware
    first = result.plans[0] if result.plans else None
    weights = first.weights_per_gpu if first else 0
    reserved = first.reserved_bytes if first else hw.reserved()
    budget = hw.budget()
    available = budget - weights - reserved
    attention = model.attention_type
    if model.is_hybrid:
        attention += f", hybrid window={model.sliding_window}"
    return [
        f"model      {model.id}",
        f"attention  {attention}, {model.layers} layers, "
        f"{'MLA latent' if model.is_mla else f'{model.num_kv_heads} KV heads'} "
        f"x head_dim {model.head_dim}",
        f"hardware   {hw.name}, budget {human_bytes(budget)} "
        f"({hw.gpu_memory_utilization:.0%} of {human_bytes(hw.total_memory_bytes)})",
        f"memory     weights/GPU {human_bytes(weights)} + reserved {human_bytes(reserved)} "
        f"= {human_bytes(weights + reserved)}; KV pool {human_bytes(available)}",
        f"request    context {human_count(result.request.context)}, "
        f"sequences {result.request.concurrency}, TP {result.request.tp}, "
        f"workload {result.request.workload}, max risk {result.request.max_quality_risk:.0%}",
        f"verdict    {'feasible' if result.feasible else 'no plan meets the quality bar'}",
    ]


def render_report(result: CompileResult, top: int = 12) -> str:
    lines = summary_lines(result)
    lines.append("")

    if result.recommended:
        labels = {
            "safest": "SAFEST",
            "balanced": "BALANCED",
            "max_context": "MAX CONTEXT",
            "max_concurrency": "MAX CONCURRENCY",
        }
        seen: set[str] = set()
        selected: list[Plan] = []
        for key in ("safest", "balanced", "max_context", "max_concurrency"):
            plan = result.recommended.get(key)
            if plan is None or plan.name in seen:
                continue
            seen.add(plan.name)
            selected.append(plan)
        lines.append("Recommended" if result.feasible else "Best-effort plans")
        lines.append(plans_table(selected))
        if len(selected) < len(labels):
            lines.append(
                "  (labels share plans when one plan wins on several axes)"
            )
        lines.append("")

        if result.frontier:
            lines.append("Pareto frontier: lowest memory for each risk/speed trade-off")
            lines.append(plans_table(list(result.frontier), top=top))
            lines.append("")

    if not result.feasible:
        attempts = sorted(
            result.plans,
            key=lambda p: (p.max_context, -p.quality_risk, p.total_memory_bytes),
            reverse=True,
        )
        if attempts:
            lines.append("Closest attempts (why they miss)")
            for plan in attempts[:5]:
                reason = "; ".join(plan.reasons) if plan.reasons else "quality bar not met"
                lines.append(f"  {plan.name}: {reason}")
            lines.append("")

    if result.notes:
        lines.append("Notes")
        for note in result.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip()
