#!/usr/bin/env python3
"""Command-line KV-cache budget compiler."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvbudget import (  # noqa: E402
    CompileRequest,
    HardwareSpec,
    ZooError,
    compile_plans,
    find_model,
    load_zoo,
    model_table,
    plans_table,
    render_report,
)
from kvbudget.quality import EVICTIONS, FORMATS  # noqa: E402


def csv_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile KV-cache budgets into feasible deployment plans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-models", action="store_true", help="print the public model zoo")
    parser.add_argument("--model", help="model id or unique substring")
    parser.add_argument("--gpu-memory-gb", type=float, default=24.0, help="per-GPU memory in decimal GB")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--reserved-gb", type=float, default=None)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-risk", type=float, default=0.10)
    parser.add_argument("--allow-eviction", action="store_true")
    parser.add_argument(
        "--evictions",
        type=csv_strings,
        default=None,
        help=f"comma-separated policies: {', '.join(EVICTIONS)}",
    )
    parser.add_argument(
        "--formats",
        type=csv_strings,
        default=("bf16", "fp8", "int8", "int4", "int2"),
        help=f"comma-separated KV formats: {', '.join(FORMATS)}",
    )
    parser.add_argument("--block-sizes", type=csv_ints, default=(16, 32, 64))
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel degree (number of GPUs)")
    parser.add_argument("--weight-dtype", default=None)
    parser.add_argument(
        "--workload",
        choices=("chat", "retrieval", "summarization", "reasoning", "code"),
        default="chat",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        models = load_zoo()
    except ZooError as exc:
        parser.error(str(exc))

    if args.list_models:
        print(model_table(models))
        return 0
    if not args.model:
        parser.error("--model is required (or use --list-models)")

    try:
        model = find_model(models, args.model)
    except ZooError as exc:
        parser.error(str(exc))
    hardware = HardwareSpec.from_gb(
        args.gpu_memory_gb,
        utilization=args.gpu_memory_utilization,
        reserved_gb=args.reserved_gb,
        num_gpus=args.tp,
    )
    request = CompileRequest(
        model=model,
        hardware=hardware,
        context=args.context,
        concurrency=args.concurrency,
        max_quality_risk=args.max_risk,
        allow_eviction=args.allow_eviction,
        block_sizes=args.block_sizes,
        formats=args.formats,
        evictions=args.evictions,
        tp=args.tp,
        weight_dtype=args.weight_dtype,
        keep_ratio=args.keep_ratio,
        workload=args.workload,
    )
    try:
        result = compile_plans(request)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_report(result, top=args.top))
        feasible = [p for p in result.plans if p.feasible]
        print()
        if feasible:
            print("All feasible plans (top)")
            print(plans_table(feasible, top=args.top))
        else:
            print("All attempts (top)")
            print(plans_table(result.plans, top=args.top))
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
