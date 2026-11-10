#!/usr/bin/env python3
"""Run experiments and write an aggregate report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fsl.common import RESULTS
from fsl.registry import EXPERIMENTS, load

SMOKE = ["e01_least_action", "e08_fourier_dtmf", "e20_qlearning"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", help="experiment ids (default: all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k in EXPERIMENTS:
            print(k)
        return 0
    if args.smoke:
        names = SMOKE
    elif args.all or not args.experiments:
        names = list(EXPERIMENTS)
    else:
        names = args.experiments

    summary_path = RESULTS / "summary.json"
    merged: dict[str, dict] = {}
    if summary_path.exists():
        try:
            for d in json.loads(summary_path.read_text()):
                merged[d.get("name")] = d
        except (json.JSONDecodeError, TypeError):
            merged = {}

    summary = []
    failed = 0
    for name in names:
        print(f"== {name} ==", flush=True)
        res = load(name)()
        print(f"   status={res.status} time={res.seconds}s", flush=True)
        for c in res.checks:
            print(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['label']}: {c['detail']}")
        if res.error:
            print(res.error)
        if res.status != "ok":
            failed += 1
        merged[res.name] = {"name": res.name, "status": res.status, "seconds": res.seconds,
                            "checks": res.checks, "metrics": res.metrics}

    order = [n for n in EXPERIMENTS if n in merged] + [n for n in merged if n not in EXPERIMENTS]
    summary = [merged[n] for n in order]
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{len(names) - failed}/{len(names)} experiments passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
