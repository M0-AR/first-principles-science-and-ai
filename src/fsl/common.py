"""Shared infrastructure: deterministic runs, result capture, verification."""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentResult:
    name: str
    status: str = "ok"
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        self.checks.append({"label": label, "passed": bool(passed), "detail": detail})
        if not passed:
            self.status = "failed"
        return passed


def run_experiment(name: str, fn: Callable[[ExperimentResult], None], seed: int = 0) -> ExperimentResult:
    np.random.seed(seed)
    res = ExperimentResult(name=name)
    t0 = time.time()
    try:
        fn(res)
    except Exception:
        res.status = "error"
        res.error = traceback.format_exc(limit=3)
    res.seconds = round(time.time() - t0, 2)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(res.__dict__, indent=2, default=str))
    return res


def base_url() -> str:
    return os.environ.get("FSL_BASE_URL", "")
