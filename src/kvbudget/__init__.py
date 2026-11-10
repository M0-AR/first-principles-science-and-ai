"""KV-cache budget compiler for long-context inference."""
from .compiler import CompileRequest, CompileResult, Plan, compile_plans, pareto
from .model import HardwareSpec, ModelSpec, dtype_bytes
from .quality import (
    EVICTIONS,
    FORMATS,
    evidence_table,
    format_speed,
    plan_risk,
)
from .report import human_bytes, human_count, model_table, plans_table, render_report
from .zoo import DEFAULT_ZOO, ZooError, find_model, find_models, load_zoo

__all__ = [
    "CompileRequest",
    "CompileResult",
    "DEFAULT_ZOO",
    "EVICTIONS",
    "FORMATS",
    "HardwareSpec",
    "ModelSpec",
    "Plan",
    "ZooError",
    "compile_plans",
    "dtype_bytes",
    "evidence_table",
    "find_model",
    "find_models",
    "format_speed",
    "human_bytes",
    "human_count",
    "load_zoo",
    "model_table",
    "pareto",
    "plan_risk",
    "plans_table",
    "render_report",
]
