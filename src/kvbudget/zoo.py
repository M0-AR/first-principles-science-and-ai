"""Public model zoo loading and lookup."""
from __future__ import annotations

import json
from pathlib import Path

from .model import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZOO = PROJECT_ROOT / "data" / "model_zoo.json"


class ZooError(LookupError):
    pass


def load_zoo(path: str | Path = DEFAULT_ZOO) -> list[ModelSpec]:
    zoo_path = Path(path)
    if not zoo_path.exists():
        raise ZooError(
            f"model zoo not found at {zoo_path}; run scripts/fetch_model_zoo.py first"
        )
    raw = json.loads(zoo_path.read_text())
    models = [ModelSpec.from_dict(item) for item in raw["models"]]
    if not models:
        raise ZooError(f"model zoo at {zoo_path} is empty")
    return models


def find_models(models: list[ModelSpec], query: str) -> list[ModelSpec]:
    exact = [m for m in models if m.id == query]
    if exact:
        return exact
    needle = query.lower()
    matches = [m for m in models if needle in m.id.lower()]
    if not matches:
        available = ", ".join(m.id for m in models)
        raise ZooError(f"no model matching {query!r}; available: {available}")
    return matches


def find_model(models: list[ModelSpec], query: str) -> ModelSpec:
    matches = find_models(models, query)
    if len(matches) > 1:
        names = ", ".join(m.id for m in matches)
        raise ZooError(f"{query!r} is ambiguous; matches: {names}")
    return matches[0]
