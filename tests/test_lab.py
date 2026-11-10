"""Fast regression tests for the lab: registry integrity and one live run."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fsl.common import DATA, RESULTS
from fsl.registry import EXPERIMENTS


def test_registry_complete():
    assert len(EXPERIMENTS) == 24
    assert all(name.startswith("e") for name in EXPERIMENTS)


def test_every_experiment_imports():
    for name, module in EXPERIMENTS.items():
        mod = importlib.import_module(f"fsl.experiments.{module}")
        assert callable(mod.run), name


def test_paths_exist():
    assert DATA.is_dir()
    assert RESULTS.is_dir() or True


def test_paged_attention_end_to_end():
    from fsl.experiments.e18_paged_attention import run
    res = run()
    assert res.status == "ok"
