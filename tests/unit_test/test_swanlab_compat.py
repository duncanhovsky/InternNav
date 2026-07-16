import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts.train.base_train.swanlab_compat import initialize_swanlab_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _install_fake_swanlab(monkeypatch, get_run_result=None, get_run_error=None):
    calls = {"get_run": 0, "init": []}
    module = ModuleType("swanlab")

    def get_run():
        calls["get_run"] += 1
        if get_run_error is not None:
            raise get_run_error
        return get_run_result

    def init(**kwargs):
        calls["init"].append(kwargs)
        return "new-run"

    module.get_run = get_run
    module.init = init
    monkeypatch.setitem(sys.modules, "swanlab", module)
    return calls


def test_initialize_swanlab_run_skips_disabled_reporting(monkeypatch):
    calls = _install_fake_swanlab(monkeypatch)

    result = initialize_swanlab_run("tensorboard", is_main_process=True, run_name="run")

    assert result is None
    assert calls == {"get_run": 0, "init": []}


def test_initialize_swanlab_run_skips_non_main_process(monkeypatch):
    calls = _install_fake_swanlab(monkeypatch)

    result = initialize_swanlab_run("swanlab", is_main_process=False, run_name="run")

    assert result is None
    assert calls == {"get_run": 0, "init": []}


def test_initialize_swanlab_run_reuses_active_run(monkeypatch):
    active_run = object()
    calls = _install_fake_swanlab(monkeypatch, get_run_result=active_run)

    result = initialize_swanlab_run("swanlab", is_main_process=True, run_name="run")

    assert result is active_run
    assert calls == {"get_run": 1, "init": []}


def test_initialize_swanlab_run_initializes_when_get_run_returns_none(monkeypatch):
    calls = _install_fake_swanlab(monkeypatch, get_run_result=None)
    monkeypatch.setenv("SWANLAB_PROJ_NAME", "ArcDP-project")
    monkeypatch.setenv("SWANLAB_EXP_NAME", "ArcDP-experiment")

    result = initialize_swanlab_run("swanlab", is_main_process=True, run_name="fallback")

    assert result == "new-run"
    assert calls["init"] == [
        {"project": "ArcDP-project", "experiment_name": "ArcDP-experiment"}
    ]


def test_initialize_swanlab_run_handles_swanlab_085_no_active_run(monkeypatch):
    calls = _install_fake_swanlab(
        monkeypatch,
        get_run_error=RuntimeError("No active Run. Call swanlab.init() first."),
    )
    monkeypatch.setenv("SWANLAB_PROJ_NAME", "ArcDP-project")
    monkeypatch.delenv("SWANLAB_EXP_NAME", raising=False)

    result = initialize_swanlab_run("swanlab", is_main_process=True, run_name="fallback")

    assert result == "new-run"
    assert calls["init"] == [
        {"project": "ArcDP-project", "experiment_name": "fallback"}
    ]


def test_initialize_swanlab_run_does_not_hide_other_runtime_errors(monkeypatch):
    calls = _install_fake_swanlab(
        monkeypatch,
        get_run_error=RuntimeError("SwanLab backend failed"),
    )

    with pytest.raises(RuntimeError, match="SwanLab backend failed"):
        initialize_swanlab_run("swanlab", is_main_process=True, run_name="run")

    assert calls["init"] == []


def test_training_entry_initializes_swanlab_before_trainer_train():
    text = (PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py").read_text(
        encoding="utf-8"
    )
    call = """initialize_swanlab_run(
            config.il.report_to,
            is_main_process=is_main_process,
            run_name=config.name,
        )"""

    assert (
        "from scripts.train.base_train.swanlab_compat import initialize_swanlab_run"
        in text
    )
    assert call in text
    assert text.index(call) < text.index(
        "trainer.train(resume_from_checkpoint=resume_checkpoint)"
    )
