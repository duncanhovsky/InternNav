import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "train" / "base_train" / "progress_metrics.py"


def _load_metrics_module():
    spec = importlib.util.spec_from_file_location("arcdp_progress_metrics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch48_profile_reports_true_global_throughput():
    metrics = _load_metrics_module().compute_progress_metrics(
        global_step=10,
        max_steps=100,
        elapsed_seconds=20.0,
        per_device_batch_size=48,
        world_size=8,
        gradient_accumulation_steps=1,
        start_global_step=0,
        dataset_size=38_400,
    )

    assert metrics["effective_global_batch"] == 384
    assert metrics["samples_completed"] == 3_840
    assert metrics["samples_per_second"] == 192.0
    assert metrics["avg_step_time"] == 2.0
    assert metrics["eta_seconds"] == 180.0
    assert metrics["steps_per_epoch"] == 100


def test_batch24_accum2_profile_keeps_effective_global_batch_384():
    metrics = _load_metrics_module().compute_progress_metrics(
        global_step=10,
        max_steps=100,
        elapsed_seconds=20.0,
        per_device_batch_size=24,
        world_size=8,
        gradient_accumulation_steps=2,
        start_global_step=0,
        dataset_size=38_400,
    )

    assert metrics["effective_global_batch"] == 384
    assert metrics["samples_per_second"] == 192.0
    assert metrics["avg_step_time"] == 2.0


def test_resumed_run_uses_only_new_steps_for_runtime_throughput():
    metrics = _load_metrics_module().compute_progress_metrics(
        global_step=60,
        max_steps=100,
        elapsed_seconds=20.0,
        per_device_batch_size=48,
        world_size=8,
        gradient_accumulation_steps=1,
        start_global_step=50,
        dataset_size=38_400,
    )

    assert metrics["optimizer_steps_this_run"] == 10
    assert metrics["samples_completed"] == 23_040
    assert metrics["samples_this_run"] == 3_840
    assert metrics["samples_per_second"] == 192.0
    assert metrics["avg_step_time"] == 2.0
    assert metrics["eta_seconds"] == 80.0
    assert metrics["progress_pct"] == 60.0
