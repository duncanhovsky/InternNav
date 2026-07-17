import json


def test_percent_targets_cover_every_integer_percent_and_exact_final_step():
    from scripts.train.base_train.percent_checkpoint_utils import compute_percent_checkpoint_targets

    targets = compute_percent_checkpoint_targets(
        total_steps=25_590,
        interval_percent=1,
        permanent_interval_percent=10,
    )

    assert len(targets) == 100
    assert (targets[0].percent, targets[0].step, targets[0].permanent) == (1, 256, False)
    assert (targets[9].percent, targets[9].step, targets[9].permanent) == (10, 2_559, True)
    assert (targets[-1].percent, targets[-1].step, targets[-1].permanent) == (100, 25_590, True)
    assert [target.step for target in targets] == sorted({target.step for target in targets})


def test_retention_keeps_all_permanent_and_only_latest_five_rolling(tmp_path):
    from scripts.train.base_train.percent_checkpoint_utils import (
        COMPLETE_MARKER_FILE,
        apply_percent_checkpoint_retention,
    )

    output_dir = tmp_path / "ckpts"
    output_dir.mkdir()
    checkpoints = {}
    for percent in range(1, 13):
        step = percent * 100
        checkpoint = output_dir / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / COMPLETE_MARKER_FILE).write_text(
            json.dumps(
                {
                    "global_step": step,
                    "percent": percent,
                    "permanent": percent % 10 == 0,
                    "required_files": [],
                }
            ),
            encoding="utf-8",
        )
        checkpoints[percent] = checkpoint

    incomplete = output_dir / "checkpoint-9999"
    incomplete.mkdir()

    removed = apply_percent_checkpoint_retention(output_dir, rolling_keep=5)

    assert {path.name for path in removed} == {
        "checkpoint-100",
        "checkpoint-200",
        "checkpoint-300",
        "checkpoint-400",
        "checkpoint-500",
        "checkpoint-600",
    }
    assert checkpoints[10].is_dir()
    assert {percent for percent, path in checkpoints.items() if path.exists()} == {7, 8, 9, 10, 11, 12}
    assert incomplete.is_dir()


def test_retention_manifest_lists_completed_checkpoints(tmp_path):
    from scripts.train.base_train.percent_checkpoint_utils import (
        COMPLETE_MARKER_FILE,
        MANIFEST_FILE,
        apply_percent_checkpoint_retention,
    )

    output_dir = tmp_path / "ckpts"
    checkpoint = output_dir / "checkpoint-1000"
    checkpoint.mkdir(parents=True)
    (checkpoint / COMPLETE_MARKER_FILE).write_text(
        json.dumps(
            {
                "global_step": 1000,
                "percent": 10,
                "permanent": True,
                "required_files": ["pytorch_model.bin"],
            }
        ),
        encoding="utf-8",
    )

    apply_percent_checkpoint_retention(output_dir, rolling_keep=5)

    manifest = json.loads((output_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["completed_count"] == 1
    assert manifest["checkpoints"][0]["path"] == "checkpoint-1000"
    assert manifest["checkpoints"][0]["permanent"] is True
