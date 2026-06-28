from pathlib import Path


def _checkpoint(root: Path, step: int, *, trainer_state: bool = True) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if trainer_state:
        (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    return path


def test_auto_resume_returns_none_when_output_dir_has_no_valid_checkpoint(tmp_path):
    from scripts.train.base_train.resume_utils import resolve_resume_checkpoint

    output_dir = tmp_path / "ckpts"
    assert resolve_resume_checkpoint(output_dir, requested="", auto_resume=True) is None

    output_dir.mkdir()
    _checkpoint(output_dir, 100, trainer_state=False)
    assert resolve_resume_checkpoint(output_dir, requested="auto", auto_resume=False) is None


def test_auto_resume_uses_latest_valid_checkpoint(tmp_path):
    from scripts.train.base_train.resume_utils import resolve_resume_checkpoint

    output_dir = tmp_path / "ckpts"
    _checkpoint(output_dir, 5)
    expected = _checkpoint(output_dir, 25)
    _checkpoint(output_dir, 50, trainer_state=False)

    assert resolve_resume_checkpoint(output_dir, requested="auto", auto_resume=False) == str(expected)


def test_explicit_missing_checkpoint_can_fall_back_to_fresh_start(tmp_path):
    from scripts.train.base_train.resume_utils import resolve_resume_checkpoint

    missing = tmp_path / "checkpoint-999"

    assert resolve_resume_checkpoint(tmp_path / "ckpts", requested=str(missing), auto_resume=False) is None


def test_explicit_none_disables_auto_resume(tmp_path):
    from scripts.train.base_train.resume_utils import resolve_resume_checkpoint

    output_dir = tmp_path / "ckpts"
    _checkpoint(output_dir, 30)

    assert resolve_resume_checkpoint(output_dir, requested="none", auto_resume=True) is None


def test_legacy_bridgedp_checkpoint_is_mirrored_to_transformers_weight_name(tmp_path):
    from scripts.train.base_train.resume_utils import ensure_checkpoint_model_weight

    checkpoint_dir = _checkpoint(tmp_path / "ckpts", 10)
    bridge_ckpt = checkpoint_dir / "bridgedp.ckpt"
    hf_ckpt = checkpoint_dir / "pytorch_model.bin"
    bridge_ckpt.write_bytes(b"legacy-weights")

    ensure_checkpoint_model_weight(checkpoint_dir)

    assert hf_ckpt.read_bytes() == b"legacy-weights"
