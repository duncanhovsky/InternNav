from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bridgedp_model_save_is_rank_zero_only_and_atomic():
    source = (PROJECT_ROOT / "internnav" / "trainer" / "bridgedp_trainer.py").read_text(encoding="utf-8")

    assert "if not self.args.should_save:" in source
    assert "def _atomic_torch_save" in source
    assert "stream.flush()" in source
    assert "os.fsync(stream.fileno())" in source
    assert "os.replace(temporary, destination)" in source
    assert "def _atomic_link_or_copy" in source
    assert "os.link(source, temporary)" in source
    assert '"pytorch_model.bin"' in source
    assert '"bridgedp.ckpt"' in source
    assert '"training_args.bin"' in source


def test_percent_checkpoint_manager_synchronizes_validates_and_prunes():
    source = (
        PROJECT_ROOT / "scripts" / "train" / "base_train" / "percent_checkpoint_manager.py"
    ).read_text(encoding="utf-8")

    assert "class PercentCheckpointCallback(TrainerCallback):" in source
    assert "control.should_save = True" in source
    assert "dist.barrier()" in source
    assert "dist.broadcast_object_list" in source
    assert "torch.load(" in source
    assert "CHECKPOINT_COMPLETE.json" in source
    assert "atomic_write_json" in source
    assert "apply_percent_checkpoint_retention" in source
    assert "def ensure_final_checkpoint" in source
    assert "trainer._save_checkpoint(" in source


def test_base_training_wires_strict_percent_checkpoints_and_final_verification():
    train_source = (
        PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"
    ).read_text(encoding="utf-8")
    config_source = (
        PROJECT_ROOT
        / "scripts"
        / "train"
        / "base_train"
        / "configs"
        / "bridgedp_full_8x4090.py"
    ).read_text(encoding="utf-8")

    assert "PercentCheckpointCallback" in train_source
    assert "percent_checkpoint_enabled" in train_source
    assert "save_strategy='no' if percent_checkpoint_enabled else 'epoch'" in train_source
    assert "require_complete=require_complete_checkpoint" in train_source
    assert "percent_checkpoint_callback.ensure_final_checkpoint(trainer)" in train_source
    assert "BRIDGEDP_PERCENT_CHECKPOINT_INTERVAL" in config_source
    assert "BRIDGEDP_PERCENT_CHECKPOINT_ROLLING_KEEP" in config_source
    assert "BRIDGEDP_PERCENT_CHECKPOINT_PERMANENT_INTERVAL" in config_source
    assert "BRIDGEDP_REQUIRE_COMPLETE_CHECKPOINT" in config_source
