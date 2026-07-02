from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cache_rotation_training_entry_is_independent_from_epoch_train_path():
    train_py = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"
    cache_train_py = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train_cache_rotation.py"

    train_text = train_py.read_text(encoding="utf-8")
    cache_text = cache_train_py.read_text(encoding="utf-8")

    assert "StageStopCallback" not in train_text
    assert "BRIDGEDP_STAGE_END_STEP" not in train_text
    assert "class StageStopCallback" in cache_text
    assert "max_steps=total_max_steps" in cache_text
    assert "ignore_data_skip=ignore_data_skip" in cache_text
    assert "bridgedp_cache_rotation_exp_cfg" in cache_text


def test_cache_rotation_config_and_shell_entrypoints_exist():
    config = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "bridgedp_cache_rotation.py"
    train_shell = PROJECT_ROOT / "train_bridgedp_cache_rotation.sh"
    prepare_shell = PROJECT_ROOT / "scripts" / "cache_rotation" / "prepare_bridgedp_cache_rotation.sh"

    config_text = config.read_text(encoding="utf-8")
    train_shell_text = train_shell.read_text(encoding="utf-8")
    prepare_shell_text = prepare_shell.read_text(encoding="utf-8")

    assert 'BRIDGEDP_PROJECT_ROOT", "/nvme/MyResearch/InternNav"' in config_text
    assert "BRIDGEDP_TOTAL_MAX_STEPS" in config_text
    assert "BRIDGEDP_STAGE_END_STEP" in config_text
    assert "--gpus" in train_shell_text
    assert "--nvme-size" in train_shell_text
    assert "train_cache_rotation.py" in train_shell_text
    assert "make_bridgedp_shards.py" in prepare_shell_text
    assert "build_bridgedp_cache_shard.py" in prepare_shell_text
