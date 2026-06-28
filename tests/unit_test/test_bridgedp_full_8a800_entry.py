from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bridgedp_full_8a800_config_is_registered():
    configs_init = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "__init__.py"
    train_entry = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"

    assert "bridgedp_full_8a800_exp_cfg" in configs_init.read_text(encoding="utf-8")
    assert "'bridgedp_full_8a800': [bridgedp_full_8a800_exp_cfg, \"BridgeDP_Policy\"]" in train_entry.read_text(
        encoding="utf-8"
    )


def test_bridgedp_full_8a800_defaults_are_for_eight_a800s():
    config_path = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "bridgedp_full_8a800.py"
    script_path = PROJECT_ROOT / "train_bridgedp_full_8a800.sh"

    config_text = config_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")

    assert 'BRIDGEDP_RUN_NAME", "bridgedp_full_8a800"' in config_text
    assert 'BRIDGEDP_NUM_GPUS", 8' in config_text
    assert 'BRIDGEDP_BATCH_SIZE", 96' in config_text
    assert 'BRIDGEDP_NUM_GPUS:-8' in script_text
    assert 'CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7' in script_text
    assert "--model-name bridgedp_full_8a800" in script_text
