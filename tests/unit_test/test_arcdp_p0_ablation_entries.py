from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


P0_CONFIGS = {
    "bridgedp_p0_rel": "bridgedp_p0_rel_exp_cfg",
    "bridgedp_p0_no_bridge": "bridgedp_p0_no_bridge_exp_cfg",
    "bridgedp_p0_no_ordered_init": "bridgedp_p0_no_ordered_init_exp_cfg",
    "bridgedp_p0_no_scale_cond": "bridgedp_p0_no_scale_cond_exp_cfg",
    "bridgedp_p0_no_anchor_train": "bridgedp_p0_no_anchor_train_exp_cfg",
    "bridgedp_p0_no_gcs": "bridgedp_p0_no_gcs_exp_cfg",
}


def test_arcdp_p0_configs_are_registered_without_reusing_full_entry():
    configs_init = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "__init__.py"
    train_entry = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"

    configs_text = configs_init.read_text(encoding="utf-8")
    train_text = train_entry.read_text(encoding="utf-8")

    for model_name, cfg_var in P0_CONFIGS.items():
        assert cfg_var in configs_text
        assert f"'{model_name}': [{cfg_var}, \"BridgeDP_Policy\"]" in train_text


def test_arcdp_p0_configs_use_independent_checkpoint_root_and_run_names():
    config_dir = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs"

    for model_name, _ in P0_CONFIGS.items():
        text = (config_dir / f"{model_name}.py").read_text(encoding="utf-8")
        assert "ARCDP_P0_CHECKPOINT_ROOT" in text
        assert f"arcdp_p0_{model_name.removeprefix('bridgedp_p0_')}" in text
        assert "bridgedp_full" not in text


def test_arcdp_p0_train_scripts_select_matching_config_and_separate_root():
    script_dir = PROJECT_ROOT / "scripts" / "train" / "bridgedp_ablation"

    for model_name in P0_CONFIGS:
        script = script_dir / f"train_{model_name}.sh"
        text = script.read_text(encoding="utf-8")
        assert f"--model-name {model_name}" in text
        assert "ARCDP_P0_CHECKPOINT_ROOT" in text
        assert "checkpoints/arcdp_p0" in text


def test_bridgedp_policy_exposes_p0_ablation_switches():
    policy_path = PROJECT_ROOT / "internnav" / "model" / "basemodel" / "bridgedp" / "bridgedp_policy.py"
    text = policy_path.read_text(encoding="utf-8")

    assert "ablation_prediction_space" in text
    assert "ablation_diffusion_mode" in text
    assert "ablation_initialization_mode" in text
    assert "DDPMScheduler" in text
    assert "_trajectory_to_relative_delta" in text
    assert "_relative_delta_to_trajectory" in text
