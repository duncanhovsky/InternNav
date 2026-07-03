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


def test_arcdp_cache_rotation_config_supports_full_and_p0_variants():
    config = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "bridgedp_cache_rotation.py"
    text = config.read_text(encoding="utf-8")

    assert "ARCDP_CACHE_VARIANT" in text
    for variant in [
        "full",
        "rel",
        "no_bridge",
        "no_ordered_init",
        "no_scale_cond",
        "no_anchor_train",
        "no_gcs",
    ]:
        assert f'"{variant}"' in text

    assert '"ablation_prediction_space": "relative_delta"' in text
    assert '"ablation_diffusion_mode": "ddpm"' in text
    assert '"enable_scale_condition_token": False' in text
    assert '"enable_bridge_anchor_sampling": False' in text
    assert '"enable_goal_consistency_score": False' in text


def test_arcdp_cache_rotation_launchers_cover_gpu_epoch_matrix_and_swanlab():
    script_dir = PROJECT_ROOT / "scripts" / "train" / "arcdp_cache_rotation"
    generic = script_dir / "train_arcdp_cache_rotation.sh"
    gpu4 = script_dir / "train_arcdp_cache_rotation_4a800.sh"
    gpu8 = script_dir / "train_arcdp_cache_rotation_8a800.sh"
    matrix = script_dir / "print_arcdp_cache_rotation_matrix.sh"

    for path in [generic, gpu4, gpu8, matrix]:
        assert path.exists(), f"missing launcher: {path}"

    generic_text = generic.read_text(encoding="utf-8")
    assert "--epochs 100|200|500|1000" in generic_text
    assert "ARCDP_CACHE_VARIANT" in generic_text
    assert "BRIDGEDP_REPORT_TO" in generic_text
    assert "swanlab" in generic_text
    assert "SWANLAB_PROJECT" in generic_text
    assert "SWANLAB_PROJ_NAME" in generic_text
    assert "SWANLAB_API_KEY" in generic_text
    assert "swanlab login" in generic_text
    assert "SwanLab enabled" in generic_text
    assert "SwanLab project" in generic_text
    assert "BRIDGEDP_ETA_LOG_STEPS" in generic_text
    assert "train_bridgedp_cache_rotation.sh" in generic_text

    assert '--gpus "4"' in gpu4.read_text(encoding="utf-8")
    assert '--gpus "8"' in gpu8.read_text(encoding="utf-8")

    matrix_text = matrix.read_text(encoding="utf-8")
    for epoch in ["100", "200", "500", "1000"]:
        assert epoch in matrix_text
    for variant in ["full", "rel", "no_bridge", "no_ordered_init", "no_scale_cond", "no_anchor_train", "no_gcs"]:
        assert variant in matrix_text


def test_detailed_progress_callback_exposes_low_frequency_eta_metrics():
    train_py = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"
    text = train_py.read_text(encoding="utf-8")

    assert "BRIDGEDP_ETA_LOG_STEPS" in text
    assert "time/eta_hours" in text
    assert "progress/percent" in text
    assert "[ETA]" in text


def test_arcdp_training_readiness_checker_guides_missing_prereqs():
    checker = PROJECT_ROOT / "scripts" / "train" / "arcdp_cache_rotation" / "check_arcdp_training_ready.sh"
    assert checker.exists(), f"missing readiness checker: {checker}"

    text = checker.read_text(encoding="utf-8")
    assert "set -uo pipefail" in text
    assert "full scan" in text
    assert "does not stop at the first failed check" in text
    assert "--gpus 4|8" in text
    assert "--variant full|rel|no_bridge|no_ordered_init|no_scale_cond|no_anchor_train|no_gcs" in text
    assert "--epochs 100|200|500|1000" in text
    assert "prepare_bridgedp_cache_rotation.sh" in text
    assert "train_arcdp_cache_rotation" in text
    assert "SWANLAB_API_KEY" in text
    assert "swanlab login" in text
    assert "torchrun" in text
    assert "nvidia-smi" in text
    assert "check_bridgedp_cache.py" in text
    assert "depth_anything_v2_vits.pth" in text
    assert "manifest" in text
    assert "cache_A" in text
    assert "cache_B" in text


def test_swanlab_is_part_of_setup_and_cache_preparation_guidance():
    setup = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    prepare = PROJECT_ROOT / "scripts" / "cache_rotation" / "prepare_bridgedp_cache_rotation.sh"

    setup_text = setup.read_text(encoding="utf-8")
    prepare_text = prepare.read_text(encoding="utf-8")

    assert "configure_swanlab" in setup_text
    assert "pip install \"swanlab" in setup_text
    assert "SWANLAB_API_KEY" in setup_text
    assert "swanlab login" in setup_text
    assert "ArcDP-cache-rotation" in setup_text

    assert "check_swanlab_setup" in prepare_text
    assert "SWANLAB_API_KEY" in prepare_text
    assert "swanlab login" in prepare_text
    assert "check_arcdp_training_ready.sh" in prepare_text
    assert "full prerequisite scan" in prepare_text
