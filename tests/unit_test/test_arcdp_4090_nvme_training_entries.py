from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


P0_MODELS = [
    "bridgedp_p0_rel",
    "bridgedp_p0_no_bridge",
    "bridgedp_p0_no_ordered_init",
    "bridgedp_p0_no_scale_cond",
    "bridgedp_p0_no_anchor_train",
    "bridgedp_p0_no_gcs",
]


def test_arcdp_4090_full_configs_are_registered():
    configs_init = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "__init__.py"
    train_entry = PROJECT_ROOT / "scripts" / "train" / "base_train" / "train.py"

    configs_text = configs_init.read_text(encoding="utf-8")
    train_text = train_entry.read_text(encoding="utf-8")

    assert "bridgedp_full_8x4090_exp_cfg" in configs_text
    assert "bridgedp_full_4x4090_exp_cfg" in configs_text
    assert "'bridgedp_full_8x4090': [bridgedp_full_8x4090_exp_cfg, \"BridgeDP_Policy\"]" in train_text
    assert "'bridgedp_full_4x4090': [bridgedp_full_4x4090_exp_cfg, \"BridgeDP_Policy\"]" in train_text


def test_arcdp_4090_full_configs_default_to_full10_with_effective_batch_384():
    config_dir = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs"
    expected = {
        "bridgedp_full_8x4090.py": {
            "name": 'BRIDGEDP_RUN_NAME", "arcdp_full10_8x4090_nvme"',
            "gpus": 'BRIDGEDP_NUM_GPUS", 8',
            "batch": 'BRIDGEDP_BATCH_SIZE", 16',
            "accum": 'BRIDGEDP_GRAD_ACCUM", 3',
        },
        "bridgedp_full_4x4090.py": {
            "name": 'BRIDGEDP_RUN_NAME", "arcdp_full10_4x4090_nvme"',
            "gpus": 'BRIDGEDP_NUM_GPUS", 4',
            "batch": 'BRIDGEDP_BATCH_SIZE", 16',
            "accum": 'BRIDGEDP_GRAD_ACCUM", 6',
        },
    }

    for filename, fragments in expected.items():
        text = (config_dir / filename).read_text(encoding="utf-8")
        assert fragments["name"] in text
        assert fragments["gpus"] in text
        assert 'BRIDGEDP_EPOCHS", 10' in text
        assert fragments["batch"] in text
        assert fragments["accum"] in text


def test_p0_common_honors_4090_runtime_overrides():
    p0_common = PROJECT_ROOT / "scripts" / "train" / "base_train" / "configs" / "bridgedp_p0_common.py"
    text = p0_common.read_text(encoding="utf-8")

    assert 'BRIDGEDP_EPOCHS"' in text
    assert 'BRIDGEDP_BATCH_SIZE"' in text
    assert 'BRIDGEDP_GRAD_ACCUM"' in text
    assert 'BRIDGEDP_NUM_WORKERS"' in text


def test_arcdp_4090_nvme_suite_scripts_launch_full10_and_six_p0_one_epoch():
    script_dir = PROJECT_ROOT / "scripts" / "train" / "arcdp_4090"
    expected = {
        "train_arcdp_full10_p0_1ep_8x4090_nvme.sh": {
            "full_model": "--model-name bridgedp_full_8x4090",
            "gpus": "BRIDGEDP_NUM_GPUS:-8",
            "cuda": "CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7",
            "accum": "BRIDGEDP_GRAD_ACCUM:-3",
        },
        "train_arcdp_full10_p0_1ep_4x4090_nvme.sh": {
            "full_model": "--model-name bridgedp_full_4x4090",
            "gpus": "BRIDGEDP_NUM_GPUS:-4",
            "cuda": "CUDA_VISIBLE_DEVICES:-0,1,2,3",
            "accum": "BRIDGEDP_GRAD_ACCUM:-6",
        },
    }

    for filename, fragments in expected.items():
        text = (script_dir / filename).read_text(encoding="utf-8")
        assert "ARCDP_NVME_ROOT:-/nvme" in text
        assert "BRIDGEDP_DATASET_ROOT" in text
        assert "v0.5-full-vln-n1/vln_n1/traj_data" in text
        assert "BRIDGEDP_EPOCHS=10" in text
        assert "BRIDGEDP_EPOCHS=1" in text
        assert "BRIDGEDP_BATCH_SIZE:-16" in text
        assert fragments["gpus"] in text
        assert fragments["cuda"] in text
        assert fragments["accum"] in text
        assert fragments["full_model"] in text
        assert "cache_rotation" not in text
        for model_name in P0_MODELS:
            assert f"--model-name {model_name}" in text


def test_arcdp_4090_ten_day_suite_runs_full1_and_no_bridge1_with_swanlab():
    script = (
        PROJECT_ROOT
        / "scripts"
        / "train"
        / "arcdp_4090"
        / "train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert 'BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-8}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in text
    assert 'BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-48}"' in text
    assert 'BRIDGEDP_GRAD_ACCUM="${BRIDGEDP_GRAD_ACCUM:-1}"' in text
    assert 'BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"' in text
    assert "unset SWANLAB_PROJECT" in text
    assert "export SWANLAB_PROJECT=" not in text
    assert 'export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT_VALUE}"' in text
    assert 'echo "  swanlab project: ${SWANLAB_PROJ_NAME}"' in text
    assert "ArcDP-8x4090-full1-no-bridge" in text
    assert 'python -c "import swanlab"' in text
    assert 'python -c "import swanlab" >/dev/null 2>&1' not in text
    assert text.count("export BRIDGEDP_EPOCHS=1") == 2
    assert "--model-name bridgedp_full_8x4090" in text
    assert "--model-name bridgedp_p0_no_bridge" in text
    assert "bridgedp_p0_rel" not in text
    assert "bridgedp_p0_no_ordered_init" not in text
    assert "bridgedp_p0_no_scale_cond" not in text
    assert "bridgedp_p0_no_anchor_train" not in text
    assert "bridgedp_p0_no_gcs" not in text


def test_arcdp_4090_root_wrappers_point_to_nvme_suite_scripts():
    expected = {
        "train_arcdp_full10_p0_1ep_8x4090_nvme.sh": "scripts/train/arcdp_4090/train_arcdp_full10_p0_1ep_8x4090_nvme.sh",
        "train_arcdp_full10_p0_1ep_4x4090_nvme.sh": "scripts/train/arcdp_4090/train_arcdp_full10_p0_1ep_4x4090_nvme.sh",
        "train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh": "scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh",
    }

    for wrapper, target in expected.items():
        text = (PROJECT_ROOT / wrapper).read_text(encoding="utf-8")
        assert target in text
