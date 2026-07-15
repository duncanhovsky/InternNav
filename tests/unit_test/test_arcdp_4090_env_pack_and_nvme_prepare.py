from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cpu_env_pack_script_builds_arcdp_conda_pack_from_pytorch270_image():
    script = PROJECT_ROOT / "scripts" / "arcdp_pack_training_env_cpu.sh"
    text = script.read_text(encoding="utf-8")

    assert 'ENV_NAME="${ENV_NAME:-arcdp}"' in text
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-/root/data/conda_env_packs}"' in text
    assert "setup_internnav_pytorch270_cu126.sh" in text
    assert "pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04" in text
    assert 'INTERNNAV_INSTALL_GIT_DEPS="${INTERNNAV_INSTALL_GIT_DEPS:-skip}"' in text
    assert 'PACKAGING_VERSION_SPEC="${PACKAGING_VERSION_SPEC:-packaging>=23.0,<25}"' in text
    assert "conda_pack" in text
    assert '"${PACKAGING_VERSION_SPEC}" conda-pack' in text
    assert "torch.version.cuda" in text
    assert "EXPECTED_TORCH_CUDA" in text
    assert "arcdp_deploy_training_env_gpu.sh" in text


def test_gpu_env_deploy_script_unpacks_without_network_install():
    script = PROJECT_ROOT / "scripts" / "arcdp_deploy_training_env_gpu.sh"
    text = script.read_text(encoding="utf-8")

    assert 'ENV_NAME="${ENV_NAME:-arcdp}"' in text
    assert "conda info --base" in text
    assert "TARGET_PREFIX" in text
    assert "conda-unpack" in text
    assert "torch.cuda.is_available" in text
    assert "BridgeDP_Policy" in text
    assert "pip install" not in text
    assert "conda install" not in text


def test_nvme_prepare_script_extracts_root_data_full_dataset_without_cache_rotation():
    script = PROJECT_ROOT / "scripts" / "arcdp_prepare_full_nvme_from_root_data.sh"
    text = script.read_text(encoding="utf-8")

    assert 'SOURCE_ROOT="${SOURCE_ROOT:-/root/data}"' in text
    assert 'NVME_ROOT="${NVME_ROOT:-/nvme}"' in text
    assert "MIN_FREE_INODES" in text
    assert "100000000" in text
    assert "prepare_ssd_full.sh" in text
    assert "v0.5-full-vln-n1/vln_n1/traj_data" in text
    assert "CONDA_ENV" in text
    assert "cache_rotation" not in text
    assert "sharp" not in text.lower()


def test_operation_doc_references_env_pack_nvme_prepare_and_4090_training():
    doc = PROJECT_ROOT / "docs" / "ArcDP-4090-CPU环境打包-NVMe全量数据训练操作说明.md"
    text = doc.read_text(encoding="utf-8")

    assert "pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04" in text
    assert "arcdp_pack_training_env_cpu.sh" in text
    assert "arcdp_deploy_training_env_gpu.sh" in text
    assert "arcdp_prepare_full_nvme_from_root_data.sh" in text
    assert "train_arcdp_full10_p0_1ep_8x4090_nvme.sh" in text
    assert "/root/data" in text
    assert "/nvme" in text
    assert "不使用 sharp" in text
    assert "不使用 cache rotation" in text
