from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pack_base_conda_env_script_is_offline_archive_oriented():
    script = PROJECT_ROOT / "scripts" / "pack_base_conda_env.sh"
    text = script.read_text(encoding="utf-8")

    assert 'ENV_NAME="${ENV_NAME:-base}"' in text
    assert "info --base" in text
    assert "conda-pack" in text
    assert "raw-tar" in text
    assert "REQUIRE_CUDA_TORCH" in text
    assert "torch.version.cuda" in text
    assert "--exclude='./pkgs'" in text
    assert "--exclude='./envs'" in text
    assert "pip install" not in text
    assert "conda install" not in text


def test_deploy_base_conda_env_script_restores_into_base_without_network():
    script = PROJECT_ROOT / "scripts" / "deploy_base_conda_env.sh"
    text = script.read_text(encoding="utf-8")

    assert 'ENV_NAME="${ENV_NAME:-base}"' in text
    assert "info --base" in text
    assert "conda-unpack" in text
    assert "YES=\"${YES:-0}\"" in text
    assert "TARGET_PREFIX" in text
    assert "Refusing to deploy into unsafe TARGET_PREFIX" in text
    assert "pip install" not in text
    assert "conda install" not in text
