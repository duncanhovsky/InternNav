from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_setup_script_targets_pytorch270_cuda126_python310():
    script = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    text = script.read_text(encoding="utf-8")

    assert "ENV_NAME=\"${ENV_NAME:-base}\"" in text
    assert "USE_CONDA=\"${USE_CONDA:-auto}\"" in text
    assert "PYTHON_VERSION=\"${PYTHON_VERSION:-3.10}\"" in text
    assert "TORCH_VERSION=\"${TORCH_VERSION:-2.7.0}\"" in text
    assert "TORCHVISION_VERSION=\"${TORCHVISION_VERSION:-0.22.0}\"" in text
    assert "TORCHAUDIO_VERSION=\"${TORCHAUDIO_VERSION:-2.7.0}\"" in text
    assert "https://download.pytorch.org/whl/cu126" in text


def test_setup_script_avoids_known_torch27_conflict_pins():
    script = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    text = script.read_text(encoding="utf-8")

    assert "flash_attn" in text
    assert "triton" in text
    assert "sympy" in text
    assert "huggingface-hub" in text
    assert "grep -Ev '^(flash_attn|triton|sympy|huggingface-hub)=='" in text


def test_setup_script_reuses_matching_prebuilt_environment():
    script = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    text = script.read_text(encoding="utf-8")

    assert "current_env_matches_base_stack()" in text
    assert "torch_stack_matches()" in text
    assert "Current Python already matches Python ${PYTHON_VERSION}, torch ${TORCH_VERSION}, CUDA 12.6; reuse it." in text
    assert "Requested PyTorch stack is already installed; skip PyTorch reinstall." in text


def test_setup_script_prevents_requirements_from_resolving_other_torch_versions():
    script = PROJECT_ROOT / "scripts" / "setup_internnav_pytorch270_cu126.sh"
    text = script.read_text(encoding="utf-8")

    assert "make_torch_constraints()" in text
    assert "require_torch_stack_ready()" in text
    assert "torch==${TORCH_VERSION}" in text
    assert "torchvision==${TORCHVISION_VERSION}" in text
    assert "torchaudio==${TORCHAUDIO_VERSION}" in text
    assert "sympy==${SYMPY_VERSION}" in text
    assert "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" in text
    assert "uvicorn==${UVICORN_VERSION}" in text
    assert "fastapi==${FASTAPI_VERSION}" in text
    assert "starlette==${STARLETTE_VERSION}" in text
    assert "dash==${DASH_VERSION}" in text
    assert "Flask==${FLASK_VERSION}" in text
    assert "python -m pip install -c \"${TORCH_CONSTRAINTS}\" -r requirements/core_requirements.txt" in text
    assert "python -m pip install -c \"${TORCH_CONSTRAINTS}\" -r \"${model_req_tmp}\"" in text
