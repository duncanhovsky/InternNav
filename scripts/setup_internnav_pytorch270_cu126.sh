#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-base}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0}"
SYMPY_VERSION="${SYMPY_VERSION:-1.13.3}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-0.34.4}"
FASTAPI_VERSION="${FASTAPI_VERSION:-0.110.0}"
STARLETTE_VERSION="${STARLETTE_VERSION:-0.36.3}"
UVICORN_VERSION="${UVICORN_VERSION:-0.30.6}"
DASH_VERSION="${DASH_VERSION:-2.18.2}"
FLASK_VERSION="${FLASK_VERSION:-3.0.3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

INSTALL_APT="${INSTALL_APT:-1}"
USE_CONDA="${USE_CONDA:-auto}"
INTERNNAV_INSTALL_FLASH_ATTN="${INTERNNAV_INSTALL_FLASH_ATTN:-try}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV_ACTIVATED=0
TORCH_CONSTRAINTS=""

log() {
    echo "[InternNav setup] $*"
}

warn() {
    echo "[InternNav setup][WARN] $*" >&2
}

run_apt_install() {
    if [[ "${INSTALL_APT}" != "1" ]]; then
        log "Skip apt packages because INSTALL_APT=${INSTALL_APT}"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found; skip system packages."
        return
    fi

    local apt_prefix=()
    if [[ "${EUID}" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            apt_prefix=(sudo)
        else
            warn "Not root and sudo not found; skip apt packages."
            return
        fi
    fi

    log "Install Ubuntu system packages"
    "${apt_prefix[@]}" apt-get update
    "${apt_prefix[@]}" apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        git-lfs \
        libegl1 \
        libglib2.0-0 \
        libgl1 \
        libglvnd0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxkbcommon0 \
        libxcb1 \
        libxcb-glx0 \
        ninja-build \
        pkg-config \
        python3-dev \
        wget
    git lfs install --skip-repo || true
}

activate_python_env() {
    if [[ "${USE_CONDA}" == "auto" ]]; then
        if current_env_matches_base_stack; then
            log "Current Python already matches Python ${PYTHON_VERSION}, torch ${TORCH_VERSION}, CUDA 12.6; reuse it."
            return
        fi
        log "Current Python does not match requested stack; use/create conda env ${ENV_NAME}."
    elif [[ "${USE_CONDA}" != "1" ]]; then
        log "Use current Python because USE_CONDA=${USE_CONDA}"
        return
    fi

    local conda_sh=""
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        conda_sh="${conda_base}/etc/profile.d/conda.sh"
    else
        for candidate in \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [[ -f "${candidate}" ]]; then
                conda_sh="${candidate}"
                break
            fi
        done
    fi

    if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
        warn "conda not found; use current Python environment."
        return
    fi

    # shellcheck disable=SC1090
    source "${conda_sh}"

    if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        log "Create conda env ${ENV_NAME} with Python ${PYTHON_VERSION}"
        conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
    else
        log "Use existing conda env ${ENV_NAME}"
    fi
    conda activate "${ENV_NAME}"
    CONDA_ENV_ACTIVATED=1
}

current_env_matches_base_stack() {
    python - "${PYTHON_VERSION}" "${TORCH_VERSION}" <<'PY' >/dev/null 2>&1
import sys

want_python, want_torch = sys.argv[1:3]
if f"{sys.version_info.major}.{sys.version_info.minor}" != want_python:
    raise SystemExit(1)

try:
    import torch
except Exception:
    raise SystemExit(1)

torch_version = torch.__version__.split("+", 1)[0]
torch_cuda = str(torch.version.cuda or "")
if torch_version != want_torch or not torch_cuda.startswith("12.6"):
    raise SystemExit(1)
PY
}

torch_stack_matches() {
    python - "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${TORCHAUDIO_VERSION}" <<'PY' >/dev/null 2>&1
import importlib.metadata as metadata
import sys

want_torch, want_torchvision, want_torchaudio = sys.argv[1:4]

try:
    import torch
except Exception:
    raise SystemExit(1)

def normalize(version):
    return version.split("+", 1)[0]

if normalize(torch.__version__) != want_torch:
    raise SystemExit(1)
if not str(torch.version.cuda or "").startswith("12.6"):
    raise SystemExit(1)

for package, wanted in (("torchvision", want_torchvision), ("torchaudio", want_torchaudio)):
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        raise SystemExit(1)
    if normalize(installed) != wanted:
        raise SystemExit(1)
PY
}

install_torch_stack() {
    if torch_stack_matches; then
        log "Requested PyTorch stack is already installed; skip PyTorch reinstall."
        return
    fi

    log "Install PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} / torchaudio ${TORCHAUDIO_VERSION} from ${PYTORCH_INDEX_URL}"
    python -m pip install --upgrade --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"
}

require_torch_stack_ready() {
    if torch_stack_matches; then
        return
    fi

    cat >&2 <<EOF
PyTorch ${TORCH_VERSION} / torchvision ${TORCHVISION_VERSION} / torchaudio ${TORCHAUDIO_VERSION}
with CUDA 12.6 is not ready in the current Python environment.

Stop here to prevent pip requirements from resolving and downloading unrelated torch versions.
If this is a CPU-only staging machine, either install the CUDA torch wheel first, or skip
environment packaging for torch and rely on the GPU machine's preinstalled torch stack.
EOF
    exit 1
}

make_torch_constraints() {
    TORCH_CONSTRAINTS="$(mktemp)"
    cat > "${TORCH_CONSTRAINTS}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
sympy==${SYMPY_VERSION}
huggingface-hub==${HUGGINGFACE_HUB_VERSION}
fastapi==${FASTAPI_VERSION}
starlette==${STARLETTE_VERSION}
uvicorn==${UVICORN_VERSION}
dash==${DASH_VERSION}
Flask==${FLASK_VERSION}
EOF
    export TORCH_CONSTRAINTS
}

install_internnav_requirements() {
    cd "${PROJECT_ROOT}"
    require_torch_stack_ready
    make_torch_constraints

    log "Install Python packaging tools"
    python -m pip install --upgrade pip wheel packaging ninja
    python -m pip install "setuptools<81"

    log "Install InternNav core requirements"
    python -m pip install -c "${TORCH_CONSTRAINTS}" -r requirements/core_requirements.txt

    local model_req_tmp
    model_req_tmp="$(mktemp)"
    grep -Ev '^(flash_attn|triton|sympy|huggingface-hub)==' requirements/model_requirements.txt > "${model_req_tmp}"
    log "Install InternNav model requirements except flash_attn/triton/sympy/huggingface-hub compatibility pins"
    python -m pip install -c "${TORCH_CONSTRAINTS}" -r "${model_req_tmp}"
    rm -f "${model_req_tmp}"

    log "Register InternNav package in editable mode"
    python -m pip install -e . --no-deps

    log "Install Bridge-DP runtime compatibility packages"
    python -m pip install -c "${TORCH_CONSTRAINTS}" numpy-quaternion flask psutil "transformers==4.51.0"

    # Some upstream requirements can pull a different torch-adjacent stack. Re-pin torch last.
    install_torch_stack
    rm -f "${TORCH_CONSTRAINTS}"
}

install_flash_attn_if_requested() {
    case "${INTERNNAV_INSTALL_FLASH_ATTN}" in
        0|false|False|no|No|skip)
            warn "Skip flash_attn installation. Set INTERNNAV_INSTALL_FLASH_ATTN=try or required to install it."
            return
            ;;
        try|1|true|True|required)
            ;;
        *)
            warn "Unknown INTERNNAV_INSTALL_FLASH_ATTN=${INTERNNAV_INSTALL_FLASH_ATTN}; treat as try."
            INTERNNAV_INSTALL_FLASH_ATTN="try"
            ;;
    esac

    log "Try installing flash_attn==2.7.2.post1"
    if python -m pip install --no-build-isolation "flash_attn==2.7.2.post1"; then
        log "flash_attn installed."
        return
    fi

    if [[ "${INTERNNAV_INSTALL_FLASH_ATTN}" == "required" || "${INTERNNAV_INSTALL_FLASH_ATTN}" == "1" || "${INTERNNAV_INSTALL_FLASH_ATTN}" == "true" || "${INTERNNAV_INSTALL_FLASH_ATTN}" == "True" ]]; then
        echo "flash_attn installation failed and INTERNNAV_INSTALL_FLASH_ATTN=${INTERNNAV_INSTALL_FLASH_ATTN}." >&2
        echo "Install a CUDA devel image with nvcc or provide a compatible flash_attn wheel, then rerun." >&2
        exit 1
    fi

    warn "flash_attn installation failed, continuing because mode is '${INTERNNAV_INSTALL_FLASH_ATTN}'."
    warn "Bridge-DP training can usually run without flash_attn; InternVLA/Qwen-VL paths may require it."
}

verify_install() {
    cd "${PROJECT_ROOT}"
    log "Verify Python, PyTorch, CUDA, and InternNav imports"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python - <<'PY'
import sys

import torch
import transformers
from internnav.model import get_config, get_policy

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("BridgeDP policy:", get_policy("BridgeDP_Policy").__name__)
print("BridgeDP config:", get_config("BridgeDP_Policy").__name__)
PY
}

main() {
    log "Project root: ${PROJECT_ROOT}"
    run_apt_install
    activate_python_env
    install_torch_stack
    install_internnav_requirements
    install_flash_attn_if_requested
    verify_install

    log "Done."
    log "To use the environment later:"
    if [[ "${CONDA_ENV_ACTIVATED}" == "1" ]]; then
        echo "  conda activate ${ENV_NAME}"
    fi
    echo "  cd ${PROJECT_ROOT}"
    echo "  export PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH:-}"
}

main "$@"
