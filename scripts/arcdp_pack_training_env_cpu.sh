#!/usr/bin/env bash
set -euo pipefail

# Expected base image: pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04

ENV_NAME="${ENV_NAME:-arcdp}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/conda_env_packs}"
ARCHIVE="${ARCHIVE:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_SETUP="${RUN_SETUP:-1}"
PACK_UNINSTALL_EDITABLE_INTERNNAV="${PACK_UNINSTALL_EDITABLE_INTERNNAV:-1}"
INSTALL_APT="${INSTALL_APT:-1}"
INTERNNAV_INSTALL_GIT_DEPS="${INTERNNAV_INSTALL_GIT_DEPS:-skip}"
INTERNNAV_INSTALL_FLASH_ATTN="${INTERNNAV_INSTALL_FLASH_ATTN:-skip}"
PACKAGING_VERSION_SPEC="${PACKAGING_VERSION_SPEC:-packaging>=23.0,<25}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.6}"
EXPECTED_IMAGE="${EXPECTED_IMAGE:-pytorch:2.7.0-cuda12.6-python3.10-ubuntu22.04}"

log() {
    echo "[arcdp-pack] $*"
}

die() {
    echo "[arcdp-pack] ERROR: $*" >&2
    exit 1
}

remove_editable_internnav_before_pack() {
    if [[ "${PACK_UNINSTALL_EDITABLE_INTERNNAV}" != "1" ]]; then
        log "skip removing editable internnav because PACK_UNINSTALL_EDITABLE_INTERNNAV=${PACK_UNINSTALL_EDITABLE_INTERNNAV}"
        return
    fi

    log "remove editable internnav registration before conda-pack"
    conda run -n "${ENV_NAME}" python -m pip uninstall -y internnav >/dev/null 2>&1 || true

    log "verify source import still works through PYTHONPATH"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" conda run -n "${ENV_NAME}" python - "${PROJECT_ROOT}" <<'PY'
from pathlib import Path
import sys

project_root = Path(sys.argv[1]).resolve()
import internnav

source_file = Path(internnav.__file__).resolve()
print("internnav:", source_file)
if project_root not in source_file.parents:
    raise SystemExit(f"internnav is not imported from PROJECT_ROOT={project_root}: {source_file}")
PY
}

find_conda_sh() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        echo "${conda_base}/etc/profile.d/conda.sh"
        return
    fi

    for candidate in \
        "/opt/conda/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done
}

CONDA_SH="$(find_conda_sh)"
[[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]] || die "conda.sh not found in this image"
# shellcheck disable=SC1090
source "${CONDA_SH}"

SETUP_SCRIPT="${PROJECT_ROOT}/scripts/setup_internnav_pytorch270_cu126.sh"
[[ -f "${SETUP_SCRIPT}" ]] || die "missing setup script: ${SETUP_SCRIPT}"

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HOSTNAME_SAFE="$(hostname 2>/dev/null | tr '/: ' '___' || echo host)"
if [[ -z "${ARCHIVE}" ]]; then
    ARCHIVE="${OUTPUT_DIR}/arcdp_${ENV_NAME}_pytorch270_cu126_${HOSTNAME_SAFE}_${TIMESTAMP}.tar.gz"
fi
ARCHIVE="$(readlink -m "${ARCHIVE}")"
META_DIR="${ARCHIVE}.metadata"

log "expected image: ${EXPECTED_IMAGE}"
log "project root:    ${PROJECT_ROOT}"
log "env name:        ${ENV_NAME}"
log "archive:         ${ARCHIVE}"

if [[ "${RUN_SETUP}" == "1" ]]; then
    log "configure InternNav runtime before packing"
    ENV_NAME="${ENV_NAME}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    USE_CONDA=1 \
    INSTALL_APT="${INSTALL_APT}" \
    INTERNNAV_INSTALL_GIT_DEPS="${INTERNNAV_INSTALL_GIT_DEPS}" \
    INTERNNAV_INSTALL_FLASH_ATTN="${INTERNNAV_INSTALL_FLASH_ATTN}" \
    PACKAGING_VERSION_SPEC="${PACKAGING_VERSION_SPEC}" \
    bash "${SETUP_SCRIPT}"
else
    log "skip runtime setup because RUN_SETUP=${RUN_SETUP}"
fi

log "verify torch stack inside ${ENV_NAME}"
conda run -n "${ENV_NAME}" python - "${EXPECTED_TORCH_CUDA}" <<'PY'
import sys

expected_cuda = sys.argv[1]
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
if str(torch.version.cuda) != expected_cuda:
    raise SystemExit(f"expected torch CUDA {expected_cuda}, got {torch.version.cuda}")
PY

log "ensure conda_pack is available inside ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install "${PACKAGING_VERSION_SPEC}"
if ! conda run -n "${ENV_NAME}" python -c "import conda_pack" >/dev/null 2>&1; then
    conda run -n "${ENV_NAME}" python -m pip install "${PACKAGING_VERSION_SPEC}" conda-pack
fi
conda run -n "${ENV_NAME}" python - "${PACKAGING_VERSION_SPEC}" <<'PY'
import importlib.metadata as metadata
import sys
from packaging.version import Version

version = metadata.version("packaging")
print("packaging:", version)
if Version(version) >= Version("25"):
    raise SystemExit(f"packaging version is incompatible with InternNav: {version}")
PY

remove_editable_internnav_before_pack

rm -rf "${META_DIR}"
mkdir -p "${META_DIR}"
{
    echo "expected_image=${EXPECTED_IMAGE}"
    echo "env_name=${ENV_NAME}"
    echo "project_root=${PROJECT_ROOT}"
    echo "archive=${ARCHIVE}"
    echo "created_at=${TIMESTAMP}"
    echo "host=${HOSTNAME_SAFE}"
    echo "expected_torch_cuda=${EXPECTED_TORCH_CUDA}"
} > "${META_DIR}/manifest.env"
conda run -n "${ENV_NAME}" python -V > "${META_DIR}/python_version.txt" 2>&1 || true
conda run -n "${ENV_NAME}" python -m pip freeze > "${META_DIR}/pip_freeze.txt" 2>&1 || true
conda list -n "${ENV_NAME}" > "${META_DIR}/conda_list.txt" 2>&1 || true

log "pack conda env ${ENV_NAME}"
if conda run -n "${ENV_NAME}" conda-pack --help >/dev/null 2>&1; then
    conda run -n "${ENV_NAME}" conda-pack -n "${ENV_NAME}" -o "${ARCHIVE}" --force --ignore-missing-files
else
    conda run -n "${ENV_NAME}" python -m conda_pack -n "${ENV_NAME}" -o "${ARCHIVE}" --force --ignore-missing-files
fi

tar -czf "${ARCHIVE}.metadata.tar.gz" -C "${META_DIR}" .
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
fi

DEPLOY_SCRIPT="${PROJECT_ROOT}/scripts/arcdp_deploy_training_env_gpu.sh"
if [[ -f "${DEPLOY_SCRIPT}" ]]; then
    cp -f "${DEPLOY_SCRIPT}" "${OUTPUT_DIR}/arcdp_deploy_training_env_gpu.sh"
fi

log "done"
echo "archive:  ${ARCHIVE}"
echo "metadata: ${ARCHIVE}.metadata.tar.gz"
if [[ -f "${ARCHIVE}.sha256" ]]; then
    echo "sha256:   ${ARCHIVE}.sha256"
fi
echo "deploy:   ${OUTPUT_DIR}/arcdp_deploy_training_env_gpu.sh"
