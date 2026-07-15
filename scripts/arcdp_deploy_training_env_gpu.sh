#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-arcdp}"
ARCHIVE="${1:-${ARCHIVE:-}}"
TARGET_PREFIX="${TARGET_PREFIX:-}"
OVERWRITE="${OVERWRITE:-0}"
VERIFY_TORCH="${VERIFY_TORCH:-1}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.6}"
PROJECT_ROOT="${PROJECT_ROOT:-/nvme/MyResearch/InternNav_v1.0.3}"

log() {
    echo "[arcdp-deploy] $*"
}

die() {
    echo "[arcdp-deploy] ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/arcdp_deploy_training_env_gpu.sh /root/data/conda_env_packs/arcdp_*.tar.gz

Environment variables:
  ENV_NAME=arcdp
  TARGET_PREFIX=/opt/conda/envs/arcdp
  OVERWRITE=1
  PROJECT_ROOT=/nvme/MyResearch/InternNav_v1.0.3
EOF
}

[[ -n "${ARCHIVE}" && "${ARCHIVE}" != "-h" && "${ARCHIVE}" != "--help" ]] || { usage; exit 1; }
[[ -f "${ARCHIVE}" ]] || die "archive not found: ${ARCHIVE}"
command -v conda >/dev/null 2>&1 || die "conda command not found"
command -v tar >/dev/null 2>&1 || die "tar command not found"

CONDA_BASE="$(conda info --base)"
if [[ -z "${TARGET_PREFIX}" ]]; then
    TARGET_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
fi
TARGET_PREFIX="$(readlink -m "${TARGET_PREFIX}")"

case "${TARGET_PREFIX}" in
    ""|"/"|"/root"|"/home"|"/usr"|"/opt"|"/tmp"|"/var"|"/nvme")
        die "refusing unsafe TARGET_PREFIX: ${TARGET_PREFIX}"
        ;;
esac

if [[ -e "${TARGET_PREFIX}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
        die "target env already exists: ${TARGET_PREFIX}; set OVERWRITE=1 to replace it"
    fi
    BACKUP="${TARGET_PREFIX}.before_arcdp_deploy_$(date +%Y%m%d_%H%M%S)"
    log "move existing target to ${BACKUP}"
    mv "${TARGET_PREFIX}" "${BACKUP}"
fi

log "archive:       ${ARCHIVE}"
log "target prefix: ${TARGET_PREFIX}"
mkdir -p "${TARGET_PREFIX}"

if [[ -f "${ARCHIVE}.sha256" ]] && command -v sha256sum >/dev/null 2>&1; then
    log "verify sha256"
    (cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${ARCHIVE}.sha256")")
fi

log "extract environment"
tar -xzf "${ARCHIVE}" -C "${TARGET_PREFIX}"

if [[ -x "${TARGET_PREFIX}/bin/conda-unpack" ]]; then
    log "run conda-unpack"
    "${TARGET_PREFIX}/bin/conda-unpack"
else
    log "conda-unpack not found; continue"
fi

if [[ "${VERIFY_TORCH}" == "1" ]]; then
    log "verify ArcDP runtime on GPU machine"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${TARGET_PREFIX}/bin/python" - "${EXPECTED_TORCH_CUDA}" <<'PY'
import sys

expected_cuda = sys.argv[1]
import torch
from internnav.model import get_config, get_policy

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("BridgeDP policy:", get_policy("BridgeDP_Policy").__name__)
print("BridgeDP config:", get_config("BridgeDP_Policy").__name__)
if str(torch.version.cuda) != expected_cuda:
    raise SystemExit(f"expected torch CUDA {expected_cuda}, got {torch.version.cuda}")
PY
fi

log "done"
echo "Open a new shell, then run:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo "  export PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH:-}"
