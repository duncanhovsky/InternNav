#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-base}"
ARCHIVE="${1:-${ARCHIVE:-}}"
TARGET_PREFIX="${TARGET_PREFIX:-}"
YES="${YES:-0}"
ALLOW_PREFIX_MISMATCH="${ALLOW_PREFIX_MISMATCH:-0}"
VERIFY_TORCH="${VERIFY_TORCH:-1}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.6}"

log() {
    echo "[deploy-base] $*"
}

die() {
    echo "[deploy-base] ERROR: $*" >&2
    exit 1
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        echo "${CONDA_EXE}"
        return 0
    fi
    die "conda command not found. Activate base or source conda.sh first."
}

usage() {
    cat <<'EOF'
Usage:
  YES=1 bash scripts/deploy_base_conda_env.sh /path/to/internnav_base_*.tar.gz

Environment variables:
  TARGET_PREFIX=/root/miniconda3   Override target base prefix.
  ALLOW_PREFIX_MISMATCH=1          Allow raw-tar archives from a different prefix.
  VERIFY_TORCH=0                   Skip torch verification after deploy.
EOF
}

[[ "${ENV_NAME}" == "base" ]] || die "this script only deploys into ENV_NAME=base"
if [[ -z "${ARCHIVE}" || "${ARCHIVE}" == "-h" || "${ARCHIVE}" == "--help" ]]; then
    usage
    exit 1
fi
[[ -f "${ARCHIVE}" ]] || die "archive not found: ${ARCHIVE}"
command -v tar >/dev/null 2>&1 || die "missing command: tar"

CONDA_CMD="$(find_conda)"
if [[ -z "${TARGET_PREFIX}" ]]; then
    TARGET_PREFIX="$("${CONDA_CMD}" info --base)"
fi
TARGET_PREFIX="$(readlink -m "${TARGET_PREFIX}")"

case "${TARGET_PREFIX}" in
    ""|"/"|"/root"|"/home"|"/usr"|"/opt"|"/tmp"|"/var")
        die "Refusing to deploy into unsafe TARGET_PREFIX: ${TARGET_PREFIX}"
        ;;
esac
[[ -d "${TARGET_PREFIX}/conda-meta" ]] || die "target prefix does not look like conda base: ${TARGET_PREFIX}"

log "archive:       ${ARCHIVE}"
log "target prefix: ${TARGET_PREFIX}"

if [[ -f "${ARCHIVE}.sha256" ]] && command -v sha256sum >/dev/null 2>&1; then
    log "verify sha256"
    (cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${ARCHIVE}.sha256")")
fi

RAW_SOURCE_PREFIX="$(tar -xOzf "${ARCHIVE}" manifest.env 2>/dev/null | awk -F= '$1=="source_prefix" {print $2}' || true)"
if [[ -n "${RAW_SOURCE_PREFIX}" && "${RAW_SOURCE_PREFIX}" != "${TARGET_PREFIX}" && "${ALLOW_PREFIX_MISMATCH}" != "1" ]]; then
    die "raw-tar archive source_prefix=${RAW_SOURCE_PREFIX}, target=${TARGET_PREFIX}. Use a conda-pack archive or set ALLOW_PREFIX_MISMATCH=1 only if you understand the risk."
fi

if [[ "${YES}" != "1" ]]; then
    echo "This will overlay the current conda base environment at:"
    echo "  ${TARGET_PREFIX}"
    echo "Type exactly 'yes' to continue:"
    read -r CONFIRM
    [[ "${CONFIRM}" == "yes" ]] || die "cancelled"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${TARGET_PREFIX}/.internnav_base_env_before_deploy_${TIMESTAMP}"
mkdir -p "${BACKUP_DIR}"
"${CONDA_CMD}" list -n base > "${BACKUP_DIR}/conda_list.txt" 2>&1 || true
if [[ -x "${TARGET_PREFIX}/bin/python" ]]; then
    "${TARGET_PREFIX}/bin/python" -V > "${BACKUP_DIR}/python_version.txt" 2>&1 || true
    "${TARGET_PREFIX}/bin/python" -m pip freeze > "${BACKUP_DIR}/pip_freeze.txt" 2>&1 || true
fi
if [[ -d "${TARGET_PREFIX}/conda-meta" ]]; then
    tar -czf "${BACKUP_DIR}/conda_meta.tar.gz" -C "${TARGET_PREFIX}" conda-meta
fi
log "metadata backup: ${BACKUP_DIR}"

log "extract archive into base prefix"
tar -xzf "${ARCHIVE}" -C "${TARGET_PREFIX}"

if [[ -x "${TARGET_PREFIX}/bin/conda-unpack" ]]; then
    log "run conda-unpack"
    "${TARGET_PREFIX}/bin/conda-unpack"
else
    log "conda-unpack not found; this is expected for raw-tar archives with the same prefix"
fi

if [[ "${VERIFY_TORCH}" == "1" ]]; then
    log "verify python and torch"
    "${TARGET_PREFIX}/bin/python" - <<PY
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    if "${EXPECTED_TORCH_CUDA}" and torch.version.cuda != "${EXPECTED_TORCH_CUDA}":
        raise SystemExit("torch CUDA version mismatch")
except ImportError as exc:
    raise SystemExit(f"torch import failed: {exc}")
PY
fi

log "done. Open a new shell and run: conda activate base"
