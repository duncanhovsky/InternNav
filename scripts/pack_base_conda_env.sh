#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-base}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/conda_env_packs}"
ARCHIVE="${ARCHIVE:-}"
PACK_METHOD="${PACK_METHOD:-auto}"
REQUIRE_CUDA_TORCH="${REQUIRE_CUDA_TORCH:-1}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.6}"

log() {
    echo "[pack-base] $*"
}

die() {
    echo "[pack-base] ERROR: $*" >&2
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

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

safe_basename() {
    printf '%s' "$1" | tr '/: ' '___'
}

CONDA_CMD="$(find_conda)"
BASE_PREFIX="$("${CONDA_CMD}" info --base)"
PYTHON_BIN="${BASE_PREFIX}/bin/python"

[[ "${ENV_NAME}" == "base" ]] || die "this script only packs ENV_NAME=base"
[[ -d "${BASE_PREFIX}/conda-meta" ]] || die "base prefix does not look like conda: ${BASE_PREFIX}"
[[ -x "${PYTHON_BIN}" ]] || die "python not found in base prefix: ${PYTHON_BIN}"
require_cmd tar

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
HOSTNAME_SAFE="$(safe_basename "$(hostname 2>/dev/null || echo host)")"
mkdir -p "${OUTPUT_DIR}"
if [[ -z "${ARCHIVE}" ]]; then
    ARCHIVE="${OUTPUT_DIR}/internnav_base_${HOSTNAME_SAFE}_${TIMESTAMP}.tar.gz"
fi
ARCHIVE="$(readlink -m "${ARCHIVE}")"
META_DIR="${ARCHIVE}.metadata"

log "base prefix: ${BASE_PREFIX}"
log "output:      ${ARCHIVE}"
log "method:      ${PACK_METHOD}"

TORCH_INFO="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    print(f"version={torch.__version__}")
    print(f"cuda={torch.version.cuda or ''}")
except Exception as exc:
    print(f"error={type(exc).__name__}: {exc}")
PY
)"
TORCH_CUDA="$(printf '%s\n' "${TORCH_INFO}" | awk -F= '$1=="cuda" {print $2}')"
if [[ "${REQUIRE_CUDA_TORCH}" == "1" && "${TORCH_CUDA}" != "${EXPECTED_TORCH_CUDA}" ]]; then
    printf '%s\n' "${TORCH_INFO}" >&2
    die "base torch is not CUDA ${EXPECTED_TORCH_CUDA}. Set REQUIRE_CUDA_TORCH=0 only if you intentionally pack CPU torch."
fi

rm -rf "${META_DIR}"
mkdir -p "${META_DIR}"
{
    echo "env_name=${ENV_NAME}"
    echo "source_prefix=${BASE_PREFIX}"
    echo "created_at=${TIMESTAMP}"
    echo "host=${HOSTNAME_SAFE}"
    echo "pack_method=${PACK_METHOD}"
    echo "expected_torch_cuda=${EXPECTED_TORCH_CUDA}"
    printf '%s\n' "${TORCH_INFO}" | sed 's/^/torch_/'
} > "${META_DIR}/manifest.env"
"${PYTHON_BIN}" -V > "${META_DIR}/python_version.txt" 2>&1 || true
"${PYTHON_BIN}" -m pip freeze > "${META_DIR}/pip_freeze.txt" 2>&1 || true
"${CONDA_CMD}" list -n base > "${META_DIR}/conda_list.txt" 2>&1 || true
"${CONDA_CMD}" list -n base --explicit > "${META_DIR}/conda_explicit.txt" 2>&1 || true

pack_with_conda_pack() {
    if command -v conda-pack >/dev/null 2>&1; then
        conda-pack -p "${BASE_PREFIX}" -o "${ARCHIVE}" --force --ignore-missing-files
        return 0
    fi
    if "${PYTHON_BIN}" -c 'import conda_pack' >/dev/null 2>&1; then
        "${PYTHON_BIN}" -m conda_pack -p "${BASE_PREFIX}" -o "${ARCHIVE}" --force --ignore-missing-files
        return 0
    fi
    return 1
}

pack_with_raw_tar() {
    log "conda-pack is unavailable; use raw-tar fallback."
    log "raw-tar is safest when the new machine has the same base prefix: ${BASE_PREFIX}"
    tar --warning=no-file-changed -czf "${ARCHIVE}" \
        --exclude='./pkgs' \
        --exclude='./envs' \
        --exclude='./conda-bld' \
        --exclude='./.conda' \
        --exclude='./.cache' \
        --exclude='./logs' \
        -C "${META_DIR}" manifest.env \
        -C "${BASE_PREFIX}" .
}

case "${PACK_METHOD}" in
    auto)
        if ! pack_with_conda_pack; then
            pack_with_raw_tar
        fi
        ;;
    conda-pack)
        pack_with_conda_pack || die "conda-pack is not available in base"
        ;;
    raw-tar)
        pack_with_raw_tar
        ;;
    *)
        die "unknown PACK_METHOD=${PACK_METHOD}; use auto, conda-pack, or raw-tar"
        ;;
esac

tar -czf "${ARCHIVE}.metadata.tar.gz" -C "${META_DIR}" .
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/deploy_base_conda_env.sh" ]]; then
    cp -f "${SCRIPT_DIR}/deploy_base_conda_env.sh" "${OUTPUT_DIR}/deploy_base_conda_env.sh"
fi

log "done"
echo "archive:  ${ARCHIVE}"
echo "metadata: ${ARCHIVE}.metadata.tar.gz"
if [[ -f "${ARCHIVE}.sha256" ]]; then
    echo "sha256:   ${ARCHIVE}.sha256"
fi
echo
echo "Copy the archive, metadata, sha256 file, and deploy_base_conda_env.sh to the new machine."
