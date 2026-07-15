#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/root/data}"
NVME_ROOT="${NVME_ROOT:-/nvme}"
ENV_NAME="${ENV_NAME:-arcdp}"
CONDA_ENV="${CONDA_ENV:-${ENV_NAME}}"
PROJECT_SUBDIR="${PROJECT_SUBDIR:-MyResearch/InternNav_v1.0.3}"
SRC_PROJECT="${SRC_PROJECT:-${SOURCE_ROOT}/${PROJECT_SUBDIR}}"
DST_PROJECT="${DST_PROJECT:-${NVME_ROOT}/${PROJECT_SUBDIR}}"
SRC_TRAJ="${SRC_TRAJ:-${SOURCE_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
DST_TRAJ="${DST_TRAJ:-${NVME_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
EXTRACT_JOBS="${EXTRACT_JOBS:-8}"
MIN_FREE_INODES="${MIN_FREE_INODES:-100000000}"
ENFORCE_MIN_INODES="${ENFORCE_MIN_INODES:-1}"
REQUIRE_SSD_MOUNT="${REQUIRE_SSD_MOUNT:-1}"

log() {
    echo "[arcdp-nvme-prepare] $*"
}

die() {
    echo "[arcdp-nvme-prepare] ERROR: $*" >&2
    exit 1
}

if [[ ! -d "${SRC_PROJECT}" ]]; then
    for candidate in \
        "${SOURCE_ROOT}/MyResearch/InternNav" \
        "${SOURCE_ROOT}/InternNav_v1.0.3" \
        "${SOURCE_ROOT}/InternNav"; do
        if [[ -d "${candidate}" ]]; then
            SRC_PROJECT="${candidate}"
            break
        fi
    done
fi

[[ -d "${SRC_PROJECT}" ]] || die "missing source project: ${SRC_PROJECT}"
[[ -d "${SRC_TRAJ}" ]] || die "missing source dataset archives: ${SRC_TRAJ}"
[[ -f "${SRC_PROJECT}/scripts/prepare_ssd_full.sh" ]] || die "missing prepare script in ${SRC_PROJECT}"

mkdir -p "${NVME_ROOT}"
if command -v findmnt >/dev/null 2>&1; then
    findmnt -T "${NVME_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true
    mount_target="$(findmnt -T "${NVME_ROOT}" -n -o TARGET 2>/dev/null || true)"
    if [[ "${REQUIRE_SSD_MOUNT}" == "1" && "${mount_target}" != "${NVME_ROOT}" ]]; then
        die "${NVME_ROOT} is not the direct mount target; set REQUIRE_SSD_MOUNT=0 only if intentional"
    fi
fi

df -hT "${NVME_ROOT}"
df -ih "${NVME_ROOT}"
stat -f -c 'mount=%m fs_type=%T files_total=%c files_free=%d blocks_total=%b blocks_free=%f block_size=%S' "${NVME_ROOT}"

free_inodes="$(stat -f -c %d "${NVME_ROOT}")"
if [[ "${free_inodes}" -lt "${MIN_FREE_INODES}" ]]; then
    message="free inodes ${free_inodes} < MIN_FREE_INODES ${MIN_FREE_INODES}"
    if [[ "${ENFORCE_MIN_INODES}" == "1" ]]; then
        die "${message}"
    fi
    log "WARN: ${message}"
fi

log "source root: ${SOURCE_ROOT}"
log "nvme root:   ${NVME_ROOT}"
log "project:     ${SRC_PROJECT} -> ${DST_PROJECT}"
log "dataset:     ${SRC_TRAJ} -> ${DST_TRAJ}"
log "conda env:   ${CONDA_ENV}"
log "jobs:        ${EXTRACT_JOBS}"

SOURCE_ROOT="${SOURCE_ROOT}" \
SSD_ROOT="${NVME_ROOT}" \
CONDA_ENV="${CONDA_ENV}" \
SRC_PROJECT="${SRC_PROJECT}" \
DST_PROJECT="${DST_PROJECT}" \
SRC_TRAJ="${SRC_TRAJ}" \
DST_TRAJ="${DST_TRAJ}" \
EXTRACT_JOBS="${EXTRACT_JOBS}" \
REQUIRE_SSD_MOUNT="${REQUIRE_SSD_MOUNT}" \
bash "${SRC_PROJECT}/scripts/prepare_ssd_full.sh"

log "done"
echo "project: ${DST_PROJECT}"
echo "dataset: ${DST_TRAJ}"
echo "preload: ${DST_PROJECT}/checkpoints/preload_index.json"
