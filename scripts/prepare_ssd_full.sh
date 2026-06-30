#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/data}"
if [[ ! -d "${SOURCE_ROOT}" && -d "${HOME}/data" ]]; then
    SOURCE_ROOT="${HOME}/data"
fi

SSD_ROOT="${SSD_ROOT:-/ssd}"
CONDA_ENV="${CONDA_ENV:-internnav}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"
REQUIRE_SSD_MOUNT="${REQUIRE_SSD_MOUNT:-1}"

SRC_PROJECT="${SRC_PROJECT:-${SOURCE_ROOT}/MyResearch/InternNav}"
DST_PROJECT="${DST_PROJECT:-${SSD_ROOT}/MyResearch/InternNav}"
SRC_TRAJ="${SRC_TRAJ:-${SOURCE_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
DST_TRAJ="${DST_TRAJ:-${SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
BAD_ARCHIVE_LOG="${BAD_ARCHIVE_LOG:-${DST_PROJECT}/logs/bad_archives.txt}"

echo "Preparing SSD workspace"
echo "  source root: ${SOURCE_ROOT}"
echo "  ssd root:    ${SSD_ROOT}"
echo "  project:     ${SRC_PROJECT} -> ${DST_PROJECT}"
echo "  dataset:     ${SRC_TRAJ} -> ${DST_TRAJ}"
echo "  jobs:        ${EXTRACT_JOBS}"

if [[ ! -d "${SRC_PROJECT}" ]]; then
    echo "Missing source project: ${SRC_PROJECT}" >&2
    exit 1
fi

if [[ ! -d "${SRC_TRAJ}" ]]; then
    echo "Missing source trajectory archive root: ${SRC_TRAJ}" >&2
    exit 1
fi

check_ssd_storage() {
    mkdir -p "${SSD_ROOT}"

    echo
    echo "[0/4] Check SSD mount and free space"
    if command -v findmnt >/dev/null 2>&1; then
        findmnt -T "${SSD_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true
        local mount_target
        mount_target="$(findmnt -T "${SSD_ROOT}" -n -o TARGET 2>/dev/null || true)"
        if [[ "${REQUIRE_SSD_MOUNT}" == "1" && "${mount_target}" != "${SSD_ROOT}" ]]; then
            echo "SSD_ROOT is not a mount point: ${SSD_ROOT}" >&2
            echo "findmnt target for ${SSD_ROOT}: ${mount_target:-<none>}" >&2
            echo "If your SSD is listed by lsblk but not mounted at ${SSD_ROOT}, mount it first." >&2
            echo "To bypass this check intentionally: REQUIRE_SSD_MOUNT=0 bash scripts/prepare_ssd_full.sh" >&2
            exit 1
        fi
    else
        echo "findmnt not found; skip mount-point check." >&2
    fi

    df -h "${SSD_ROOT}"
    df -i "${SSD_ROOT}"

    local free_bytes free_inodes
    free_bytes="$(df -P -B1 "${SSD_ROOT}" | awk 'NR==2 {print $4}')"
    free_inodes="$(df -Pi "${SSD_ROOT}" | awk 'NR==2 {print $4}')"
    if [[ -n "${free_bytes}" && "${free_bytes}" -le 0 ]]; then
        echo "No free bytes on ${SSD_ROOT}" >&2
        exit 1
    fi
    if [[ -n "${free_inodes}" && "${free_inodes}" -le 0 ]]; then
        echo "No free inodes on ${SSD_ROOT}" >&2
        exit 1
    fi
}

check_ssd_storage
mkdir -p "${DST_PROJECT}" "${DST_TRAJ}" "${DST_PROJECT}/checkpoints" "$(dirname "${BAD_ARCHIVE_LOG}")"

copy_project_to_ssd() {
    if command -v rsync >/dev/null 2>&1; then
        rsync -aH --info=progress2 \
            --exclude '__pycache__/' \
            --exclude '.pytest_cache/' \
            "${SRC_PROJECT}/" "${DST_PROJECT}/"
        return
    fi

    echo "rsync not found; falling back to cp -a." >&2
    echo "Tip: apt-get update && apt-get install -y rsync gives faster resumable project copies." >&2
    cp -a "${SRC_PROJECT}/." "${DST_PROJECT}/"
}

echo
echo "[1/4] Copy InternNav to SSD"
copy_project_to_ssd

echo
echo "[2/4] Ensure DepthAnything encoder is on SSD"
if [[ ! -f "${DST_PROJECT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    for candidate in \
        "${SRC_PROJECT}/checkpoints/depth_anything_v2_vits.pth" \
        "${SOURCE_ROOT}/checkpoints/depth_anything_v2_vits.pth"; do
        if [[ -f "${candidate}" ]]; then
            cp -av "${candidate}" "${DST_PROJECT}/checkpoints/depth_anything_v2_vits.pth"
            break
        fi
    done
fi

if [[ ! -f "${DST_PROJECT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    echo "DepthAnything encoder is still missing." >&2
    echo "Place it at: ${DST_PROJECT}/checkpoints/depth_anything_v2_vits.pth" >&2
    exit 1
fi

extract_one() {
    local archive="$1"
    local rel_path rel_dir dst_dir marker tar_mode
    rel_path="${archive#${SRC_TRAJ}/}"
    rel_dir="$(dirname "${rel_path}")"
    dst_dir="${DST_TRAJ}/${rel_dir}"
    marker="${dst_dir}/.extracted_$(basename "${archive}").done"

    mkdir -p "${dst_dir}"
    if [[ -f "${marker}" ]]; then
        echo "skip $(basename "${archive}")"
        return 0
    fi

    echo "extract ${archive} -> ${dst_dir}"
    if gzip -t "${archive}" >/dev/null 2>&1; then
        tar_mode="gzip"
    elif tar -tf "${archive}" >/dev/null 2>&1; then
        tar_mode="tar"
        echo "warning: ${archive} is not gzip-compressed; extracting as plain tar" >&2
    else
        echo "bad or incomplete archive: ${archive}" >&2
        printf '%s\n' "${archive}" >> "${BAD_ARCHIVE_LOG}"
        return 1
    fi

    if [[ "${tar_mode}" == "gzip" ]]; then
        tar -xzf "${archive}" -C "${dst_dir}"
    else
        tar -xf "${archive}" -C "${dst_dir}"
    fi
    touch "${marker}"
}
export -f extract_one
export SRC_TRAJ DST_TRAJ BAD_ARCHIVE_LOG

echo
echo "[3/4] Extract trajectory archives to SSD"
: > "${BAD_ARCHIVE_LOG}"
if ! find "${SRC_TRAJ}" -mindepth 2 -maxdepth 2 -type f -name '*.tar.gz' -print0 \
    | xargs -0 -r -n 1 -P "${EXTRACT_JOBS}" bash -c 'extract_one "$@"' _; then
    echo "One or more trajectory archives failed to extract." >&2
    if [[ -s "${BAD_ARCHIVE_LOG}" ]]; then
        echo "Bad archive list: ${BAD_ARCHIVE_LOG}" >&2
        cat "${BAD_ARCHIVE_LOG}" >&2
    fi
    exit 1
fi

echo
echo "[4/4] Generate preload index on SSD"
cd "${DST_PROJECT}"
PYTHONPATH="${DST_PROJECT}" conda run -n "${CONDA_ENV}" env PYTHONPATH="${DST_PROJECT}" \
    python scripts/dataset/generate_preload_index.py \
    --root_dir "${DST_TRAJ}" \
    --output "${DST_PROJECT}/checkpoints/preload_index.json"

echo
echo "SSD preparation complete."
echo "  project: ${DST_PROJECT}"
echo "  dataset: ${DST_TRAJ}"
echo "  preload: ${DST_PROJECT}/checkpoints/preload_index.json"
df -h "${SSD_ROOT}" || true
