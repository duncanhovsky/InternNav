#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/data}"
if [[ ! -d "${SOURCE_ROOT}" && -d "${HOME}/data" ]]; then
    SOURCE_ROOT="${HOME}/data"
fi

SSD_ROOT="${SSD_ROOT:-/ssd}"
CONDA_ENV="${CONDA_ENV:-internnav}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"

SRC_PROJECT="${SRC_PROJECT:-${SOURCE_ROOT}/MyResearch/InternNav}"
DST_PROJECT="${DST_PROJECT:-${SSD_ROOT}/MyResearch/InternNav}"
SRC_TRAJ="${SRC_TRAJ:-${SOURCE_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
DST_TRAJ="${DST_TRAJ:-${SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"

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

mkdir -p "${DST_PROJECT}" "${DST_TRAJ}" "${DST_PROJECT}/checkpoints"

echo
echo "[1/4] Copy InternNav to SSD"
rsync -aH --info=progress2 \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    "${SRC_PROJECT}/" "${DST_PROJECT}/"

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
    local rel_path rel_dir dst_dir marker
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
    tar -xzf "${archive}" -C "${dst_dir}"
    touch "${marker}"
}
export -f extract_one
export SRC_TRAJ DST_TRAJ

echo
echo "[3/4] Extract trajectory archives to SSD"
find "${SRC_TRAJ}" -mindepth 2 -maxdepth 2 -type f -name '*.tar.gz' -print0 \
    | xargs -0 -r -n 1 -P "${EXTRACT_JOBS}" bash -c 'extract_one "$@"' _

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

