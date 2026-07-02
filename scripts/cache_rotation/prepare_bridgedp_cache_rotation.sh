#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PROJECT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS=""
NVME_SIZE=""
PRESET="balanced"
HDD_ROOT="/hdd"
NVME_ROOT="/nvme"
FORCE=0
RUN_NAME="bridgedp_cache_rotation"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --hdd-root) HDD_ROOT="$2"; shift 2 ;;
        --nvme-root) NVME_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help)
            echo "Usage: $0 --gpus 4|8 --nvme-size 1tb|2tb|4tb [--preset balanced|quality|throughput] [--force]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${GPUS}" || -z "${NVME_SIZE}" ]]; then
    echo "--gpus and --nvme-size are required." >&2
    exit 1
fi

export PYTHONPATH="${SRC_PROJECT}:${PYTHONPATH:-}"
eval "$(python "${SCRIPT_DIR}/profile_bridgedp_cache.py" --gpus "${GPUS}" --nvme-size "${NVME_SIZE}" --preset "${PRESET}")"

PROJECT_ROOT="${NVME_ROOT}/MyResearch/InternNav"
HDD_TRAJ="${HDD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data"
MANIFEST="${HDD_ROOT}/bridgedp_rotation/manifests/shards_${GPUS}gpu_${BRIDGEDP_NVME_SIZE}_${PRESET}.json"
CACHE_ROOT="${NVME_ROOT}/bridgedp_cache"

echo "Preparing Bridge-DP cache rotation workspace"
echo "  source project: ${SRC_PROJECT}"
echo "  project:        ${PROJECT_ROOT}"
echo "  hdd traj:       ${HDD_TRAJ}"
echo "  cache root:     ${CACHE_ROOT}"
echo "  manifest:       ${MANIFEST}"
echo "  gpus:           ${GPUS}"
echo "  nvme size:      ${BRIDGEDP_NVME_SIZE}"
echo "  preset:         ${PRESET}"
echo "  cache slot GB:  ${BRIDGEDP_CACHE_SLOT_GB}"

if [[ ! -d "${HDD_TRAJ}" ]]; then
    echo "Missing extracted HDD dataset: ${HDD_TRAJ}" >&2
    exit 1
fi

mkdir -p "${PROJECT_ROOT}" "${CACHE_ROOT}" "$(dirname "${MANIFEST}")" "${HDD_ROOT}/bridgedp_rotation/logs"

echo
echo "[1/5] Check storage"
findmnt -T "${HDD_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true
findmnt -T "${NVME_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true
df -h "${HDD_ROOT}" "${NVME_ROOT}"
df -i "${HDD_ROOT}" "${NVME_ROOT}"

echo
echo "[2/5] Copy InternNav to NVMe"
if command -v rsync >/dev/null 2>&1; then
    rsync -aH --info=progress2 \
        --exclude '__pycache__/' \
        --exclude '.pytest_cache/' \
        "${SRC_PROJECT}/" "${PROJECT_ROOT}/"
else
    echo "rsync not found; falling back to cp -a." >&2
    cp -a "${SRC_PROJECT}/." "${PROJECT_ROOT}/"
fi

echo
echo "[3/5] Ensure DepthAnything encoder is on NVMe"
mkdir -p "${PROJECT_ROOT}/checkpoints"
if [[ ! -f "${PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    for candidate in \
        "${SRC_PROJECT}/checkpoints/depth_anything_v2_vits.pth" \
        "${HDD_ROOT}/checkpoints/depth_anything_v2_vits.pth" \
        "/data/checkpoints/depth_anything_v2_vits.pth"; do
        if [[ -f "${candidate}" ]]; then
            cp -av "${candidate}" "${PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth"
            break
        fi
    done
fi
if [[ ! -f "${PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    echo "Missing DepthAnything encoder: ${PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo
echo "[4/5] Generate shard manifest"
python "${PROJECT_ROOT}/scripts/cache_rotation/make_bridgedp_shards.py" \
    --hdd-root "${HDD_ROOT}" \
    --manifest "${MANIFEST}" \
    --gpus "${GPUS}" \
    --nvme-size "${BRIDGEDP_NVME_SIZE}" \
    --preset "${PRESET}" \
    --cache-slot-gb "${BRIDGEDP_CACHE_SLOT_GB}"

NUM_SHARDS="$(python -c "import json; print(len(json.load(open('${MANIFEST}'))['shards']))")"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
    echo "Manifest contains no shards: ${MANIFEST}" >&2
    exit 1
fi

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
    FORCE_ARGS+=(--force)
fi

echo
echo "[5/5] Prebuild cache_A and cache_B"
python "${PROJECT_ROOT}/scripts/cache_rotation/build_bridgedp_cache_shard.py" \
    --manifest "${MANIFEST}" \
    --hdd-root "${HDD_ROOT}" \
    --shard-index 0 \
    --slot-root "${CACHE_ROOT}/cache_A" \
    "${FORCE_ARGS[@]}"

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    python "${PROJECT_ROOT}/scripts/cache_rotation/build_bridgedp_cache_shard.py" \
        --manifest "${MANIFEST}" \
        --hdd-root "${HDD_ROOT}" \
        --shard-index 1 \
        --slot-root "${CACHE_ROOT}/cache_B" \
        "${FORCE_ARGS[@]}"
fi

echo
echo "Cache rotation preparation complete."
echo "  project:  ${PROJECT_ROOT}"
echo "  manifest: ${MANIFEST}"
echo "  cache_A:  ${CACHE_ROOT}/cache_A"
echo "  cache_B:  ${CACHE_ROOT}/cache_B"
echo
echo "Start training example:"
echo "  bash ${PROJECT_ROOT}/train_bridgedp_cache_rotation.sh --gpus ${GPUS} --nvme-size ${BRIDGEDP_NVME_SIZE} --total-epochs 100 --preset ${PRESET}"
