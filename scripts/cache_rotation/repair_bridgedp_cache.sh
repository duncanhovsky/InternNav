#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SLOT=""
SHARD_INDEX=""
GPUS=""
NVME_SIZE=""
PRESET="balanced"
HDD_ROOT="/hdd"
NVME_ROOT="/nvme"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot) SLOT="$2"; shift 2 ;;
        --shard-index) SHARD_INDEX="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --hdd-root) HDD_ROOT="$2"; shift 2 ;;
        --nvme-root) NVME_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --slot cache_A|cache_B --shard-index N --gpus 4|8 --nvme-size 1tb|2tb|4tb [--preset balanced]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${SLOT}" || -z "${SHARD_INDEX}" || -z "${GPUS}" || -z "${NVME_SIZE}" ]]; then
    echo "--slot, --shard-index, --gpus, and --nvme-size are required." >&2
    exit 1
fi

if [[ "${SLOT}" != "cache_A" && "${SLOT}" != "cache_B" ]]; then
    echo "--slot must be cache_A or cache_B." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
eval "$(python "${SCRIPT_DIR}/profile_bridgedp_cache.py" --gpus "${GPUS}" --nvme-size "${NVME_SIZE}" --preset "${PRESET}")"

MANIFEST="${HDD_ROOT}/bridgedp_rotation/manifests/shards_${GPUS}gpu_${BRIDGEDP_NVME_SIZE}_${PRESET}.json"
SLOT_ROOT="${NVME_ROOT}/bridgedp_cache/${SLOT}"

if [[ -f "${SLOT_ROOT}/.ACTIVE" ]]; then
    active_pid="$(cat "${SLOT_ROOT}/.ACTIVE" || true)"
    if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
        echo "Refusing to repair ACTIVE cache slot ${SLOT_ROOT}; live pid=${active_pid}" >&2
        exit 1
    fi
fi

python "${PROJECT_ROOT}/scripts/cache_rotation/build_bridgedp_cache_shard.py" \
    --manifest "${MANIFEST}" \
    --hdd-root "${HDD_ROOT}" \
    --shard-index "${SHARD_INDEX}" \
    --slot-root "${SLOT_ROOT}" \
    --force

python "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" --slot-root "${SLOT_ROOT}"
