#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS=""
NVME_SIZE=""
TOTAL_EPOCHS=""
PRESET="balanced"
HDD_ROOT="/hdd"
NVME_ROOT="/nvme"
RUN_NAME="bridgedp_cache_rotation"
START_SHARD=0
MAX_STAGES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --total-epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --hdd-root) HDD_ROOT="$2"; shift 2 ;;
        --nvme-root) NVME_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --start-shard) START_SHARD="$2"; shift 2 ;;
        --max-stages) MAX_STAGES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --gpus 4|8 --nvme-size 1tb|2tb|4tb --total-epochs N [--preset balanced|quality|throughput]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${GPUS}" || -z "${NVME_SIZE}" || -z "${TOTAL_EPOCHS}" ]]; then
    echo "--gpus, --nvme-size, and --total-epochs are required." >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
eval "$(python "${PROJECT_ROOT}/scripts/cache_rotation/profile_bridgedp_cache.py" --gpus "${GPUS}" --nvme-size "${NVME_SIZE}" --preset "${PRESET}")"

export BRIDGEDP_PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${PROJECT_ROOT}}"
export BRIDGEDP_RUN_NAME="${RUN_NAME}"
export BRIDGEDP_NUM_GPUS="${GPUS}"
export BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-96}"
export BRIDGEDP_NUM_WORKERS="${BRIDGEDP_NUM_WORKERS:-${BRIDGEDP_NUM_WORKERS}}"
export BRIDGEDP_GRAD_ACCUM="${BRIDGEDP_GRAD_ACCUM:-1}"
export BRIDGEDP_AUTO_RESUME="${BRIDGEDP_AUTO_RESUME:-1}"
export BRIDGEDP_IGNORE_DATA_SKIP="${BRIDGEDP_IGNORE_DATA_SKIP:-1}"
export BRIDGEDP_SAVE_STEPS="${BRIDGEDP_SAVE_STEPS:-500}"
export BRIDGEDP_LR="${BRIDGEDP_LR:-3e-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

MANIFEST="${HDD_ROOT}/bridgedp_rotation/manifests/shards_${GPUS}gpu_${BRIDGEDP_NVME_SIZE}_${PRESET}.json"
CACHE_ROOT="${NVME_ROOT}/bridgedp_cache"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}/ckpts"
BACKUP_DIR="${HDD_ROOT}/bridgedp_rotation/checkpoints/${RUN_NAME}"
LOG_DIR="${HDD_ROOT}/bridgedp_rotation/logs/${RUN_NAME}"
mkdir -p "${BACKUP_DIR}" "${LOG_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}" >&2
    echo "Run scripts/cache_rotation/prepare_bridgedp_cache_rotation.sh first." >&2
    exit 1
fi

NUM_SHARDS="$(python -c "import json; print(len(json.load(open('${MANIFEST}'))['shards']))")"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
    echo "Manifest contains no shards: ${MANIFEST}" >&2
    exit 1
fi

build_slot() {
    local shard_index="$1"
    local slot_root="$2"
    python "${PROJECT_ROOT}/scripts/cache_rotation/build_bridgedp_cache_shard.py" \
        --manifest "${MANIFEST}" \
        --hdd-root "${HDD_ROOT}" \
        --shard-index "${shard_index}" \
        --slot-root "${slot_root}" \
        --force
}

wait_for_ready() {
    local slot_root="$1"
    local wait_pid="${2:-}"
    if [[ -n "${wait_pid}" ]]; then
        wait "${wait_pid}"
    fi
    python "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" --slot-root "${slot_root}"
}

STAGE=0
SHARD_INDEX="${START_SHARD}"
BUILD_PID=""

echo "Bridge-DP cache rotation training"
echo "  project:      ${PROJECT_ROOT}"
echo "  manifest:     ${MANIFEST}"
echo "  cache root:   ${CACHE_ROOT}"
echo "  run name:     ${RUN_NAME}"
echo "  total epochs: ${TOTAL_EPOCHS}"
echo "  shard epochs: ${BRIDGEDP_SHARD_EPOCHS}"
echo "  gpus:         ${GPUS}"
echo "  preset:       ${PRESET}"

while true; do
    if [[ "${MAX_STAGES}" -gt 0 && "${STAGE}" -ge "${MAX_STAGES}" ]]; then
        echo "Reached --max-stages=${MAX_STAGES}; stop."
        break
    fi

    SLOT_NAME="cache_A"
    INACTIVE_SLOT_NAME="cache_B"
    if [[ $((STAGE % 2)) -eq 1 ]]; then
        SLOT_NAME="cache_B"
        INACTIVE_SLOT_NAME="cache_A"
    fi
    SLOT_ROOT="${CACHE_ROOT}/${SLOT_NAME}"
    INACTIVE_SLOT_ROOT="${CACHE_ROOT}/${INACTIVE_SLOT_NAME}"

    echo
    echo "=== Stage ${STAGE}: shard ${SHARD_INDEX} -> ${SLOT_NAME} ==="
    if [[ -n "${BUILD_PID}" ]]; then
        wait_for_ready "${SLOT_ROOT}" "${BUILD_PID}"
        BUILD_PID=""
    else
        build_slot "${SHARD_INDEX}" "${SLOT_ROOT}"
    fi

    eval "$(python "${PROJECT_ROOT}/scripts/cache_rotation/plan_bridgedp_cache_stage.py" \
        --manifest "${MANIFEST}" \
        --shard-index "${SHARD_INDEX}" \
        --output-dir "${OUTPUT_DIR}" \
        --gpus "${GPUS}" \
        --nvme-size "${BRIDGEDP_NVME_SIZE}" \
        --preset "${PRESET}" \
        --total-epochs "${TOTAL_EPOCHS}" \
        --shard-epochs "${BRIDGEDP_SHARD_EPOCHS}" \
        --per-gpu-batch "${BRIDGEDP_BATCH_SIZE}" \
        --grad-accum "${BRIDGEDP_GRAD_ACCUM}")"

    if [[ "${BRIDGEDP_CURRENT_GLOBAL_STEP}" -ge "${BRIDGEDP_TOTAL_MAX_STEPS}" ]]; then
        echo "Training already reached total max steps: ${BRIDGEDP_CURRENT_GLOBAL_STEP}/${BRIDGEDP_TOTAL_MAX_STEPS}"
        break
    fi

    NEXT_SHARD=$(((SHARD_INDEX + 1) % NUM_SHARDS))
    echo "Prebuilding next shard ${NEXT_SHARD} into ${INACTIVE_SLOT_NAME}"
    build_slot "${NEXT_SHARD}" "${INACTIVE_SLOT_ROOT}" >"${LOG_DIR}/build_stage_${STAGE}_next.log" 2>&1 &
    BUILD_PID="$!"

    export BRIDGEDP_CACHE_STAGE_ID="stage_${STAGE}_shard_${SHARD_INDEX}_${SLOT_NAME}"
    export BRIDGEDP_CACHE_SLOT_ROOT="${SLOT_ROOT}"
    export BRIDGEDP_DATASET_ROOT="${SLOT_ROOT}/vln_n1/traj_data"
    export BRIDGEDP_PRELOAD_INDEX="${SLOT_ROOT}/preload_index.json"
    export BRIDGEDP_TOTAL_MAX_STEPS
    export BRIDGEDP_STAGE_END_STEP

    echo "$$" > "${SLOT_ROOT}/.ACTIVE"
    set +e
    torchrun \
        --nproc_per_node="${GPUS}" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR:-localhost}" \
        --master_port="${MASTER_PORT:-12345}" \
        scripts/train/base_train/train_cache_rotation.py \
        --name "${RUN_NAME}"
    TRAIN_STATUS="$?"
    set -e
    rm -f "${SLOT_ROOT}/.ACTIVE"

    if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
        echo "Training failed at stage ${STAGE}; keeping caches for diagnosis." >&2
        exit "${TRAIN_STATUS}"
    fi

    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "${OUTPUT_DIR}/" "${BACKUP_DIR}/"
    else
        cp -a "${OUTPUT_DIR}/." "${BACKUP_DIR}/"
    fi

    SHARD_INDEX="${NEXT_SHARD}"
    STAGE=$((STAGE + 1))
done

echo "Bridge-DP cache rotation training finished."
echo "  output: ${OUTPUT_DIR}"
echo "  backup: ${BACKUP_DIR}"
