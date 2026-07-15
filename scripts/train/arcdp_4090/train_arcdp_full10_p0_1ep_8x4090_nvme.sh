#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export ARCDP_NVME_ROOT="${ARCDP_NVME_ROOT:-/nvme}"
export BRIDGEDP_SSD_ROOT="${BRIDGEDP_SSD_ROOT:-${ARCDP_NVME_ROOT}}"
export BRIDGEDP_PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
export BRIDGEDP_DATASET_BASE="${BRIDGEDP_DATASET_BASE:-${ARCDP_NVME_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1}"
export BRIDGEDP_DATASET_ROOT="${BRIDGEDP_DATASET_ROOT:-${ARCDP_NVME_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
export BRIDGEDP_PRELOAD_INDEX="${BRIDGEDP_PRELOAD_INDEX:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/preload_index.json}"
export ARCDP_P0_CHECKPOINT_ROOT="${ARCDP_P0_CHECKPOINT_ROOT:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/arcdp_p0_4090_nvme}"

export BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-8}"
export BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-16}"
export BRIDGEDP_GRAD_ACCUM="${BRIDGEDP_GRAD_ACCUM:-3}"
export BRIDGEDP_NUM_WORKERS="${BRIDGEDP_NUM_WORKERS:-4}"
export BRIDGEDP_LR="${BRIDGEDP_LR:-3e-4}"
export BRIDGEDP_SAVE_INTERVAL_EPOCHS="${BRIDGEDP_SAVE_INTERVAL_EPOCHS:-1}"
export BRIDGEDP_SAVE_TOTAL_LIMIT="${BRIDGEDP_SAVE_TOTAL_LIMIT:-20}"
export BRIDGEDP_AUTO_RESUME="${BRIDGEDP_AUTO_RESUME:-1}"
export BRIDGEDP_RESUME_FROM="${BRIDGEDP_RESUME_FROM:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${BRIDGEDP_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

cd "${BRIDGEDP_PROJECT_ROOT}"

if [[ ! -d "${BRIDGEDP_DATASET_ROOT}" ]]; then
    echo "Missing full dataset root: ${BRIDGEDP_DATASET_ROOT}" >&2
    echo "Expected InternData-N1 v0.5-full extracted directly on NVMe." >&2
    exit 1
fi

if [[ ! -f "${BRIDGEDP_PRELOAD_INDEX}" ]]; then
    echo "Missing preload index: ${BRIDGEDP_PRELOAD_INDEX}" >&2
    echo "Generate it from ${BRIDGEDP_DATASET_ROOT} before launching training." >&2
    exit 1
fi

if [[ ! -f "${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    echo "Missing DepthAnything encoder: ${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" >&2
    exit 1
fi

run_stage() {
    local run_name="$1"
    local model_flag="$2"
    local model_name="$3"

    export BRIDGEDP_RUN_NAME="${run_name}"
    export ARCDP_P0_RUN_NAME="${run_name}"

    local global_batch=$((BRIDGEDP_BATCH_SIZE * BRIDGEDP_NUM_GPUS * BRIDGEDP_GRAD_ACCUM))
    echo "ArcDP 8x4090 NVMe stage"
    echo "  run: ${run_name}"
    echo "  model: ${model_name}"
    echo "  epochs: ${BRIDGEDP_EPOCHS}"
    echo "  project: ${BRIDGEDP_PROJECT_ROOT}"
    echo "  dataset: ${BRIDGEDP_DATASET_ROOT}"
    echo "  preload: ${BRIDGEDP_PRELOAD_INDEX}"
    echo "  gpus: ${CUDA_VISIBLE_DEVICES} (${BRIDGEDP_NUM_GPUS} processes)"
    echo "  per-device batch: ${BRIDGEDP_BATCH_SIZE}"
    echo "  grad accum: ${BRIDGEDP_GRAD_ACCUM}"
    echo "  effective global batch: ${global_batch}"
    echo "  workers/process: ${BRIDGEDP_NUM_WORKERS}"
    echo "  auto resume: ${BRIDGEDP_AUTO_RESUME}"

    local extra_args=()
    if [[ -n "${BRIDGEDP_RESUME_FROM}" ]]; then
        extra_args+=(--resume-from-checkpoint "${BRIDGEDP_RESUME_FROM}")
        echo "  requested checkpoint: ${BRIDGEDP_RESUME_FROM}"
    fi

    torchrun \
        --nproc_per_node="${BRIDGEDP_NUM_GPUS}" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR:-localhost}" \
        --master_port="${MASTER_PORT:-12345}" \
        scripts/train/base_train/train.py \
        --name "${run_name}" \
        "${model_flag}" "${model_name}" \
        "${extra_args[@]}"
}

export BRIDGEDP_EPOCHS=10
run_stage "arcdp_full10_8x4090_nvme" --model-name bridgedp_full_8x4090

export BRIDGEDP_EPOCHS=1
run_stage "arcdp_p0_rel_1ep_8x4090_nvme" --model-name bridgedp_p0_rel
run_stage "arcdp_p0_no_bridge_1ep_8x4090_nvme" --model-name bridgedp_p0_no_bridge
run_stage "arcdp_p0_no_ordered_init_1ep_8x4090_nvme" --model-name bridgedp_p0_no_ordered_init
run_stage "arcdp_p0_no_scale_cond_1ep_8x4090_nvme" --model-name bridgedp_p0_no_scale_cond
run_stage "arcdp_p0_no_anchor_train_1ep_8x4090_nvme" --model-name bridgedp_p0_no_anchor_train
run_stage "arcdp_p0_no_gcs_1ep_8x4090_nvme" --model-name bridgedp_p0_no_gcs
