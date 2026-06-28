#!/usr/bin/env bash
set -euo pipefail

export BRIDGEDP_SSD_ROOT="${BRIDGEDP_SSD_ROOT:-/ssd}"
export BRIDGEDP_PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${BRIDGEDP_SSD_ROOT}/MyResearch/InternNav}"
export BRIDGEDP_DATASET_ROOT="${BRIDGEDP_DATASET_ROOT:-${BRIDGEDP_SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
export BRIDGEDP_PRELOAD_INDEX="${BRIDGEDP_PRELOAD_INDEX:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/preload_index.json}"
export BRIDGEDP_RUN_NAME="${BRIDGEDP_RUN_NAME:-bridgedp_full}"
export BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-4}"
export BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-96}"
export BRIDGEDP_NUM_WORKERS="${BRIDGEDP_NUM_WORKERS:-10}"
export BRIDGEDP_LR="${BRIDGEDP_LR:-3e-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${BRIDGEDP_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

cd "${BRIDGEDP_PROJECT_ROOT}"

if [[ ! -d "${BRIDGEDP_DATASET_ROOT}" ]]; then
    echo "Missing dataset root: ${BRIDGEDP_DATASET_ROOT}" >&2
    echo "Run scripts/prepare_ssd_full.sh before launching training." >&2
    exit 1
fi

if [[ ! -f "${BRIDGEDP_PRELOAD_INDEX}" ]]; then
    echo "Missing preload index: ${BRIDGEDP_PRELOAD_INDEX}" >&2
    echo "Run scripts/prepare_ssd_full.sh before launching training." >&2
    exit 1
fi

if [[ ! -f "${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    echo "Missing DepthAnything encoder: ${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" >&2
    exit 1
fi

echo "Bridge-DP full training"
echo "  project: ${BRIDGEDP_PROJECT_ROOT}"
echo "  dataset: ${BRIDGEDP_DATASET_ROOT}"
echo "  preload: ${BRIDGEDP_PRELOAD_INDEX}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES} (${BRIDGEDP_NUM_GPUS} processes)"
echo "  per-device batch: ${BRIDGEDP_BATCH_SIZE}"
echo "  global batch: $((BRIDGEDP_BATCH_SIZE * BRIDGEDP_NUM_GPUS))"
echo "  workers/process: ${BRIDGEDP_NUM_WORKERS}"

torchrun \
    --nproc_per_node="${BRIDGEDP_NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR:-localhost}" \
    --master_port="${MASTER_PORT:-12345}" \
    scripts/train/base_train/train.py \
    --name "${BRIDGEDP_RUN_NAME}" \
    --model-name bridgedp_full
