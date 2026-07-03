#!/usr/bin/env bash
set -euo pipefail

export BRIDGEDP_SSD_ROOT="${BRIDGEDP_SSD_ROOT:-/ssd}"
export BRIDGEDP_PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${BRIDGEDP_SSD_ROOT}/MyResearch/InternNav}"
export BRIDGEDP_DATASET_ROOT="${BRIDGEDP_DATASET_ROOT:-${BRIDGEDP_SSD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data}"
export BRIDGEDP_PRELOAD_INDEX="${BRIDGEDP_PRELOAD_INDEX:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/preload_index.json}"
export ARCDP_P0_CHECKPOINT_ROOT="${ARCDP_P0_CHECKPOINT_ROOT:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/arcdp_p0}"
export ARCDP_P0_RUN_NAME="${ARCDP_P0_RUN_NAME:-arcdp_p0_rel}"
export BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-4}"
export BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-96}"
export BRIDGEDP_NUM_WORKERS="${BRIDGEDP_NUM_WORKERS:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${BRIDGEDP_PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${BRIDGEDP_PROJECT_ROOT}"

torchrun --nproc_per_node="${BRIDGEDP_NUM_GPUS}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR:-localhost}" --master_port="${MASTER_PORT:-12345}" \
    scripts/train/base_train/train.py \
    --name "${ARCDP_P0_RUN_NAME}" \
    --model-name bridgedp_p0_rel
