#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
    echo "Usage: $0 {b48_w2|b24_ga2_w2_pf1|b24_ga2_w4_pf1} [extra train.py arguments...]" >&2
    exit 2
fi
shift

case "${PROFILE}" in
    b48_w2)
        PROFILE_BATCH_SIZE=48
        PROFILE_GRAD_ACCUM=1
        PROFILE_NUM_WORKERS=2
        PROFILE_PREFETCH_FACTOR=2
        PROFILE_RUN_NAME="arcdp_full1_b48_w2_8x4090_ceph"
        PROFILE_MASTER_PORT=12346
        ;;
    b24_ga2_w2_pf1)
        PROFILE_BATCH_SIZE=24
        PROFILE_GRAD_ACCUM=2
        PROFILE_NUM_WORKERS=2
        PROFILE_PREFETCH_FACTOR=1
        PROFILE_RUN_NAME="arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph"
        PROFILE_MASTER_PORT=12347
        ;;
    b24_ga2_w4_pf1)
        PROFILE_BATCH_SIZE=24
        PROFILE_GRAD_ACCUM=2
        PROFILE_NUM_WORKERS=4
        PROFILE_PREFETCH_FACTOR=1
        PROFILE_RUN_NAME="arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph"
        PROFILE_MASTER_PORT=12348
        ;;
    *)
        echo "Unknown ArcDP Ceph I/O profile: ${PROFILE}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export ARCDP_NVME_ROOT="${ARCDP_NVME_ROOT:-/nvme}"
export BRIDGEDP_SSD_ROOT="${BRIDGEDP_SSD_ROOT:-${ARCDP_NVME_ROOT}}"
export BRIDGEDP_PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
export BRIDGEDP_DATASET_BASE="${BRIDGEDP_DATASET_BASE:-${ARCDP_NVME_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1}"
export BRIDGEDP_DATASET_ROOT="${BRIDGEDP_DATASET_ROOT:-${BRIDGEDP_DATASET_BASE}/vln_n1/traj_data}"
export BRIDGEDP_PRELOAD_INDEX="${BRIDGEDP_PRELOAD_INDEX:-${BRIDGEDP_PROJECT_ROOT}/checkpoints/preload_index.json}"

export BRIDGEDP_NUM_GPUS=8
export BRIDGEDP_EPOCHS=1
export BRIDGEDP_BATCH_SIZE="${PROFILE_BATCH_SIZE}"
export BRIDGEDP_GRAD_ACCUM="${PROFILE_GRAD_ACCUM}"
export BRIDGEDP_NUM_WORKERS="${PROFILE_NUM_WORKERS}"
export BRIDGEDP_PREFETCH_FACTOR="${PROFILE_PREFETCH_FACTOR}"
export BRIDGEDP_LR="${BRIDGEDP_LR:-3e-4}"
export BRIDGEDP_SAVE_INTERVAL_EPOCHS=1
export BRIDGEDP_SAVE_TOTAL_LIMIT="${BRIDGEDP_SAVE_TOTAL_LIMIT:-3}"
export BRIDGEDP_AUTO_RESUME="${BRIDGEDP_AUTO_RESUME:-1}"
export BRIDGEDP_RESUME_FROM="${BRIDGEDP_RESUME_FROM:-}"
export BRIDGEDP_LOGGING_STEPS="${BRIDGEDP_LOGGING_STEPS:-10}"
export BRIDGEDP_ETA_LOG_STEPS="${BRIDGEDP_ETA_LOG_STEPS:-10}"
export BRIDGEDP_RUN_NAME="${PROFILE_RUN_NAME}"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="${BRIDGEDP_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

export BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"
SWANLAB_PROJECT_VALUE="${ARCDP_SWANLAB_PROJECT:-ArcDP-8x4090-full1-ceph-io}"
unset SWANLAB_PROJECT
export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT_VALUE}"
export SWANLAB_EXP_NAME="${PROFILE_RUN_NAME}"

MASTER_PORT_VALUE="${ARCDP_MASTER_PORT:-${PROFILE_MASTER_PORT}}"
effective_global_batch=$((BRIDGEDP_BATCH_SIZE * BRIDGEDP_NUM_GPUS * BRIDGEDP_GRAD_ACCUM))

TRAIN_COMMAND=(
    torchrun
    --nproc_per_node="${BRIDGEDP_NUM_GPUS}"
    --nnodes=1
    --node_rank=0
    --master_addr="${MASTER_ADDR:-localhost}"
    --master_port="${MASTER_PORT_VALUE}"
    scripts/train/base_train/train.py
    --name "${BRIDGEDP_RUN_NAME}"
    --model-name bridgedp_full_8x4090
)
if [[ "$#" -gt 0 ]]; then
    TRAIN_COMMAND+=("$@")
fi

echo "ArcDP full1 8x4090 Ceph I/O profile"
echo "  profile: ${PROFILE}"
echo "  run: ${BRIDGEDP_RUN_NAME}"
echo "  epochs: ${BRIDGEDP_EPOCHS}"
echo "  project: ${BRIDGEDP_PROJECT_ROOT}"
echo "  dataset: ${BRIDGEDP_DATASET_ROOT}"
echo "  preload: ${BRIDGEDP_PRELOAD_INDEX}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES} (${BRIDGEDP_NUM_GPUS} processes)"
echo "  per-device batch: ${BRIDGEDP_BATCH_SIZE}"
echo "  gradient accumulation: ${BRIDGEDP_GRAD_ACCUM}"
echo "  effective global batch: ${effective_global_batch}"
echo "  workers/process: ${BRIDGEDP_NUM_WORKERS}"
echo "  prefetch factor/worker: ${BRIDGEDP_PREFETCH_FACTOR}"
echo "  report_to: ${BRIDGEDP_REPORT_TO}"
echo "  SwanLab project: ${SWANLAB_PROJ_NAME}"
echo "  SwanLab experiment: ${SWANLAB_EXP_NAME}"
echo "  checkpoint root: ${BRIDGEDP_PROJECT_ROOT}/checkpoints/${BRIDGEDP_RUN_NAME}"

if [[ "${ARCDP_DRY_RUN:-0}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${TRAIN_COMMAND[@]}"
    printf '\n'
    exit 0
fi

cd "${BRIDGEDP_PROJECT_ROOT}"
mkdir -p logs

if [[ ! -d "${BRIDGEDP_DATASET_ROOT}" ]]; then
    echo "Missing full dataset root: ${BRIDGEDP_DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BRIDGEDP_PRELOAD_INDEX}" ]]; then
    echo "Missing preload index: ${BRIDGEDP_PRELOAD_INDEX}" >&2
    exit 1
fi
if [[ ! -f "${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" ]]; then
    echo "Missing DepthAnything encoder: ${BRIDGEDP_PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" >&2
    exit 1
fi

if command -v findmnt >/dev/null 2>&1; then
    echo "Dataset filesystem:"
    findmnt -T "${BRIDGEDP_DATASET_ROOT}" -o TARGET,SOURCE,FSTYPE,OPTIONS || true
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
if [[ "${gpu_count}" -lt "${BRIDGEDP_NUM_GPUS}" ]]; then
    echo "Need ${BRIDGEDP_NUM_GPUS} GPUs, but nvidia-smi reports ${gpu_count}." >&2
    exit 1
fi

busy_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -u | paste -sd, -)"
if [[ -n "${busy_pids}" && "${ARCDP_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
    echo "Refusing to start because GPU compute processes already exist: ${busy_pids}" >&2
    echo "Stop the existing training first, or explicitly set ARCDP_ALLOW_BUSY_GPUS=1." >&2
    exit 1
fi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 8, f"expected 8 visible GPUs, got {torch.cuda.device_count()}"
print("python/torch preflight:", torch.__version__, torch.version.cuda, torch.cuda.device_count())
PY

if [[ "${BRIDGEDP_REPORT_TO,,}" == *swanlab* ]]; then
    if ! python -c "import swanlab" >/dev/null; then
        echo "SwanLab is enabled but the Python package is not importable." >&2
        echo "Activate arcdp and run: python -m pip install swanlab" >&2
        exit 1
    fi
    if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
        echo "[SwanLab][WARN] SWANLAB_API_KEY is not set; use an existing login or run: swanlab login" >&2
    fi
fi

exec "${TRAIN_COMMAND[@]}"
