#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GPUS=""
VARIANT="full"
EPOCHS=""
NVME_SIZE="2tb"
PRESET="balanced"
HDD_ROOT="/hdd"
NVME_ROOT="/nvme"
RUN_NAME=""
START_SHARD=""
MAX_STAGES=""
ENABLE_SWANLAB=1
DRY_RUN=0
SWANLAB_PROJECT_VALUE="${SWANLAB_PROJECT:-${SWANLAB_PROJ_NAME:-ArcDP-cache-rotation}}"
SWANLAB_WORKSPACE_VALUE="${SWANLAB_WORKSPACE:-}"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  train_arcdp_cache_rotation.sh --gpus 4|8 --variant full|rel|no_bridge|no_ordered_init|no_scale_cond|no_anchor_train|no_gcs --epochs 10|100|200|500|1000 [options]" \
        "" \
        "Options:" \
        "  --nvme-size 1tb|1.5tb|2tb|4tb NVMe cache profile size. Default: 2tb" \
        "  --preset balanced|quality|throughput" \
        "                                Cache rotation profile. Default: balanced" \
        "  --hdd-root PATH               HDD root used by cache rotation. Default: /hdd" \
        "  --nvme-root PATH              NVMe root used by cache rotation. Default: /nvme" \
        "  --run-name NAME               Override checkpoint/log run name." \
        "  --start-shard N               Resume from shard index N." \
        "  --max-stages N                Stop after N cache-rotation stages." \
        "  --swanlab-project NAME        SwanLab project name. Default: ArcDP-cache-rotation" \
        "  --swanlab-workspace NAME      SwanLab workspace/entity name." \
        "  --no-swanlab                  Use tensorboard instead of SwanLab." \
        "  --dry-run                     Print the resolved command without launching." \
        "  -h, --help                    Show this help." \
        "" \
        "Equivalent epoch budgets:" \
        "  --epochs 10|100|200|500|1000 maps to cache-rotation total steps with:" \
        "  ceil(50 * total_episodes * epochs / (gpus * per_gpu_batch * grad_accum))."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --hdd-root) HDD_ROOT="$2"; shift 2 ;;
        --nvme-root) NVME_ROOT="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --start-shard) START_SHARD="$2"; shift 2 ;;
        --max-stages) MAX_STAGES="$2"; shift 2 ;;
        --swanlab-project) SWANLAB_PROJECT_VALUE="$2"; shift 2 ;;
        --swanlab-workspace) SWANLAB_WORKSPACE_VALUE="$2"; shift 2 ;;
        --no-swanlab) ENABLE_SWANLAB=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "${GPUS}" in
    4|8) ;;
    "") echo "--gpus is required." >&2; usage >&2; exit 1 ;;
    *) echo "--gpus must be 4 or 8, got: ${GPUS}" >&2; exit 1 ;;
esac

case "${VARIANT}" in
    full|rel|no_bridge|no_ordered_init|no_scale_cond|no_anchor_train|no_gcs) ;;
    *) echo "Unsupported --variant: ${VARIANT}" >&2; usage >&2; exit 1 ;;
esac

case "${EPOCHS}" in
    10|100|200|500|1000) ;;
    "") echo "--epochs is required." >&2; usage >&2; exit 1 ;;
    *) echo "--epochs must be one of 10, 100, 200, 500, 1000; got: ${EPOCHS}" >&2; exit 1 ;;
esac

case "${NVME_SIZE}" in
    1tb|1.5tb|2tb|4tb) ;;
    *) echo "--nvme-size must be 1tb, 1.5tb, 2tb, or 4tb; got: ${NVME_SIZE}" >&2; exit 1 ;;
esac

case "${PRESET}" in
    balanced|quality|throughput) ;;
    *) echo "--preset must be balanced, quality, or throughput; got: ${PRESET}" >&2; exit 1 ;;
esac

if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="arcdp_${VARIANT}_${EPOCHS}ep_${GPUS}a800_${NVME_SIZE}_${PRESET}"
fi

export ARCDP_CACHE_VARIANT="${VARIANT}"
export BRIDGEDP_LOGGING_STEPS="${BRIDGEDP_LOGGING_STEPS:-100}"
export BRIDGEDP_ETA_LOG_STEPS="${BRIDGEDP_ETA_LOG_STEPS:-500}"
export BRIDGEDP_SAVE_TOTAL_LIMIT="${BRIDGEDP_SAVE_TOTAL_LIMIT:-3}"
export BRIDGEDP_UNIFORM_CKPT_COUNT="${BRIDGEDP_UNIFORM_CKPT_COUNT:-20}"

if [[ "${ENABLE_SWANLAB}" -eq 1 ]]; then
    export BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"
    export SWANLAB_PROJECT="${SWANLAB_PROJECT_VALUE}"
    export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT_VALUE}"
    export SWANLAB_EXP_NAME="${RUN_NAME}"
    if [[ -n "${SWANLAB_WORKSPACE_VALUE}" ]]; then
        export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE_VALUE}"
    fi
    if ! python -c "import swanlab" >/dev/null 2>&1; then
        echo "SwanLab is enabled but the Python package is not importable." >&2
        echo "Install it in the training environment, for example: pip install swanlab" >&2
        echo "Or pass --no-swanlab to use tensorboard only." >&2
        exit 1
    fi
    printf '%s\n' \
        "[SwanLab] SwanLab enabled for this training run." \
        "[SwanLab] SwanLab project: ${SWANLAB_PROJECT}" \
        "[SwanLab] SwanLab experiment: ${SWANLAB_EXP_NAME}" \
        "[SwanLab] report_to=${BRIDGEDP_REPORT_TO}"
    if [[ -n "${SWANLAB_WORKSPACE_VALUE}" ]]; then
        printf '%s\n' "[SwanLab] SwanLab workspace: ${SWANLAB_WORKSPACE_VALUE}"
    fi
    if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
        printf '%s\n' "[SwanLab] SWANLAB_API_KEY is set; cloud logging can run non-interactively."
    else
        printf '%s\n' \
            "[SwanLab][WARN] SWANLAB_API_KEY is not set." \
            "[SwanLab][WARN] If this node has not run swanlab login before, set: export SWANLAB_API_KEY=<your_api_key>" \
            "[SwanLab][WARN] Or run once interactively: swanlab login <your_api_key>"
    fi
else
    export BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-tensorboard}"
    printf '%s\n' "[SwanLab] Disabled by --no-swanlab; report_to=${BRIDGEDP_REPORT_TO}"
fi

CMD=(
    "${BASH:-bash}" "${PROJECT_ROOT}/train_bridgedp_cache_rotation.sh"
    --gpus "${GPUS}"
    --nvme-size "${NVME_SIZE}"
    --total-epochs "${EPOCHS}"
    --preset "${PRESET}"
    --hdd-root "${HDD_ROOT}"
    --nvme-root "${NVME_ROOT}"
    --run-name "${RUN_NAME}"
)

if [[ -n "${START_SHARD}" ]]; then
    CMD+=(--start-shard "${START_SHARD}")
fi
if [[ -n "${MAX_STAGES}" ]]; then
    CMD+=(--max-stages "${MAX_STAGES}")
fi

printf '%s\n' \
    "ArcDP cache-rotation training" \
    "  project:      ${PROJECT_ROOT}" \
    "  variant:      ${VARIANT}" \
    "  gpus:         ${GPUS} x A800" \
    "  epochs(eq):   ${EPOCHS}" \
    "  nvme profile: ${NVME_SIZE}/${PRESET}" \
    "  run name:     ${RUN_NAME}" \
    "  report_to:    ${BRIDGEDP_REPORT_TO}" \
    "  log steps:    ${BRIDGEDP_LOGGING_STEPS}" \
    "  eta steps:    ${BRIDGEDP_ETA_LOG_STEPS}" \
    "  live ckpts:   keep latest ${BRIDGEDP_SAVE_TOTAL_LIMIT}" \
    "  uniform ckpt: ${BRIDGEDP_UNIFORM_CKPT_COUNT} evenly spaced archives" \
    "  formula:      ceil(50 * total_episodes * ${EPOCHS} / (${GPUS} * per_gpu_batch * grad_accum))"

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
    exit 0
fi

exec "${CMD[@]}"
