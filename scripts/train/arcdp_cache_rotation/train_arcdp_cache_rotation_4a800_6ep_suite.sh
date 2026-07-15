#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"

NVME_SIZE="1.5tb"
PRESET="balanced"
HDD_ROOT="/ssd"
NVME_ROOT="/nvme"
SWANLAB_PROJECT_VALUE="${SWANLAB_PROJECT:-ArcDP-6ep}"
SWANLAB_WORKSPACE_VALUE="${SWANLAB_WORKSPACE:-}"
ENABLE_SWANLAB=1
DRY_RUN=0
ONLY_VARIANT=""
START_AT=""
FULL_EPOCHS="6"
ABLATION_EPOCHS="1"
SHARD_EPOCHS="10"
FULL_UNIFORM_CKPT_COUNT="6"
ABLATION_UNIFORM_CKPT_COUNT="1"

VARIANTS=(full rel no_bridge no_ordered_init no_scale_cond no_anchor_train no_gcs)

usage() {
    printf '%s\n' \
        "Usage:" \
        "  train_arcdp_cache_rotation_4a800_6ep_suite.sh [options]" \
        "" \
        "Runs full ArcDP for 6 equivalent epochs and all P0 ablations for 1 equivalent epoch on 4 x A800." \
        "The full run uses 6 uniform checkpoint archives, so the first archive is the 1-epoch full baseline." \
        "" \
        "Options:" \
        "  --nvme-size 1tb|1.5tb|2tb|4tb  Default: 1.5tb" \
        "  --preset balanced|quality|throughput" \
        "                                   Default: balanced" \
        "  --hdd-root PATH                  Default: /ssd" \
        "  --nvme-root PATH                 Default: /nvme" \
        "  --swanlab-project NAME           Default: ArcDP-6ep" \
        "  --swanlab-workspace NAME" \
        "  --full-epochs N                  Default: 6" \
        "  --ablation-epochs N              Default: 1" \
        "  --shard-epochs N                 Default: 10" \
        "  --full-uniform-ckpts N           Default: 6" \
        "  --ablation-uniform-ckpts N       Default: 1" \
        "  --only VARIANT                   Run one variant only." \
        "  --start-at VARIANT               Skip variants before VARIANT." \
        "  --no-swanlab                     Use tensorboard only." \
        "  --dry-run                        Print commands without launching." \
        "  -h, --help                       Show this help."
}

is_supported_variant() {
    local candidate="$1"
    local variant
    for variant in "${VARIANTS[@]}"; do
        if [[ "${variant}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

require_positive_int() {
    local label="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${label} must be a positive integer; got: ${value}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --hdd-root) HDD_ROOT="$2"; shift 2 ;;
        --nvme-root) NVME_ROOT="$2"; shift 2 ;;
        --swanlab-project) SWANLAB_PROJECT_VALUE="$2"; shift 2 ;;
        --swanlab-workspace) SWANLAB_WORKSPACE_VALUE="$2"; shift 2 ;;
        --full-epochs) FULL_EPOCHS="$2"; shift 2 ;;
        --ablation-epochs) ABLATION_EPOCHS="$2"; shift 2 ;;
        --shard-epochs) SHARD_EPOCHS="$2"; shift 2 ;;
        --full-uniform-ckpts) FULL_UNIFORM_CKPT_COUNT="$2"; shift 2 ;;
        --ablation-uniform-ckpts) ABLATION_UNIFORM_CKPT_COUNT="$2"; shift 2 ;;
        --only) ONLY_VARIANT="$2"; shift 2 ;;
        --start-at) START_AT="$2"; shift 2 ;;
        --no-swanlab) ENABLE_SWANLAB=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "${NVME_SIZE}" in
    1tb|1.5tb|2tb|4tb) ;;
    *) echo "--nvme-size must be 1tb, 1.5tb, 2tb, or 4tb; got: ${NVME_SIZE}" >&2; exit 1 ;;
esac

case "${PRESET}" in
    balanced|quality|throughput) ;;
    *) echo "--preset must be balanced, quality, or throughput; got: ${PRESET}" >&2; exit 1 ;;
esac

for epoch_value in "${FULL_EPOCHS}" "${ABLATION_EPOCHS}"; do
    case "${epoch_value}" in
        1|2|6|10|16|100|200|500|1000) ;;
        *) echo "suite epochs must be one of 1, 2, 6, 10, 16, 100, 200, 500, 1000; got: ${epoch_value}" >&2; exit 1 ;;
    esac
done
require_positive_int "--shard-epochs" "${SHARD_EPOCHS}"
require_positive_int "--full-uniform-ckpts" "${FULL_UNIFORM_CKPT_COUNT}"
require_positive_int "--ablation-uniform-ckpts" "${ABLATION_UNIFORM_CKPT_COUNT}"

if [[ -n "${ONLY_VARIANT}" ]] && ! is_supported_variant "${ONLY_VARIANT}"; then
    echo "Unsupported --only variant: ${ONLY_VARIANT}" >&2
    exit 1
fi
if [[ -n "${START_AT}" ]] && ! is_supported_variant "${START_AT}"; then
    echo "Unsupported --start-at variant: ${START_AT}" >&2
    exit 1
fi

printf '%s\n' \
    "ArcDP 4 x A800 full-6ep / P0-1ep suite" \
    "  variants:       ${VARIANTS[*]}" \
    "  full epochs:    ${FULL_EPOCHS}" \
    "  ablation epochs:${ABLATION_EPOCHS}" \
    "  shard epochs:   ${SHARD_EPOCHS}" \
    "  full ckpts:     ${FULL_UNIFORM_CKPT_COUNT} uniform archives" \
    "  ablation ckpts: ${ABLATION_UNIFORM_CKPT_COUNT} uniform archives" \
    "  nvme profile:   ${NVME_SIZE}/${PRESET}" \
    "  hdd root:       ${HDD_ROOT}" \
    "  nvme root:      ${NVME_ROOT}" \
    "  swanlab:        ${ENABLE_SWANLAB}" \
    "  swanlab project:${SWANLAB_PROJECT_VALUE}" \
    "  dry run:        ${DRY_RUN}"

started=0
if [[ -z "${START_AT}" ]]; then
    started=1
fi

for variant in "${VARIANTS[@]}"; do
    if [[ -n "${ONLY_VARIANT}" && "${variant}" != "${ONLY_VARIANT}" ]]; then
        continue
    fi
    if [[ "${started}" -eq 0 ]]; then
        if [[ "${variant}" == "${START_AT}" ]]; then
            started=1
        else
            continue
        fi
    fi

    variant_epochs="${FULL_EPOCHS}"
    variant_uniform_ckpts="${FULL_UNIFORM_CKPT_COUNT}"
    if [[ "${variant}" != "full" ]]; then
        variant_epochs="${ABLATION_EPOCHS}"
        variant_uniform_ckpts="${ABLATION_UNIFORM_CKPT_COUNT}"
    fi

    args=(
        --variant "${variant}"
        --epochs "${variant_epochs}"
        --nvme-size "${NVME_SIZE}"
        --preset "${PRESET}"
        --hdd-root "${HDD_ROOT}"
        --nvme-root "${NVME_ROOT}"
        --shard-epochs "${SHARD_EPOCHS}"
        --swanlab-project "${SWANLAB_PROJECT_VALUE}"
    )
    if [[ -n "${SWANLAB_WORKSPACE_VALUE}" ]]; then
        args+=(--swanlab-workspace "${SWANLAB_WORKSPACE_VALUE}")
    fi
    if [[ "${ENABLE_SWANLAB}" -eq 0 ]]; then
        args+=(--no-swanlab)
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        args+=(--dry-run)
    fi

    printf '\n%s\n' "=== Launch variant: ${variant} (${variant_epochs} equivalent epochs, shard_epochs=${SHARD_EPOCHS}) ==="
    BRIDGEDP_UNIFORM_CKPT_COUNT="${variant_uniform_ckpts}" \
        "${BASH:-bash}" "${SCRIPT_DIR}/train_arcdp_cache_rotation_4a800.sh" "${args[@]}"
done
