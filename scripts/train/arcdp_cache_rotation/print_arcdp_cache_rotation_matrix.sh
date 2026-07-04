#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"

GPU_LIST=("4" "8")
NVME_SIZE="1.5tb"
PRESET="balanced"
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage:" \
        "  print_arcdp_cache_rotation_matrix.sh [--gpus 4|8|both] [--nvme-size 1tb|1.5tb|2tb|4tb] [--preset balanced|quality|throughput] [extra launcher args]" \
        "" \
        "Prints the ArcDP full/P0 cache-rotation training command matrix." \
        "It does not start training."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            case "$2" in
                4|8) GPU_LIST=("$2") ;;
                both) GPU_LIST=("4" "8") ;;
                *) echo "--gpus must be 4, 8, or both; got: $2" >&2; exit 1 ;;
            esac
            shift 2
            ;;
        --nvme-size) NVME_SIZE="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

VARIANTS=(full rel no_bridge no_ordered_init no_scale_cond no_anchor_train no_gcs)
EPOCHS=(10 100 200 500 1000)

for gpus in "${GPU_LIST[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            printf '%q' "${SCRIPT_DIR}/train_arcdp_cache_rotation_${gpus}a800.sh"
            printf ' --variant %q --epochs %q --nvme-size %q --preset %q' \
                "${variant}" "${epochs}" "${NVME_SIZE}" "${PRESET}"
            if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
                printf ' %q' "${EXTRA_ARGS[@]}"
            fi
            printf '\n'
        done
    done
done
