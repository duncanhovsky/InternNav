#!/usr/bin/env bash
set -uo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_PROJECT_ROOT="${PROJECT_ROOT}"
PY_PATH_SEP=":"
if command -v cygpath >/dev/null 2>&1; then
    PY_PROJECT_ROOT="$(cygpath -w "${PROJECT_ROOT}" 2>/dev/null || printf '%s' "${PROJECT_ROOT}")"
elif [[ "${PROJECT_ROOT}" == /[a-zA-Z]/* ]]; then
    PY_DRIVE="${PROJECT_ROOT:1:1}"
    PY_REST="${PROJECT_ROOT:3}"
    PY_PROJECT_ROOT="${PY_DRIVE}:/${PY_REST}"
fi
case "${PY_PROJECT_ROOT}" in
    [a-zA-Z]:/*|[a-zA-Z]:\\*) PY_PATH_SEP=";" ;;
esac

GPUS=""
VARIANT="full"
EPOCHS="100"
NVME_SIZE="2tb"
PRESET="balanced"
HDD_ROOT="/hdd"
NVME_ROOT="/nvme"
ENABLE_SWANLAB=1

FAILURES=0
WARNINGS=0
PROFILE_READY=0
BRIDGEDP_NVME_SIZE=""
BRIDGEDP_SHARD_EPOCHS=""
BRIDGEDP_BATCH_SIZE=""
BRIDGEDP_GRAD_ACCUM=""

usage() {
    printf '%s\n' \
        "Usage:" \
        "  check_arcdp_training_ready.sh --gpus 4|8 --variant full|rel|no_bridge|no_ordered_init|no_scale_cond|no_anchor_train|no_gcs --epochs 100|200|500|1000 [options]" \
        "" \
        "Options:" \
        "  --nvme-size 1tb|2tb|4tb       NVMe cache profile size. Default: 2tb" \
        "  --preset balanced|quality|throughput" \
        "                                Cache rotation profile. Default: balanced" \
        "  --hdd-root PATH               HDD root used by cache rotation. Default: /hdd" \
        "  --nvme-root PATH              NVMe root used by cache rotation. Default: /nvme" \
        "  --no-swanlab                  Skip SwanLab package/key checks." \
        "  -h, --help                    Show this help."
}

pass() {
    printf '[OK]   %s\n' "$1"
}

warn() {
    WARNINGS=$((WARNINGS + 1))
    printf '[WARN] %s\n' "$1"
    if [[ "${2:-}" != "" ]]; then
        printf '       fix: %s\n' "$2"
    fi
}

fail() {
    FAILURES=$((FAILURES + 1))
    printf '[FAIL] %s\n' "$1"
    if [[ "${2:-}" != "" ]]; then
        printf '       fix: %s\n' "$2"
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    local fix="$3"
    if [[ -f "${path}" ]]; then
        pass "${label}: ${path}"
    else
        fail "${label} missing: ${path}" "${fix}"
    fi
}

require_dir() {
    local path="$1"
    local label="$2"
    local fix="$3"
    if [[ -d "${path}" ]]; then
        pass "${label}: ${path}"
    else
        fail "${label} missing: ${path}" "${fix}"
    fi
}

require_cmd() {
    local cmd="$1"
    local fix="$2"
    if command -v "${cmd}" >/dev/null 2>&1; then
        pass "command found: ${cmd}"
    else
        fail "command not found: ${cmd}" "${fix}"
    fi
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
        --no-swanlab) ENABLE_SWANLAB=0; shift ;;
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
    100|200|500|1000) ;;
    *) echo "--epochs must be one of 100, 200, 500, 1000; got: ${EPOCHS}" >&2; exit 1 ;;
esac

case "${NVME_SIZE}" in
    1tb|2tb|4tb) ;;
    *) echo "--nvme-size must be 1tb, 2tb, or 4tb; got: ${NVME_SIZE}" >&2; exit 1 ;;
esac

case "${PRESET}" in
    balanced|quality|throughput) ;;
    *) echo "--preset must be balanced, quality, or throughput; got: ${PRESET}" >&2; exit 1 ;;
esac

PREPARE_CMD="bash ${PROJECT_ROOT}/scripts/cache_rotation/prepare_bridgedp_cache_rotation.sh --gpus ${GPUS} --nvme-size ${NVME_SIZE} --preset ${PRESET} --hdd-root ${HDD_ROOT} --nvme-root ${NVME_ROOT}"
TRAIN_CMD="bash ${SCRIPT_DIR}/train_arcdp_cache_rotation_${GPUS}a800.sh --variant ${VARIANT} --epochs ${EPOCHS} --nvme-size ${NVME_SIZE} --preset ${PRESET} --hdd-root ${HDD_ROOT} --nvme-root ${NVME_ROOT}"

printf '%s\n' \
    "ArcDP training readiness check" \
    "  project:    ${PROJECT_ROOT}" \
    "  pythonpath: ${PY_PROJECT_ROOT}" \
    "  variant:    ${VARIANT}" \
    "  gpus:       ${GPUS}" \
    "  epochs(eq): ${EPOCHS}" \
    "  profile:    ${NVME_SIZE}/${PRESET}" \
    "  hdd root:   ${HDD_ROOT}" \
    "  nvme root:  ${NVME_ROOT}" \
    "  swanlab:    ${ENABLE_SWANLAB}"

printf '\n[1/8] Scripts and project files\n'
require_file "${SCRIPT_DIR}/train_arcdp_cache_rotation.sh" "ArcDP launcher" "Restore scripts/train/arcdp_cache_rotation/train_arcdp_cache_rotation.sh."
require_file "${SCRIPT_DIR}/train_arcdp_cache_rotation_${GPUS}a800.sh" "A800 launcher" "Use --gpus 4 or --gpus 8, or restore the matching launcher."
require_file "${PROJECT_ROOT}/train_bridgedp_cache_rotation.sh" "cache rotation trainer" "Restore train_bridgedp_cache_rotation.sh."
require_file "${PROJECT_ROOT}/scripts/train/base_train/train_cache_rotation.py" "cache rotation Python entry" "Restore scripts/train/base_train/train_cache_rotation.py."
require_file "${PROJECT_ROOT}/scripts/train/base_train/configs/bridgedp_cache_rotation.py" "ArcDP cache config" "Restore scripts/train/base_train/configs/bridgedp_cache_rotation.py."
require_file "${PROJECT_ROOT}/scripts/cache_rotation/prepare_bridgedp_cache_rotation.sh" "cache preparation script" "Restore scripts/cache_rotation/prepare_bridgedp_cache_rotation.sh."
require_file "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" "cache validator" "Restore scripts/cache_rotation/check_bridgedp_cache.py."
require_file "${PROJECT_ROOT}/checkpoints/depth_anything_v2_vits.pth" "DepthAnything encoder" "Run: ${PREPARE_CMD}; or copy depth_anything_v2_vits.pth into ${PROJECT_ROOT}/checkpoints/."

printf '\n[2/8] Shell commands\n'
require_cmd python "Activate the ArcDP training conda/env and install dependencies: pip install -r ${PROJECT_ROOT}/requirements/model_requirements.txt"
require_cmd torchrun "Install PyTorch distributed tools in the active env: pip install torch"
require_cmd nvidia-smi "Load NVIDIA driver/CUDA runtime on the training node."
if command -v rsync >/dev/null 2>&1; then
    pass "command found: rsync"
else
    warn "command not found: rsync" "Training can fall back to cp -a, but prepare/copy is faster with rsync."
fi

printf '\n[3/8] Cache profile\n'
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${PY_PROJECT_ROOT}${PY_PATH_SEP}${PYTHONPATH}"
else
    export PYTHONPATH="${PY_PROJECT_ROOT}"
fi
PROFILE_OUTPUT="$(python "${PROJECT_ROOT}/scripts/cache_rotation/profile_bridgedp_cache.py" --gpus "${GPUS}" --nvme-size "${NVME_SIZE}" --preset "${PRESET}" 2>&1)"
PROFILE_STATUS="$?"
if [[ "${PROFILE_STATUS}" -eq 0 ]]; then
    eval "${PROFILE_OUTPUT}"
    BRIDGEDP_NVME_SIZE="${BRIDGEDP_NVME_SIZE:-${NVME_SIZE}}"
    PROFILE_READY=1
    pass "cache profile resolved: gpus=${BRIDGEDP_NUM_GPUS}, nvme=${BRIDGEDP_NVME_SIZE}, shard_epochs=${BRIDGEDP_SHARD_EPOCHS}, batch=${BRIDGEDP_BATCH_SIZE}, grad_accum=${BRIDGEDP_GRAD_ACCUM}"
else
    BRIDGEDP_NVME_SIZE="${NVME_SIZE}"
    fail "cache profile failed" "Check --gpus/--nvme-size/--preset, then rerun this checker. Error: ${PROFILE_OUTPUT}"
fi

printf '\n[4/8] Python packages and ArcDP config\n'
if python -c "import torch, transformers, accelerate, diffusers, tyro, pydantic, pandas, pyarrow; import internnav" >/dev/null 2>&1; then
    pass "core Python packages import"
else
    fail "core Python package import failed" "Activate the training env, then run: pip install -r ${PROJECT_ROOT}/requirements/model_requirements.txt"
fi

if ARCDP_CACHE_VARIANT="${VARIANT}" BRIDGEDP_NUM_GPUS="${GPUS}" python -c "from scripts.train.base_train.configs.bridgedp_cache_rotation import bridgedp_cache_rotation_exp_cfg as cfg; assert cfg.arcdp_cache_variant == '${VARIANT}'; assert len(cfg.torch_gpu_ids) == ${GPUS}" >/dev/null 2>&1; then
    pass "ArcDP cache config imports with variant=${VARIANT} and gpus=${GPUS}"
else
    fail "ArcDP cache config import failed" "Check Python dependencies and ARCDP_CACHE_VARIANT. Try: python scripts/train/base_train/train_cache_rotation.py --help"
fi

printf '\n[5/8] GPU and CUDA\n'
if python -c "import sys, torch; n=torch.cuda.device_count(); print(n); sys.exit(0 if torch.cuda.is_available() and n >= ${GPUS} else 1)" >/dev/null 2>&1; then
    pass "PyTorch sees at least ${GPUS} CUDA GPUs"
else
    fail "PyTorch does not see ${GPUS} CUDA GPUs" "Check CUDA_VISIBLE_DEVICES, NVIDIA driver, CUDA/PyTorch build, or choose --gpus 4/8 to match the node."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        pass "nvidia-smi can query GPUs"
    else
        fail "nvidia-smi exists but cannot query GPUs" "Check NVIDIA driver and node allocation."
    fi
fi

printf '\n[6/8] SwanLab\n'
if [[ "${ENABLE_SWANLAB}" -eq 1 ]]; then
    if python -c "import swanlab" >/dev/null 2>&1; then
        pass "swanlab Python package imports"
    else
        fail "swanlab Python package missing" "Run: pip install swanlab; or reinstall requirements/model_requirements.txt."
    fi
    if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
        pass "SWANLAB_API_KEY is set for non-interactive cloud logging"
    else
        warn "SWANLAB_API_KEY is not set" "For cloud logging on a cluster, run: export SWANLAB_API_KEY=<your_api_key>; or run interactively once: swanlab login <your_api_key>."
    fi
else
    pass "SwanLab checks skipped by --no-swanlab"
fi

printf '\n[7/8] HDD dataset, manifest, and NVMe cache\n'
HDD_TRAJ="${HDD_ROOT}/datasets/InternData-N1/v0.5-full-vln-n1/vln_n1/traj_data"
MANIFEST="${HDD_ROOT}/bridgedp_rotation/manifests/shards_${GPUS}gpu_${BRIDGEDP_NVME_SIZE}_${PRESET}.json"
CACHE_ROOT="${NVME_ROOT}/bridgedp_cache"

require_dir "${HDD_ROOT}" "HDD root" "Mount or create ${HDD_ROOT}, then place the extracted dataset under ${HDD_TRAJ}."
require_dir "${NVME_ROOT}" "NVMe root" "Mount or create ${NVME_ROOT}, then run: ${PREPARE_CMD}"
require_dir "${HDD_TRAJ}" "extracted HDD trajectory dataset" "Extract/mount InternData-N1 to ${HDD_TRAJ}; after that run: ${PREPARE_CMD}"

if [[ -f "${MANIFEST}" ]]; then
    if python -c "import json, sys; m=json.load(open('${MANIFEST}', encoding='utf-8')); assert m.get('summary', {}).get('total_episodes', 0) > 0; assert len(m.get('shards', [])) > 0" >/dev/null 2>&1; then
        pass "manifest is valid: ${MANIFEST}"
    else
        fail "manifest is invalid: ${MANIFEST}" "Regenerate it with: ${PREPARE_CMD}"
    fi
else
    fail "manifest missing: ${MANIFEST}" "Generate it with: ${PREPARE_CMD}"
fi

NEED_CACHE_B=1
if [[ -f "${MANIFEST}" ]]; then
    NUM_SHARDS="$(python -c "import json; print(len(json.load(open('${MANIFEST}', encoding='utf-8')).get('shards', [])))" 2>/dev/null || printf '2')"
    if [[ "${NUM_SHARDS}" -le 1 ]]; then
        NEED_CACHE_B=0
    fi
fi

check_cache_slot() {
    local slot_name="$1"
    local required="$2"
    local slot_root="${CACHE_ROOT}/${slot_name}"
    if [[ ! -d "${slot_root}" ]]; then
        if [[ "${required}" == "1" ]]; then
            fail "${slot_name} missing: ${slot_root}" "Build cache slots with: ${PREPARE_CMD}"
        else
            warn "${slot_name} missing but manifest has one shard" "No action required unless you want double-buffered cache rotation."
        fi
        return
    fi
    if python "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" --slot-root "${slot_root}" >/dev/null 2>&1; then
        pass "${slot_name} is READY and preload paths validate"
    else
        if [[ "${required}" == "1" ]]; then
            fail "${slot_name} is not READY or preload paths are invalid" "Rebuild cache slots with: ${PREPARE_CMD} --force"
        else
            warn "${slot_name} is not READY" "No action required for a one-shard manifest; otherwise run: ${PREPARE_CMD} --force"
        fi
    fi
}

check_cache_slot "cache_A" "1"
check_cache_slot "cache_B" "${NEED_CACHE_B}"

printf '\n[8/8] Launcher dry-run\n'
DRY_RUN_ARGS=(--variant "${VARIANT}" --epochs "${EPOCHS}" --nvme-size "${NVME_SIZE}" --preset "${PRESET}" --hdd-root "${HDD_ROOT}" --nvme-root "${NVME_ROOT}" --dry-run)
if [[ "${ENABLE_SWANLAB}" -eq 0 ]]; then
    DRY_RUN_ARGS+=(--no-swanlab)
elif [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    DRY_RUN_ARGS+=(--no-swanlab)
fi

if "${BASH:-bash}" "${SCRIPT_DIR}/train_arcdp_cache_rotation_${GPUS}a800.sh" "${DRY_RUN_ARGS[@]}" >/dev/null 2>&1; then
    pass "training launcher dry-run parses"
else
    fail "training launcher dry-run failed" "Run manually to inspect: ${TRAIN_CMD} --dry-run"
fi

printf '\nSummary\n'
printf '  failures: %s\n' "${FAILURES}"
printf '  warnings: %s\n' "${WARNINGS}"
printf '  prepare:  %s\n' "${PREPARE_CMD}"
printf '  train:    %s\n' "${TRAIN_CMD}"

if [[ "${FAILURES}" -eq 0 ]]; then
    printf '%s\n' "READY: hard prerequisites passed. Review warnings before a long run."
    exit 0
fi

printf '%s\n' "NOT READY: fix the failed items above, then rerun this checker."
exit 1
