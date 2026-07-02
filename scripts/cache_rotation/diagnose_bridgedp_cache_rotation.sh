#!/usr/bin/env bash
set -euo pipefail

HDD_ROOT="${HDD_ROOT:-/hdd}"
NVME_ROOT="${NVME_ROOT:-/nvme}"
RUN_NAME="${BRIDGEDP_RUN_NAME:-bridgedp_cache_rotation}"
PROJECT_ROOT="${BRIDGEDP_PROJECT_ROOT:-${NVME_ROOT}/MyResearch/InternNav}"
CACHE_ROOT="${NVME_ROOT}/bridgedp_cache"

echo "Bridge-DP cache rotation diagnosis"
echo "  hdd root:     ${HDD_ROOT}"
echo "  nvme root:    ${NVME_ROOT}"
echo "  project root: ${PROJECT_ROOT}"
echo "  run name:     ${RUN_NAME}"

echo
echo "[mounts]"
findmnt -T "${HDD_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true
findmnt -T "${NVME_ROOT}" -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE% || true

echo
echo "[space]"
df -h "${HDD_ROOT}" "${NVME_ROOT}" || true
df -i "${HDD_ROOT}" "${NVME_ROOT}" || true

echo
echo "[cache slots]"
for slot in cache_A cache_B; do
    slot_root="${CACHE_ROOT}/${slot}"
    echo "--- ${slot_root}"
    ls -la "${slot_root}" 2>/dev/null | head -40 || true
    if [[ -f "${slot_root}/metadata.json" ]]; then
        python -m json.tool "${slot_root}/metadata.json" || true
    fi
    if [[ -f "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" ]]; then
        PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python "${PROJECT_ROOT}/scripts/cache_rotation/check_bridgedp_cache.py" --slot-root "${slot_root}" || true
    fi
done

echo
echo "[latest checkpoints]"
find "${PROJECT_ROOT}/checkpoints/${RUN_NAME}/ckpts" -maxdepth 2 -name trainer_state.json -print 2>/dev/null | sort | tail -5 || true

echo
echo "[gpu]"
nvidia-smi || true

echo
echo "[io]"
iostat -xz 1 2 || true
