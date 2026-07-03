#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
exec "${BASH:-bash}" "${SCRIPT_DIR}/train_arcdp_cache_rotation.sh" --gpus "8" "$@"
