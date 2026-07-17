#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh" "b48_w2" "$@"
