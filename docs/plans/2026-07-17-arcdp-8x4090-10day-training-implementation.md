# ArcDP 8×4090 Ten-Day Training Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an isolated 8×RTX 4090 launcher that trains full ArcDP for one epoch followed by the no-bridge ablation for one epoch, using batch 48, gradient accumulation 1, and SwanLab by default.

**Architecture:** Preserve the historical full10 launcher and add one new root wrapper plus one new implementation script. Reuse the existing `bridgedp_full_8x4090` and `bridgedp_p0_no_bridge` registered configs through environment overrides, so no model or trainer behavior changes are required. Protect the behavior with static entrypoint tests and document the launch, monitoring, OOM fallback, and timing assumptions separately from the old full10 manual.

**Tech Stack:** Bash, PyTorch `torchrun`, Hugging Face Trainer configuration, SwanLab, pytest.

---

### Task 1: Add failing entrypoint tests

**Files:**
- Modify: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`

**Step 1: Write the failing test for the new implementation script**

Append a test that reads `scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh` and verifies the complete contract:

```python
def test_arcdp_4090_ten_day_suite_runs_full1_and_no_bridge1_with_swanlab():
    script = (
        PROJECT_ROOT
        / "scripts"
        / "train"
        / "arcdp_4090"
        / "train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert 'BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-8}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in text
    assert 'BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-48}"' in text
    assert 'BRIDGEDP_GRAD_ACCUM="${BRIDGEDP_GRAD_ACCUM:-1}"' in text
    assert 'BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"' in text
    assert "ArcDP-8x4090-full1-no-bridge" in text
    assert 'python -c "import swanlab"' in text
    assert text.count("export BRIDGEDP_EPOCHS=1") == 2
    assert "--model-name bridgedp_full_8x4090" in text
    assert "--model-name bridgedp_p0_no_bridge" in text
    assert "bridgedp_p0_rel" not in text
    assert "bridgedp_p0_no_ordered_init" not in text
    assert "bridgedp_p0_no_scale_cond" not in text
    assert "bridgedp_p0_no_anchor_train" not in text
    assert "bridgedp_p0_no_gcs" not in text
```

**Step 2: Write the failing root-wrapper test**

Extend `test_arcdp_4090_root_wrappers_point_to_nvme_suite_scripts` with:

```python
"train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh": (
    "scripts/train/arcdp_4090/"
    "train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh"
),
```

**Step 3: Run the new test to verify RED**

Run:

```bash
pytest -q tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  -k "ten_day_suite or root_wrappers"
```

Expected: FAIL with `FileNotFoundError` for the new implementation script and/or root wrapper.

**Step 4: Commit the failing tests**

```bash
git add tests/unit_test/test_arcdp_4090_nvme_training_entries.py
git commit -m "test: specify ten-day 8x4090 ArcDP suite"
```

### Task 2: Implement the 8×4090 full1 + no-bridge launcher

**Files:**
- Create: `scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh`
- Create: `train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh`
- Test: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`

**Step 1: Create the implementation script**

Base path and prerequisite handling on the existing full10 8×4090 launcher, with these defaults:

```bash
export BRIDGEDP_NUM_GPUS="${BRIDGEDP_NUM_GPUS:-8}"
export BRIDGEDP_BATCH_SIZE="${BRIDGEDP_BATCH_SIZE:-48}"
export BRIDGEDP_GRAD_ACCUM="${BRIDGEDP_GRAD_ACCUM:-1}"
export BRIDGEDP_NUM_WORKERS="${BRIDGEDP_NUM_WORKERS:-4}"
export BRIDGEDP_LR="${BRIDGEDP_LR:-3e-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export BRIDGEDP_REPORT_TO="${BRIDGEDP_REPORT_TO:-swanlab}"
```

Resolve the SwanLab project while keeping environment overrides:

```bash
SWANLAB_PROJECT_VALUE="${SWANLAB_PROJECT:-${SWANLAB_PROJ_NAME:-ArcDP-8x4090-full1-no-bridge}}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT_VALUE}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT_VALUE}"
```

When `BRIDGEDP_REPORT_TO` contains `swanlab`, verify the package and print API-key/login guidance:

```bash
if [[ "${BRIDGEDP_REPORT_TO,,}" == *swanlab* ]]; then
    if ! python -c "import swanlab" >/dev/null 2>&1; then
        echo "SwanLab is enabled but the Python package is not importable." >&2
        echo "Install it in the ArcDP environment: python -m pip install swanlab" >&2
        exit 1
    fi
    if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
        echo "[SwanLab][WARN] SWANLAB_API_KEY is not set; use an existing login or run swanlab login." >&2
    fi
fi
```

Within `run_stage`, set a per-stage experiment name and print the resolved monitoring fields:

```bash
export SWANLAB_EXP_NAME="${run_name}"
echo "  report_to: ${BRIDGEDP_REPORT_TO}"
echo "  swanlab project: ${SWANLAB_PROJECT}"
echo "  swanlab experiment: ${SWANLAB_EXP_NAME}"
```

Launch only these stages:

```bash
export BRIDGEDP_EPOCHS=1
run_stage "arcdp_full1_bs${BRIDGEDP_BATCH_SIZE}_8x4090_nvme" \
  --model-name bridgedp_full_8x4090

export BRIDGEDP_EPOCHS=1
run_stage "arcdp_p0_no_bridge_1ep_bs${BRIDGEDP_BATCH_SIZE}_8x4090_nvme" \
  --model-name bridgedp_p0_no_bridge
```

Keep `set -euo pipefail`, existing dataset/preload/checkpoint preflight checks, `torchrun` options, environment overrides, and `BRIDGEDP_AUTO_RESUME` behavior.

**Step 2: Create the root wrapper**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh" "$@"
```

**Step 3: Run the targeted tests to verify GREEN**

Run:

```bash
pytest -q tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  -k "ten_day_suite or root_wrappers"
```

Expected: all selected tests PASS.

**Step 4: Run Bash syntax checks**

Run:

```bash
bash -n scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh
bash -n train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh
```

Expected: both commands exit 0 with no output.

**Step 5: Commit the launcher**

```bash
git add \
  scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh \
  train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh
git commit -m "feat: add ten-day 8x4090 ArcDP suite"
```

### Task 3: Add the operator guide

**Files:**
- Create: `docs/ArcDP-4090-full1-no-bridge-10天训练操作手册.md`
- Modify: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`

**Step 1: Write the failing documentation test**

Add:

```python
def test_arcdp_4090_ten_day_operator_guide_documents_launch_and_fallback():
    guide = PROJECT_ROOT / "docs" / "ArcDP-4090-full1-no-bridge-10天训练操作手册.md"
    text = guide.read_text(encoding="utf-8")

    assert "train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh" in text
    assert "BRIDGEDP_BATCH_SIZE=48" in text
    assert "BRIDGEDP_GRAD_ACCUM=1" in text
    assert "BRIDGEDP_BATCH_SIZE=24" in text
    assert "BRIDGEDP_GRAD_ACCUM=2" in text
    assert "SwanLab" in text
    assert "full" in text
    assert "no_bridge" in text
    assert "10 天" in text
```

**Step 2: Run the documentation test to verify RED**

Run:

```bash
pytest -q tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  -k ten_day_operator_guide
```

Expected: FAIL with `FileNotFoundError` for the guide.

**Step 3: Write the operator guide**

Document:

- The two-stage schedule and why the other training ablations were removed.
- The default `48/1` and fallback `24/2` effective-global-batch calculation.
- SwanLab login/API-key setup and default project/experiment names.
- Exact launch command with timestamped local log.
- Commands to inspect GPU utilization, the latest `training_status.json`, and running processes.
- The 2204.74-hour baseline extrapolation, 1.84× required speedup, and the 9.2–10 day target with explicit uncertainty.
- The rule to evaluate `no_ordered_init` and `no_gcs` from the full checkpoint later rather than retrain them.

**Step 4: Run the documentation test to verify GREEN**

Run:

```bash
pytest -q tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  -k ten_day_operator_guide
```

Expected: PASS.

**Step 5: Commit the guide**

```bash
git add \
  docs/ArcDP-4090-full1-no-bridge-10天训练操作手册.md \
  tests/unit_test/test_arcdp_4090_nvme_training_entries.py
git commit -m "docs: add ten-day ArcDP operator guide"
```

### Task 4: Run regression verification

**Files:**
- Verify: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`
- Verify: `tests/unit_test/test_arcdp_p0_ablation_entries.py`
- Verify: `tests/unit_test/test_arcdp_4090_env_pack_and_nvme_prepare.py`

**Step 1: Run the related unit tests**

Run:

```bash
pytest -q \
  tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  tests/unit_test/test_arcdp_p0_ablation_entries.py \
  tests/unit_test/test_arcdp_4090_env_pack_and_nvme_prepare.py
```

Expected: all tests PASS.

**Step 2: Re-run Bash syntax checks**

Run:

```bash
bash -n scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh
bash -n train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh
```

Expected: exit 0 with no output.

**Step 3: Inspect the final diff and worktree status**

Run:

```bash
git diff HEAD~3 -- \
  scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh \
  train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh \
  tests/unit_test/test_arcdp_4090_nvme_training_entries.py \
  docs/ArcDP-4090-full1-no-bridge-10天训练操作手册.md
git status --short
```

Expected: only the intended feature commits plus the pre-existing unrelated modification to the CPU environment deployment guide remain visible.

