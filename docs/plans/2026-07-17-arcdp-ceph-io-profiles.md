# ArcDP Ceph I/O Profiles Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two isolated full-one-epoch 8x4090 launch profiles for CephFS A/B testing and make the training status report true global throughput.

**Architecture:** Keep one shared shell launcher with two validated profiles and expose each profile through a root-level executable wrapper. Pass DataLoader prefetch settings through the existing 8x4090 config into Transformers 4.51.0 `TrainingArguments`. Move throughput arithmetic into a dependency-free helper so world size, gradient accumulation, resume state, and ETA can be unit tested without importing the full training stack.

**Tech Stack:** Bash, Python 3.10, PyTorch DDP/torchrun, Hugging Face Transformers 4.51.0, pytest.

---

### Task 1: Correct global progress metrics

**Files:**
- Create: `scripts/train/base_train/progress_metrics.py`
- Create: `tests/unit_test/test_training_progress_metrics.py`
- Modify: `scripts/train/base_train/train.py`

**Step 1: Write failing tests**

Add tests for batch 48 / accumulation 1, batch 24 / accumulation 2, and resumed training. Assert that both profiles report effective global batch 384, throughput uses only optimizer steps completed in the current invocation, and ETA uses the measured per-optimizer-step time.

**Step 2: Run tests and verify failure**

Run: `python -m pytest tests/unit_test/test_training_progress_metrics.py -q`

Expected: FAIL because `progress_metrics.py` does not exist.

**Step 3: Implement the helper and callback integration**

Create a pure `compute_progress_metrics(...)` helper. In `DetailedProgressCallback`, record the starting global step, derive world size and gradient accumulation from `TrainingArguments`, report `effective_global_batch`, and calculate per-step time between logging callbacks instead of treating a multi-step logging interval as one step.

**Step 4: Run tests and verify pass**

Run: `python -m pytest tests/unit_test/test_training_progress_metrics.py -q`

Expected: PASS.

### Task 2: Pass DataLoader prefetch settings

**Files:**
- Modify: `internnav/configs/trainer/il.py`
- Modify: `scripts/train/base_train/configs/bridgedp_full_8x4090.py`
- Modify: `scripts/train/base_train/train.py`
- Create: `tests/unit_test/test_arcdp_ceph_io_profiles.py`

**Step 1: Add failing static contract tests**

Assert that the IL config declares a prefetch field, the 8x4090 config honors `BRIDGEDP_PREFETCH_FACTOR`, and `TrainingArguments` receives `dataloader_prefetch_factor` only when workers are enabled.

**Step 2: Run the focused test**

Run: `python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py -q`

Expected: FAIL on missing prefetch wiring.

**Step 3: Implement minimal parameter wiring**

Add `dataloader_prefetch_factor` to `IlCfg`, load it from the environment in the 8x4090 config, and pass a validated optional integer to `TrainingArguments`.

**Step 4: Run the focused test**

Run: `python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py -q`

Expected: prefetch contract tests PASS.

### Task 3: Add the two launch profiles

**Files:**
- Create: `scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh`
- Create: `train_arcdp_full1_b48_w2_8x4090_ceph.sh`
- Create: `train_arcdp_full1_b24_ga2_w2_pf1_8x4090_ceph.sh`
- Modify: `tests/unit_test/test_arcdp_ceph_io_profiles.py`

**Step 1: Extend failing launcher tests**

Assert profile A resolves to batch 48, accumulation 1, workers 2, prefetch 2, and profile B resolves to batch 24, accumulation 2, workers 2, prefetch 1. Assert both use 8 GPUs, one epoch, effective global batch 384, unique run names, full-model-only execution, SwanLab, dataset preflight checks, and busy-GPU protection.

**Step 2: Run the launcher tests**

Run: `python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py -q`

Expected: FAIL because the launchers do not exist.

**Step 3: Implement the shared launcher and wrappers**

Implement a strict profile `case`, fixed profile parameters, common ArcDP environment setup, visible-GPU and busy-process checks, Ceph filesystem reporting, SwanLab import validation, configuration summary, and one `torchrun` full-model stage. Add `ARCDP_DRY_RUN=1` to print the resolved command without starting training.

**Step 4: Mark scripts executable and verify syntax**

Run: `git update-index --chmod=+x` on all three shell files.

Run when Bash is available: `bash -n` for each script.

Expected: zero exit status.

### Task 4: Add the operator runbook

**Files:**
- Create: `docs/ArcDP-8x4090-Ceph-IO-AB测试.md`

**Step 1: Document preflight and launch commands**

Document conda activation, SwanLab authentication, dry-run checks, profile A launch first, profile B launch only if needed, unique log files, and safe termination.

**Step 2: Document metric collection**

Include commands for status JSON, 120-second per-GPU SM averages, GPU power/memory monitoring, worker wait-channel inspection, and the exact result bundle to send back for comparison.

### Task 5: Regression verification

**Files:**
- Test: `tests/unit_test/test_training_progress_metrics.py`
- Test: `tests/unit_test/test_arcdp_ceph_io_profiles.py`
- Test: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`

**Step 1: Run focused tests**

Run: `python -m pytest tests/unit_test/test_training_progress_metrics.py tests/unit_test/test_arcdp_ceph_io_profiles.py tests/unit_test/test_arcdp_4090_nvme_training_entries.py -q`

Expected: all PASS.

**Step 2: Compile modified Python files**

Run: `python -m py_compile scripts/train/base_train/progress_metrics.py scripts/train/base_train/train.py scripts/train/base_train/configs/bridgedp_full_8x4090.py internnav/configs/trainer/il.py`

Expected: zero exit status.

**Step 3: Review the diff and preserve unrelated work**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; the pre-existing modified operator manual remains untouched and uncommitted.
