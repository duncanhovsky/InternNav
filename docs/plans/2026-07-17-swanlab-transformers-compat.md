# SwanLab Transformers Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make ArcDP's `transformers==4.51.0` training work with SwanLab 0.8.5 by creating exactly one SwanLab Run on distributed rank 0 before the Trainer callback starts.

**Architecture:** Add a dependency-light compatibility helper under `scripts/train/base_train/`. It handles both historical `get_run() -> None` behavior and SwanLab 0.8.5's no-active-run exception, then the training entry calls it immediately before `trainer.train()`.

**Tech Stack:** Python 3.10, PyTorch distributed, Hugging Face Transformers 4.51.0, SwanLab 0.8.x, pytest.

---

### Task 1: Specify the compatibility behavior

**Files:**
- Create: `tests/unit_test/test_swanlab_compat.py`

**Step 1: Write the failing tests**

Cover disabled reporting, non-main rank, existing Run reuse, `None` return, and SwanLab 0.8.5's `No active Run` exception. Use a fake `swanlab` module through `sys.modules` so the tests do not need the package installed locally.

**Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/unit_test/test_swanlab_compat.py -q`

Expected: collection failure because `scripts.train.base_train.swanlab_compat` does not exist.

### Task 2: Implement the compatibility helper

**Files:**
- Create: `scripts/train/base_train/swanlab_compat.py`
- Test: `tests/unit_test/test_swanlab_compat.py`

**Step 1: Implement the minimal helper**

Create `initialize_swanlab_run(report_to, is_main_process, run_name)` that dynamically imports SwanLab only when enabled on the main process, reuses an active Run, converts only the known no-active-run exception into `None`, and otherwise calls `swanlab.init()` with `SWANLAB_PROJ_NAME` and `SWANLAB_EXP_NAME`.

**Step 2: Run the tests to verify GREEN**

Run: `python -m pytest tests/unit_test/test_swanlab_compat.py -q`

Expected: all compatibility tests pass.

### Task 3: Integrate before training starts

**Files:**
- Modify: `scripts/train/base_train/train.py`
- Modify: `tests/unit_test/test_swanlab_compat.py`

**Step 1: Add a failing static integration test**

Assert that `train.py` imports `initialize_swanlab_run`, invokes it with `config.il.report_to`, `is_main_process`, and `config.name`, and that the invocation appears before `trainer.train(...)`.

**Step 2: Run the integration test to verify RED**

Run: `python -m pytest tests/unit_test/test_swanlab_compat.py -q`

Expected: failure because the training entry does not call the helper yet.

**Step 3: Add the minimal training-entry call**

Import the helper and call it after resume preparation and immediately before `trainer.train()`.

**Step 4: Run the integration test to verify GREEN**

Run: `python -m pytest tests/unit_test/test_swanlab_compat.py -q`

Expected: all tests pass.

### Task 4: Regression verification

**Files:**
- Verify: `scripts/train/base_train/train.py`
- Verify: `scripts/train/base_train/swanlab_compat.py`
- Verify: ArcDP unit tests

**Step 1: Run focused and related tests**

Run: `python -m pytest tests/unit_test/test_swanlab_compat.py tests/unit_test/test_arcdp_4090_nvme_training_entries.py tests/unit_test/test_arcdp_4090_env_pack_and_nvme_prepare.py tests/unit_test/test_arcdp_p0_ablation_entries.py -q`

Expected: all tests pass.

**Step 2: Validate source and shell syntax**

Run: `python -m py_compile scripts/train/base_train/swanlab_compat.py scripts/train/base_train/train.py`

Run: `bash -n scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh`

Expected: both commands exit zero.

**Step 3: Inspect the final diff**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended compatibility, test, and plan files are changed.
