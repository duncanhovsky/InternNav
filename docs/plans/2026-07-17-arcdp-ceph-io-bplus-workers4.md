# ArcDP Ceph I/O B+ Workers4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an isolated 8x4090 full-model one-epoch B+ launcher with batch 24, gradient accumulation 2, four DataLoader workers per rank, and prefetch factor 1.

**Architecture:** Extend the existing shared Ceph I/O launcher with one named profile so all safety and preflight behavior stays identical to A/B. Add a dedicated root wrapper and profile-specific run/checkpoint/log/PID names. Preserve effective global batch 384 and test the shell configuration statically plus through dry-run output.

**Tech Stack:** Bash, PyTorch `torchrun`, Transformers TrainingArguments, pytest.

---

### Task 1: Specify the B+ profile with a failing test

**Files:**
- Modify: `tests/unit_test/test_arcdp_ceph_io_profiles.py`

**Step 1: Write the failing test**

Extend the launcher assertions to require:

```python
assert 'b24_ga2_w4_pf1)' in launcher
assert 'PROFILE_NUM_WORKERS=4' in launcher
assert 'PROFILE_PREFETCH_FACTOR=1' in launcher
assert 'arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph' in launcher
```

Extend the wrapper mapping with:

```python
"train_arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph.sh": "b24_ga2_w4_pf1"
```

**Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py -q
```

Expected: failure because the B+ profile and wrapper do not exist.

### Task 2: Implement the profile and wrapper

**Files:**
- Modify: `scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh`
- Create: `train_arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph.sh`

**Step 1: Add the profile**

Add a `b24_ga2_w4_pf1` case with:

```bash
PROFILE_BATCH_SIZE=24
PROFILE_GRAD_ACCUM=2
PROFILE_NUM_WORKERS=4
PROFILE_PREFETCH_FACTOR=1
PROFILE_RUN_NAME="arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph"
PROFILE_MASTER_PORT=12348
```

Update usage text to list the profile.

**Step 2: Add the dedicated wrapper**

Create an executable-style Bash wrapper that resolves its project directory and executes the shared launcher with `b24_ga2_w4_pf1`.

**Step 3: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py -q
```

Expected: all focused tests pass.

### Task 3: Document B+ operation

**Files:**
- Modify: `docs/ArcDP-8x4090-Ceph-IO-AB测试.md`

**Step 1: Add B+ commands**

Document chmod, dry-run, foreground/background launch, exact PID/log/checkpoint/status paths, safe B shutdown, and the minimum 100-step comparison procedure.

**Step 2: Verify identifiers**

Search the runbook and scripts for `b24_ga2_w4_pf1` and confirm all names agree.

### Task 4: Verify the complete change

**Files:**
- Test: `tests/unit_test/test_arcdp_ceph_io_profiles.py`
- Test: `tests/unit_test/test_training_progress_metrics.py`
- Test: `tests/unit_test/test_arcdp_4090_nvme_training_entries.py`

**Step 1: Run relevant tests**

```bash
python -m pytest tests/unit_test/test_arcdp_ceph_io_profiles.py tests/unit_test/test_training_progress_metrics.py tests/unit_test/test_arcdp_4090_nvme_training_entries.py -q
```

Expected: all tests pass.

**Step 2: Check Bash syntax**

```bash
bash -n scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh
bash -n train_arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph.sh
```

Expected: both commands exit zero without output.

**Step 3: Run B+ dry-run**

```bash
ARCDP_DRY_RUN=1 ./train_arcdp_full1_b24_ga2_w4_pf1_8x4090_ceph.sh
```

Expected: output reports batch 24, accumulation 2, workers 4, prefetch 1, effective global batch 384, and the isolated B+ run name.

**Step 4: Review scope**

Run `git diff --check`, inspect `git diff`, and verify the unrelated user-modified document remains unstaged and unchanged by this task.
