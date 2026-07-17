# ArcDP Safe Percentage Checkpoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an isolated one-epoch B+ training profile that creates verified resumable checkpoints every 1%, permanently retains every 10%, and keeps only five rolling non-permanent checkpoints.

**Architecture:** A pure helper module computes percentage targets and applies marker-driven retention. A distributed Trainer callback requests saves, synchronizes ranks, validates complete checkpoint contents, writes an atomic completion marker, and performs retention. BridgeDP model writes become rank-zero-only and atomic; strict auto-resume and a final fallback prevent partial checkpoints from being accepted.

**Tech Stack:** Python 3.10, PyTorch 2.7, Transformers 4.51, torch.distributed, Bash, pytest.

---

### Task 1: Specify percentage targets and retention

**Files:**
- Create: `tests/unit_test/test_percent_checkpoint_utils.py`
- Create: `scripts/train/base_train/percent_checkpoint_utils.py`

**Step 1: Write failing tests**

Test that 25,590 total steps produce 100 targets, including 1%=256, 10%=2,559, and 100%=25,590. Test that 10% checkpoints are permanent and that retention preserves all permanent checkpoints plus only the newest five non-permanent checkpoints.

**Step 2: Verify RED**

```bash
python -m pytest tests/unit_test/test_percent_checkpoint_utils.py -q
```

Expected: import failure because the helper module does not exist.

**Step 3: Implement pure helpers**

Provide percentage target records, atomic JSON writing, marker parsing, manifest generation, and rolling deletion based only on completed markers.

**Step 4: Verify GREEN**

Run the focused test and expect all tests to pass.

### Task 2: Require complete checkpoints for the safe profile

**Files:**
- Modify: `tests/unit_test/test_train_resume_utils.py`
- Modify: `scripts/train/base_train/resume_utils.py`

**Step 1: Write failing tests**

Test that strict resolution skips a newer checkpoint without `CHECKPOINT_COMPLETE.json`, selects the newest complete checkpoint, and rejects markers whose declared files are missing. Preserve legacy behavior when strict mode is disabled.

**Step 2: Verify RED**

```bash
python -m pytest tests/unit_test/test_train_resume_utils.py -q
```

Expected: failure because strict marker validation is unsupported.

**Step 3: Implement strict validation**

Add `require_complete` parameters to checkpoint validation, discovery, and resolution. Parse marker JSON and require every marker-listed file to exist and be non-empty.

**Step 4: Verify GREEN**

Run the focused test and expect all tests to pass.

### Task 3: Make BridgeDP model saving rank-zero-only and atomic

**Files:**
- Create: `tests/unit_test/test_bridgedp_checkpoint_safety.py`
- Modify: `internnav/trainer/bridgedp_trainer.py`

**Step 1: Write a failing static contract test**

Require `save_model()` to guard `self.args.should_save`, write through a PID-specific temporary file, flush and `fsync`, atomically replace `pytorch_model.bin`, create an atomic `bridgedp.ckpt` alias, and save `training_args.bin`.

**Step 2: Verify RED**

Run the focused test and expect failure on the missing guard.

**Step 3: Implement the save path**

Add small atomic save/link helpers and update `BridgeDPTrainer.save_model()` without changing model-state semantics.

**Step 4: Verify GREEN**

Run the focused test and expect it to pass.

### Task 4: Add distributed percentage checkpoint management

**Files:**
- Create: `scripts/train/base_train/percent_checkpoint_manager.py`
- Modify: `scripts/train/base_train/train.py`
- Modify: `scripts/train/base_train/configs/bridgedp_full_8x4090.py`
- Modify: `tests/unit_test/test_bridgedp_checkpoint_safety.py`

**Step 1: Write failing integration-contract tests**

Require a `PercentCheckpointCallback`, distributed barrier and error broadcast, model/optimizer/scheduler/RNG loading, atomic completion marker, rolling retention, strict resume wiring, `save_strategy='no'` while percentage saves are active, and a final checkpoint fallback.

**Step 2: Verify RED**

Run the focused test and expect failure because the manager is absent.

**Step 3: Implement the manager**

The callback computes targets lazily from `state.max_steps`, requests saves exactly at target steps, validates after all ranks finish, marks completed checkpoints, updates the manifest, and prunes rolling checkpoints. Add a post-training function that verifies or creates the final 100% checkpoint.

**Step 4: Wire configuration and strict resume**

Expose environment-driven interval, rolling count, permanent interval, load verification, and strict resume settings. Register the callback only when percentage checkpointing is enabled.

**Step 5: Verify GREEN**

Run focused safety and resume tests.

### Task 5: Add the isolated safe launcher

**Files:**
- Modify: `scripts/train/arcdp_4090/train_arcdp_full1_8x4090_ceph_io.sh`
- Create: `train_arcdp_full1_b24_ga2_w4_pf1_safeckpt_8x4090_ceph.sh`
- Modify: `tests/unit_test/test_arcdp_ceph_io_profiles.py`
- Modify: `docs/ArcDP-8x4090-Ceph-IO-AB测试.md`

**Step 1: Write a failing launcher test**

Require an isolated `b24_ga2_w4_pf1_safeckpt` profile with port 12349, batch 24, accumulation 2, workers 4, prefetch 1, one-percent interval, five rolling checkpoints, ten-percent permanent interval, strict resume, and its own wrapper/run name.

**Step 2: Verify RED**

Run the profile test and expect failure on the missing profile.

**Step 3: Implement launcher and tmux documentation**

Add the profile and wrapper. Document stopping the old process, pushing/pulling, dry-run, tmux foreground launch, auto-resume, retention layout, manifest inspection, and completion verification.

**Step 4: Verify GREEN**

Run the profile test, Bash syntax checks, and dry-run.

### Task 6: Full verification and commits

**Files:**
- Test all files above plus existing progress/resume/profile tests.

**Step 1: Run relevant tests**

```bash
python -m pytest tests/unit_test/test_percent_checkpoint_utils.py tests/unit_test/test_train_resume_utils.py tests/unit_test/test_bridgedp_checkpoint_safety.py tests/unit_test/test_arcdp_ceph_io_profiles.py tests/unit_test/test_training_progress_metrics.py tests/unit_test/test_arcdp_4090_nvme_training_entries.py -q
```

**Step 2: Run static verification**

Run `py_compile`, Bash `-n`, safe-profile dry-run, `git diff --check`, and inspect staged scope.

**Step 3: Commit only task files**

Leave the user's unrelated modified document untouched and unstaged.
