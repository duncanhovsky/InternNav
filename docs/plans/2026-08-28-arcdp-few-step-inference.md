# ArcDP Few-Step Inference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the canonical ArcDP model execute 2/4/8/10 full-range reverse bridge steps without retraining.

**Architecture:** The scheduler selects sparse indices over the original ten-step axis and accepts the next selected timestep in each reverse update. PointGoal and NoGoal policy loops pass that next timestep explicitly; training and checkpoint structure remain unchanged.

**Tech Stack:** Python 3.10, PyTorch, unittest.

---

### Task 1: Scheduler behavior

**Files:**
- Modify: `internnav/model/basemodel/bridgedp/bridge_scheduler.py`
- Test: `tests/unit_test/test_bridgedp_few_step_scheduler.py`

1. Write failing tests for exact schedules, input validation, sparse jumps, and ten-step compatibility.
2. Run and verify the expected failure.
3. Implement full-range schedule selection and `prev_timestep` support.
4. Run the focused tests.

### Task 2: Policy integration

**Files:**
- Modify: `internnav/model/basemodel/bridgedp/bridgedp_policy.py`
- Test: `tests/unit_test/test_bridgedp_few_step_policy_contract.py`

1. Write a failing source-contract test for next-timestep propagation in both inference branches.
2. Pass the next selected timestep in PointGoal and NoGoal reverse loops.
3. Run focused tests, compile checks, and `git diff --check`.
4. Commit and push to the established ArcDP training branch.

