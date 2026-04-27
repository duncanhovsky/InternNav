# FlowNav + DataEngine Baseline Snapshot (2026-04-27)

## Purpose

This document records the current implementation baseline of FlowNav and its paired DataEngine stack.
Future FlowNav development should branch from this baseline on `flownav-dev`.
A separate method line based on NavDP baseline should branch from the same baseline on `bridgedp-dev`.

## Repository Baseline Commit Policy

- Commit 1: data artifact hygiene (`dataengine/out` ignored and untracked).
- Commit 2: implementation snapshot + this status document.

## FlowNav Current Status

### Trainer and Runtime Integration

- `internnav/trainer/flownav_trainer.py` includes FlowNav training runtime and dynamic voxel handling paths.
- Dynamic voxel resolution supports:
  - direct consumption from dynamic datasets when `batch_dynamic_voxels` exists,
  - mixed-batch fallback with validity masks,
  - online fallback construction for static datasets via dyn-module runtime.
- Depth preprocessing metadata is propagated into intrinsic correction before point cloud reconstruction, to reduce geometric inconsistency in online voxel fallback.

### Dataset/Baseline Context

- The repository maintains FlowNav-related dataset and trainer entries under `internnav/dataset` and `internnav/trainer`.
- NavDP remains the known baseline context for VN/System-1 related method extensions.

## DataEngine Current Status

### Dynamic Layer (IRA-based)

Main implementation is present under `dataengine/src/dynamic_layer/`:

- CLI entry and module entry:
  - `__main__.py`
  - `__init__.py`
- Core orchestration:
  - `dynamic_layer.py`
- Config and models:
  - `config.py`
  - `models.py`
- IRA pipeline components:
  - `ira_config_generator.py`
  - `command_generator.py`
  - `ira_driver.py`
  - `post_processor.py`
- User-facing docs and smoke command:
  - `README.md`
  - `test.sh`

Related default configs are added under:

- `dataengine/src/config/IRA/`
- `dataengine/src/config/dynamic_layer/`

### Scene Layer Extensions

`dataengine/src/scene_layer/` includes new/updated integration points:

- dynamic-layer coupling and worker plumbing:
  - `dynamics_layer.py`
  - `dynamics_worker.py`
- updated pipeline/compiler/config/navmesh/metrics/models/open_stage docs and behavior.
- additional schema/plan/smoke config artifacts for scene generation and validation.

### Test Coverage Added

New tests are present for both unit and function levels:

- `tests/unit_test/test_dynamic_layer.py`
- `tests/function_test/test_dynamic_layer_integration.py`
- `tests/unit_test/test_scene_layer_contracts.py`
- `tests/unit_test/test_scene_layer_dynamics.py`
- `tests/unit_test/test_scene_layer_metrics_policy.py`
- `tests/unit_test/test_scene_layer_validate_artifacts.py`
- `tests/function_test/test_scene_layer_smoke_stub.py`

## Verified Command (Current Session)

The following dry-run command has been executed successfully in this session (exit code 0):

```bash
python -m dataengine.src.dynamic_layer \
    --scene-usd /home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
    --scene-dir /home/monika/dyishere/project/MyResearch/InternNav/dataengine/out/dynamic \
    --seed 42 \
    --dry-run
```

## Branching Plan Locked by This Baseline

- `master`: baseline branch at the same commit as this snapshot.
- `flownav-dev`: created from `master` for all subsequent FlowNav work.
- `bridgedp-dev`: created from `master` for new method development using NavDP as baseline.

## Notes

- Generated data under `dataengine/out/` is ignored to keep commits reviewable and reproducible.
- Follow-up feature work should avoid mixing generated artifacts with source changes.
