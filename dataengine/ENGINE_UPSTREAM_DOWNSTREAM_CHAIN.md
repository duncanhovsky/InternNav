# DataEngine Upstream/Downstream Chain

## 1. Scope

This document clarifies how the current `dataengine/src/scene_layer` implementation connects with upstream configuration/assets and downstream task generation.

## 2. Upstream Inputs

1. Runtime config (`SceneLayerConfig`)
- Source: `dataengine/src/scene_layer/default_config.json` or CLI overrides.
- Carries sampling strategy, backend mode, output paths, retry policy, and quality thresholds.

2. Asset templates (`AssetRegistry`)
- Source root: `asset_root`.
- Output: scene template pool grouped by `scene_type` with mode capability flags.

3. Deterministic seed stream
- Base: `global_seed + scene_seed_offset`.
- Per-scene: `global_seed + scene_seed_offset + i * 1009`.

## 3. Scene-Layer Internal Pipeline

1. Selector stage (`WeightedSceneSelector`)
- Samples `(scene_mode, template)` by configured weights.

2. Compiler stage (`SceneCompiler`)
- `composer.compose(candidate)`
- `navmesh_builder.build_and_eval(...)`
- `metrics_eval.validate_and_build(...)`
- Produces one `SceneCompileResult`.

3. Composer stage (`SceneComposer`)
- `stub` backend: writes `stage_spec.json` only.
- `isaac_replicator` backend: writes `COMPOSING` spec, invokes worker subprocess, then writes `DONE`/`FAILED` final spec.
- Placement quality controls now include:
  - class-ratio asset selection,
  - zone-based sampling,
  - keepout-region rejection,
  - minimum spacing,
  - axis-aligned yaw jitter.

4. Nav/Metrics stage
- Current navmesh is deterministic placeholder logic.
- Metrics gate enforces reachability and free-space thresholds.

## 4. Downstream Outputs

1. Scene manifest
- Path: `scene_manifest_path`
- Contains scene status, stage paths, metrics, complexity bucket, and template metadata.

2. Compile logs
- Path: `compile_log_path`
- Appended per scene compile attempt to preserve failure evidence.

3. Trajectory task manifest
- Path: `task_manifest_path`
- Generated only from scenes where `status == DONE`.
- This manifest is the direct upstream input for trajectory/episode generation.

4. Per-scene artifacts
- `stage_spec.json`
- `stage_composed.usda` (replicator backend)
- `navmesh.bin` (current stub navmesh output)

## 5. Failure Propagation Contract

1. Composer failures are converted to `SceneCompileError` with standardized `ErrorCode`.
2. `compile_with_retry` retries only retryable codes.
3. Pipeline always appends scene row + compile log row, so failures remain observable.
4. Downstream task generation automatically skips failed scenes.

## 6. Current Integration Boundary

1. Upstream of scene-layer:
- asset indexing/curation,
- config orchestration,
- run scheduling.

2. Downstream of scene-layer:
- trajectory generation,
- dataset materialization,
- evaluator/trainer consumption.

This boundary keeps scene generation deterministic, auditable, and restartable.
