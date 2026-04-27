# Scene Layer Schema v1alpha

本文件定义 `scene_layer` 在 Phase 0/1 的最小稳定数据契约，覆盖 `scene_manifest` 与 `task_manifest`。

## scene_manifest.v1alpha

### Required

- `schema_version`: string, must be `v1alpha`
- `contract`: string, must be `scene_manifest.v1alpha`
- `scene_id`: string
- `scene_type`: string
- `mode`: string (`complete|modular`)
- `seed`: int
- `status`: string (`DONE|FAILED`)
- `stage_usd`: string
- `navmesh_file`: string
- `complexity_bucket`: string (`easy|medium|hard`)
- `layout_hash`: string
- `template_id`: string
- `metrics`: object
- `metrics.path_count`: int
- `metrics.shortest_path_m`: float
- `metrics.second_shortest_path_m`: float
- `metrics.detour_margin_m`: float
- `metrics.free_space_ratio`: float
- `metrics.static_density`: float
- `metrics.static_complexity_score`: float

### Optional

- `reason`: string or null
- `navmesh`: object
- `navmesh.metrics_by_profile`: object, key=`agent_profile`
- `navmesh.navmesh_files_by_profile`: object, key=`agent_profile`
- `navmesh.debug_files_by_profile`: object, key=`agent_profile`
- `dynamic`: object
- `dynamic.enabled`: bool
- `dynamic.backend`: string
- `dynamic.track_file`: string
- `dynamic.behavior_event_file`: string, optional behavior state transition event stream
- `dynamic.overlay_usd`: string, optional overlay stage for asset-driven dynamics preview
- `dynamic.object_count`: int
- `dynamic.sample_count`: int

### Deprecated

- none

## dynamic_tracks.v1alpha

### Required

- `schema_version`: string, must be `v1alpha`
- `contract`: string, must be `dynamic_tracks.v1alpha`
- `scene_id`: string
- `object_id`: string
- `category`: string (`people|object`)
- `timestamp_ns`: int
- `position_xyz`: array[3] of float
- `velocity_xyz`: array[3] of float
- `bbox_xyz`: array[3] of float
- `yaw_deg`: float

### Optional

- `source_backend`: string (`synthetic|asset_driven|ira_character_graph|ira`)
- `motion_mode`: string
	- people: `synthetic|stand_idle|walk`
	- object: `synthetic|parked|patrol`
- `asset_relpath`: string, path relative to `asset_root`
- `animation_relpath`: string, people animation path relative to `asset_root`
- `animation_behavior`: string (`none|idle|walk`), semantic animation intent bound to `motion_mode`

## dynamic_events.v1alpha

### Required

- `schema_version`: string, must be `v1alpha`
- `contract`: string, must be `dynamic_events.v1alpha`
- `scene_id`: string
- `object_id`: string
- `category`: string (`people|object`)
- `timestamp_ns`: int
- `event_type`: string (`state_enter`)
- `to_motion_mode`: string
- `source_backend`: string (`ira_character_graph|asset_driven|synthetic`)

### Optional

- `from_motion_mode`: string
- `command_name`: string
- `animation_behavior`: string (`none|idle|walk`)

### Deprecated

- none

### Deprecated

- none

## task_manifest.v1alpha

### Required

- `schema_version`: string, must be `v1alpha`
- `contract`: string, must be `task_manifest.v1alpha`
- `task_id`: string
- `scene_id`: string
- `agent_type`: string
- `episode_idx`: int
- `global_seed`: int
- `scene_seed`: int
- `complexity_bucket`: string
- `scene_type`: string
- `mode`: string
- `scene_layout_hash`: string

### Optional

- none

### Deprecated

- none
