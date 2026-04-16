from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


SUPPORTED_SCENE_TYPES = {"warehouse", "hospital", "outdoor"}
SUPPORTED_AGENT_PROFILES = {"go2", "g1"}
SUPPORTED_MODES = {"train", "eval", "build_manifest"}
SUPPORTED_SCENE_MODES = {"complete", "modular"}


@dataclass
class SceneLayerConfig:
    """场景层配置 schema（精简实现版）。"""

    schema_version: str = "1.1.0"
    mode: str = "train"
    asset_root: str = "/home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/"
    enabled_scene_types: List[str] = field(default_factory=lambda: ["warehouse", "hospital", "outdoor"])
    scene_type_weights: Dict[str, float] = field(
        default_factory=lambda: {"warehouse": 1.0, "hospital": 1.0, "outdoor": 1.0}
    )
    use_equal_weights_if_missing: bool = True

    global_seed: int = 0
    scene_seed_offset: int = 0

    # 生成模式：完整场景与模块化场景可混合。
    scene_mode_weights: Dict[str, float] = field(default_factory=lambda: {"complete": 0.7, "modular": 0.3})

    freeze_eval_manifest: bool = True
    frozen_manifest_path: str = "dataengine/manifests/scene_frozen_v1.jsonl"

    asset_allowlist_path: str = "dataengine/configs/asset_allowlist.json"

    usd_output_root: str = "dataengine/out/scenes/usd"
    scene_manifest_path: str = "dataengine/out/manifests/scene_manifest.jsonl"
    compile_log_path: str = "dataengine/out/logs/scene_compile_log.jsonl"
    task_manifest_path: str = "dataengine/out/manifests/trajectory_tasks.jsonl"
    write_tmp_then_rename: bool = True
    save_stage_flattened_usd: bool = True

    compose_timeout_s: int = 180

    enable_navmesh: bool = True
    navmesh_agent_profiles: List[str] = field(default_factory=lambda: ["go2", "g1"])
    navmesh_bake_timeout_s: int = 300
    navmesh_cell_size_m: float = 0.05
    navmesh_cell_height_m: float = 0.05
    navmesh_max_slope_deg: float = 35.0
    navmesh_step_height_m: float = 0.25
    agent_radius_m_by_profile: Dict[str, float] = field(default_factory=lambda: {"go2": 0.28, "g1": 0.34})

    start_goal_pairs_per_scene: int = 128
    pair_sampling_retry_limit: int = 200
    min_pair_distance_m: float = 6.0
    max_pair_distance_m: float = 60.0

    min_path_count: int = 3
    min_detour_margin_m: float = 2.0
    detour_margin_ratio_floor: float = 0.15
    min_free_space_ratio: float = 0.22

    static_density_target: float = 0.30
    static_density_tolerance: float = 0.10

    # easy : medium : hard
    complexity_ratio_easy_medium_hard: List[int] = field(default_factory=lambda: [10, 5, 2])
    complexity_balance_tolerance: float = 0.15

    length_bins_m: List[float] = field(default_factory=lambda: [0.0, 10.0, 25.0, 1_000_000.0])
    static_complexity_bins: List[float] = field(default_factory=lambda: [0.0, 0.35, 0.65, 1.01])
    dynamic_complexity_bins: List[float] = field(default_factory=lambda: [0.0, 0.35, 0.65, 1.01])

    max_compile_retries: int = 2
    retryable_error_codes: List[str] = field(
        default_factory=lambda: ["RUNTIME_FAIL", "IO_TRANSIENT", "GPU_TIMEOUT"]
    )
    non_retryable_error_codes: List[str] = field(
        default_factory=lambda: ["ASSET_MISSING", "NAVMESH_FAIL", "REACHABILITY_FAIL", "METRIC_REJECT"]
    )

    heartbeat_timeout_s: int = 60
    reclaim_stale_running_tasks: bool = True

    checksum_algo: str = "xxh64"
    dedup_key_fields: List[str] = field(
        default_factory=lambda: ["scene_id", "scene_seed", "scene_type", "layout_hash", "asset_pack_version"]
    )
    export_failure_cases: bool = True

    # 规模配置
    target_scene_count: int = 30
    trajectories_per_scene: int = 100
    per_scene_agents: List[str] = field(default_factory=lambda: ["go2", "g1"])

    # 可选：外部覆盖路径，便于实验管理
    run_name: str = "scene_layer_run"

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode 必须属于 {SUPPORTED_MODES}, got={self.mode}")

        if len(self.enabled_scene_types) == 0:
            raise ValueError("enabled_scene_types 不能为空")
        for scene_type in self.enabled_scene_types:
            if scene_type not in SUPPORTED_SCENE_TYPES:
                raise ValueError(f"不支持的 scene_type: {scene_type}")

        if not os.path.isdir(self.asset_root):
            raise ValueError(f"asset_root 不存在或不可读: {self.asset_root}")

        weight_keys = set(self.scene_type_weights.keys())
        enabled_keys = set(self.enabled_scene_types)
        if weight_keys != enabled_keys:
            raise ValueError(
                "scene_type_weights 的键必须与 enabled_scene_types 完全一致: "
                f"weights={weight_keys}, enabled={enabled_keys}"
            )
        if any(v <= 0 for v in self.scene_type_weights.values()):
            raise ValueError("scene_type_weights 的值必须大于 0")

        mode_keys = set(self.scene_mode_weights.keys())
        if mode_keys != SUPPORTED_SCENE_MODES:
            raise ValueError(
                "scene_mode_weights 必须包含 complete/modular 两个键，"
                f"got={mode_keys}"
            )
        if any(v <= 0 for v in self.scene_mode_weights.values()):
            raise ValueError("scene_mode_weights 的值必须大于 0")

        if self.freeze_eval_manifest and self.mode == "eval":
            if not os.path.isfile(self.frozen_manifest_path):
                raise ValueError(
                    f"eval+freeze 模式下 frozen_manifest_path 必须存在: {self.frozen_manifest_path}"
                )

        if self.max_pair_distance_m <= self.min_pair_distance_m:
            raise ValueError("max_pair_distance_m 必须大于 min_pair_distance_m")

        if self.min_path_count < 2:
            raise ValueError("min_path_count 不能小于 2")

        if not (0.0 < self.min_free_space_ratio < 0.9):
            raise ValueError("min_free_space_ratio 必须在 (0, 0.9) 区间内")

        if len(self.complexity_ratio_easy_medium_hard) != 3:
            raise ValueError("complexity_ratio_easy_medium_hard 必须是长度为 3 的数组")
        if any(v <= 0 for v in self.complexity_ratio_easy_medium_hard):
            raise ValueError("complexity_ratio_easy_medium_hard 每项必须 > 0")

        if self.trajectories_per_scene <= 0:
            raise ValueError("trajectories_per_scene 必须 > 0")

        for profile in self.navmesh_agent_profiles:
            if profile not in SUPPORTED_AGENT_PROFILES:
                raise ValueError(f"不支持的 navmesh_agent_profiles: {profile}")
            if profile not in self.agent_radius_m_by_profile:
                raise ValueError(f"agent_radius_m_by_profile 缺少 {profile}")

        for profile in self.per_scene_agents:
            if profile not in SUPPORTED_AGENT_PROFILES:
                raise ValueError(f"不支持的 per_scene_agents: {profile}")

        self._validate_increasing_bins(self.length_bins_m, "length_bins_m")
        self._validate_increasing_bins(self.static_complexity_bins, "static_complexity_bins")
        self._validate_increasing_bins(self.dynamic_complexity_bins, "dynamic_complexity_bins")

    @staticmethod
    def _validate_increasing_bins(values: List[float], name: str) -> None:
        if len(values) < 4:
            raise ValueError(f"{name} 长度至少为 4")
        for i in range(1, len(values)):
            if not values[i] > values[i - 1]:
                raise ValueError(f"{name} 必须严格递增: {values}")

    @classmethod
    def from_dict(cls, cfg_dict: Dict) -> "SceneLayerConfig":
        cfg = cls(**cfg_dict)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, json_path: str) -> "SceneLayerConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()
