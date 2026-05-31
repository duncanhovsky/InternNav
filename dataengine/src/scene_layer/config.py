from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


SUPPORTED_SCENE_TYPES = {"warehouse", "hospital", "outdoor"}
SUPPORTED_AGENT_PROFILES = {"go2", "g1"}
SUPPORTED_MODES = {"train", "eval", "build_manifest"}
SUPPORTED_SCENE_MODES = {"complete", "modular"}
SUPPORTED_SCENE_BACKENDS = {"stub", "isaac_replicator"}
SUPPORTED_DYNAMICS_BACKENDS = {"synthetic", "asset_driven", "ira_character_graph", "ira"}
SUPPORTED_PROP_CLASSES = {"large", "medium", "small"}
SUPPORTED_PLACEMENT_ZONES = {"aisle", "wall", "corner", "open"}


@dataclass
class SceneLayerConfig:
    """场景层配置契约。

    该对象是 scene-layer 的唯一运行时输入源，覆盖：
    1. 模板采样策略与随机种子。
    2. 场景组合后端与 Replicator 摆放策略。
    3. NavMesh/Metrics 质量阈值。
    4. manifest 与任务输出路径。

    所有字段都可通过 JSON 或 CLI 覆盖，`validate()` 负责统一约束检查。
    """

    schema_version: str = "v1alpha"
    mode: str = "train"
    # Isaac 5.1 资产通常位于 Assets/Isaac/5.1 下。
    # asset_registry.py 中模板路径默认以 "Isaac/..." 开头拼接。
    asset_root: str = "/home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/Assets/Isaac/5.1"
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

    # 动态层最小闭环配置。
    enable_dynamics: bool = False
    dynamics_backend: str = "synthetic"
    dynamics_timeout_s: int = 90
    dynamic_steps: int = 20
    dynamic_dt_s: float = 0.1
    dynamic_num_people_min: int = 1
    dynamic_num_people_max: int = 2
    dynamic_num_objects_min: int = 1
    dynamic_num_objects_max: int = 2
    dynamic_asset_strict: bool = False
    dynamic_people_character_relpaths: List[str] = field(
        default_factory=lambda: ["Isaac/People/Characters/F_Business_02/F_Business_02.usd"]
    )
    dynamic_people_idle_animation_relpaths: List[str] = field(
        default_factory=lambda: ["Isaac/People/Animations/stand_idle_loop.skelanim.usd"]
    )
    dynamic_people_walk_animation_relpaths: List[str] = field(
        default_factory=lambda: ["Isaac/People/Animations/stand_walk_loop.skelanim.usd"]
    )
    dynamic_vehicle_relpaths: List[str] = field(
        default_factory=lambda: ["Isaac/Props/Forklift/forklift.usd"]
    )
    # Ensure animation clips are present and semantically matched to behavior.
    dynamic_require_animation_binding: bool = True
    dynamic_people_idle_ratio: float = 0.35
    dynamic_vehicle_parked_ratio: float = 0.40

    # ─── IRA (isaacsim.replicator.agent) 动态层配置 ───
    # 当 dynamics_backend == "ira" 时生效。
    ira_simulation_length: int = 300
    ira_character_asset_path: str = ""
    ira_character_filters: List[str] = field(default_factory=lambda: ["male", "medical"])
    ira_character_spawn_area: List[str] = field(default_factory=lambda: ["Walkable"])
    ira_character_navigation_area: List[str] = field(default_factory=lambda: ["Walkable"])
    ira_max_commands_per_character: int = 5
    ira_camera_num: int = 5
    ira_isaac_sim_python: str = ""
    ira_fallback_to_synthetic: bool = True
    ira_output_dir: str = ""

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

    # 场景组合后端：
    # - stub: 仅写 stage_spec，不依赖 Isaac Runtime
    # - isaac_replicator: 通过 Isaac Sim + Replicator 程序化生成并导出 stage
    scene_backend: str = "stub"
    isaac_headless: bool = True
    isaac_renderer: str = "RayTracedLighting"
    isaac_enable_replicator: bool = True
    isaac_close_on_finish: bool = False
    # GUI 调试模式：为 True 时保持窗口运行，直到用户手动关闭。
    isaac_gui_inspect_mode: bool = False
    # 导出前额外执行的 update 帧数，确保资产加载/渲染状态稳定。
    isaac_warmup_frames: int = 12

    # 复制器可复现参数。相同 scene_seed + 相同配置应得到一致布局。
    replicator_seed_offset: int = 17
    replicator_min_props: int = 8
    replicator_max_props: int = 28
    replicator_xy_range_m: float = 16.0
    replicator_z_offset_m: float = 0.05
    replicator_yaw_min_deg: float = -180.0
    replicator_yaw_max_deg: float = 180.0
    replicator_scale_min: float = 0.9
    replicator_scale_max: float = 1.1
    # 物体摆放约束：最小间距（米）+ 每个物体位置采样重试次数。
    replicator_min_spacing_m: float = 1.2
    replicator_position_max_retries: int = 24
    # 朝向策略：按概率吸附到主轴方向（0/90/180/270），再叠加小抖动。
    replicator_axis_align_prob: float = 0.75
    replicator_axis_jitter_deg: float = 8.0
    replicator_axis_candidates_deg: List[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    # 资产类别比例控制：用于降低“全随机导致的杂乱分布”。
    replicator_prop_class_ratio: Dict[str, float] = field(
        default_factory=lambda: {"large": 0.25, "medium": 0.35, "small": 0.40}
    )
    # 分区采样策略：zoned 时按功能区采样，uniform 时全局采样。
    replicator_zone_sampling_strategy: str = "zoned"
    # aisle 采用中心纵向走廊：|x| <= xy_range * aisle_half_width_ratio
    replicator_aisle_half_width_ratio: float = 0.14
    # wall 采用四边条带：边缘厚度 = xy_range * wall_band_ratio
    replicator_wall_band_ratio: float = 0.18
    # corner 采用四角方区：边长约为 xy_range * corner_zone_ratio
    replicator_corner_zone_ratio: float = 0.22
    # 类别 -> 分区权重。每类键集合必须完整覆盖 aisle/wall/corner/open。
    replicator_zone_weights_large: Dict[str, float] = field(
        default_factory=lambda: {"aisle": 0.05, "wall": 0.50, "corner": 0.35, "open": 0.10}
    )
    replicator_zone_weights_medium: Dict[str, float] = field(
        default_factory=lambda: {"aisle": 0.12, "wall": 0.28, "corner": 0.20, "open": 0.40}
    )
    replicator_zone_weights_small: Dict[str, float] = field(
        default_factory=lambda: {"aisle": 0.08, "wall": 0.20, "corner": 0.12, "open": 0.60}
    )
    # 禁入区矩形列表，每项格式：[xmin, xmax, ymin, ymax]（单位米）。
    replicator_keepout_rects: List[List[float]] = field(default_factory=list)
    generated_stage_filename: str = "stage_composed.usda"

    # 可选固定候选资产，路径相对于 asset_root。
    # 为空时将自动扫描 Modular_Warehouse/Props/*.usd。
    replicator_prop_relpaths: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """执行配置合法性校验。"""
        if self.schema_version.strip() == "":
            raise ValueError("schema_version 不能为空")

        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode 必须属于 {SUPPORTED_MODES}, got={self.mode}")

        if self.scene_backend not in SUPPORTED_SCENE_BACKENDS:
            raise ValueError(
                f"scene_backend 必须属于 {SUPPORTED_SCENE_BACKENDS}, got={self.scene_backend}"
            )

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

        if self.replicator_min_props < 0:
            raise ValueError("replicator_min_props 不能小于 0")
        if self.replicator_max_props < self.replicator_min_props:
            raise ValueError("replicator_max_props 必须 >= replicator_min_props")
        if self.replicator_xy_range_m <= 0:
            raise ValueError("replicator_xy_range_m 必须 > 0")
        if self.replicator_scale_min <= 0 or self.replicator_scale_max <= 0:
            raise ValueError("replicator_scale_min/max 必须 > 0")
        if self.replicator_scale_max < self.replicator_scale_min:
            raise ValueError("replicator_scale_max 必须 >= replicator_scale_min")
        if self.replicator_min_spacing_m < 0:
            raise ValueError("replicator_min_spacing_m 不能小于 0")
        if self.replicator_position_max_retries <= 0:
            raise ValueError("replicator_position_max_retries 必须 > 0")
        if not (0.0 <= self.replicator_axis_align_prob <= 1.0):
            raise ValueError("replicator_axis_align_prob 必须在 [0, 1] 区间")
        if self.replicator_axis_jitter_deg < 0:
            raise ValueError("replicator_axis_jitter_deg 不能小于 0")
        if len(self.replicator_axis_candidates_deg) == 0:
            raise ValueError("replicator_axis_candidates_deg 不能为空")
        if set(self.replicator_prop_class_ratio.keys()) != SUPPORTED_PROP_CLASSES:
            raise ValueError(
                "replicator_prop_class_ratio 必须包含 large/medium/small 三个键"
            )
        if any(v <= 0 for v in self.replicator_prop_class_ratio.values()):
            raise ValueError("replicator_prop_class_ratio 各项必须 > 0")
        if self.replicator_zone_sampling_strategy not in {"uniform", "zoned"}:
            raise ValueError("replicator_zone_sampling_strategy 必须是 uniform 或 zoned")
        if not (0.0 < self.replicator_aisle_half_width_ratio < 0.95):
            raise ValueError("replicator_aisle_half_width_ratio 必须在 (0, 0.95) 区间")
        if not (0.0 < self.replicator_wall_band_ratio < 0.95):
            raise ValueError("replicator_wall_band_ratio 必须在 (0, 0.95) 区间")
        if not (0.0 < self.replicator_corner_zone_ratio < 0.95):
            raise ValueError("replicator_corner_zone_ratio 必须在 (0, 0.95) 区间")
        self._validate_zone_weights(self.replicator_zone_weights_large, "replicator_zone_weights_large")
        self._validate_zone_weights(self.replicator_zone_weights_medium, "replicator_zone_weights_medium")
        self._validate_zone_weights(self.replicator_zone_weights_small, "replicator_zone_weights_small")
        for idx, rect in enumerate(self.replicator_keepout_rects):
            if len(rect) != 4:
                raise ValueError(f"replicator_keepout_rects[{idx}] 必须是 [xmin,xmax,ymin,ymax]")
            xmin, xmax, ymin, ymax = rect
            if not (xmax > xmin and ymax > ymin):
                raise ValueError(f"replicator_keepout_rects[{idx}] 非法范围: {rect}")
        if self.generated_stage_filename.strip() == "":
            raise ValueError("generated_stage_filename 不能为空")
        if self.isaac_warmup_frames < 0:
            raise ValueError("isaac_warmup_frames 不能小于 0")

        for profile in self.navmesh_agent_profiles:
            if profile not in SUPPORTED_AGENT_PROFILES:
                raise ValueError(f"不支持的 navmesh_agent_profiles: {profile}")
            if profile not in self.agent_radius_m_by_profile:
                raise ValueError(f"agent_radius_m_by_profile 缺少 {profile}")

        for profile in self.per_scene_agents:
            if profile not in SUPPORTED_AGENT_PROFILES:
                raise ValueError(f"不支持的 per_scene_agents: {profile}")

        if self.dynamics_backend not in SUPPORTED_DYNAMICS_BACKENDS:
            raise ValueError(f"dynamics_backend 必须属于 {SUPPORTED_DYNAMICS_BACKENDS}, got={self.dynamics_backend}")
        if self.dynamics_timeout_s <= 0:
            raise ValueError("dynamics_timeout_s 必须 > 0")
        if self.dynamic_steps <= 0:
            raise ValueError("dynamic_steps 必须 > 0")
        if self.dynamic_dt_s <= 0:
            raise ValueError("dynamic_dt_s 必须 > 0")
        if self.dynamic_num_people_min < 0 or self.dynamic_num_objects_min < 0:
            raise ValueError("dynamic_num_people_min/dynamic_num_objects_min 不能小于 0")
        if self.dynamic_num_people_max < self.dynamic_num_people_min:
            raise ValueError("dynamic_num_people_max 必须 >= dynamic_num_people_min")
        if self.dynamic_num_objects_max < self.dynamic_num_objects_min:
            raise ValueError("dynamic_num_objects_max 必须 >= dynamic_num_objects_min")
        if not (0.0 <= self.dynamic_people_idle_ratio <= 1.0):
            raise ValueError("dynamic_people_idle_ratio 必须在 [0, 1] 区间")
        if not (0.0 <= self.dynamic_vehicle_parked_ratio <= 1.0):
            raise ValueError("dynamic_vehicle_parked_ratio 必须在 [0, 1] 区间")
        self._validate_relpath_list(self.dynamic_people_character_relpaths, "dynamic_people_character_relpaths")
        self._validate_relpath_list(
            self.dynamic_people_idle_animation_relpaths,
            "dynamic_people_idle_animation_relpaths",
        )
        self._validate_relpath_list(
            self.dynamic_people_walk_animation_relpaths,
            "dynamic_people_walk_animation_relpaths",
        )
        self._validate_relpath_list(self.dynamic_vehicle_relpaths, "dynamic_vehicle_relpaths")

        if self.enable_dynamics and self.dynamics_backend == "asset_driven" and self.dynamic_asset_strict:
            self._validate_relpath_exists(
                self.asset_root,
                self.dynamic_people_character_relpaths,
                "dynamic_people_character_relpaths",
            )
            self._validate_relpath_exists(
                self.asset_root,
                self.dynamic_people_idle_animation_relpaths,
                "dynamic_people_idle_animation_relpaths",
            )
            self._validate_relpath_exists(
                self.asset_root,
                self.dynamic_people_walk_animation_relpaths,
                "dynamic_people_walk_animation_relpaths",
            )
            self._validate_relpath_exists(
                self.asset_root,
                self.dynamic_vehicle_relpaths,
                "dynamic_vehicle_relpaths",
            )

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

    @staticmethod
    def _validate_zone_weights(weights: Dict[str, float], name: str) -> None:
        if set(weights.keys()) != SUPPORTED_PLACEMENT_ZONES:
            raise ValueError(f"{name} 必须包含 {SUPPORTED_PLACEMENT_ZONES}")
        if any(v <= 0 for v in weights.values()):
            raise ValueError(f"{name} 的权重必须全部 > 0")

    @staticmethod
    def _validate_relpath_list(values: List[str], name: str) -> None:
        if len(values) == 0:
            raise ValueError(f"{name} 不能为空")
        for idx, relpath in enumerate(values):
            item = relpath.strip()
            if item == "":
                raise ValueError(f"{name}[{idx}] 不能为空")
            if os.path.isabs(item):
                raise ValueError(f"{name}[{idx}] 必须是相对 asset_root 的路径")

    @staticmethod
    def _validate_relpath_exists(asset_root: str, relpaths: List[str], name: str) -> None:
        missing: List[str] = []
        for relpath in relpaths:
            candidates = [relpath]
            if not relpath.startswith("Isaac/"):
                candidates.append(f"Isaac/{relpath}")

            exists = False
            for candidate in candidates:
                full_path = os.path.join(asset_root, candidate)
                if os.path.isfile(full_path):
                    exists = True
                    break

            if not exists:
                missing.append(relpath)
        if missing:
            raise ValueError(f"{name} 中存在不存在的资产: {missing}")

    @classmethod
    def from_dict(cls, cfg_dict: Dict) -> "SceneLayerConfig":
        """从字典构建并校验配置。"""
        cfg = cls(**cfg_dict)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, json_path: str) -> "SceneLayerConfig":
        """从 JSON 文件读取并构建配置。"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        """导出配置字典，用于 run snapshot 与审计。"""
        return self.__dict__.copy()
