"""FlowNav 动态场景 LeRobot 数据集。

本文件实现 `flownav_dyn_lerobot_dataset.py`，用于在包含动态障碍物的自制数据集上
训练 FlowNav。该实现在接口上尽量对齐 `flownav_lerobot_dataset.py`，并新增动态信息
读取与体素化逻辑。

====================================
一、与 flownav_lerobot_dataset.py 的关键差异
====================================
1. 障碍物来源不同
   - 静态版: 仅使用 `meta/pointcloud.ply` 提取静态障碍点。
   - 动态版: 同时使用
       a) `meta/pointcloud_static.ply` 或 `meta/pointcloud.ply`（静态障碍）
       b) `meta/dynamic_tracks.parquet`（动态对象轨迹）

2. Critic 监督来源不同
   - 静态版: pred/augment 都在线用静态障碍启发式计算。
   - 动态版:
       a) `pred_critic` 优先读取离线预计算结果（episode parquet 的 `critic_base_pred`
          或 `cache/critic/<episode>.parquet`）
       b) `augment_critic` 在线按“静态 + 动态”距离启发式计算

3. 新增动态体素输出
   - 动态版新增 `dynamic_voxels`，形状为 `(predict_frames, 4, X, Y, Z)`，
     4 个通道分别为 `[Occ, Vx, Vy, Vz]`。
   - 对应的 collate 输出新增 `batch_dynamic_voxels`。

4. 时间对齐策略
   - 动态版要求轨迹和动态对象均带 `timestamp_ns`。
   - 以轨迹时间为主时钟，动态对象按最近邻（带阈值）对齐；超阈值时该对象不参与该帧成本。

====================================
二、自制数据集目录与字段要求（Isaac Sim 可直接导出）
====================================
建议目录（与现有扫描逻辑兼容）：

root/group_name/scene_name/
  data/chunk_name/*.parquet
  videos/chunk_name/observation.images.rgb/*
  videos/chunk_name/observation.images.depth/*
  meta/episodes_stats.jsonl
  meta/pointcloud_static.ply            # 推荐
  meta/pointcloud.ply                   # 兼容旧版
  meta/dynamic_tracks.parquet           # 必需（动态场景）
  cache/dyn_voxel/<episode>.npz         # 可选（预缓存动态体素）
  cache/dyn_module_voxel/<episode>.npz  # 可选（dyn_module 估计体素）
  cache/critic/<episode>.parquet        # 可选（离线 critic）

------------------------------------
A. episode parquet 最低字段要求
------------------------------------
必须字段（至少满足以下之一）：
1) 位姿/动作
   - `action`：与现有 FlowNav/NavDP 一致，按帧存 4x4 位姿（或可还原到 4x4）
2) 相机参数
   - `observation.camera_intrinsic`：3x3
   - `observation.camera_extrinsic`：4x4
3) 时间戳（推荐必有）
   - `timestamp_ns` 或 `timestamps_ns` 或 `observation.timestamp_ns`
4) 离线原始轨迹 critic（可选但强烈建议）
   - `critic_base_pred`

------------------------------------
B. dynamic_tracks.parquet 最低字段要求
------------------------------------
必须包含时间、ID、位置；速度/尺寸建议提供：
1) 时间戳
   - `timestamp_ns`（推荐）
2) 对象标识
   - `object_id`
3) 对象位姿
   - 任选一种形式:
     a) `x`, `y`, `z`
     b) `pose_world`（长度 7 的 [x,y,z,qx,qy,qz,qw] 或长度 16 的 4x4）
4) 速度（建议）
   - 任选:
     a) `vx`, `vy`, `vz`
     b) `velocity_world`（长度 3）
5) 尺寸（建议）
   - 任选:
     a) `dx`, `dy`, `dz`
     b) `bbox_size`（长度 3）
6) 有效标记（可选）
   - `validity`（1 有效 / 0 无效）

------------------------------------
C. episodes_stats.jsonl 最低字段要求
------------------------------------
与现有流程一致：
- `image_index.min`
- `image_index.max`
- 额外推荐：`episode_id`, `start_time_ns`, `end_time_ns`, `num_frames`

====================================
三、实现原则
====================================
1. 兼容优先：尽量复用 `FlowNav_Base_Datset` 的图像/深度/动作构造逻辑。
2. 动态增强：只替换障碍读取、critic 计算、dynamic_voxels 构造相关逻辑。
3. 可诊断：对关键字段做显式校验，报错信息尽量可定位。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from internnav.dataset.flownav_lerobot_dataset import FlowNav_Base_Datset


class FlowNav_Dyn_Lerobot_Dataset(FlowNav_Base_Datset):
    """FlowNav 动态版数据集。

    该类继承自 `FlowNav_Base_Datset`，保留了静态版的数据清洗与轨迹构造流程，
    并在以下方面做动态扩展：

    1. 场景级动态对象轨迹读取（`dynamic_tracks.parquet`）
    2. 基于时间对齐的动态距离代价估计
    3. 动态体素 `dynamic_voxels` 构造/读取
    4. pred_critic 离线优先 + augment_critic 在线计算

    注意：
    - 本类不修改模型结构；只负责数据侧契约。
    - 若训练端尚未消费 `batch_dynamic_voxels`，需在 trainer 中补齐对该 key 的前向传参。

    Args:
        root_dirs: 数据根目录。
        preload_path: 预加载索引路径（沿用父类语义）。
        memory_size: 历史 RGB 序列长度。
        history_frames: 深度历史长度。
        predict_frames: 未来动态体素帧数（通常为 8）。
        predict_size: 动作预测步数。
        batch_size: DataLoader batch 大小，仅用于统计打印。
        image_size: 图像分辨率。
        scene_data_scale: 场景下采样比例。
        trajectory_data_scale: 轨迹下采样比例（当前保留参数）。
        pixel_channel: 像素目标通道。
        action_dim: 动作维度。
        fallback_fps: 当 parquet 缺少时间戳时使用的回退帧率（Hz）。
        debug: 调试开关。
        preload: 是否从 preload_path 直接加载索引。
        random_digit: 是否随机时间步长采样。
        prior_sample: 是否按障碍密度优先采样。
        dynamic_time_tolerance_ns: 动态轨迹与机器人轨迹的时间对齐阈值（纳秒）。
        dynamic_grid_shape: 动态体素网格尺寸 `(X, Y, Z)`。
        dynamic_grid_resolution: 体素边长（米）。
        dynamic_grid_z_min: 体素网格最小 z（相对机器人局部坐标）。
        use_cached_dynamic_voxels: 是否优先读取 `cache/dyn_voxel/<episode>.npz`。
        use_cached_est_dynamic_voxels: 是否读取 dyn_module 估计体素缓存。
        est_dynamic_voxel_subdir: 估计体素缓存子目录（默认 `dyn_module_voxel`）。
        est_voxel_ratio: 当前估计体素混入比例，0 表示全 GT，1 表示全估计。
        est_voxel_seed: 估计/真值混入采样随机种子（可复现实验）。
        use_cached_pred_critic: 是否优先读取离线 pred critic。
        dynamic_weight: critic 中动态距离权重。
        static_weight: critic 中静态距离权重。
        near_threshold: 启发式近距离惩罚阈值（米）。
    """

    def __init__(
        self,
        root_dirs,
        preload_path=False,
        memory_size=1,
        history_frames=2,
        predict_frames=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        pixel_channel=7,
        action_dim=3,
        fallback_fps: float = 30.0,
        debug=False,
        preload=False,
        random_digit=False,
        prior_sample=False,
        dynamic_time_tolerance_ns: int = 20_000_000,
        dynamic_grid_shape: Tuple[int, int, int] = (32, 32, 16),
        dynamic_grid_resolution: float = 0.25,
        dynamic_grid_z_min: float = -1.0,
        use_cached_dynamic_voxels: bool = True,
        use_cached_est_dynamic_voxels: bool = True,
        est_dynamic_voxel_subdir: str = "dyn_module_voxel",
        est_voxel_ratio: float = 0.0,
        est_voxel_seed: Optional[int] = 0,
        use_cached_pred_critic: bool = True,
        dynamic_weight: float = 1.0,
        static_weight: float = 1.0,
        near_threshold: float = 0.1,
    ):
        super().__init__(
            root_dirs=root_dirs,
            preload_path=preload_path,
            memory_size=memory_size,
            history_frames=history_frames,
            predict_frames=predict_frames,
            predict_size=predict_size,
            batch_size=batch_size,
            image_size=image_size,
            scene_data_scale=scene_data_scale,
            trajectory_data_scale=trajectory_data_scale,
            pixel_channel=pixel_channel,
            action_dim=action_dim,
            fallback_fps=fallback_fps,
            debug=debug,
            preload=preload,
            random_digit=random_digit,
            prior_sample=prior_sample,
        )

        # ==============================
        # 动态版新增配置
        # ==============================
        self.dynamic_time_tolerance_ns = int(dynamic_time_tolerance_ns)
        self.dynamic_grid_shape = tuple(dynamic_grid_shape)
        self.dynamic_grid_resolution = float(dynamic_grid_resolution)
        self.dynamic_grid_z_min = float(dynamic_grid_z_min)
        # GT 体素缓存开关：用于“预热阶段只喂真值”或回退逻辑。
        self.use_cached_dynamic_voxels = bool(use_cached_dynamic_voxels)
        # 估计体素缓存开关：用于“部署对齐”阶段喂 dyn_module 输出。
        self.use_cached_est_dynamic_voxels = bool(use_cached_est_dynamic_voxels)
        self.est_dynamic_voxel_subdir = str(est_dynamic_voxel_subdir)
        # 估计体素混入比例，核心课程学习控制量：
        #   0.0 -> 全 GT，1.0 -> 全估计。
        self.est_voxel_ratio = float(np.clip(est_voxel_ratio, 0.0, 1.0))
        # 使用独立随机数发生器，避免受全局 np.random 状态干扰。
        self._voxel_mix_rng = np.random.default_rng(est_voxel_seed)
        # 调试诊断字段：记录当前样本体素来源，便于日志审计。
        self._last_dynamic_voxel_source = "unknown"
        self.use_cached_pred_critic = bool(use_cached_pred_critic)
        self.dynamic_weight = float(dynamic_weight)
        self.static_weight = float(static_weight)
        self.near_threshold = float(near_threshold)

        # 每个 episode 衍生路径（长度与 trajectory_data_dir 一致）
        self.trajectory_dynamic_tracks_path: List[str] = []
        self.trajectory_dynamic_voxel_cache_path: List[str] = []
        self.trajectory_est_dynamic_voxel_cache_path: List[str] = []
        self.trajectory_critic_cache_path: List[str] = []

        # 文件级缓存，避免 DataLoader 高频重复读取磁盘
        self._dynamic_tracks_cache: Dict[str, pd.DataFrame] = {}
        self._critic_cache: Dict[str, Optional[float]] = {}

        self._build_dynamic_side_paths()

    # ------------------------------------------------------------------
    # 目录推导与字段校验
    # ------------------------------------------------------------------
    def _build_dynamic_side_paths(self) -> None:
        """从 `trajectory_data_dir` 推导动态相关路径。

        父类已经构建了每个 episode 的 parquet 路径，本方法按该路径回推 scene 目录，
        并生成如下附加路径：
        1) `meta/dynamic_tracks.parquet`
        2) `cache/dyn_voxel/<episode_stem>.npz`
        3) `cache/<est_dynamic_voxel_subdir>/<episode_stem>.npz`
        4) `cache/critic/<episode_stem>.parquet`

        同时若检测到 `meta/pointcloud_static.ply`，则覆盖父类默认的
        `meta/pointcloud.ply`，确保静态障碍读取更明确。
        """
        for i, episode_parquet_path in enumerate(self.trajectory_data_dir):
            episode_path = Path(episode_parquet_path)
            # 结构约定: scene/data/chunk/episode.parquet
            # 因此 scene 目录是 parents[2]
            scene_dir = episode_path.parents[2]

            dynamic_tracks_path = scene_dir / "meta" / "dynamic_tracks.parquet"
            dynamic_voxel_cache_path = scene_dir / "cache" / "dyn_voxel" / f"{episode_path.stem}.npz"
            est_dynamic_voxel_cache_path = (
                scene_dir / "cache" / self.est_dynamic_voxel_subdir / f"{episode_path.stem}.npz"
            )
            critic_cache_path = scene_dir / "cache" / "critic" / f"{episode_path.stem}.parquet"

            static_pcd_prefer = scene_dir / "meta" / "pointcloud_static.ply"
            if static_pcd_prefer.is_file():
                # 与静态版差异点：优先使用 pointcloud_static.ply
                self.trajectory_afford_path[i] = str(static_pcd_prefer)

            self.trajectory_dynamic_tracks_path.append(str(dynamic_tracks_path))
            self.trajectory_dynamic_voxel_cache_path.append(str(dynamic_voxel_cache_path))
            self.trajectory_est_dynamic_voxel_cache_path.append(str(est_dynamic_voxel_cache_path))
            self.trajectory_critic_cache_path.append(str(critic_cache_path))

    def set_est_voxel_ratio(self, ratio: float) -> float:
        """手动设置估计体素混入比例。

        设计目的：
        1) 训练器可按 epoch/step 外部控制课程学习。
        2) 所有采样都读同一个比例参数，便于复现实验。

        Args:
            ratio: 期望混入比例，自动裁剪到 [0,1]。

        Returns:
            float: 裁剪后的实际比例。
        """
        self.est_voxel_ratio = float(np.clip(ratio, 0.0, 1.0))
        return self.est_voxel_ratio

    def set_est_voxel_ratio_by_step(self, global_step: int, warmup_steps: int, mix_steps: int) -> float:
        """按训练步数更新“GT -> 估计体素”混入比例。

        课程学习策略：
        1) 预热阶段（global_step < warmup_steps）：比例=0，仅用 GT。
        2) 混合阶段：比例从 0 线性上升到 1。
        3) 收敛阶段：比例=1，仅用估计体素，逼近部署分布。

        Args:
            global_step: 当前全局训练步。
            warmup_steps: 预热步数。
            mix_steps: 线性上升阶段步数。

        Returns:
            float: 更新后的 `est_voxel_ratio`。
        """
        s = max(int(global_step), 0)
        w = max(int(warmup_steps), 0)
        m = max(int(mix_steps), 1)

        if s < w:
            ratio = 0.0
        else:
            ratio = (s - w) / float(m)
        return self.set_est_voxel_ratio(ratio)

    @staticmethod
    def _find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        """在候选列名中找到第一个存在的列。"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    # ------------------------------------------------------------------
    # 轨迹 parquet 读取（动态版扩展）
    # ------------------------------------------------------------------
    def process_data_parquet(self, index):
        """读取 episode parquet，并额外返回时间戳与离线 pred critic。

        与父类 `process_data_parquet` 的区别：
        1) 增加 `timestamps_ns` 解析（用于动态轨迹时间对齐）
        2) 尝试读取离线 `pred_critic`（优先 episode parquet，其次 cache/critic）

        Returns:
            tuple:
                camera_intrinsic: (3, 3)
                camera_extrinsic: (4, 4)
                camera_trajectory: (T, 4, 4)
                trajectory_length: int
                timestamps_ns: (T,) int64
                offline_pred_critic: Optional[float]
        """
        parquet_path = self.trajectory_data_dir[index]
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(parquet_path)

        df = pd.read_parquet(parquet_path)

        camera_intrinsic = np.vstack(np.array(df["observation.camera_intrinsic"].tolist()[0])).reshape(3, 3)
        camera_extrinsic = np.vstack(np.array(df["observation.camera_extrinsic"].tolist()[0])).reshape(4, 4)
        trajectory_length = len(df["action"].tolist())
        camera_trajectory = np.array([np.stack(frame) for frame in df["action"]], dtype=np.float64).reshape(-1, 4, 4)

        # 动态版新增：时间戳读取
        ts_col = self._find_first_existing_column(
            df,
            [
                "timestamp_ns",
                "timestamps_ns",
                "observation.timestamp_ns",
                "time_ns",
            ],
        )
        if ts_col is None:
            # 若源数据没有时间戳，退化为等间隔伪时间戳，保证流程可跑。
            # 强烈建议在 Isaac Sim 导出时提供真实 ns 时间戳。
            timestamps_ns = np.arange(trajectory_length, dtype=np.int64) * 100_000_000
        else:
            timestamps_ns = np.asarray(df[ts_col].tolist(), dtype=np.int64)
            if timestamps_ns.shape[0] != trajectory_length:
                # 防御性对齐：时间戳长度异常时截断/填充到轨迹长度
                if timestamps_ns.shape[0] > trajectory_length:
                    timestamps_ns = timestamps_ns[:trajectory_length]
                else:
                    pad = np.full((trajectory_length - timestamps_ns.shape[0],), timestamps_ns[-1], dtype=np.int64)
                    timestamps_ns = np.concatenate([timestamps_ns, pad], axis=0)

        offline_pred_critic = self._load_offline_pred_critic(index, df)

        return (
            camera_intrinsic,
            camera_extrinsic,
            camera_trajectory,
            trajectory_length,
            timestamps_ns,
            offline_pred_critic,
        )

    def _load_offline_pred_critic(self, index: int, episode_df: pd.DataFrame) -> Optional[float]:
        """加载离线 pred critic。

        优先级：
        1) episode parquet 内列 `critic_base_pred`
        2) `cache/critic/<episode>.parquet` 内列 `critic_base_pred` 或 `pred_critic`

        Returns:
            float 或 None
        """
        if not self.use_cached_pred_critic:
            return None

        # 1) 先查 episode parquet
        if "critic_base_pred" in episode_df.columns:
            vals = np.asarray(episode_df["critic_base_pred"].tolist(), dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                return float(vals.mean())

        # 2) 再查 critic cache
        cache_path = self.trajectory_critic_cache_path[index]
        if cache_path in self._critic_cache:
            return self._critic_cache[cache_path]

        pred_critic = None
        if os.path.isfile(cache_path):
            cdf = pd.read_parquet(cache_path)
            col = self._find_first_existing_column(cdf, ["critic_base_pred", "pred_critic"])
            if col is not None:
                vals = np.asarray(cdf[col].tolist(), dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    pred_critic = float(vals.mean())

        self._critic_cache[cache_path] = pred_critic
        return pred_critic

    # ------------------------------------------------------------------
    # 动态轨迹读取与标准化
    # ------------------------------------------------------------------
    def _load_dynamic_tracks(self, index: int) -> pd.DataFrame:
        """读取并标准化 `dynamic_tracks.parquet`。

        标准化后的 DataFrame 至少包含以下列：
        - timestamp_ns (int64)
        - object_id (str)
        - x, y, z (float64)
        - vx, vy, vz (float64)
        - dx, dy, dz (float64)
        - validity (int8)
        """
        path = self.trajectory_dynamic_tracks_path[index]
        if path in self._dynamic_tracks_cache:
            return self._dynamic_tracks_cache[path]

        if not os.path.isfile(path):
            empty = pd.DataFrame(
                columns=[
                    "timestamp_ns",
                    "object_id",
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "dx",
                    "dy",
                    "dz",
                    "validity",
                ]
            )
            self._dynamic_tracks_cache[path] = empty
            return empty

        raw = pd.read_parquet(path)
        out = pd.DataFrame()

        # 时间戳
        ts_col = self._find_first_existing_column(raw, ["timestamp_ns", "timestamps_ns", "time_ns"])
        if ts_col is None:
            raise ValueError(
                f"动态轨迹文件缺少时间戳列: {path}。"
                "需要至少一个列名: timestamp_ns / timestamps_ns / time_ns"
            )
        out["timestamp_ns"] = np.asarray(raw[ts_col].tolist(), dtype=np.int64)

        # object_id
        if "object_id" in raw.columns:
            out["object_id"] = raw["object_id"].astype(str)
        else:
            # 退化处理：若没有 object_id，则用行号分配（不推荐）
            out["object_id"] = np.arange(len(raw)).astype(str)

        # 位置
        if all(c in raw.columns for c in ["x", "y", "z"]):
            out["x"] = pd.to_numeric(raw["x"], errors="coerce").fillna(0.0)
            out["y"] = pd.to_numeric(raw["y"], errors="coerce").fillna(0.0)
            out["z"] = pd.to_numeric(raw["z"], errors="coerce").fillna(0.0)
        elif "pose_world" in raw.columns:
            xyz = raw["pose_world"].apply(self._parse_pose_world_xyz)
            out[["x", "y", "z"]] = np.vstack(xyz.to_numpy())
        else:
            raise ValueError(
                f"动态轨迹文件缺少位置列: {path}。"
                "需要 (x,y,z) 或 pose_world"
            )

        # 速度（可选，缺失则置 0）
        if all(c in raw.columns for c in ["vx", "vy", "vz"]):
            out["vx"] = pd.to_numeric(raw["vx"], errors="coerce").fillna(0.0)
            out["vy"] = pd.to_numeric(raw["vy"], errors="coerce").fillna(0.0)
            out["vz"] = pd.to_numeric(raw["vz"], errors="coerce").fillna(0.0)
        elif "velocity_world" in raw.columns:
            vxyz = raw["velocity_world"].apply(self._parse_vec3)
            out[["vx", "vy", "vz"]] = np.vstack(vxyz.to_numpy())
        else:
            out["vx"] = 0.0
            out["vy"] = 0.0
            out["vz"] = 0.0

        # 尺寸（可选，缺失给默认尺寸）
        if all(c in raw.columns for c in ["dx", "dy", "dz"]):
            out["dx"] = pd.to_numeric(raw["dx"], errors="coerce").fillna(0.6)
            out["dy"] = pd.to_numeric(raw["dy"], errors="coerce").fillna(0.6)
            out["dz"] = pd.to_numeric(raw["dz"], errors="coerce").fillna(1.7)
        elif "bbox_size" in raw.columns:
            dxyz = raw["bbox_size"].apply(self._parse_vec3)
            out[["dx", "dy", "dz"]] = np.vstack(dxyz.to_numpy())
        else:
            # 默认尺寸：近似“人体”占据
            out["dx"] = 0.6
            out["dy"] = 0.6
            out["dz"] = 1.7

        # 有效性
        if "validity" in raw.columns:
            out["validity"] = pd.to_numeric(raw["validity"], errors="coerce").fillna(1).astype(np.int8)
        else:
            out["validity"] = np.int8(1)

        # 排序便于后续时间窗口查询
        out = out.sort_values(["timestamp_ns", "object_id"]).reset_index(drop=True)

        self._dynamic_tracks_cache[path] = out
        return out

    @staticmethod
    def _parse_pose_world_xyz(value) -> np.ndarray:
        """解析 pose_world，返回 `[x, y, z]`。

        支持:
        - 长度 7: [x,y,z,qx,qy,qz,qw]
        - 长度 16: 4x4 展平
        """
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 7:
            return arr[:3]
        if arr.size == 16:
            mat = arr.reshape(4, 4)
            return mat[:3, 3]
        # 异常值兜底
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _parse_vec3(value) -> np.ndarray:
        """解析长度为 3 的向量，异常时返回全零。"""
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # 动态距离与 critic
    # ------------------------------------------------------------------
    def _nearest_dynamic_states(self, tracks_df: pd.DataFrame, timestamp_ns: int) -> pd.DataFrame:
        """按时间戳获取每个 object 的最近状态。

        策略：
        1) 先过滤 `validity == 1`
        2) 在 `|t_obj - t| <= tolerance` 窗口内做每个 object 最近邻
        3) 超出窗口的对象不参与该时刻成本
        """
        if tracks_df.empty:
            return tracks_df

        valid = tracks_df[tracks_df["validity"] == 1]
        if valid.empty:
            return valid

        dt = np.abs(valid["timestamp_ns"].to_numpy(dtype=np.int64) - int(timestamp_ns))
        within = valid.loc[dt <= self.dynamic_time_tolerance_ns]
        if within.empty:
            return within

        # 每个 object 仅保留当前时刻最近的一条
        within = within.copy()
        within["_abs_dt"] = np.abs(within["timestamp_ns"] - int(timestamp_ns))
        nearest = within.sort_values("_abs_dt").groupby("object_id", sort=False).head(1)
        return nearest.drop(columns=["_abs_dt"])

    @staticmethod
    def _min_l1_distance_to_static(points_world: np.ndarray, static_points: np.ndarray) -> np.ndarray:
        """计算每个轨迹点到静态点云的最小 L1 距离（xy 平面）。"""
        if static_points.shape[0] == 0:
            return np.full((points_world.shape[0],), np.inf, dtype=np.float64)
        # (T,1,2) - (1,N,2) -> (T,N)
        dist = np.abs(points_world[:, None, 0:2] - static_points[None, :, 0:2]).sum(axis=-1)
        return dist.min(axis=-1)

    def _min_dynamic_distance_series(
        self,
        points_world: np.ndarray,
        timestamps_ns: np.ndarray,
        tracks_df: pd.DataFrame,
    ) -> np.ndarray:
        """计算轨迹每一步到动态对象最近“边界距离”。

        距离定义：
        - 先算机器人点到对象中心的平面欧氏距离
        - 再减去对象近似半径 `0.5 * max(dx, dy)`
        - 下限截断为 0
        """
        if tracks_df.empty:
            return np.full((points_world.shape[0],), np.inf, dtype=np.float64)

        out = np.full((points_world.shape[0],), np.inf, dtype=np.float64)
        for i in range(points_world.shape[0]):
            ts = int(timestamps_ns[min(i, timestamps_ns.shape[0] - 1)])
            states = self._nearest_dynamic_states(tracks_df, ts)
            if states.empty:
                continue

            centers = states[["x", "y"]].to_numpy(dtype=np.float64)
            radius = 0.5 * np.maximum(
                states["dx"].to_numpy(dtype=np.float64),
                states["dy"].to_numpy(dtype=np.float64),
            )
            d_center = np.linalg.norm(centers - points_world[i, 0:2][None, :], axis=1)
            d_surface = np.maximum(0.0, d_center - radius)
            out[i] = float(d_surface.min())

        return out

    def _compute_critic_from_dist(
        self,
        effective_distance: np.ndarray,
        action_indexes: np.ndarray,
    ) -> float:
        """沿用静态版启发式公式，根据有效距离序列计算 critic。"""
        if effective_distance.shape[0] == 0:
            return 2.0

        safe_idx = np.clip(action_indexes, 0, effective_distance.shape[0] - 1)
        # 近距离惩罚 + 趋势奖励
        collision_penalty = -5.0 * (effective_distance[safe_idx[:-1]] < self.near_threshold).mean()
        progress_term = 0.5 * (effective_distance[safe_idx][1:] - effective_distance[safe_idx][:-1]).sum()
        return float(collision_penalty + progress_term)

    def _compute_dynamic_aware_critic(
        self,
        points_world: np.ndarray,
        timestamps_ns: np.ndarray,
        action_indexes: np.ndarray,
        static_points: np.ndarray,
        tracks_df: pd.DataFrame,
    ) -> float:
        """计算动态感知 critic（静态 + 动态融合）。

        融合策略：
        - 先分别计算静态最小距离 `d_static` 和动态最小距离 `d_dynamic`
        - 再用加权最小值构造有效距离：
            `d_eff = min(static_weight * d_static, dynamic_weight * d_dynamic)`

        说明：
        该策略强调“最危险来源主导”，适合用于导航安全惩罚。
        """
        d_static = self._min_l1_distance_to_static(points_world, static_points)
        d_dynamic = self._min_dynamic_distance_series(points_world, timestamps_ns, tracks_df)

        # 若动态不存在（全 inf），则退化为静态距离
        if np.isinf(d_dynamic).all():
            d_eff = self.static_weight * d_static
        else:
            d_eff = np.minimum(self.static_weight * d_static, self.dynamic_weight * d_dynamic)

        return self._compute_critic_from_dist(d_eff, action_indexes)

    # ------------------------------------------------------------------
    # 动态体素构造
    # ------------------------------------------------------------------
    def _normalize_dynamic_voxel_shape(self, vox: np.ndarray) -> Optional[np.ndarray]:
        """对 dynamic_voxels 形状做统一规范。

        规范目标：
        1) 确保返回 shape 为 `(predict_frames, 4, X, Y, Z)`。
        2) 若通道维不为 4，直接判定不可用并回退，避免静默错误。
        3) 对帧数进行裁剪/补零，保证 batch 内维度稳定。
        """
        arr = np.asarray(vox, dtype=np.float32)
        if arr.ndim != 5:
            return None
        if arr.shape[1] != 4:
            return None

        # 体素空间维不强制与配置一致（允许离线缓存先验），
        # 但至少要求非空，避免后续 reshape/conv 异常。
        if min(arr.shape[2], arr.shape[3], arr.shape[4]) <= 0:
            return None

        if arr.shape[0] > self.predict_frames:
            arr = arr[: self.predict_frames]
        elif arr.shape[0] < self.predict_frames:
            pad_shape = (self.predict_frames - arr.shape[0],) + arr.shape[1:]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        return arr

    def _load_cached_dynamic_voxels(self, index: int) -> Optional[np.ndarray]:
        """读取 GT 预缓存动态体素（若可用）。"""
        if not self.use_cached_dynamic_voxels:
            return None

        cache_path = self.trajectory_dynamic_voxel_cache_path[index]
        if not os.path.isfile(cache_path):
            return None

        npz = np.load(cache_path)
        if "dynamic_voxels" not in npz:
            return None

        vox = self._normalize_dynamic_voxel_shape(npz["dynamic_voxels"])
        return vox

    def _load_cached_est_dynamic_voxels(self, index: int) -> Optional[np.ndarray]:
        """读取 dyn_module 估计体素缓存（若可用）。

        说明：
        - 这里默认读取 `cache/<est_dynamic_voxel_subdir>/<episode>.npz`。
        - 文件内键仍统一使用 `dynamic_voxels`，便于复用离线导出脚本。
        """
        if not self.use_cached_est_dynamic_voxels:
            return None

        cache_path = self.trajectory_est_dynamic_voxel_cache_path[index]
        if not os.path.isfile(cache_path):
            return None

        npz = np.load(cache_path)
        if "dynamic_voxels" not in npz:
            return None

        vox = self._normalize_dynamic_voxel_shape(npz["dynamic_voxels"])
        return vox

    def _build_dynamic_voxels_online(
        self,
        tracks_df: pd.DataFrame,
        target_timestamps_ns: np.ndarray,
        anchor_pose_world: np.ndarray,
    ) -> np.ndarray:
        """在线构造动态体素 `(T, 4, X, Y, Z)`。

        通道定义：
        - ch0: Occ（占据，0/1）
        - ch1: Vx  （局部坐标系 x 速度）
        - ch2: Vy
        - ch3: Vz

        坐标系：
        - 体素网格在 `anchor_pose_world` 对应的机器人局部坐标系中构造。
        """
        T = int(target_timestamps_ns.shape[0])
        X, Y, Z = self.dynamic_grid_shape
        vox = np.zeros((T, 4, X, Y, Z), dtype=np.float32)
        count = np.zeros((T, X, Y, Z), dtype=np.float32)

        # 世界->局部变换
        world_to_local = np.linalg.inv(anchor_pose_world)
        rot_world_to_local = world_to_local[:3, :3]

        x_half = (X * self.dynamic_grid_resolution) / 2.0
        y_half = (Y * self.dynamic_grid_resolution) / 2.0

        for t_idx in range(T):
            ts = int(target_timestamps_ns[t_idx])
            states = self._nearest_dynamic_states(tracks_df, ts)
            if states.empty:
                continue

            for row in states.itertuples(index=False):
                # 位置 world -> local
                p_world = np.array([row.x, row.y, row.z, 1.0], dtype=np.float64)
                p_local = world_to_local @ p_world

                # 速度 world -> local（仅旋转）
                v_world = np.array([row.vx, row.vy, row.vz], dtype=np.float64)
                v_local = rot_world_to_local @ v_world

                ix = int(np.floor((p_local[0] + x_half) / self.dynamic_grid_resolution))
                iy = int(np.floor((p_local[1] + y_half) / self.dynamic_grid_resolution))
                iz = int(np.floor((p_local[2] - self.dynamic_grid_z_min) / self.dynamic_grid_resolution))

                if ix < 0 or ix >= X or iy < 0 or iy >= Y or iz < 0 or iz >= Z:
                    continue

                vox[t_idx, 0, ix, iy, iz] = 1.0
                vox[t_idx, 1, ix, iy, iz] += float(v_local[0])
                vox[t_idx, 2, ix, iy, iz] += float(v_local[1])
                vox[t_idx, 3, ix, iy, iz] += float(v_local[2])
                count[t_idx, ix, iy, iz] += 1.0

        # 对速度通道做体素内平均
        valid_mask = count > 0
        for c in (1, 2, 3):
            ch = vox[:, c]
            ch[valid_mask] = ch[valid_mask] / count[valid_mask]
            vox[:, c] = ch

        return vox

    # ------------------------------------------------------------------
    # 采样主流程
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        """获取单条动态 FlowNav 训练样本。

        与静态版的差异总结：
        1) 读取 `timestamps_ns` 与动态轨迹；
        2) `pred_critic` 离线优先；
        3) `augment_critic` 使用静态+动态融合距离在线计算；
        4) 产出新增字段 `dynamic_voxels`。

        Returns:
            tuple:
                0  point_goal
                1  image_goal
                2  pixel_goal
                3  memory_images
                4  depth_images
                5  pred_actions
                6  augment_actions
                7  pred_critic
                8  augment_critic
                9  odom_pose
                10 odom_delta
                11 dynamic_voxels
                12 pixel_flag
        """
        import time

        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
            trajectory_timestamps_ns,
            offline_pred_critic,
        ) = self.process_data_parquet(index)

        if trajectory_length < 3:
            # 极短轨迹无法构造动作监督，回退采样邻近样本
            return self.__getitem__((index + 1) % len(self))

        # 静态障碍（继承父类逻辑）
        trajectory_static_points, _ = self.process_obstacle_points(index)

        # 动态轨迹
        dynamic_tracks_df = self._load_dynamic_tracks(index)

        # 采样起终点
        if self.prior_sample and trajectory_static_points.shape[0] > 0:
            pixel_start_choice, target_choice = self.rank_steps(trajectory_extrinsics, trajectory_static_points)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, max(trajectory_length // 2, 1))
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        # 时间步长采样
        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        # 构造历史观测
        memory_images, depth_images, _, _ = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )

        # 构造标签动作和增强动作
        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            memory_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )

        # xyz -> xyt
        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)

        # odometry
        odom_global_idx = np.clip(memory_start_choice + action_indexes, 0, trajectory_extrinsics.shape[0] - 1).astype(np.int64)
        odom_pose = trajectory_extrinsics[odom_global_idx]
        odom_delta = np.matmul(np.linalg.inv(odom_pose[:-1]), odom_pose[1:])

        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]

        # 时间对齐：为每个世界轨迹点取对应时间戳
        full_world_idx = np.arange(memory_start_choice, target_choice + 1)
        full_world_idx = np.clip(full_world_idx, 0, trajectory_timestamps_ns.shape[0] - 1)
        full_world_ts = trajectory_timestamps_ns[full_world_idx]

        # ------------------------------
        # Critic 计算（动态版核心差异）
        # ------------------------------
        if offline_pred_critic is not None:
            pred_critic = float(offline_pred_critic)
        else:
            pred_critic = self._compute_dynamic_aware_critic(
                points_world=target_world_points,
                timestamps_ns=full_world_ts,
                action_indexes=action_indexes,
                static_points=trajectory_static_points,
                tracks_df=dynamic_tracks_df,
            )

        augment_critic = self._compute_dynamic_aware_critic(
            points_world=augment_world_points,
            timestamps_ns=full_world_ts,
            action_indexes=action_indexes,
            static_points=trajectory_static_points,
            tracks_df=dynamic_tracks_df,
        )

        # 目标条件
        point_goal = target_xyt_actions[-1]
        image_goal = np.concatenate(
            (
                self.process_image(self.trajectory_rgb_path[index][target_choice]),
                self.process_image(self.trajectory_rgb_path[index][memory_start_choice]),
            ),
            axis=-1,
        )

        # 像素目标监督
        pixel_target_local_points, _, _, _, _ = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            pixel_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )
        pixel_init_vector = pixel_target_local_points[1] - pixel_target_local_points[0]
        pixel_xyt_actions = self.xyz_to_xyt(pixel_target_local_points, pixel_init_vector)
        pixel_goal, pixel_flag = self.process_pixel_goal(
            self.trajectory_rgb_path[index][pixel_start_choice],
            pixel_xyt_actions[-1],
            camera_intrinsic,
            trajectory_base_extrinsic,
        )
        if self.pixel_channel == 7:
            pixel_goal = np.concatenate((pixel_goal, memory_images[-1]), axis=-1)

        # 增量动作
        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        pred_actions = np.pad(
            pred_actions,
            ((0, 0), (0, self.action_dim - pred_actions.shape[-1])),
            mode="constant",
            constant_values=(0, 0),
        )
        augment_actions = np.pad(
            augment_actions,
            ((0, 0), (0, self.action_dim - augment_actions.shape[-1])),
            mode="constant",
            constant_values=(0, 0),
        )

        # ------------------------------
        # dynamic_voxels（动态版核心差异）
        # ------------------------------
        # 策略说明（本次改动核心）：
        # 1) GT 体素：代表理想监督分布，利于前期稳定收敛。
        # 2) 估计体素：代表部署真实分布（dyn_module 输出），利于后期对齐。
        # 3) 课程学习：按 est_voxel_ratio 在两者间随机混合，平滑过渡。
        cached_gt_vox = self._load_cached_dynamic_voxels(index)
        cached_est_vox = self._load_cached_est_dynamic_voxels(index)

        dynamic_voxels = None
        # 仅当估计体素存在时才允许采样到估计分支；避免“抽中估计却拿不到数据”。
        if cached_est_vox is not None:
            # 若 GT 不存在，强制使用估计体素，保证样本可训练。
            if cached_gt_vox is None:
                use_est = True
            else:
                use_est = bool(self._voxel_mix_rng.random() < self.est_voxel_ratio)

            if use_est:
                dynamic_voxels = cached_est_vox
                self._last_dynamic_voxel_source = "est_cache"
            else:
                dynamic_voxels = cached_gt_vox
                self._last_dynamic_voxel_source = "gt_cache"
        elif cached_gt_vox is not None:
            # 只有 GT 可用时，保持旧行为。
            dynamic_voxels = cached_gt_vox
            self._last_dynamic_voxel_source = "gt_cache"

        if dynamic_voxels is None:
            # 双缓存都缺失时，在线构造 GT 体素兜底。
            # 这样即使离线缓存未准备完成，训练也不会中断。
            voxel_frame_idx = np.clip(
                memory_start_choice + np.arange(self.predict_frames, dtype=np.int64) * pred_digit,
                0,
                trajectory_timestamps_ns.shape[0] - 1,
            )
            voxel_ts = trajectory_timestamps_ns[voxel_frame_idx]
            anchor_pose_world = trajectory_extrinsics[memory_start_choice]
            dynamic_voxels = self._build_dynamic_voxels_online(
                tracks_df=dynamic_tracks_df,
                target_timestamps_ns=voxel_ts,
                anchor_pose_world=anchor_pose_world,
            )
            self._last_dynamic_voxel_source = "gt_online"

        # 统计耗时
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(
                f"__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s, cnt={self.item_cnt}"
            )
            self.batch_time_sum = 0.0

        # 转 tensor
        point_goal = torch.tensor(point_goal, dtype=torch.float32)
        image_goal = torch.tensor(image_goal, dtype=torch.float32)
        pixel_goal = torch.tensor(pixel_goal, dtype=torch.float32)
        memory_images = torch.tensor(memory_images, dtype=torch.float32)
        depth_images = torch.tensor(depth_images, dtype=torch.float32)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        augment_actions = torch.tensor(augment_actions, dtype=torch.float32)
        pred_critic = torch.tensor(pred_critic, dtype=torch.float32)
        augment_critic = torch.tensor(augment_critic, dtype=torch.float32)
        odom_pose = torch.tensor(odom_pose, dtype=torch.float32)
        odom_delta = torch.tensor(odom_delta, dtype=torch.float32)
        dynamic_voxels = torch.tensor(dynamic_voxels, dtype=torch.float32)

        return (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_images,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            odom_pose,
            odom_delta,
            dynamic_voxels,
            float(pixel_flag),
        )


def flownav_dyn_collate_fn(batch):
    """FlowNav 动态数据集 collate 函数。

    与静态版 `flownav_collate_fn` 的差异：
    1) 新增 `batch_dynamic_voxels`
    2) 新增 `batch_pixel_flag`

    Returns:
        dict: 训练输入字典。
    """
    collated = {
        "batch_pg": torch.stack([item[0] for item in batch]),
        "batch_ig": torch.stack([item[1] for item in batch]),
        "batch_tg": torch.stack([item[2] for item in batch]),
        "batch_rgb": torch.stack([item[3] for item in batch]),
        "batch_depth": torch.stack([item[4] for item in batch]),
        "batch_labels": torch.stack([item[5] for item in batch]),
        "batch_augments": torch.stack([item[6] for item in batch]),
        "batch_label_critic": torch.stack([item[7] for item in batch]),
        "batch_augment_critic": torch.stack([item[8] for item in batch]),
        "batch_odom_pose": torch.stack([item[9] for item in batch]),
        "batch_odom_delta": torch.stack([item[10] for item in batch]),
        "batch_dynamic_voxels": torch.stack([item[11] for item in batch]),
        "batch_pixel_flag": torch.tensor([item[12] for item in batch], dtype=torch.float32),
    }
    return collated
