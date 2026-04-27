"""IRA 输出后处理器。

将 IRA (isaacsim.replicator.agent) 的原始输出转换为 dataengine 标准格式。

IRA 的 IRABasicWriter 输出结构（典型）::

    output_dir/
    ├── 0000/           # 第 0 帧
    │   ├── rgb_0000.png
    │   ├── camera_params_0000.json
    │   ├── bounding_box_2d_tight_0000.npy
    │   ├── bounding_box_2d_loose_0000.npy
    │   ├── bounding_box_3d_0000.npy
    │   └── ...
    ├── 0001/
    │   └── ...
    └── ...

后处理器负责：
1. 扫描 IRA 输出目录，收集每帧数据
2. 将 bounding box 数据转换为 dynamic_tracks.v1alpha 格式
3. 统计帧数、对象数等元信息
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _try_load_npy(path: str) -> Optional[Any]:
    """尝试加载 .npy 文件。"""
    try:
        import numpy as np  # type: ignore[import-not-found]
        return np.load(path, allow_pickle=True)
    except (ImportError, Exception):
        return None


def _try_load_json(path: str) -> Optional[Dict]:
    """尝试加载 JSON 文件。"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class PostProcessor:
    """IRA 输出后处理器。

    将 IRA 的帧级输出汇总为 dataengine 标准化的动态轨迹和统计信息。
    """

    def __init__(self, scene_id: str, output_dir: str, dt_s: float = 1.0) -> None:
        """
        Args:
            scene_id: 场景 ID。
            output_dir: IRA 输出根目录。
            dt_s: 帧间隔（秒），用于计算 timestamp_ns。
        """
        self.scene_id = scene_id
        self.output_dir = output_dir
        self.dt_s = dt_s

    def scan_frames(self) -> List[str]:
        """扫描输出目录下的帧子目录。

        Returns:
            按帧号排序的子目录路径列表。
        """
        if not os.path.isdir(self.output_dir):
            logger.warning("IRA output directory not found: %s", self.output_dir)
            return []

        frame_dirs: List[str] = []
        for entry in sorted(os.listdir(self.output_dir)):
            full = os.path.join(self.output_dir, entry)
            if os.path.isdir(full):
                frame_dirs.append(full)

        return frame_dirs

    def _extract_bbox3d_tracks(
        self,
        frame_dir: str,
        frame_idx: int,
        timestamp_ns: int,
    ) -> List[Dict[str, Any]]:
        """从单帧的 3D bounding box 数据中提取动态轨迹。

        IRA 的 bounding_box_3d 输出（npy 格式）包含：
        - object_id / semantic_id
        - 中心位置 (x, y, z)
        - 尺寸 (dx, dy, dz)
        - 朝向 (quaternion 或 euler)

        Returns:
            该帧的所有对象轨迹行。
        """
        tracks: List[Dict[str, Any]] = []

        # 尝试加载 3D bounding box
        bbox3d_pattern = f"bounding_box_3d_{frame_idx:04d}"
        npy_candidates = [
            os.path.join(frame_dir, f"{bbox3d_pattern}.npy"),
            os.path.join(frame_dir, f"bounding_box_3d.npy"),
        ]

        bbox_data = None
        for npy_path in npy_candidates:
            if os.path.isfile(npy_path):
                bbox_data = _try_load_npy(npy_path)
                break

        # 同时查看 JSON 格式的 bounding box（某些 IRA 版本输出 JSON）
        json_candidates = [
            os.path.join(frame_dir, f"{bbox3d_pattern}.json"),
            os.path.join(frame_dir, "bounding_box_3d.json"),
        ]
        bbox_json = None
        for json_path in json_candidates:
            bbox_json = _try_load_json(json_path)
            if bbox_json is not None:
                break

        if bbox_data is not None:
            tracks.extend(self._parse_bbox3d_npy(bbox_data, timestamp_ns))
        elif bbox_json is not None:
            tracks.extend(self._parse_bbox3d_json(bbox_json, timestamp_ns))

        return tracks

    def _parse_bbox3d_npy(
        self,
        bbox_data: Any,
        timestamp_ns: int,
    ) -> List[Dict[str, Any]]:
        """解析 npy 格式的 3D bounding box 数据。"""
        tracks: List[Dict[str, Any]] = []
        try:
            # IRA 的 npy 通常是 structured array 或 dict
            if hasattr(bbox_data, "item"):
                bbox_data = bbox_data.item()

            if isinstance(bbox_data, dict):
                # 格式: {"data": [...], "info": {...}}
                data_list = bbox_data.get("data", [])
                for obj in data_list:
                    track = self._bbox_obj_to_track(obj, timestamp_ns)
                    if track:
                        tracks.append(track)
            elif hasattr(bbox_data, "__iter__"):
                for obj in bbox_data:
                    if isinstance(obj, dict):
                        track = self._bbox_obj_to_track(obj, timestamp_ns)
                        if track:
                            tracks.append(track)
        except Exception as e:
            logger.debug("Failed to parse bbox3d npy: %s", e)

        return tracks

    def _parse_bbox3d_json(
        self,
        bbox_json: Dict,
        timestamp_ns: int,
    ) -> List[Dict[str, Any]]:
        """解析 JSON 格式的 3D bounding box 数据。"""
        tracks: List[Dict[str, Any]] = []
        data_list = bbox_json.get("data", [])
        if isinstance(data_list, list):
            for obj in data_list:
                track = self._bbox_obj_to_track(obj, timestamp_ns)
                if track:
                    tracks.append(track)
        return tracks

    def _bbox_obj_to_track(
        self,
        obj: Dict,
        timestamp_ns: int,
    ) -> Optional[Dict[str, Any]]:
        """将单个 bounding box 对象转换为 dynamic_tracks.v1alpha 格式。"""
        try:
            object_id = str(obj.get("semanticId", obj.get("object_id", "")))
            if not object_id:
                return None

            # 位置：优先 center，其次 transform
            center = obj.get("center", obj.get("position", [0.0, 0.0, 0.0]))
            if isinstance(center, (list, tuple)) and len(center) >= 3:
                pos = [float(center[0]), float(center[1]), float(center[2])]
            else:
                pos = [0.0, 0.0, 0.0]

            # 尺寸
            extent = obj.get("extent", obj.get("size", [0.6, 0.6, 1.7]))
            if isinstance(extent, (list, tuple)) and len(extent) >= 3:
                bbox = [float(extent[0]), float(extent[1]), float(extent[2])]
            else:
                bbox = [0.6, 0.6, 1.7]

            # 类别推断
            semantic_label = str(obj.get("semanticLabel", obj.get("class", "character")))
            category = "people" if "character" in semantic_label.lower() else "object"

            return {
                "schema_version": "v1alpha",
                "contract": "dynamic_tracks.v1alpha",
                "scene_id": self.scene_id,
                "object_id": object_id,
                "category": category,
                "timestamp_ns": int(timestamp_ns),
                "position_xyz": pos,
                "velocity_xyz": [0.0, 0.0, 0.0],  # IRA 不直接输出速度，需要差分计算
                "bbox_xyz": bbox,
                "yaw_deg": float(obj.get("yaw", obj.get("rotation_z", 0.0))),
                "source_backend": "ira",
                "motion_mode": "walk",
                "asset_relpath": str(obj.get("asset_path", "")),
            }
        except Exception as e:
            logger.debug("Failed to convert bbox object to track: %s", e)
            return None

    def _compute_velocities(self, all_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """通过相邻帧位置差分计算速度。

        IRA 不直接输出速度信息，需要后处理差分计算。
        """
        if not all_tracks:
            return all_tracks

        # 按 object_id 分组，按 timestamp_ns 排序
        by_object: Dict[str, List[Dict[str, Any]]] = {}
        for track in all_tracks:
            oid = track["object_id"]
            if oid not in by_object:
                by_object[oid] = []
            by_object[oid].append(track)

        for oid, obj_tracks in by_object.items():
            obj_tracks.sort(key=lambda t: t["timestamp_ns"])
            for i in range(1, len(obj_tracks)):
                prev = obj_tracks[i - 1]
                curr = obj_tracks[i]
                dt_ns = curr["timestamp_ns"] - prev["timestamp_ns"]
                if dt_ns > 0:
                    dt_s = dt_ns / 1e9
                    vx = (curr["position_xyz"][0] - prev["position_xyz"][0]) / dt_s
                    vy = (curr["position_xyz"][1] - prev["position_xyz"][1]) / dt_s
                    vz = (curr["position_xyz"][2] - prev["position_xyz"][2]) / dt_s
                    curr["velocity_xyz"] = [round(vx, 4), round(vy, 4), round(vz, 4)]

        return all_tracks

    def process(self, base_timestamp_ns: int = 0) -> Dict[str, Any]:
        """执行完整的后处理流程。

        Args:
            base_timestamp_ns: 起始时间戳（纳秒）。

        Returns:
            包含以下字段的结果字典：
            - track_file: 输出的 dynamic_tracks.jsonl 路径
            - frame_count: 处理的帧数
            - object_count: 检测到的唯一对象数
            - sample_count: 总轨迹样本数
            - object_ids: 所有检测到的对象 ID 列表
        """
        frame_dirs = self.scan_frames()
        if not frame_dirs:
            return self._empty_result()

        all_tracks: List[Dict[str, Any]] = []
        object_ids: set = set()

        for frame_idx, frame_dir in enumerate(frame_dirs):
            timestamp_ns = base_timestamp_ns + int(frame_idx * self.dt_s * 1e9)
            frame_tracks = self._extract_bbox3d_tracks(
                frame_dir=frame_dir,
                frame_idx=frame_idx,
                timestamp_ns=timestamp_ns,
            )
            for track in frame_tracks:
                object_ids.add(track["object_id"])
            all_tracks.extend(frame_tracks)

        # 差分计算速度
        all_tracks = self._compute_velocities(all_tracks)

        # 写入 dynamic_tracks.jsonl
        track_file = os.path.join(self.output_dir, "dynamic_tracks.jsonl")
        with open(track_file, "w", encoding="utf-8") as f:
            for track in all_tracks:
                f.write(json.dumps(track, ensure_ascii=False) + "\n")

        return {
            "track_file": os.path.abspath(track_file),
            "frame_count": len(frame_dirs),
            "object_count": len(object_ids),
            "sample_count": len(all_tracks),
            "object_ids": sorted(object_ids),
        }

    def _empty_result(self) -> Dict[str, Any]:
        """无数据时的空结果。"""
        return {
            "track_file": "",
            "frame_count": 0,
            "object_count": 0,
            "sample_count": 0,
            "object_ids": [],
        }

    def generate_summary(self) -> Dict[str, Any]:
        """生成 IRA 输出的摘要统计。

        扫描输出目录，统计帧数、图像数、标注文件数等。
        """
        frame_dirs = self.scan_frames()
        total_images = 0
        total_annotations = 0

        for fd in frame_dirs:
            for fname in os.listdir(fd):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    total_images += 1
                elif fname.endswith((".npy", ".json")):
                    total_annotations += 1

        return {
            "output_dir": self.output_dir,
            "frame_count": len(frame_dirs),
            "total_images": total_images,
            "total_annotations": total_annotations,
        }
