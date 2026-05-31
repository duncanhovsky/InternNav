from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import SceneLayerConfig
from .errors import ErrorCode, SceneCompileError
from .models import SceneCandidate
from .utils import atomic_write_text, stable_hash


@dataclass
class SceneComposer:
    """场景组合器。

    支持两种后端：
    1. stub: 仅生成 stage_spec，离线快速联调。
    2. isaac_replicator: 使用 Isaac Sim + Replicator 进行可复现程序化组合并导出 USD。
    """

    cfg: SceneLayerConfig

    def _ensure_template_exists(self, candidate: SceneCandidate) -> None:
        """确保模板 USD 可访问，避免后续组合阶段才失败。"""
        if not os.path.isfile(candidate.template.usd_path):
            raise SceneCompileError(
                code=ErrorCode.ASSET_MISSING,
                message=f"模板 USD 不存在: {candidate.template.usd_path}",
                scene_id=candidate.scene_id,
            )

    def _build_layout_variant(self, candidate: SceneCandidate) -> Dict[str, float]:
        """构建轻量布局参数摘要，用于审计和复现。"""
        rng = random.Random(candidate.seed)
        layout_variant = {
            "obstacle_density": round(rng.uniform(0.2, 0.6), 3),
            "aisle_width_m": round(rng.uniform(1.0, 2.8), 3),
            "door_open_ratio": round(rng.uniform(0.4, 1.0), 3),
            "light_intensity_scale": round(rng.uniform(0.7, 1.3), 3),
        }

        if candidate.mode == "modular":
            layout_variant.update(
                {
                    "module_rows": int(rng.randint(2, 6)),
                    "module_cols": int(rng.randint(2, 8)),
                    "turn_probability": round(rng.uniform(0.1, 0.5), 3),
                }
            )
        return layout_variant

    def _scene_dir(self, candidate: SceneCandidate) -> str:
        """返回当前场景输出目录，不存在则创建。"""
        scene_dir = os.path.join(self.cfg.usd_output_root, candidate.scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        return scene_dir

    def _write_spec(self, spec_path: str, stage_spec: Dict) -> None:
        """原子写入 JSON 规格文件。"""
        atomic_write_text(spec_path, json.dumps(stage_spec, ensure_ascii=False, indent=2))

    @staticmethod
    def _read_json_file(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compose_stub(self, candidate: SceneCandidate, scene_dir: str) -> Dict[str, str]:
        """离线模式组合：仅生成 stage_spec，不依赖 Isaac Runtime。"""
        layout_variant = self._build_layout_variant(candidate)

        stage_spec = {
            "backend": "stub",
            "scene_id": candidate.scene_id,
            "scene_type": candidate.scene_type,
            "mode": candidate.mode,
            "seed": candidate.seed,
            "template_id": candidate.template.template_id,
            "template_usd": candidate.template.usd_path,
            "layout_variant": layout_variant,
        }
        spec_path = os.path.join(scene_dir, "stage_spec.json")
        self._write_spec(spec_path, stage_spec)

        stage_usd = candidate.template.usd_path
        layout_hash = stable_hash(json.dumps(stage_spec, ensure_ascii=False, sort_keys=True))

        return {
            "scene_dir": scene_dir,
            "stage_usd": stage_usd,
            "stage_spec_path": spec_path,
            "layout_hash": layout_hash,
        }

    def _resolve_replicator_props(self) -> List[str]:
        """解析可用于程序化摆放的资产池。"""
        if self.cfg.replicator_prop_relpaths:
            resolved = [os.path.join(self.cfg.asset_root, p) for p in self.cfg.replicator_prop_relpaths]
            return [p for p in resolved if os.path.isfile(p)]

        props_root = os.path.join(
            self.cfg.asset_root,
            "Isaac/Environments/Modular_Warehouse/Props",
        )
        if not os.path.isdir(props_root):
            return []

        out: List[str] = []
        for name in sorted(os.listdir(props_root)):
            if not name.endswith(".usd"):
                continue
            out.append(os.path.join(props_root, name))
        return out

    @staticmethod
    def _classify_prop_asset(usd_path: str) -> str:
        """基于文件名做轻量类别映射，用于比例采样与分区放置。"""
        name = os.path.basename(usd_path).lower()
        large_keys = ["rack", "shelf", "forklift", "pallet", "container", "table", "cabinet", "crate"]
        medium_keys = ["barrel", "bin", "box", "cart", "chair", "plant", "cone", "stand"]
        if any(k in name for k in large_keys):
            return "large"
        if any(k in name for k in medium_keys):
            return "medium"
        return "small"

    @staticmethod
    def _allocate_counts_by_ratio(total: int, ratio: Dict[str, float]) -> Dict[str, int]:
        """按比例分配整数数量（最大余数法），保证和为 total。"""
        sum_ratio = sum(ratio.values())
        norm = {k: v / sum_ratio for k, v in ratio.items()}
        raw = {k: total * p for k, p in norm.items()}
        base = {k: int(math.floor(v)) for k, v in raw.items()}
        remain = total - sum(base.values())
        order = sorted(raw.keys(), key=lambda k: raw[k] - base[k], reverse=True)
        for i in range(remain):
            base[order[i % len(order)]] += 1
        return base

    def _pick_prop_assets(self, candidate: SceneCandidate, available_props: List[str]) -> List[str]:
        """按照类别比例采样资产列表。"""
        if not available_props:
            return []
        seed = candidate.seed + self.cfg.replicator_seed_offset
        rng = random.Random(seed)
        n_props = rng.randint(self.cfg.replicator_min_props, self.cfg.replicator_max_props)

        class_pool: Dict[str, List[str]] = {"large": [], "medium": [], "small": []}
        for usd in available_props:
            class_pool[self._classify_prop_asset(usd)].append(usd)

        target = self._allocate_counts_by_ratio(n_props, self.cfg.replicator_prop_class_ratio)

        selected: List[str] = []
        for cls, cnt in target.items():
            pool = class_pool.get(cls, [])
            if not pool:
                continue
            # 样本不足时允许有放回采样，避免因池大小限制破坏比例。
            for _ in range(cnt):
                selected.append(rng.choice(pool))

        if len(selected) < n_props:
            for _ in range(n_props - len(selected)):
                selected.append(rng.choice(available_props))

        rng.shuffle(selected)
        return selected

    def _zone_weights_for_class(self, prop_class: str) -> Dict[str, float]:
        """返回指定类别的分区采样权重。"""
        if prop_class == "large":
            return self.cfg.replicator_zone_weights_large
        if prop_class == "medium":
            return self.cfg.replicator_zone_weights_medium
        return self.cfg.replicator_zone_weights_small

    def _sample_position_for_zone(self, rng: random.Random, zone: str) -> Tuple[float, float]:
        """在指定功能分区内采样一个 (x, y) 点。"""
        r = float(self.cfg.replicator_xy_range_m)
        half_aisle = r * float(self.cfg.replicator_aisle_half_width_ratio)
        wall_band = r * float(self.cfg.replicator_wall_band_ratio)
        corner_span = r * float(self.cfg.replicator_corner_zone_ratio)

        if zone == "aisle":
            x = rng.uniform(-half_aisle, half_aisle)
            y = rng.uniform(-r, r)
            return x, y

        if zone == "wall":
            side = rng.choice(["left", "right", "bottom", "top"])
            if side == "left":
                x = rng.uniform(-r, -r + wall_band)
                y = rng.uniform(-r, r)
            elif side == "right":
                x = rng.uniform(r - wall_band, r)
                y = rng.uniform(-r, r)
            elif side == "bottom":
                x = rng.uniform(-r, r)
                y = rng.uniform(-r, -r + wall_band)
            else:
                x = rng.uniform(-r, r)
                y = rng.uniform(r - wall_band, r)
            return x, y

        if zone == "corner":
            cx = rng.choice([-r + corner_span * 0.5, r - corner_span * 0.5])
            cy = rng.choice([-r + corner_span * 0.5, r - corner_span * 0.5])
            x = rng.uniform(cx - corner_span * 0.5, cx + corner_span * 0.5)
            y = rng.uniform(cy - corner_span * 0.5, cy + corner_span * 0.5)
            return x, y

        # open: 偏中部且避开最靠边区域，提升可通行性。
        x = rng.uniform(-r + wall_band, r - wall_band)
        y = rng.uniform(-r + wall_band, r - wall_band)
        return x, y

    def _sample_position(self, rng: random.Random, prop_class: str) -> Tuple[float, float, str]:
        """按策略采样位置并返回所属分区标签。"""
        if self.cfg.replicator_zone_sampling_strategy == "uniform":
            r = float(self.cfg.replicator_xy_range_m)
            return rng.uniform(-r, r), rng.uniform(-r, r), "open"

        weights = self._zone_weights_for_class(prop_class)
        zones = list(weights.keys())
        probs = list(weights.values())
        zone = rng.choices(zones, weights=probs, k=1)[0]
        x, y = self._sample_position_for_zone(rng, zone)
        return x, y, zone

    def _is_in_keepout(self, x: float, y: float) -> bool:
        """检查点位是否落入禁入区。"""
        for rect in self.cfg.replicator_keepout_rects:
            xmin, xmax, ymin, ymax = rect
            if float(xmin) <= x <= float(xmax) and float(ymin) <= y <= float(ymax):
                return True
        return False

    def _build_replicator_layout(self, candidate: SceneCandidate, selected_props: List[str]) -> List[Dict]:
        """生成可复现的程序化摆放布局。

        该流程同时实现：
        1. 资产类别比例控制。
        2. 分区采样（aisle/wall/corner/open）。
        3. 禁入区约束。
        4. 最小间距拒绝采样。
        5. 弱轴向对齐 + 抖动朝向。
        """
        seed = candidate.seed + self.cfg.replicator_seed_offset
        rng = random.Random(seed)
        layout: List[Dict] = []

        def distance_xy(p1: List[float], p2: List[float]) -> float:
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])
            return math.sqrt(dx * dx + dy * dy)

        def normalize_yaw(yaw_deg: float) -> float:
            # 统一到 [-180, 180) 便于后续分析和可视化审计。
            return ((yaw_deg + 180.0) % 360.0) - 180.0

        def sample_yaw() -> float:
            if rng.random() < self.cfg.replicator_axis_align_prob:
                base = float(rng.choice(self.cfg.replicator_axis_candidates_deg))
                jitter = rng.uniform(-self.cfg.replicator_axis_jitter_deg, self.cfg.replicator_axis_jitter_deg)
                yaw = base + jitter
            else:
                yaw = rng.uniform(self.cfg.replicator_yaw_min_deg, self.cfg.replicator_yaw_max_deg)
            return round(normalize_yaw(yaw), 4)

        for idx, usd_path in enumerate(selected_props):
            prop_class = self._classify_prop_asset(usd_path)
            # 先采样尺度，后续最小间距可根据尺度做线性放缩。
            scale = round(rng.uniform(self.cfg.replicator_scale_min, self.cfg.replicator_scale_max), 4)
            z = round(self.cfg.replicator_z_offset_m, 4)
            yaw = sample_yaw()

            chosen_position = [0.0, 0.0, z]
            chosen_zone = "open"
            used_retry = 0
            relaxed = False
            for retry in range(self.cfg.replicator_position_max_retries):
                used_retry = retry
                x_raw, y_raw, zone = self._sample_position(rng, prop_class)
                x = round(x_raw, 4)
                y = round(y_raw, 4)
                candidate_pos = [x, y, z]

                if self._is_in_keepout(x, y):
                    continue

                ok = True
                for placed in layout:
                    placed_scale = float(placed["scale_xyz"][0])
                    min_spacing = self.cfg.replicator_min_spacing_m * (scale + placed_scale) * 0.5
                    if distance_xy(candidate_pos, placed["position"]) < min_spacing:
                        ok = False
                        break

                if ok:
                    chosen_position = candidate_pos
                    chosen_zone = zone
                    break
            else:
                # 若多次采样仍冲突，保留最后一次候选，避免流程卡死。
                relaxed = True
                x_raw, y_raw, zone = self._sample_position(rng, prop_class)
                x = round(x_raw, 4)
                y = round(y_raw, 4)
                chosen_position = [x, y, z]
                chosen_zone = zone

            layout.append(
                {
                    "idx": idx,
                    "usd_path": usd_path,
                    "prop_class": prop_class,
                    "placement_zone": chosen_zone,
                    "position": chosen_position,
                    "rotation_euler_deg": [0.0, 0.0, yaw],
                    "scale_xyz": [scale, scale, scale],
                    "placement_retry": used_retry,
                    "placement_relaxed": relaxed,
                }
            )
        return layout

    def _replicator_config_snapshot(self) -> Dict:
        """提取本次组合关键配置，写入 stage_spec 便于审计。"""
        return {
            "scene_backend": self.cfg.scene_backend,
            "isaac_headless": self.cfg.isaac_headless,
            "isaac_gui_inspect_mode": self.cfg.isaac_gui_inspect_mode,
            "isaac_warmup_frames": self.cfg.isaac_warmup_frames,
            "isaac_renderer": self.cfg.isaac_renderer,
            "isaac_enable_replicator": self.cfg.isaac_enable_replicator,
            "replicator_seed_offset": self.cfg.replicator_seed_offset,
            "replicator_min_props": self.cfg.replicator_min_props,
            "replicator_max_props": self.cfg.replicator_max_props,
            "replicator_xy_range_m": self.cfg.replicator_xy_range_m,
            "replicator_scale_min": self.cfg.replicator_scale_min,
            "replicator_scale_max": self.cfg.replicator_scale_max,
            "replicator_min_spacing_m": self.cfg.replicator_min_spacing_m,
            "replicator_position_max_retries": self.cfg.replicator_position_max_retries,
            "replicator_axis_align_prob": self.cfg.replicator_axis_align_prob,
            "replicator_axis_jitter_deg": self.cfg.replicator_axis_jitter_deg,
            "replicator_axis_candidates_deg": self.cfg.replicator_axis_candidates_deg,
            "replicator_prop_class_ratio": self.cfg.replicator_prop_class_ratio,
            "replicator_zone_sampling_strategy": self.cfg.replicator_zone_sampling_strategy,
            "replicator_aisle_half_width_ratio": self.cfg.replicator_aisle_half_width_ratio,
            "replicator_wall_band_ratio": self.cfg.replicator_wall_band_ratio,
            "replicator_corner_zone_ratio": self.cfg.replicator_corner_zone_ratio,
            "replicator_zone_weights_large": self.cfg.replicator_zone_weights_large,
            "replicator_zone_weights_medium": self.cfg.replicator_zone_weights_medium,
            "replicator_zone_weights_small": self.cfg.replicator_zone_weights_small,
            "replicator_keepout_rects": self.cfg.replicator_keepout_rects,
        }

    def _base_replicator_spec(self, candidate: SceneCandidate, prop_layout: List[Dict]) -> Dict:
        """构建 replicator 规格基线结构。"""
        return {
            "backend": "isaac_replicator",
            "scene_id": candidate.scene_id,
            "scene_type": candidate.scene_type,
            "mode": candidate.mode,
            "seed": candidate.seed,
            "template_id": candidate.template.template_id,
            "template_usd": candidate.template.usd_path,
            "layout_variant": self._build_layout_variant(candidate),
            "config": self._replicator_config_snapshot(),
            "replicator": {
                "prop_count": len(prop_layout),
                "props": prop_layout,
            },
        }

    @staticmethod
    def _error_code_from_str(code: str) -> ErrorCode:
        try:
            return ErrorCode(code)
        except ValueError:
            return ErrorCode.RUNTIME_FAIL

    def _run_replicator_subprocess(
        self,
        candidate: SceneCandidate,
        scene_dir: str,
        prop_layout: List[Dict],
    ) -> Dict[str, str]:
        """在子进程中执行 Isaac 组合，避免主进程被 Kit 崩溃带走。"""
        request_path = os.path.join(scene_dir, "compose_request.json")
        response_path = os.path.join(scene_dir, "compose_response.json")
        stage_path = os.path.join(scene_dir, self.cfg.generated_stage_filename)

        request_payload = {
            "scene_id": candidate.scene_id,
            "scene_seed": candidate.seed,
            "scene_dir": scene_dir,
            "template_usd": candidate.template.usd_path,
            "stage_path": stage_path,
            "headless": bool(self.cfg.isaac_headless),
            "renderer": str(self.cfg.isaac_renderer),
            "warmup_frames": int(self.cfg.isaac_warmup_frames),
            "enable_replicator": bool(self.cfg.isaac_enable_replicator),
            "replicator_seed": int(candidate.seed + self.cfg.replicator_seed_offset),
            "props": prop_layout,
        }
        self._write_spec(request_path, request_payload)

        cmd = [
            sys.executable,
            "-m",
            "dataengine.src.scene_layer.replicator_worker",
            "--request",
            request_path,
            "--response",
            response_path,
        ]

        try:
            completed = subprocess.run(  # noqa: S603
                cmd,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=max(1, int(self.cfg.compose_timeout_s)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SceneCompileError(
                code=ErrorCode.GPU_TIMEOUT,
                message=(
                    f"Replicator 子进程超时({self.cfg.compose_timeout_s}s): {exc}. "
                    "可尝试减小 target_scene_count 或提升 compose_timeout_s"
                ),
                scene_id=candidate.scene_id,
            ) from exc

        if os.path.isfile(response_path):
            response = self._read_json_file(response_path)
            if response.get("ok"):
                out_stage = str(response.get("stage_usd", stage_path))
                layout_hash = str(response.get("layout_hash", ""))
                return {
                    "scene_dir": scene_dir,
                    "stage_usd": out_stage,
                    "stage_spec_path": os.path.join(scene_dir, "stage_spec.json"),
                    "layout_hash": layout_hash,
                }

            err_code = self._error_code_from_str(str(response.get("error_code", ErrorCode.RUNTIME_FAIL.value)))
            message = str(response.get("message", "replicator worker failed"))
            raise SceneCompileError(code=err_code, message=message, scene_id=candidate.scene_id)

        stderr_tail = (completed.stderr or "").strip().splitlines()[-3:]
        stderr_summary = " | ".join(stderr_tail) if stderr_tail else "no stderr"
        raise SceneCompileError(
            code=ErrorCode.RUNTIME_FAIL,
            message=(
                "Replicator 子进程未返回 response 文件; "
                f"returncode={completed.returncode}; stderr_tail={stderr_summary}"
            ),
            scene_id=candidate.scene_id,
        )

    def _write_final_replicator_spec(
        self,
        spec_path: str,
        base_spec: Dict,
        status: str,
        error: str = "",
    ) -> None:
        """回写最终状态（COMPOSING/DONE/FAILED）到 stage_spec。"""
        final_spec = dict(base_spec)
        final_spec["status"] = status
        if error:
            final_spec["error"] = error
        self._write_spec(spec_path, final_spec)

    def _compose_with_isaac_replicator(self, candidate: SceneCandidate, scene_dir: str) -> Dict[str, str]:
        """Replicator 后端主入口：采样布局 + 子进程组合 + 状态落盘。"""
        prop_assets = self._resolve_replicator_props()
        selected_props = self._pick_prop_assets(candidate, prop_assets)
        prop_layout = self._build_replicator_layout(candidate, selected_props)

        spec_path = os.path.join(scene_dir, "stage_spec.json")
        base_spec = self._base_replicator_spec(candidate, prop_layout)
        self._write_final_replicator_spec(spec_path=spec_path, base_spec=base_spec, status="COMPOSING")

        try:
            result = self._run_replicator_subprocess(candidate, scene_dir, prop_layout)
            self._write_final_replicator_spec(spec_path=spec_path, base_spec=base_spec, status="DONE")
            if not result.get("layout_hash"):
                result["layout_hash"] = stable_hash(
                    json.dumps(self._read_json_file(spec_path), ensure_ascii=False, sort_keys=True)
                )
            return result
        except SceneCompileError as exc:
            self._write_final_replicator_spec(
                spec_path=spec_path,
                base_spec=base_spec,
                status="FAILED",
                error=str(exc),
            )
            raise

    def compose(self, candidate: SceneCandidate) -> Dict[str, str]:
        """根据 scene_backend 分派组合流程。"""
        self._ensure_template_exists(candidate)
        scene_dir = self._scene_dir(candidate)

        if self.cfg.scene_backend == "stub":
            return self._compose_stub(candidate, scene_dir)
        if self.cfg.scene_backend == "isaac_replicator":
            return self._compose_with_isaac_replicator(candidate, scene_dir)

        raise SceneCompileError(
            code=ErrorCode.USD_COMPOSE_FAIL,
            message=f"不支持的 scene_backend: {self.cfg.scene_backend}",
            scene_id=candidate.scene_id,
        )
