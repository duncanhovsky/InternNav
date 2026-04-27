from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Dict, List

from .asset_registry import AssetRegistry
from .compiler import SceneCompiler
from .config import SceneLayerConfig
from .manifest_store import ManifestStore
from .models import SceneCandidate, SceneCompileResult, TrajectoryTask
from .selector import WeightedSceneSelector
from .utils import atomic_write_text, load_jsonl


@dataclass
class SceneLayerPipeline:
    """场景层主流程。

    功能：
    1. 混合模式采样（complete + modular）
    2. 场景编译与指标校验
    3. manifest 导出
    4. 每场景轨迹任务生成（默认 100）
    5. 断点恢复（跳过已 DONE 场景）
    """

    cfg: SceneLayerConfig

    def __post_init__(self) -> None:
        """初始化核心组件并执行配置校验。"""
        self.cfg.validate()
        self.registry = AssetRegistry(self.cfg.asset_root)
        self.compiler = SceneCompiler(self.cfg)
        self.store = ManifestStore(
            scene_manifest_path=self.cfg.scene_manifest_path,
            compile_log_path=self.cfg.compile_log_path,
            task_manifest_path=self.cfg.task_manifest_path,
        )

    def _build_run_snapshot(self) -> None:
        """写入本次运行配置快照，便于复现实验。"""
        snapshot_path = self.cfg.scene_manifest_path.replace(".jsonl", "_run_config.json")
        payload = {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "config": self.cfg.to_dict(),
        }
        atomic_write_text(snapshot_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _load_frozen_rows(self) -> List[Dict]:
        """加载 eval 冻结清单。"""
        return load_jsonl(self.cfg.frozen_manifest_path)

    def _make_candidate(self, idx: int, mode: str, template, seed: int) -> SceneCandidate:
        """将采样结果封装为 SceneCandidate。"""
        scene_id = f"scene_{idx:06d}_{template.scene_type}_{mode}"
        return SceneCandidate(
            scene_id=scene_id,
            scene_type=template.scene_type,
            mode=mode,
            seed=seed,
            template=template,
            params={},
        )

    def _result_to_row(self, result: SceneCompileResult) -> Dict:
        """将编译结果转换为 manifest 可落盘结构。"""
        m = result.metrics
        return {
            "scene_id": result.scene_id,
            "scene_type": result.scene_type,
            "mode": result.mode,
            "seed": result.seed,
            "status": result.status,
            "stage_usd": result.stage_usd,
            "navmesh_file": result.navmesh_file,
            "complexity_bucket": result.complexity_bucket,
            "reason": result.reason,
            "layout_hash": result.layout_hash,
            "template_id": result.template_id,
            "metrics": {
                "path_count": m.path_count,
                "shortest_path_m": m.shortest_path_m,
                "second_shortest_path_m": m.second_shortest_path_m,
                "detour_margin_m": m.detour_margin_m,
                "free_space_ratio": m.free_space_ratio,
                "static_density": m.static_density,
                "static_complexity_score": m.static_complexity_score,
            },
        }

    def _build_tasks_for_scene(self, row: Dict) -> List[TrajectoryTask]:
        """根据单场景结果生成轨迹任务列表。"""
        tasks: List[TrajectoryTask] = []
        agents = self.cfg.per_scene_agents
        for ep_idx in range(self.cfg.trajectories_per_scene):
            agent_type = agents[ep_idx % len(agents)]
            task_id = f"{row['scene_id']}_{agent_type}_ep{ep_idx:04d}"
            tasks.append(
                TrajectoryTask(
                    task_id=task_id,
                    scene_id=row["scene_id"],
                    agent_type=agent_type,
                    episode_idx=ep_idx,
                    global_seed=self.cfg.global_seed,
                    scene_seed=int(row["seed"]),
                    complexity_bucket=row["complexity_bucket"],
                    scene_type=row["scene_type"],
                    mode=row["mode"],
                )
            )
        return tasks

    def run(self) -> Dict[str, int]:
        """执行场景层主流程并返回统计摘要。

        返回值字段：
            done: 编译成功场景数。
            failed: 编译失败场景数。
            tasks: 生成的轨迹任务数。
        """
        self._build_run_snapshot()

        done_scene_ids = self.store.load_done_scene_ids()

        if self.cfg.mode == "eval" and self.cfg.freeze_eval_manifest:
            frozen_rows = self._load_frozen_rows()
            selected_rows = [x for x in frozen_rows if x.get("status") == "DONE"]
            for row in selected_rows:
                if row["scene_id"] in done_scene_ids:
                    continue
                self.store.append_scene_rows([row])
                done_scene_ids.add(row["scene_id"])

            task_rows: List[Dict] = []
            for row in selected_rows:
                for task in self._build_tasks_for_scene(row):
                    task_rows.append(task.__dict__)
            self.store.append_task_rows(task_rows)
            return {"done": len(selected_rows), "failed": 0, "tasks": len(task_rows)}

        templates = self.registry.build_registry()
        selector = WeightedSceneSelector(
            scene_type_weights={k: self.cfg.scene_type_weights[k] for k in self.cfg.enabled_scene_types},
            scene_mode_weights=self.cfg.scene_mode_weights,
            seed=self.cfg.global_seed + self.cfg.scene_seed_offset,
        )

        results: List[SceneCompileResult] = []

        for i in range(self.cfg.target_scene_count):
            mode, template = selector.sample(templates)
            scene_seed = self.cfg.global_seed + self.cfg.scene_seed_offset + i * 1009
            candidate = self._make_candidate(i, mode, template, scene_seed)

            if candidate.scene_id in done_scene_ids:
                continue

            result = self.compiler.compile_with_retry(candidate)
            results.append(result)
            scene_row = self._result_to_row(result)
            self.store.append_scene_rows([scene_row])
            self.store.append_compile_logs(
                [
                    {
                        "scene_id": result.scene_id,
                        "status": result.status,
                        "reason": result.reason,
                        "scene_type": result.scene_type,
                        "mode": result.mode,
                        "seed": result.seed,
                        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                    }
                ]
            )

        scene_rows = [self._result_to_row(r) for r in results]

        task_rows: List[Dict] = []
        for row in scene_rows:
            if row["status"] != "DONE":
                continue
            for task in self._build_tasks_for_scene(row):
                task_rows.append(task.__dict__)
        self.store.append_task_rows(task_rows)

        done_count = sum(1 for r in results if r.status == "DONE")
        failed_count = sum(1 for r in results if r.status == "FAILED")

        return {"done": done_count, "failed": failed_count, "tasks": len(task_rows)}
