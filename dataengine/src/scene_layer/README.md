# Scene Layer

## 1. 目标

该模块用于数据引擎的场景层：

1. 以 complete/modular 混合策略自动采样场景模板。
2. 执行场景编译（stage 规格落盘）。
3. 执行 NavMesh 与可达性指标校验（当前为可替换的轻量实现）。
4. 导出 scene manifest 与每场景轨迹任务清单。
5. 支持断点恢复：跳过已 DONE 场景。

## 1.1 组合后端

`scene_backend` 支持两种模式：

1. `stub`
	- 不依赖 Isaac Runtime。
	- 仅生成 `stage_spec.json`，用于快速联调。
2. `isaac_replicator`
	- 依赖 Isaac Sim Python 环境。
	- 使用 Replicator 程序化生成场景并导出真实 stage USD。
	- 在相同 `scene_seed + replicator_* 配置` 下可复现相同布局。

## 2. 主要文件

1. `config.py`: 参数 schema 与校验。
2. `asset_registry.py`: 可用模板注册。
3. `selector.py`: 场景类型/模式加权采样。
4. `composer.py`: 场景组合与 stage_spec 导出。
5. `navmesh.py`: NavMesh 构建接口（当前为 deterministic stub）。
6. `metrics.py`: 指标阈值校验与复杂度分桶。
7. `compiler.py`: 单场景编译 + 重试策略。
8. `pipeline.py`: 批量场景生成与任务清单导出。
9. `cli.py`: 命令行入口。

## 3. 输出文件

默认输出：

1. `dataengine/out/manifests/scene_manifest.jsonl`
2. `dataengine/out/logs/scene_compile_log.jsonl`
3. `dataengine/out/manifests/trajectory_tasks.jsonl`
4. `dataengine/out/scenes/usd/<scene_id>/stage_spec.json`
5. `dataengine/out/scenes/usd/<scene_id>/navmesh.bin`

## 4. 运行方式

使用默认配置：

```bash
python -m dataengine.src.scene_layer
```

使用指定配置：

```bash
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json
```

覆盖目标场景数/每场景轨迹数：

```bash
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json --target-scene-count 50 --trajectories-per-scene 100
```

### Isaac Replicator 模式建议

1. 在 Isaac Sim 对应 Python 环境中运行（如 `isaaclab51`）。
2. 将配置中的 `scene_backend` 设为 `isaac_replicator`。
3. 首次调试建议 `target_scene_count=1`，确认 stage 导出后再放量。

### 有头/无头切换

1. 使用配置文件（推荐单次 smoke）：

```bash
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/gui_smoke_config.json
```

2. 命令行覆盖：

```bash
# 强制有头（显示窗口）
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json --isaac-gui

# 强制无头
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json --isaac-headless

# 有头调试并保持窗口（手动关闭）
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json --isaac-gui --inspect-gui
```

说明：
1. `--inspect-gui` 现在只作为调试标记，不再阻塞生成流程，避免关闭窗口时影响结果落盘。
2. Outdoor/Rivermark 场景可能出现 `PopulatePointInstancerBucket ... numPrototypes=0` 警告，这通常是资产内部实例化警告，不一定导致场景加载失败。
3. 默认 `isaac_close_on_finish=false`，避免部分 Isaac 版本在 `SimulationApp.close()` 时提前退出进程，导致调试信息/落盘不完整。
4. `isaac_replicator` 组合在子进程执行，主进程负责兜底写 `stage_spec.status` 与 compile log；即使 Isaac 子进程崩溃，也应得到 `FAILED` 记录而不是静默中断。

### stage_spec 状态语义

1. `COMPOSING`：已完成布局采样与请求落盘，正在执行 Isaac 组合。
2. `DONE`：`stage_composed.usda` 已导出成功，可用于后续 navmesh 与任务生成。
3. `FAILED`：组合失败，`stage_spec.error` 给出错误摘要；同时 compile log 中应存在对应失败行。

稳定建议：采用“先生成，后查看”两步法。

1. 先生成：

```bash
python -m dataengine.src.scene_layer --config-json dataengine/src/scene_layer/default_config.json --target-scene-count 1 --trajectories-per-scene 1
```

2. 再打开生成场景查看：

```bash
python -m dataengine.src.scene_layer.open_stage --stage dataengine/out/scenes/usd/<scene_id>/stage_composed.usda
```

### 摆放质量参数（新增）

用于降低“物体摆放杂乱”问题的关键参数：

1. 最小间距拒绝采样（第 2 点）
`replicator_min_spacing_m`: 物体中心最小间距基准值（会按两物体 scale 做线性缩放）。
`replicator_position_max_retries`: 单物体位置采样最大重试次数。

2. 弱轴向对齐 + 抖动（第 4 点）
`replicator_axis_align_prob`: 以多大概率吸附到主轴方向。
`replicator_axis_candidates_deg`: 主轴候选（默认 0/90/180/270）。
`replicator_axis_jitter_deg`: 吸附后叠加的随机抖动角度。

建议起步值：
1. `replicator_min_spacing_m=1.2`
2. `replicator_position_max_retries=24`
3. `replicator_axis_align_prob=0.75`
4. `replicator_axis_jitter_deg=8.0`

### 其余摆放优化点（本轮已实现）

1. 资产类别比例控制
`replicator_prop_class_ratio` 用于控制 `large/medium/small` 三类资产占比。

2. 分区采样策略
`replicator_zone_sampling_strategy=zoned` 时按 `aisle/wall/corner/open` 区域采样。

3. 类别到分区偏好
`replicator_zone_weights_large|medium|small` 控制不同类别资产更倾向出现在哪些区域。

4. 禁入区
`replicator_keepout_rects` 提供矩形禁入区（格式 `[xmin, xmax, ymin, ymax]`）。

5. 审计字段
`stage_spec.replicator.props` 现在包含 `prop_class`、`placement_zone`、`placement_retry`、`placement_relaxed`，可用于排查布局质量问题。

### 上下游链路文档

完整的数据引擎上下游说明见：

`dataengine/ENGINE_UPSTREAM_DOWNSTREAM_CHAIN.md`

## 5. 与 Isaac Sim 真正对接点

当前实现支持 `stub` 与 `isaac_replicator` 双后端。

后续需要替换的关键位置：

1. `composer.py`: 持续增强 Replicator 随机化策略（语义约束、禁入区、资产类型比例等）。
2. `navmesh.py`: 接入 Isaac Sim 5.1 NavMesh 烘焙与路径查询 API。
3. `asset_registry.py`: 根据本地资产库扫描结果自动扩展模板池。
