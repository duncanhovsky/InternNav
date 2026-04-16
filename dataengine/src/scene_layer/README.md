# Scene Layer

## 1. 目标

该模块用于数据引擎的场景层：

1. 以 complete/modular 混合策略自动采样场景模板。
2. 执行场景编译（stage 规格落盘）。
3. 执行 NavMesh 与可达性指标校验（当前为可替换的轻量实现）。
4. 导出 scene manifest 与每场景轨迹任务清单。
5. 支持断点恢复：跳过已 DONE 场景。

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

## 5. 与 Isaac Sim 真正对接点

当前实现为“流程可跑 + 规则完整”的离线版本。

后续需要替换的关键位置：

1. `composer.py`: 将 stage_spec 驱动为真实 USD 组合/写盘。
2. `navmesh.py`: 接入 Isaac Sim 5.1 NavMesh 烘焙与路径查询 API。
3. `asset_registry.py`: 根据本地资产库扫描结果自动扩展模板池。
