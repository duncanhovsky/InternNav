## Plan: DataEngine 分阶段落地路线

目标是在不推翻现有 `scene_layer` 调度骨架的前提下，优先复用 Isaac Sim 5.1 官方插件（Replicator、NavMesh、Action/Event、Anim.People、RTX Placement），将当前启发式原型演进为可审计、可扩展、可训练闭环的数据引擎。

**Steps**
1. 阶段 0（对齐与基线固化）: 冻结当前行为并补齐可观测性。
2. 阶段 0（对齐与基线固化）: 定义统一数据契约版本（`v1alpha`），明确 `scene_manifest/task_manifest` 与后续 `episodes/dynamic_tracks/critic/voxel` 的字段映射；把“必填字段/可选字段/废弃字段”写入 schema 文档。*parallel with step 3*
3. 阶段 0（对齐与基线固化）: 为 `scene_layer` 增加契约测试与回归快照测试（固定 seed 对比布局哈希、NavMesh 指标、输出文件完整性），作为后续重构守门。*parallel with step 2*
4. 阶段 1（静态场景能力替换，P0）: 保留 `SceneLayerPipeline` 与 `SceneCompiler`，替换 `composer/replicator_worker` 内手写采样逻辑为 `omni.replicator.core` 随机化节点（语义采样、散布、姿态/尺度随机化、writer/annotator 管线）。*depends on 1*
5. 阶段 1（静态场景能力替换，P0）: 资产元数据体系化：将当前文件名启发式分类替换为语义标签与资产索引（类别、物理属性、可摆放区域、禁入规则），并在组合时强校验资产可用性。*depends on 4*
6. 阶段 1（静态场景能力替换，P0）: `navmesh_worker` 从“可查询 + 占位文件”升级为“可复核烘焙产物 + 多 agent profile 指标”，统一输出 `navmesh_debug` 与路径采样报告；保留失败可审计日志。*depends on 4*
7. 阶段 1（静态场景能力替换，P0）: 将 `metrics.py` 启发式复杂度拆分为“官方可查询指标 + 业务阈值策略”两层，避免指标与实现耦合。*depends on 6*
8. 阶段 2（动态层接入，P1）: 新增 `dynamics_layer`，先接入 `omni.anim.people` 与 `isaacsim.replicator.agent/object` 形成最小闭环（人/车生成、行为模式、时间同步输出）。*depends on 6*
9. 阶段 2（动态层接入，P1）: 输出 `dynamic_tracks.parquet`（object_id、pose、velocity、bbox、timestamp_ns）并与 agent 时钟对齐；把动态信息写回场景级 manifest。*depends on 8*
10. 阶段 2（动态层接入，P1）: 接入 `isaacsim.sensors.rtx.placement` 做自动传感器位姿与覆盖率检查，替代手工相机布设；建立覆盖率阈值与失败重试。*parallel with step 9*
11. 阶段 3（训练数据闭环，P2）: 新增导出器把 scene/task 输出扩展到 InternNav 动态契约目录（episodes parquet + dynamic_tracks + critic cache + dyn_voxel cache），并保留 JSONL 向后兼容一段过渡期。*depends on 9*
12. 阶段 3（训练数据闭环，P2）: 实现 Oracle/Reactive 双专家最小版本与 critic 真值离线计算，先支持小规模回归（1 scene x 10 episodes），再扩大批量。*depends on 11*
13. 阶段 4（性能与规模化，P3）: 分离“仿真生成”和“离线标注”队列，基于 4090 显存做并发上限与分片策略（按 shard 100-300 episodes）；加入失败重放队列。*depends on 12*
14. 阶段 4（性能与规模化，P3）: 做稳定性治理（超时、内存峰值、子进程崩溃自动回收、重试分级）并形成运行手册。*depends on 13*

**官方插件优先级与调用顺序**
1. P0 必选: `omni.replicator.core`（场景随机化/标注/写出）、`omni.anim.navigation.core`（NavMesh 烘焙与查询）。
2. P1 必选: `isaacsim.replicator.agent`、`isaacsim.replicator.object`、`omni.anim.people`（动态人车与事件生成）。
3. P1 推荐: `isaacsim.sensors.rtx.placement`（自动传感器布局与覆盖率校验）。
4. P2 推荐: Materials 相关扩展（外观域随机化），在拓扑与行为真实性稳定后启用。
5. P2 选配: Digital Twin/Design 相关扩展用于资产治理与工作流编排，不作为首批阻塞项。

**Relevant files**
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/pipeline.py` — 保留主调度骨架，扩展阶段状态与导出分支。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/compiler.py` — 定义“组合/烘焙/指标/导出”的阶段化执行与重试策略。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/composer.py` — 从手写启发式采样迁移到 Replicator 驱动。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/replicator_worker.py` — 建立官方 randomizer/annotator/writer 工作流。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/navmesh_worker.py` — 升级 NavMesh 烘焙产物、事件监听、指标采样与调试导出。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/navmesh.py` — 支持多 agent profile 的烘焙参数与结果聚合。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/metrics.py` — 重构为“原始指标计算层 + 策略阈值层”。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/asset_registry.py` — 资产语义、物理属性和可放置约束索引。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/src/scene_layer/manifest_store.py` — 增加 Parquet 导出与 schema version 写入。
- `/home/monika/dyishere/project/MyResearch/InternNav/internnav/dataset/flownav_dyn_lerobot_dataset.py` — 作为契约对齐基准，验证最小可读闭环。
- `/home/monika/dyishere/project/MyResearch/InternNav/dataengine/Plan.md` — 更新总体里程碑与阶段验收口径。
- `/home/monika/dyishere/project/MyResearch/InternNav/tests/unit_test` — 新增契约与回归测试。
- `/home/monika/dyishere/project/MyResearch/InternNav/tests/function_test` — 新增端到端 smoke 与小规模集成验证。

**Verification**
1. 静态回归: 固定 seed 运行 10 个 scene，对比 `layout_hash`、`status`、关键指标分布，确保迁移前后可控差异。
2. NavMesh 真实性: 每个 scene 至少输出一份可复核调试文件（volume 配置、事件流、路径采样统计），并验证多 agent profile 可达性。
3. 动态契约: 生成最小数据集（1 scene x 10 episodes），确保 `flownav_dyn_lerobot_dataset.py` 能完整加载 `episodes` 与 `dynamic_tracks`。
4. 训练前检查: 启用 cache 开关验证 `critic` 与 `dyn_voxel` 文件存在性、schema、时间戳单调性。
5. 稳定性: 连续运行 200 episodes，统计失败率、重试成功率、平均吞吐、显存峰值，达到预设阈值后再扩大规模。

**Decisions**
- Included: 先做场景层/动态层/导出契约三条主线的最小闭环，再做规模化优化。
- Excluded: 首阶段不做 UI 工作流定制，不依赖手工可视化操作作为必需步骤。
- Included: 保留现有 JSONL 输出一段过渡期，避免下游一次性切换风险。
- Assumption: 当前 Isaac Sim 5.1 环境可稳定启用 `omni.replicator` 与 `omni.anim.navigation.core`。

**Further Considerations**
1. 动态体优先级建议: Option A 先做人群（People）/ Option B 先做车辆（Object）/ Option C 并行；推荐 A（更快打通导航避障训练价值）。
2. 导出格式切换建议: Option A 双写 JSONL+Parquet 两周 / Option B 一次切换；推荐 A（降低回归风险）。
3. 规模化节奏建议: Option A 先 1->10->50 scenes 阶梯放量 / Option B 直接 100 scenes；推荐 A（便于定位瓶颈）。

## 本轮实现记录（2026-04-19）

### 已落地（对应 Phase 0 / Phase 1 子项）

1. 契约版本落地到 `v1alpha`：
	- `scene_manifest` 行级新增 `schema_version` 与 `contract=scene_manifest.v1alpha`。
	- `task_manifest` 行级新增 `schema_version` 与 `contract=task_manifest.v1alpha`。
	- `task_manifest` 追加 `scene_layout_hash`，用于场景到 episode 的可追溯映射。

2. NavMesh 多 agent profile 可复核输出：
	- `NavMeshBuilder` 按 `navmesh_agent_profiles` 逐个调用 worker 烘焙/评估。
	- 输出 `metrics_by_profile`、`navmesh_files_by_profile`、`debug_files_by_profile`。
	- 聚合层采用保守策略（取最差可达性）回填上游阈值校验。

3. 阶段0守门测试（contract + regression）新增：
	- 固定 seed 下 `composer(stub)` 布局哈希稳定性测试。
	- `scene_manifest/task_manifest` v1alpha 字段完整性测试。
	- 多 profile 指标聚合策略测试。

### 本轮未覆盖

1. `dynamics_layer` 与 `dynamic_tracks.parquet`（Phase 2）。
2. `critic/dyn_voxel` 训练闭环导出（Phase 3）。
3. 规模化队列/失败重放治理（Phase 4）。

## 第二轮实现记录（2026-04-20）

### 已落地（Phase 1 Step 7 + Phase 0 Step 3 扩展）

1. `metrics.py` 分层重构：
	- 原始指标层：`RawSceneMetrics` + `RawMetricsAdapter`（只做结构化与归一化）。
	- 策略阈值层：`SceneMetricPolicy`（负责阈值判定、复杂度计算、分桶）。
	- 兼容入口：`MetricsEvaluator` 保持原接口，内部组合两层，避免影响 `SceneCompiler`。

2. 测试补强：
	- `tests/unit_test/test_scene_layer_metrics_policy.py`：覆盖原始层归一化、策略层通过/拒绝、复杂度分桶。
	- `tests/function_test/test_scene_layer_smoke_stub.py`：新增场景层最小 smoke function test，校验 `scene/task manifest` 契约字段。

## 第三轮实现记录（2026-04-20）

### 已落地（Phase 0 守门与可观测性扩展）

1. 产物一致性校验器落地：
	- 新增 `dataengine/src/scene_layer/validate_artifacts.py`。
	- 支持 `--scene-dir` 单场景校验与 `--scene-manifest` 批量校验。
	- 支持 `--only-done`、`--strict-warn`、`--json` 输出。

2. 守门测试补齐：
	- 新增 `tests/unit_test/test_scene_layer_validate_artifacts.py`。
	- 覆盖：正常通过、缺少 debug 告警、strict-warn 失败、缺少 navmesh 错误。

## 第四轮实现记录（2026-04-21）

### 已落地（Phase 2 最小动态闭环 + 守门扩展）

1. 动态层最小闭环：
	- 新增 `dataengine/src/scene_layer/dynamics_layer.py` 与 `dynamics_worker.py`。
	- 编译流程升级为 `compose -> navmesh -> dynamics -> metrics`。
	- 每场景输出 `dynamic_tracks.jsonl`（`dynamic_tracks.v1alpha`）。

2. 契约与产物回写：
	- `scene_manifest` 新增 `dynamic` 摘要：`enabled/backend/track_file/object_count/sample_count`。
	- `schema_v1alpha.md` 补充 `dynamic_tracks.v1alpha` 字段定义。

3. 校验器升级：
	- `validate_artifacts.py` 支持按 manifest 中 `dynamic.enabled` 自动校验动态轨迹。
	- 新增动态轨迹 schema/scene_id/时间戳单调性检查。

4. 测试新增：
	- `tests/unit_test/test_scene_layer_dynamics.py`：动态层生成与关闭路径。
	- `tests/unit_test/test_scene_layer_validate_artifacts.py` 增加动态轨迹校验用例。
	- `tests/function_test/test_scene_layer_smoke_stub.py` 增加动态字段断言。

## 第五轮实现记录（2026-04-21）

### 已落地（asset-driven 预览闭环）

1. 动态 overlay 产物：
	- `dynamics_worker.py` 在 `asset_driven` 模式新增 `dynamic_overlay.usda` 生成。
	- overlay subLayer 基础 `stage_usd`，并在 `/World/Dynamics/*` 引用人物/叉车资产。
	- 每个动态 prim 写入 `dynamic:*` 元数据（`motion_mode/asset_relpath/animation_relpath`）。

2. 清单与流程贯通：
	- `SceneCompileResult` 与 `scene_manifest.dynamic` 新增 `overlay_usd` 字段。
	- `compiler.py` 将 compose 结果 `stage_usd` 传给 dynamics，用于构建 overlay。

3. 校验与预览增强：
	- `validate_artifacts.py` 新增 `DYNAMIC_OVERLAY_MISSING` 校验。
	- `open_stage.py` 增加 `--overlay` 参数，支持直接打开动态叠加场景。

## 第六轮实现记录（2026-04-23）

### 已落地（IRA 动态层完整实现 — Phase 2 核心）

1. 新增 `dataengine/src/dynamic_layer/` 独立模块：
	- `config.py`：动态层配置契约 `DynamicLayerConfig`，覆盖 IRA 全部参数。
	- `models.py`：数据模型 `IRAConfig`、`CommandEntry`、`DynamicLayerResult` 等，
	  `IRAConfig.to_dict()` 一一映射 IRA YAML schema。
	- `ira_config_generator.py`：从场景配置 + 运行参数生成 IRA `default_config.yaml`。
	- `command_generator.py`：按行为策略（idle/walk/mixed）为每个 NPC 生成
	  `command.txt`（Idle/GoTo/LookAround），支持确定性种子。
	- `ira_driver.py`：驱动 Isaac Sim + IRA 执行仿真，
	  支持子进程模式/进程内模式/自动 Python 路径探测。
	- `post_processor.py`：解析 IRA 输出（bbox3d npy/json），
	  差分计算速度，导出 `dynamic_tracks.v1alpha` JSONL。
	- `dynamic_layer.py`：主编排器 `IRADynamicLayer`，
	  串联配置 → 命令 → 驱动 → 后处理，
	  暴露 `generate()` + `generate_legacy_dict()` 兼容旧接口。
	- `__main__.py`：CLI 入口，支持 `--dry-run`。
	- `README.md`：完整模块文档。

2. `SceneCompiler` 集成（`compiler.py`）：
	- 新增 `_build_ira_dynamic_layer()` 工厂函数。
	- `SceneCompiler.__post_init__` 根据 `dynamics_backend == "ira"` 自动切换实现。
	- 新增 `_run_dynamics()` 方法，优先 IRA 后端，fallback 到旧合成后端。
	- 旧 `DynamicsLayer` 完全保留作为 fallback。

3. `SceneLayerConfig` 扩展（`config.py`）：
	- `SUPPORTED_DYNAMICS_BACKENDS` 新增 `"ira"`。
	- 新增 `ira_*` 前缀配置字段（`ira_simulation_length`、`ira_character_asset_path`、
	  `ira_character_filters`、`ira_camera_num`、`ira_isaac_sim_python` 等）。

4. 预定义场景行为方案（`config/dynamic_layer/scenarios/`）：
	- `warehouse_busy.yaml`：繁忙仓库，高密度 NPC + 频繁移动。
	- `warehouse_sparse.yaml`：稀疏仓库，少量 NPC + 以站立为主。
	- `hospital_corridor.yaml`：医院走廊，中等密度 + 医务人员。
	- `default.yaml`：通用默认配置。

5. 测试新增：
	- `tests/unit_test/test_dynamic_layer.py`：33 个用例，覆盖配置校验、
	  命令模型序列化、命令生成器确定性、IRA 配置结构、结果模型兼容。
	- `tests/function_test/test_dynamic_layer_integration.py`：10 个用例，
	  覆盖 dry-run 端到端、禁用路径、fallback 路径、旧接口兼容。
	- 全部 43 个测试通过。

### 设计决策

- **配置驱动**：动态层核心是"写好 YAML + command.txt → 交给 IRA 执行"，
  代码量可控但配置灵活度高。
- **Fallback 机制**：当 Isaac Sim 不可用时（如 CI 环境），
  自动退化到旧的 synthetic 后端，IRA 配置文件仍会生成供后续手动使用。
- **Robot 预留**：`robot_command.txt` 保持空，
  架构上预留完整的 robot 控制链路，后续接入主视角机器人只需添加 robot 命令生成逻辑。
- **旧接口兼容**：`generate_legacy_dict()` 返回与旧 `DynamicsLayer.generate()`
  相同格式的字典，`SceneCompiler` 无感切换。

### 本轮未覆盖

1. Robot 主视角控制（`robot_command.txt` 生成）。
2. Recording Layer 与 IRA `IRABasicWriter` 的训练数据导出对接。
3. NavMesh Walkable 区域联动采样 GoTo 目标点。
4. `dynamic_tracks.parquet` 格式导出（当前为 JSONL）。