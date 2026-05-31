# DataEngine Plan (v1.1)

## 1. 背景与目标

本计划面向 `InternNav/dataengine` 下的数据引擎实现，优先完成“数据引擎独立闭环”，再与 FlowNav 训练/评估对齐。

当前目标：

1. 基于 Isaac Sim 5.1 资产自动化生成大规模训练序列与评估场景。
2. 支持生成任务断点记录与恢复（长任务可中断续跑）。
3. 输出完全对齐 InternNav 动态数据结构。
4. 支持动态场景下专家轨迹（含“先知能力”）与 critic 真值生成。

## 2. 已确认约束

1. 资产根目录：
   - `/home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/`
2. 机器人：仅支持两类智能体
   - Unitree Go2
   - Unitree G1
3. 传感器：
   - 深度相机 D435i（目标 30Hz，时间对齐精度至少 0.01s）
   - 激光雷达：Hesai XT16 或 Livox Mid360（安装角不同）
4. 动态障碍：
   - 人 + 车都要支持
   - 既要自由移动，也要规则移动（交通规则模式）
   - 障碍尺寸必须与环境尺度匹配
5. 样本策略：
   - 保留失败样本作为负样本
   - 必须提供对应 critic 真值
6. 训练接口：
   - 输出完全对齐 InternNav 动态数据读取契约
7. 计算资源：
   - 单卡 4090 24GB

## 3. 目录职责

1. `dataengine/include/`
   - 第三方仓库与外部依赖镜像（只做引用/适配，不直接当核心骨架）。
2. `dataengine/src/`
   - 数据引擎核心实现（调度、生成、标注、导出、质检）。
3. `dataengine/Plan.md`
   - 当前实施方案与讨论纪要。

## 4. 总体架构（五层）

1. 场景层（Scene Layer）
   - 参数化静态场景生成、NavMesh、冗余探索空间约束。
2. 动态层（Dynamics Layer）
   - 人车动态体生成与行为调度（自由/规则）。
3. 规划层（Planning Layer）
   - 双专家轨迹：Oracle（先知） + Reactive（现实）。
4. 记录层（Recording Layer）
   - 观测、状态、轨迹、标签、缓存写盘。
5. 调度层（Orchestration Layer）
   - 大规模任务拆分、断点恢复、重试与统计。

## 5. 数据协议与对齐目标

对齐对象：InternNav 动态数据集契约。

目标目录结构：

- `root/group_name/scene_name/data/chunk_name/*.parquet`
- `root/group_name/scene_name/videos/chunk_name/observation.images.rgb/*`
- `root/group_name/scene_name/videos/chunk_name/observation.images.depth/*`
- `root/group_name/scene_name/meta/episodes_stats.jsonl`
- `root/group_name/scene_name/meta/pointcloud_static.ply`
- `root/group_name/scene_name/meta/dynamic_tracks.parquet`
- `root/group_name/scene_name/cache/dyn_voxel/<episode>.npz`
- `root/group_name/scene_name/cache/critic/<episode>.parquet`

关键字段要求（摘要）：

1. episode parquet
   - action / pose
   - camera intrinsic/extrinsic
   - timestamp_ns（或兼容列）
   - critic_base_pred（建议）
2. dynamic_tracks.parquet
   - timestamp_ns
   - object_id
   - x,y,z 或 pose_world
   - vx,vy,vz 或 velocity_world
   - dx,dy,dz 或 bbox_size
   - validity
3. episodes_stats.jsonl
   - image_index.min / image_index.max
   - episode_id / start_time_ns / end_time_ns / num_frames（建议）

## 6. 专家轨迹与 critic 设计

采用双专家并行：

1. Oracle Expert（先知）
   - 输入未来动态体真值轨迹，做时空规划（可使用时空 A* / MPC）。
   - 作为上界监督，重点对齐 dynamic voxel 的时空信息。
2. Reactive Expert（现实）
   - 仅使用当前观测和短时预测，模拟真实部署。
3. critic
   - 同时考虑静态障碍距离与动态障碍距离。
   - 失败样本保留，并给出真实惩罚分。

## 7. 复杂度分层与采样

每条 episode 记录三类复杂度：

1. 路径长度复杂度 L
2. 静态障碍复杂度 S
3. 动态障碍复杂度 D

建议在配置中控制分桶采样，保证训练/评估覆盖均衡，避免只生成“简单轨迹”。

## 8. 断点恢复机制

采用任务状态机 + 原子写入 + 心跳租约：

1. 状态流转：
   - `PENDING -> RUNNING -> DONE`
   - `RUNNING -> FAILED -> RETRY`
2. 恢复规则：
   - RUNNING 任务心跳超时自动回收为 PENDING。
   - DONE 幂等跳过。
3. 原子写入：
   - 先写 `*.tmp`，完成后 rename。
4. 幂等键：
   - `scene_id + episode_id + agent_type + label_version`

## 9. 资源约束下的执行策略（4090 24GB）

1. 仿真阶段低并发（避免多实例 Isaac 同时抢显存）。
2. 数据分片（shard）生成，建议每 shard 100~300 episodes。
3. 标签计算（critic / dyn_voxel）尽量离线异步，提升主流水吞吐。

## 10. 实施里程碑

1. M1：静态闭环
   - 场景生成 + 双机器人 + 断点恢复 + 基础导出。
2. M2：动态闭环
   - 人车动态 + 自由/规则双模式 + 轨迹录制。
3. M3：标注闭环
   - 双 expert + critic + dynamic voxel 缓存。
4. M4：评估闭环
   - 自动评估场景池 + 固定种子复现 + 报告输出。

## 11. 场景层（第一层讨论入口）

本节作为下一轮讨论与实现入口。

### 11.1 场景层职责

1. 从资产库构建参数化场景（仓储/医院/室外等）。
2. 对静态几何做尺度与可达性校验。
3. 生成 NavMesh 并输出可行路径统计。
4. 保证冗余探索空间达标（非单一路径可达）。
5. 输出可复现的 scene_manifest。

### 11.2 输入

1. 资产根目录：
   - `/home/monika/dyishere/dataset/assets/isaac/isaac-sim-assets-complete-5.1.0/`
2. 场景模板配置：
   - 场景类型（warehouse/hospital/outdoor）
   - 静态障碍密度范围
   - 通道宽度范围
   - 冗余探索空间阈值
3. 随机种子：
   - `scene_seed`

### 11.3 输出

1. `scene.usd`（或组合后的 stage）
2. `navmesh.bin`（或等价 navmesh 结果）
3. `scene_manifest.json`
   - scene_id
   - scene_seed
   - asset list
   - 关键几何统计
   - 冗余指标

### 11.4 核心指标（参数化约束）

1. `min_path_count`
   - 起终点之间最少可行路径条数。
2. `min_detour_margin_m`
   - 主最短路与次优路长度差的最小绕行冗余。
3. `min_free_space_ratio`
   - 可通行面积占比。
4. `static_complexity_score`
   - 静态障碍复杂度综合分。

### 11.5 场景生成建议策略

1. 优先资产组合
   - Digital Twin Warehouse
   - Modular Warehouse
   - Hospital
   - Outdoor/Rivermark（后续）
2. 先“模板拼接”，再“障碍扰动”
   - 保证可复现和可控性。
3. 场景通过校验后再进入 episode 生成队列
   - 失败场景直接丢弃并记录原因。

### 11.6 场景层验收标准

1. 同一 `scene_seed` 重建结果一致。
2. NavMesh 可生成且通过可达性检查。
3. 冗余探索指标满足配置阈值。
4. 场景统计写入 manifest，支持审计。

### 11.7 场景层待明确问题（下一轮讨论）

1. 首批默认启用哪些场景模板（warehouse/hospital/outdoor 的比例）？
2. `min_path_count`、`min_detour_margin_m`、`min_free_space_ratio` 的默认值是多少？
3. 静态复杂度分桶（easy/medium/hard）的具体阈值如何定义？
4. 场景编译失败时是重采样还是降级到备用模板？
5. 是否要求评估场景全部“冻结清单”，禁止后续随机增补？

## 12. 下一步

下一轮按“场景层”展开：

1. 先定场景模板与参数 schema。
2. 再定冗余探索指标计算方法。
3. 最后定场景生成状态机与落盘格式。
