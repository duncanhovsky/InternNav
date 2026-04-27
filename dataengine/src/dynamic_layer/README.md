# Dynamic Layer — 基于 IRA 的动态 NPC 层

## 概述

`dynamic_layer` 模块是 DataEngine 的动态层实现，基于 Isaac Sim 5.1 的
`isaacsim.replicator.agent` (IRA) 扩展插件驱动 NPC 行为仿真。

核心理念：**动态层的工作就是生成正确的 IRA 配置文件 + 命令文件，
然后把执行工作交给 Isaac Sim 的 IRA 插件**。

## 架构

```
┌─────────────────────────────────────────────────────┐
│                  IRADynamicLayer                     │
│                (dynamic_layer.py)                    │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ IRAConfig    │  │ Command      │                 │
│  │ Generator    │  │ Generator    │                 │
│  │              │  │              │                 │
│  │ → YAML 配置  │  │ → command.txt│                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         │                 │                         │
│         ▼                 ▼                         │
│  ┌──────────────────────────────┐                   │
│  │        IRA Driver            │                   │
│  │  (启动 Isaac Sim + IRA)      │                   │
│  └──────────────┬───────────────┘                   │
│                 │                                   │
│                 ▼                                   │
│  ┌──────────────────────────────┐                   │
│  │      Post Processor          │                   │
│  │  (IRA 输出 → 标准格式)       │                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## 文件结构

```
dataengine/src/dynamic_layer/
├── __init__.py                 # 模块入口
├── __main__.py                 # CLI 入口
├── config.py                   # 动态层配置定义
├── models.py                   # 数据模型（IRAConfig, CommandEntry, etc.）
├── ira_config_generator.py     # IRA YAML 配置生成器
├── command_generator.py        # NPC 行为命令生成器
├── ira_driver.py               # IRA 插件驱动器
├── post_processor.py           # IRA 输出后处理器
├── dynamic_layer.py            # 主编排器
└── README.md                   # 本文档

dataengine/src/config/
├── IRA/                        # 用户提供的 IRA 配置示例
│   ├── default_config.yaml
│   ├── command.txt
│   └── robot_command.txt
└── dynamic_layer/
    ├── default.yaml            # 默认配置
    └── scenarios/              # 预定义场景行为方案
        ├── warehouse_busy.yaml
        ├── warehouse_sparse.yaml
        └── hospital_corridor.yaml
```

## 快速开始

### 1. 作为 Python 模块使用

```python
from dataengine.src.dynamic_layer.config import DynamicLayerConfig
from dataengine.src.dynamic_layer.dynamic_layer import IRADynamicLayer

# 创建配置
cfg = DynamicLayerConfig(
    character_num_min=5,
    character_num_max=10,
    simulation_length=300,
)

# 创建动态层
layer = IRADynamicLayer(cfg=cfg)

# 执行（dry-run 模式只生成配置不执行 IRA）
result = layer.generate(
    scene_id="scene_001",
    scene_dir="/path/to/scene_dir",
    seed=42,
    stage_usd="/path/to/scene.usd",
    dry_run=True,
)

print(result.ira_config_path)   # 生成的 IRA YAML
print(result.command_file)       # 生成的命令文件
```

### 2. 命令行使用

```bash
python -m dataengine.src.dynamic_layer \
    --scene-usd /path/to/scene.usd \
    --scene-dir /path/to/output \
    --seed 42 \
    --dry-run
```

### 3. 集成到场景编译器

在 `SceneLayerConfig` 中设置 `dynamics_backend: "ira"` 即可自动使用 IRA 动态层：

```json
{
  "dynamics_backend": "ira",
  "enable_dynamics": true,
  "ira_simulation_length": 300,
  "ira_character_filters": ["male", "medical"],
  "ira_camera_num": 5
}
```

## IRA 命令格式

命令文件 (`command.txt`) 每行一条命令，格式：

```
<character_id> <action> <param1> [param2] [param3] [param4]
```

支持的行为：

| 行为 | 格式 | 说明 |
|------|------|------|
| `Idle` | `Character Idle 4.1` | 原地等待指定秒数 |
| `GoTo` | `Character GoTo -14.16 7.29 0.0 _` | 导航到目标坐标 |
| `LookAround` | `Character LookAround 2.5` | 环顾四周指定秒数 |

角色 ID 命名规则：
- 第一个角色：`Character`
- 后续角色：`Character_01`, `Character_02`, ...

## 运行模式

### IRA 模式（默认）
当 Isaac Sim 可用时，驱动 IRA 插件执行真实的 NPC 仿真。

### Dry-Run 模式
仅生成配置文件（YAML + command.txt），不执行 Isaac Sim。
适用于调试配置、离线准备。

### Synthetic Fallback 模式
当 Isaac Sim 不可用时，退化到旧的合成后端生成模拟轨迹。
IRA 配置文件仍会生成，可供后续手动使用。

## 环境要求

### Isaac Sim Python 路径探测

IRA Driver 按以下优先级探测 Isaac Sim：

1. 配置项 `isaac_sim_python`
2. 环境变量 `ISAAC_SIM_PYTHON`
3. 环境变量 `ISAAC_SIM_PATH` 下的 `python.sh`
4. 常见安装路径

### 推荐设置

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim-5.1.0/python.sh
```

## 测试

```bash
# 单元测试（33 个用例）
python -m pytest tests/unit_test/test_dynamic_layer.py -v

# 集成测试（10 个用例）
python -m pytest tests/function_test/test_dynamic_layer_integration.py -v
```

## 后续扩展

1. **Robot 主视角**：通过 `robot_command.txt` 控制信息采集平台
2. **Recording Layer**：利用 IRA 的 `IRABasicWriter` 输出训练数据
3. **场景行为模板**：支持更多预定义场景行为方案
4. **NavMesh 联动**：从 NavMesh Walkable 区域采样 GoTo 目标点
