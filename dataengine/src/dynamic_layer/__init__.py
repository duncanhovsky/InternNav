"""DataEngine Dynamic Layer — 基于 isaacsim.replicator.agent (IRA) 的动态 NPC 层。

该模块负责：
1. 根据场景配置自动生成 IRA YAML 配置文件
2. 生成 NPC 行为命令文件 (command.txt / robot_command.txt)
3. 驱动 IRA 插件在 Isaac Sim 中执行仿真
4. 后处理 IRA 输出，转换为 dataengine 标准格式
"""

from .config import DynamicLayerConfig
from .models import DynamicLayerResult, CharacterSpec, CommandEntry

__all__ = [
    "DynamicLayerConfig",
    "DynamicLayerResult",
    "CharacterSpec",
    "CommandEntry",
]
