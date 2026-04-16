"""场景层包入口。

本包提供：
1. 场景层参数 schema 与校验
2. 混合模式场景编译（完整场景 + 模块化场景）
3. 场景指标校验与复杂度分桶
4. manifest 与轨迹任务清单导出
"""

from .config import SceneLayerConfig
from .pipeline import SceneLayerPipeline

__all__ = ["SceneLayerConfig", "SceneLayerPipeline"]
