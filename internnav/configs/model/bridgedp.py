"""Bridge-DP 模型配置。

独立于 NavDP 的模型配置，通过 ``ModelCfg`` 基类注册到 InternNav 框架。
与 ``internnav/configs/model/navdp.py`` 平级，不修改原文件。

参考：
    - internnav/configs/model/navdp.py（NavDP 配置）
    - internnav/configs/model/base_encoders.py（ModelCfg 基类）
"""

from .base_encoders import ModelCfg

bridgedp_cfg = ModelCfg(
    policy_name='BridgeDP_Policy',
    state_encoder=None,
)
