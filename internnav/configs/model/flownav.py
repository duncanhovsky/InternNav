from .base_encoders import *

# FlowNav 与 NavDP 一样走 policy_name + ModelCfg 的统一配置入口。
flownav_cfg = ModelCfg(
    policy_name='FlowNav_Policy',
    state_encoder=None,
)
