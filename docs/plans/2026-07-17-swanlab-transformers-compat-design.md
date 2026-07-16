# SwanLab 与 Transformers 兼容设计

## 背景

ArcDP 环境固定使用 `transformers==4.51.0`，同时安装到了 SwanLab 0.8.5。Transformers 的 `SwanLabCallback.setup()` 认为未初始化时 `swanlab.get_run()` 返回 `None`，SwanLab 0.8.5 则抛出 `RuntimeError: No active Run. Call swanlab.init() first.`，导致训练在 `on_train_begin`、首个 batch 之前退出。

## 方案比较

1. **rank 0 显式初始化（采用）**：在 `Trainer.train()` 前创建唯一 SwanLab Run，Transformers 回调复用该 Run。不修改第三方包，风险和改动最小。
2. **降级 SwanLab**：需要重新确认兼容版本及其其他依赖，且环境脚本当前允许升级，问题可能复发。
3. **升级 Transformers**：可能破坏项目针对 4.51.0 的既有兼容约束，风险最大。

## 设计

新增轻量兼容模块，负责：

- 仅在 `report_to` 含 `swanlab` 且当前进程为 world-process-zero 时执行；
- 同时兼容 `get_run()` 返回 `None` 和抛出“没有活动 Run”异常两种语义；
- 已有 Run 时直接复用；
- 新建 Run 时从 `SWANLAB_PROJ_NAME` 和 `SWANLAB_EXP_NAME` 读取项目名与实验名；
- 对其他异常不做吞噬，保证认证、网络和配置错误可见。

训练入口在所有模型、数据集、Trainer 和断点恢复状态准备完成后、`trainer.train()` 之前调用该模块。非 rank 0 只执行无副作用的快速返回，因此不会创建重复实验，也不会改变 DDP 梯度同步、优化器或数据采样。

## 正确性与观测

Hugging Face Trainer 的 SwanLab 回调本来只在 `state.is_world_process_zero` 上写外部日志。训练 loss 等指标按 Trainer 的分布式流程汇总后由 rank 0 写入；单一 rank 0 Run 不降低训练指标完整性。逐 rank 调试值不作为默认实验指标记录。

## 测试

- 未启用 SwanLab时不初始化；
- 非主进程不初始化；
- `get_run()` 返回 `None` 时初始化；
- `get_run()` 抛出 SwanLab 0.8.5 的无活动 Run 异常时初始化；
- 已存在 Run 时复用；
- 训练入口确实在 `trainer.train()` 前调用兼容函数。
