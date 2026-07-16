# ArcDP 8×4090 十天训练方案设计

## 目标

在 8 张 RTX 4090 24 GB GPU 上运行 ArcDP 全量 NVMe 数据训练，并将计划压缩为最多两个训练阶段：

1. 完整 ArcDP（full）训练 1 epoch。
2. 去除 Bridge diffusion 的 `no_bridge` 消融训练 1 epoch。

训练默认使用 SwanLab 监控，并提高单卡 micro-batch 以利用当前约 11.4 GiB 的实测峰值之外的显存空间。旧的 full10 + 六项 P0 入口继续保留，用于历史实验复现。

## 训练配置

- GPU：8×RTX 4090。
- 单卡 batch：48。
- 梯度累积：1。
- 有效全局 batch：`48 × 8 × 1 = 384`。
- 学习率：保持 `3e-4`。
- 数据：NVMe 上的 InternData-N1 v0.5 full。
- full：1 epoch。
- no_bridge：1 epoch。
- 训练阶段按顺序串行执行，避免两个实验争用同一组 GPU。

从原配置 `16 × 8 × 3 = 384` 改为 `48 × 8 × 1 = 384`，保持有效全局 batch 和学习率语义不变。按照当前显存峰值粗略外推，batch 48 的目标峰值为约 18–22 GiB；若发生 OOM，人工回退到 `batch=24, grad_accum=2`，有效全局 batch 仍为 384。

## 训练入口与兼容性

新增独立入口，不改变现有 full10 脚本：

- 根目录入口：`train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh`
- 实际入口：`scripts/train/arcdp_4090/train_arcdp_full1_no_bridge_1ep_8x4090_nvme.sh`

新入口使用独立运行名，避免自动恢复到旧 full10 或旧 P0 checkpoint：

- `arcdp_full1_bs48_8x4090_nvme`
- `arcdp_p0_no_bridge_1ep_bs48_8x4090_nvme`

环境变量仍可覆盖 batch、梯度累积、worker 数、学习率和恢复行为。脚本启动时打印每个阶段的 GPU 数、batch、梯度累积、有效全局 batch、epoch、监控后端和输出名称。

## SwanLab

新入口默认设置：

- `BRIDGEDP_REPORT_TO=swanlab`
- `SWANLAB_PROJECT=ArcDP-8x4090-full1-no-bridge`
- 每个阶段设置独立 `SWANLAB_EXP_NAME`。

启动前验证 `swanlab` 可以导入。缺少 Python 包时立即失败并给出安装提示；缺少 `SWANLAB_API_KEY` 时给出登录提示，但允许已经存在本地 SwanLab 登录状态的机器继续运行。用户仍可通过显式设置 `BRIDGEDP_REPORT_TO=tensorboard` 覆盖默认监控方式。

## 被移出训练计划的消融

- `p0_rel`：与 `no_bridge` 在 DDPM、高斯初始化和 anchor 关闭方面重叠，本轮删除。
- `p0_no_scale_cond`：需要重训，本轮删除。
- `p0_no_anchor_train`：需要重训，本轮删除。
- `p0_no_ordered_init`：只改变推理初始化，不重训；后续复用 full checkpoint 评估。
- `p0_no_gcs`：只改变推理候选评分，不重训；后续复用 full checkpoint 评估。

本次实现只修改训练套件，不额外实现评估入口。

## 时间预算

当前 full10 ETA 为 2204 小时 44 分 36 秒，对应当前测量吞吐下每 epoch 约 220.47 小时。两个训练 epoch 在相同吞吐下约需 440.95 小时（18.37 天）。要压缩到 10 天，batch 48 配置需要达到至少约 1.84 倍的实际吞吐提升。

由于原 ETA 使用从启动开始的累计平均 step 时间，早期会包含模型初始化和首批数据预热，可能显著高估。脚本不承诺仅凭该早期 ETA 硬性保证 10 天完成；应在 batch 48 稳定运行 50–100 个 optimizer step 后，根据 SwanLab 的稳定吞吐重新判断。目标规划为约 9.2–10 天，若稳定 ETA 超出预算，应停止 no_bridge 阶段或另行缩减数据预算，而不是静默改变 full 训练定义。

## 错误处理

- 数据目录、preload index 和 DepthAnything checkpoint 缺失时，在启动 torchrun 前失败。
- SwanLab 包缺失且监控后端为 SwanLab 时，在启动前失败。
- batch 48 OOM 时不自动重启，避免污染 checkpoint；操作员显式使用 `BRIDGEDP_BATCH_SIZE=24 BRIDGEDP_GRAD_ACCUM=2` 重启。
- full 阶段失败时，由 `set -e` 阻止 no_bridge 阶段启动。
- 各阶段保留独立 checkpoint、日志和 SwanLab 实验名称。

## 验证

增加静态回归测试，验证：

- 新根入口指向正确的实际脚本。
- 固定使用 8 个 GPU 进程和 GPU 0–7。
- 默认 batch 48、梯度累积 1、有效全局 batch 384。
- full 和 no_bridge 都只运行 1 epoch。
- 训练列表不包含其余五个 P0 训练项。
- SwanLab 是默认后端，并且存在导入检查、项目名和实验名。
- 旧 full10 套件保持不变。

运行目标单元测试和相关 ArcDP 4090/P0 配置测试，并执行 Bash 语法检查。
