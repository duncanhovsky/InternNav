"""Bridge-DP 离线 σ_base 计算脚本。

本脚本从训练数据集中统计绝对坐标轨迹的标准差，
用于确定 Bridge-DP 布朗桥噪声调度器的基础方差参数 σ_base。

σ_base 的物理含义：
    衡量训练集中轨迹在 "中间时刻" 与线性插值之间的典型偏差幅度。
    它是一个固定常数（非可学习参数），在训练前离线计算一次即可。

计算方法：
    1. 遍历数据集中所有轨迹 τ = {(x_i, y_i, θ_i)}_{i=0}^{T-1}
    2. 计算每条轨迹的线性插值基线：lerp_i = (1 - t_i) * τ_0 + t_i * τ_{T-1}
    3. 计算偏差 δ_i = τ_i - lerp_i
    4. σ_base = std(δ) 对所有轨迹、所有时间步的全局标准差

使用方法：
    python scripts/train/base_train/compute_sigma_base.py \\
        --root_dir /path/to/data \\
        --dataset_index /path/to/preload.json \\
        --memory_size 5 \\
        --predict_size 8 \\
        --image_size 224 \\
        --scene_scale 1.0 \\
        --num_samples 10000 \\
        --output_file sigma_base_result.json

输出示例：
    {
        "sigma_base": 0.847,
        "sigma_per_dim": {"x": 0.912, "y": 0.803, "theta": 0.127},
        "num_trajectories": 10000,
        "trajectory_length": 8
    }
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中，以便直接运行脚本
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def compute_sigma_base(
    root_dir: str,
    dataset_index: str,
    memory_size: int = 5,
    predict_size: int = 8,
    image_size: int = 224,
    scene_scale: float = 1.0,
    num_samples: int = -1,
    seed: int = 42,
) -> dict:
    """从数据集中计算 σ_base。

    Args:
        root_dir: 数据集根目录
        dataset_index: 预加载索引文件路径（JSON 格式）
        memory_size: 历史帧数
        predict_size: 预测步数（轨迹长度 T）
        image_size: 图像尺寸
        scene_scale: 场景缩放系数
        num_samples: 采样数量，-1 表示使用全部数据
        seed: 随机种子

    Returns:
        包含 sigma_base 及详细统计信息的字典
    """
    # 延迟导入，避免脚本自身的导入依赖问题
    from internnav.dataset.bridgedp_lerobot_dataset import BridgeDP_Base_Dataset

    print(f"[compute_sigma_base] 初始化数据集...")
    print(f"  root_dir      = {root_dir}")
    print(f"  dataset_index  = {dataset_index}")
    print(f"  memory_size    = {memory_size}")
    print(f"  predict_size   = {predict_size}")
    print(f"  num_samples    = {num_samples if num_samples > 0 else 'ALL'}")

    # 创建数据集实例
    dataset = BridgeDP_Base_Dataset(
        root_dirs=root_dir,
        preload_path=dataset_index,
        memory_size=memory_size,
        predict_size=predict_size,
        batch_size=1,
        image_size=image_size,
        scene_data_scale=scene_scale,
        preload=False,  # 不预加载图像以节省内存
    )

    total_len = len(dataset)
    print(f"  数据集大小     = {total_len}")

    # 确定采样数量
    if num_samples <= 0 or num_samples > total_len:
        num_samples = total_len

    # 随机采样索引
    rng = np.random.RandomState(seed)
    indices = rng.choice(total_len, size=num_samples, replace=False)
    indices.sort()

    # ---------------------------------------------------------------------------
    # 收集所有轨迹的偏差
    # ---------------------------------------------------------------------------
    all_deviations = []  # 存储 (N, T, 3) 的偏差列表
    valid_count = 0
    error_count = 0

    print(f"\n[compute_sigma_base] 开始遍历 {num_samples} 条轨迹...")

    for i, idx in enumerate(indices):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  进度: {i + 1}/{num_samples} ({100.0 * (i + 1) / num_samples:.1f}%)")

        try:
            sample = dataset[int(idx)]
            # sample 返回元组，pred_actions 在 index 5 (labels)
            # BridgeDP_Base_Dataset.__getitem__ 返回：
            #   (goal_point, goal_image, target_goal, rgb, depth, pred_actions,
            #    augment_actions, pred_critic, augment_critic, prior_traj, theta_g, pixel_flag)
            pred_actions = sample[5]  # shape: (predict_size, 3) — (x, y, θ)

            if isinstance(pred_actions, torch.Tensor):
                traj = pred_actions.numpy()
            else:
                traj = np.array(pred_actions)

            # traj shape: (T, 3) 其中 3 = (x, y, θ)
            if traj.ndim != 2 or traj.shape[1] < 2:
                error_count += 1
                continue

            T = traj.shape[0]
            if T < 2:
                error_count += 1
                continue

            # 计算归一化时间
            t_vals = np.linspace(0.0, 1.0, T)  # shape: (T,)

            # 线性插值基线: lerp_i = (1 - t_i) * traj[0] + t_i * traj[-1]
            start = traj[0:1]   # shape: (1, 3)
            end = traj[-1:]     # shape: (1, 3)
            t_col = t_vals[:, None]  # shape: (T, 1)
            lerp = (1.0 - t_col) * start + t_col * end  # shape: (T, 3)

            # 偏差 = 实际轨迹 - 线性插值
            deviation = traj - lerp  # shape: (T, 3)
            all_deviations.append(deviation)
            valid_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  警告: 索引 {idx} 处理失败: {e}")
            continue

    print(f"\n[compute_sigma_base] 采集完成:")
    print(f"  有效轨迹数 = {valid_count}")
    print(f"  错误数     = {error_count}")

    if valid_count == 0:
        raise RuntimeError("没有有效轨迹可供计算 σ_base！")

    # ---------------------------------------------------------------------------
    # 计算全局标准差
    # ---------------------------------------------------------------------------
    # 合并所有偏差 → (N*T, 3)
    all_dev = np.concatenate(all_deviations, axis=0)  # shape: (N*T, D)
    D = all_dev.shape[1]

    # 各维度标准差
    sigma_per_dim = {}
    dim_names = ['x', 'y', 'theta'] if D >= 3 else [f'dim_{i}' for i in range(D)]
    for d in range(D):
        sigma_per_dim[dim_names[d]] = float(np.std(all_dev[:, d]))

    # 全局标准差（对所有维度、所有时间步）
    sigma_base_global = float(np.std(all_dev))

    # 排除首尾时间步（t=0 和 t=1 处偏差恒为 0）后的标准差
    # 这更能反映中间时刻的偏差水平
    all_dev_mid_list = []
    for dev in all_deviations:
        if dev.shape[0] > 2:
            all_dev_mid_list.append(dev[1:-1])  # 去掉首尾
    if all_dev_mid_list:
        all_dev_mid = np.concatenate(all_dev_mid_list, axis=0)
        sigma_base_mid = float(np.std(all_dev_mid))
    else:
        sigma_base_mid = sigma_base_global

    # ---------------------------------------------------------------------------
    # 推荐值：使用中间时刻的标准差作为 σ_base
    # ---------------------------------------------------------------------------
    result = {
        "sigma_base_recommended": round(sigma_base_mid, 4),
        "sigma_base_global": round(sigma_base_global, 4),
        "sigma_base_mid_only": round(sigma_base_mid, 4),
        "sigma_per_dim": {k: round(v, 4) for k, v in sigma_per_dim.items()},
        "num_trajectories": valid_count,
        "trajectory_length": predict_size,
        "total_deviation_points": all_dev.shape[0],
        "seed": seed,
    }

    print(f"\n{'=' * 60}")
    print(f"[compute_sigma_base] 计算结果:")
    print(f"  σ_base (推荐值，中间时刻) = {result['sigma_base_recommended']}")
    print(f"  σ_base (全局)             = {result['sigma_base_global']}")
    for dim_name, sigma_val in sigma_per_dim.items():
        print(f"  σ_{dim_name:5s}                   = {sigma_val:.4f}")
    print(f"  有效轨迹数                 = {valid_count}")
    print(f"  总偏差样本点               = {all_dev.shape[0]}")
    print(f"{'=' * 60}")

    return result


def main():
    """主函数：解析命令行参数并执行计算。"""
    parser = argparse.ArgumentParser(
        description="Bridge-DP 离线计算 σ_base 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/train/base_train/compute_sigma_base.py \\
      --root_dir /data/nav_dataset \\
      --dataset_index /data/nav_dataset/preload.json \\
      --predict_size 8 \\
      --num_samples 5000

  # 输出结果保存到 JSON 文件
  python scripts/train/base_train/compute_sigma_base.py \\
      --root_dir /data/nav_dataset \\
      --dataset_index /data/nav_dataset/preload.json \\
      --output_file results/sigma_base.json
        """,
    )

    parser.add_argument('--root_dir', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--dataset_index', type=str, required=True,
                        help='预加载索引文件路径（JSON 格式）')
    parser.add_argument('--memory_size', type=int, default=5,
                        help='历史帧数（默认: 5）')
    parser.add_argument('--predict_size', type=int, default=8,
                        help='预测步数 / 轨迹长度 T（默认: 8）')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像尺寸（默认: 224）')
    parser.add_argument('--scene_scale', type=float, default=1.0,
                        help='场景缩放系数（默认: 1.0）')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='采样数量，-1 表示全部（默认: -1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出 JSON 文件路径（可选）')

    args = parser.parse_args()

    result = compute_sigma_base(
        root_dir=args.root_dir,
        dataset_index=args.dataset_index,
        memory_size=args.memory_size,
        predict_size=args.predict_size,
        image_size=args.image_size,
        scene_scale=args.scene_scale,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # 保存结果到 JSON 文件
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output_file}")

    # 打印建议的配置行
    print(f"\n请在训练配置中设置:")
    print(f"  sigma_base = {result['sigma_base_recommended']}")
    print(f"  # 位于 scripts/train/base_train/configs/bridgedp.py 的 IlCfg 中")


if __name__ == '__main__':
    main()
