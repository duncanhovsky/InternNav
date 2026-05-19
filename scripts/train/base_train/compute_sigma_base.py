"""Bridge-DP 离线 σ_base 计算脚本。

直接从 parquet 的 action 列读取 4×4 世界位姿矩阵，
计算轨迹偏离线性插值的标准差。

action 列每行是一个 4×4 齐次变换矩阵，存储机器人在世界坐标系下的位姿。
位姿的平移分量 T[:,3] 包含 (x, y_up, z_forward) 世界坐标。

使用方法：
    python scripts/train/base_train/compute_sigma_base.py \\
        --dataset_index checkpoints/bridgedp_train/preload_index.json \\
        --predict_size 24 \\
        --num_samples 3000 \\
        --output_file checkpoints/bridgedp_train/sigma_base_result.json
"""

import argparse
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def compute_sigma_base(dataset_index, predict_size=24, num_samples=3000,
                       action_scale_xy=5.0, seed=42):
    """从 parquet action 列计算 σ_base。

    action 列每行是 4×4 齐次矩阵（世界坐标位姿），
    与 camera_extrinsic（固定相机安装矩阵）不同。
    """
    import pandas as pd

    with open(dataset_index, 'r') as f:
        index = json.load(f)
    data_dirs = list(dict.fromkeys(index['trajectory_data_dir']))
    total = len(data_dirs)
    print(f"[compute_sigma_base] 共 {total} 条轨迹文件")

    rng = np.random.RandomState(seed)
    n = min(num_samples, total)
    chosen = rng.choice(total, size=n, replace=False)
    chosen.sort()

    all_deviations = []
    valid = error = 0

    print(f"[compute_sigma_base] 开始遍历 {n} 条轨迹...")
    for i, idx in enumerate(chosen):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  进度: {i+1}/{n} ({100.0*(i+1)/n:.1f}%)")
        try:
            df = pd.read_parquet(data_dirs[idx])
            # action 列：每行是 4x4 矩阵或 1x16 展开数组
            actions_raw = df['action'].values
            extrinsics = np.array([np.stack(frame) for frame in actions_raw], dtype=np.float64)
            if extrinsics.ndim == 2 and extrinsics.shape[1] == 16:
                extrinsics = extrinsics.reshape(-1, 4, 4)
            L = len(actions_raw)
            if L < predict_size + 2:
                error += 1
                continue

            # 随机选起点
            max_start = L - predict_size - 1
            start = rng.randint(0, max_start + 1)

            # 均匀采样 predict_size 个帧
            end = min(start + predict_size * 4, L - 1)
            frame_indices = np.linspace(start, end, predict_size, dtype=int)

            # 以起点为原点的局部坐标
            # 这里用世界坐标的 R0 旋转到局部坐标系
            pose0 = extrinsics[start]  # 4×4
            R0 = pose0[:3, :3]
            T0 = pose0[:3, 3]
            local = np.array([
                R0.T @ (extrinsics[fi][:3, 3] - T0)
                for fi in frame_indices
            ], dtype=np.float32)
            # 取 x, z 分量 (平面导航)
            traj_xz = local[:, [0, 2]]

            # 归一化
            traj_xz = traj_xz / action_scale_xy

            # 首尾距离过小的退化轨迹跳过
            span = np.linalg.norm(traj_xz[-1] - traj_xz[0])
            if span < 0.01:  # 归一化后 0.01 对应 5cm
                error += 1
                continue

            # 线性插值基线
            t_col = np.linspace(0, 1, predict_size)[:, None]
            lerp = (1 - t_col) * traj_xz[0:1] + t_col * traj_xz[-1:]
            deviation = traj_xz - lerp  # (T, 2)

            # 只取中间帧（首尾偏差恒为 0）
            if predict_size > 2:
                deviation = deviation[1:-1]

            all_deviations.append(deviation)
            valid += 1
        except Exception as e:
            error += 1
            if error <= 3:
                print(f"  警告: {data_dirs[idx]}: {e}")

    print(f"\n[compute_sigma_base] 有效={valid}, 错误={error}")
    if valid == 0:
        raise RuntimeError("没有有效轨迹！")

    all_dev = np.concatenate(all_deviations, axis=0)  # (N*T_mid, 2)
    sigma_x = float(np.std(all_dev[:, 0]))
    sigma_z = float(np.std(all_dev[:, 1]))
    sigma_base = float(np.std(all_dev))
    p95 = float(np.percentile(np.abs(all_dev), 95))

    result = {
        "sigma_base_recommended": round(p95, 4),
        "sigma_base_std": round(sigma_base, 4),
        "sigma_per_dim": {"x": round(sigma_x, 4), "z": round(sigma_z, 4)},
        "num_trajectories": valid,
        "num_skipped_degenerate": error,
        "trajectory_length": predict_size,
        "total_deviation_points": all_dev.shape[0],
        "action_scale_xy": action_scale_xy,
        "seed": seed,
    }

    print(f"\n{'='*55}")
    print(f"  σ_base (推荐, 95分位数) = {result['sigma_base_recommended']}")
    print(f"  σ_base (全局 std)       = {result['sigma_base_std']}")
    print(f"  σ_x = {sigma_x:.4f},  σ_z = {sigma_z:.4f}")
    print(f"  有效轨迹: {valid},  跳过: {error}")
    print(f"{'='*55}")
    print(f"\n请在配置中设置:  sigma_base = {result['sigma_base_recommended']}")
    return result


def compute_d_max(dataset_index, predict_size=24, num_samples=3000,
                  action_scale_xy=5.0, pred_digit=4, redundancy=1.2, seed=42):
    """统计数据集中单次预测轨迹的长度分布，计算 d_max。

    轨迹长度定义：起点到第 predict_size*pred_digit 帧的直线距离（归一化空间）。
    d_max 取 90 分位数 × redundancy 冗余系数。

    Args:
        dataset_index: 预索引 JSON 路径。
        predict_size: 预测步数（默认 24）。
        num_samples: 采样轨迹数。
        action_scale_xy: 归一化因子（与训练一致）。
        pred_digit: 帧间隔（与训练一致）。
        redundancy: 冗余系数（默认 1.2）。
        seed: 随机种子。

    Returns:
        dict: 包含 d_max_recommended 和分布统计。
    """
    import pandas as pd

    with open(dataset_index, 'r') as f:
        index = json.load(f)
    data_dirs = list(dict.fromkeys(index['trajectory_data_dir']))
    total = len(data_dirs)
    print(f"[compute_d_max] 共 {total} 条轨迹文件")

    rng = np.random.RandomState(seed)
    n = min(num_samples, total)
    chosen = rng.choice(total, size=n, replace=False)
    chosen.sort()

    all_distances = []
    valid = error = 0

    print(f"[compute_d_max] 开始遍历 {n} 条轨迹...")
    for i, idx in enumerate(chosen):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  进度: {i+1}/{n} ({100.0*(i+1)/n:.1f}%)")
        try:
            df = pd.read_parquet(data_dirs[idx])
            actions_raw = df['action'].values
            extrinsics = np.array([np.stack(frame) for frame in actions_raw], dtype=np.float64)
            if extrinsics.ndim == 2 and extrinsics.shape[1] == 16:
                extrinsics = extrinsics.reshape(-1, 4, 4)
            L = len(actions_raw)
            min_len = predict_size * pred_digit + 2
            if L < min_len:
                error += 1
                continue

            # 随机选起点
            max_start = L - predict_size * pred_digit - 1
            start = rng.randint(0, max(1, max_start + 1))

            # 终点帧 = start + predict_size * pred_digit（与 __getitem__ 一致）
            end = min(start + predict_size * pred_digit, L - 1)

            # 以起点为原点的局部坐标
            pose0 = extrinsics[start]  # 4×4
            R0 = pose0[:3, :3]
            T0 = pose0[:3, 3]
            pose_end = extrinsics[end]
            T_end_local = R0.T @ (pose_end[:3, 3] - T0)

            # 取 x, z 分量（平面导航），归一化
            dist = np.sqrt(T_end_local[0]**2 + T_end_local[2]**2) / action_scale_xy
            all_distances.append(dist)
            valid += 1
        except Exception as e:
            error += 1
            if error <= 3:
                print(f"  警告: {data_dirs[idx]}: {e}")

    print(f"\n[compute_d_max] 有效={valid}, 错误={error}")
    if valid == 0:
        raise RuntimeError("没有有效轨迹！")

    dists = np.array(all_distances)
    p90 = float(np.percentile(dists, 90))
    d_max_rec = round(p90 * redundancy, 4)

    result = {
        "d_max_recommended": d_max_rec,
        "d_max_raw_p90": round(p90, 4),
        "redundancy_factor": redundancy,
        "distance_stats": {
            "mean": round(float(np.mean(dists)), 4),
            "median": round(float(np.median(dists)), 4),
            "p75": round(float(np.percentile(dists, 75)), 4),
            "p90": round(p90, 4),
            "p95": round(float(np.percentile(dists, 95)), 4),
            "max": round(float(np.max(dists)), 4),
        },
        "num_trajectories": valid,
        "num_skipped": error,
        "predict_size": predict_size,
        "pred_digit": pred_digit,
        "action_scale_xy": action_scale_xy,
        "seed": seed,
    }

    print(f"\n{'='*55}")
    print(f"  d_max (推荐, P90×{redundancy}) = {d_max_rec}")
    print(f"  P90 = {p90:.4f}, mean = {np.mean(dists):.4f}, median = {np.median(dists):.4f}")
    print(f"  有效轨迹: {valid},  跳过: {error}")
    print(f"{'='*55}")
    print(f"\n请在配置中设置:  d_max = {d_max_rec}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Bridge-DP 离线计算 σ_base 和 d_max")
    parser.add_argument('--dataset_index', type=str, required=True)
    parser.add_argument('--predict_size', type=int, default=24)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--action_scale_xy', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['sigma', 'd_max', 'all'],
                        help='计算模式: sigma=仅σ_base, d_max=仅d_max, all=两者都算')
    parser.add_argument('--pred_digit', type=int, default=4,
                        help='帧间隔（与训练 pred_digit 一致）')
    parser.add_argument('--redundancy', type=float, default=1.2,
                        help='d_max 冗余系数')
    # 保留旧参数以兼容
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--memory_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    results = {}

    if args.mode in ('sigma', 'all'):
        sigma_result = compute_sigma_base(
            dataset_index=args.dataset_index,
            predict_size=args.predict_size,
            num_samples=args.num_samples,
            action_scale_xy=args.action_scale_xy,
            seed=args.seed,
        )
        results['sigma_base'] = sigma_result

    if args.mode in ('d_max', 'all'):
        d_max_result = compute_d_max(
            dataset_index=args.dataset_index,
            predict_size=args.predict_size,
            num_samples=args.num_samples,
            action_scale_xy=args.action_scale_xy,
            pred_digit=args.pred_digit,
            redundancy=args.redundancy,
            seed=args.seed,
        )
        results['d_max'] = d_max_result

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()
