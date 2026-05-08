"""生成 LeRobot 格式数据集的预加载索引文件。

用于加速 Bridge-DP / NavDP 数据集初始化，避免每次训练时重新扫描目录。

输出格式（与 NavDP/BridgeDP __init__ 中 preload=True 时读取的格式一致）：
    {
        "trajectory_data_dir": [...],
        "trajectory_rgb_path": [[...], ...],
        "trajectory_depth_path": [[...], ...],
        "trajectory_afford_path": [...]
    }

支持两种目录结构：
    旧格式（scene 直接包含 data/）：
        root_dir/group/scene/data/chunk-000/
        root_dir/group/scene/meta/episodes_stats.jsonl
        root_dir/group/scene/videos/chunk-000/

    新格式（scene 下有 trajectory_XX/ 子目录）：
        root_dir/group/scene/trajectory_XX/data/chunk-000/
        root_dir/group/scene/trajectory_XX/meta/episodes_stats.jsonl
        root_dir/group/scene/trajectory_XX/videos/chunk-000/

使用方法：
    python scripts/dataset/generate_preload_index.py \\
        --root_dir /path/to/traj_data \\
        --output navdp_dataset_lerobot.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import jsonlines
except ImportError:
    print("错误: 需要安装 jsonlines 库")
    print("运行: pip install jsonlines")
    sys.exit(1)


def _process_data_unit(unit_path: Path) -> dict:
    """处理一个数据单元（scene 或 trajectory_XX），返回路径索引。

    一个数据单元必须包含：
        - data/chunk-XXX/*.parquet
        - meta/episodes_stats.jsonl
        - videos/chunk-XXX/observation.images.rgb/
        - videos/chunk-XXX/observation.images.depth/

    Args:
        unit_path: 数据单元的根路径

    Returns:
        包含 trajectory_data_dir, trajectory_rgb_path, trajectory_depth_path,
        trajectory_afford_path 四个列表的字典
    """
    result = {
        'trajectory_data_dir': [],
        'trajectory_rgb_path': [],
        'trajectory_depth_path': [],
        'trajectory_afford_path': [],
    }

    data_path = unit_path / 'data'
    meta_path = unit_path / 'meta' / 'episodes_stats.jsonl'
    afford_path = str(unit_path / 'meta' / 'pointcloud.ply')

    if not data_path.exists():
        print(f"      警告: 缺少 data/ 目录，跳过: {unit_path}")
        return result

    if not meta_path.exists():
        print(f"      警告: 缺少 meta/episodes_stats.jsonl，跳过: {unit_path}")
        return result

    # 获取 chunk 名称
    chunks = [d.name for d in data_path.iterdir() if d.is_dir() and d.name.startswith('chunk-')]
    if not chunks:
        print(f"      警告: data/ 下无 chunk 目录，跳过: {unit_path}")
        return result

    chunk_name = sorted(chunks)[0]
    data_dir = unit_path / 'data' / chunk_name

    # 读取 episodes_stats.jsonl
    with jsonlines.open(str(meta_path), 'r') as reader:
        episode_info = list(reader)

    # RGB 路径
    rgb_dir = unit_path / 'videos' / chunk_name / 'observation.images.rgb'
    if not rgb_dir.exists():
        print(f"      警告: 缺少 RGB 目录 {rgb_dir}，跳过: {unit_path}")
        return result
    rgb_paths = [str(rgb_dir / p) for p in sorted(os.listdir(str(rgb_dir)))]

    # Depth 路径
    depth_dir = unit_path / 'videos' / chunk_name / 'observation.images.depth'
    if not depth_dir.exists():
        print(f"      警告: 缺少 depth 目录 {depth_dir}，跳过: {unit_path}")
        return result
    depth_paths = [str(depth_dir / p) for p in sorted(os.listdir(str(depth_dir)))]

    # Parquet 数据路径（只取 .parquet 文件）
    data_files = sorted([f for f in os.listdir(str(data_dir)) if f.endswith('.parquet')])
    data_paths = [str(data_dir / p) for p in data_files]

    if not data_paths:
        print(f"      警告: 无 parquet 文件，跳过: {unit_path}")
        return result

    if len(episode_info) == 1 and 'image_index' not in episode_info[0]:
        # 新格式：单 episode，无 image_index，使用全部帧
        result['trajectory_data_dir'].append(data_paths[0])
        result['trajectory_rgb_path'].append(rgb_paths)
        result['trajectory_depth_path'].append(depth_paths)
        result['trajectory_afford_path'].append(afford_path)
    else:
        # 旧格式：多 episode，通过 image_index 切分帧范围
        for episode_idx, episode in enumerate(episode_info):
            try:
                if 'image_index' in episode:
                    image_start_index = episode['image_index']['min']
                    image_end_index = episode['image_index']['max']
                    episode_rgb_path = np.array(rgb_paths)[image_start_index : image_end_index + 1].tolist()
                    episode_depth_path = np.array(depth_paths)[image_start_index : image_end_index + 1].tolist()
                else:
                    episode_rgb_path = rgb_paths
                    episode_depth_path = depth_paths

                result['trajectory_data_dir'].append(data_paths[episode_idx])
                result['trajectory_rgb_path'].append(episode_rgb_path)
                result['trajectory_depth_path'].append(episode_depth_path)
                result['trajectory_afford_path'].append(afford_path)
            except Exception as e:
                print(f"      警告: 处理 episode {episode_idx} 失败: {e}")

    return result


def scan_lerobot_dataset(root_dir: str, scene_scale: float = 1.0) -> dict:
    """扫描 LeRobot 格式数据集，生成与 NavDP/BridgeDP preload 一致的索引。

    Args:
        root_dir: 数据集根目录
        scene_scale: 场景采样比例（1.0 表示使用全部场景）

    Returns:
        包含 4 个路径数组的字典，可直接被 NavDP/BridgeDP Dataset 的
        preload=True 模式读取
    """
    index_data = {
        'trajectory_data_dir': [],
        'trajectory_rgb_path': [],
        'trajectory_depth_path': [],
        'trajectory_afford_path': [],
    }
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {root_dir}")

    print(f"扫描数据集: {root_dir}")
    total_episodes = 0

    # 遍历 group
    for group_dir in sorted(root_path.iterdir()):
        if not group_dir.is_dir():
            continue

        print(f"  扫描 group: {group_dir.name}")
        all_scene_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])

        # 场景采样
        if scene_scale < 1.0:
            indices = np.arange(0, len(all_scene_dirs), 1 / scene_scale).astype(np.int32)
            selected_scenes = [all_scene_dirs[i] for i in indices if i < len(all_scene_dirs)]
        else:
            selected_scenes = all_scene_dirs

        for scene_dir in selected_scenes:
            # 检测是否有 trajectory_XX 子目录
            subdirs = [d for d in scene_dir.iterdir() if d.is_dir()]
            trajectory_dirs = [d for d in subdirs if d.name.startswith('trajectory_')]

            if trajectory_dirs:
                # 新格式：scene/trajectory_XX/data
                scene_episodes = 0
                for traj_dir in sorted(trajectory_dirs):
                    unit_result = _process_data_unit(traj_dir)
                    n = len(unit_result['trajectory_data_dir'])
                    if n > 0:
                        for key in index_data:
                            index_data[key].extend(unit_result[key])
                        scene_episodes += n
                if scene_episodes > 0:
                    print(f"    {scene_dir.name}: {scene_episodes} episodes ({len(trajectory_dirs)} trajectories)")
                    total_episodes += scene_episodes
            else:
                # 旧格式：scene/data（向后兼容）
                unit_result = _process_data_unit(scene_dir)
                n = len(unit_result['trajectory_data_dir'])
                if n > 0:
                    for key in index_data:
                        index_data[key].extend(unit_result[key])
                    print(f"    {scene_dir.name}: {n} episodes")
                    total_episodes += n

    print(f"\n总计扫描到 {total_episodes} 个 episodes")
    return index_data


def main():
    parser = argparse.ArgumentParser(
        description='生成 LeRobot 数据集预加载索引（NavDP/BridgeDP 兼容格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/dataset/generate_preload_index.py \\
      --root_dir /media/monika/TSD302/dataset/.../traj_data \\
      --output navdp_dataset_lerobot.json

  # 只采样 50% 的场景
  python scripts/dataset/generate_preload_index.py \\
      --root_dir /path/to/data \\
      --output preload.json \\
      --scene_scale 0.5
        """
    )

    parser.add_argument('--root_dir', type=str, required=True,
                        help='数据集根目录（包含 group 子目录的目录）')
    parser.add_argument('--output', type=str, default='navdp_dataset_lerobot.json',
                        help='输出索引文件路径（默认: navdp_dataset_lerobot.json）')
    parser.add_argument('--scene_scale', type=float, default=1.0,
                        help='场景采样比例，1.0=全部（默认: 1.0）')

    args = parser.parse_args()

    # 扫描数据集
    index_data = scan_lerobot_dataset(args.root_dir, args.scene_scale)

    total = len(index_data['trajectory_data_dir'])
    if total == 0:
        print("\n错误: 未找到任何有效的 episode 数据")
        print("请检查数据集目录结构，应为：")
        print("  root_dir/group/scene/trajectory_XX/data/chunk-000/")
        print("  root_dir/group/scene/trajectory_XX/meta/episodes_stats.jsonl")
        print("  root_dir/group/scene/trajectory_XX/videos/chunk-000/")
        sys.exit(1)

    # 保存索引
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"索引文件已生成: {output_path.absolute()}")
    print(f"总 episode 数: {total}")
    print(f"索引包含 4 个数组:")
    print(f"  trajectory_data_dir:   {len(index_data['trajectory_data_dir'])} 条")
    print(f"  trajectory_rgb_path:   {len(index_data['trajectory_rgb_path'])} 条")
    print(f"  trajectory_depth_path: {len(index_data['trajectory_depth_path'])} 条")
    print(f"  trajectory_afford_path:{len(index_data['trajectory_afford_path'])} 条")
    print(f"{'='*60}")

    # 使用提示
    print(f"\n使用方法:")
    print(f"  在训练配置 bridgedp.py 中设置:")
    print(f"    root_dir='{args.root_dir}'")
    print(f"    dataset_navdp='{output_path.absolute()}'")
    print(f"    preload=True")
    print(f"\n  或在 compute_sigma_base.py 中使用:")
    print(f"    python scripts/train/base_train/compute_sigma_base.py \\")
    print(f"        --root_dir '{args.root_dir}' \\")
    print(f"        --dataset_index '{output_path.absolute()}'")


if __name__ == '__main__':
    main()
