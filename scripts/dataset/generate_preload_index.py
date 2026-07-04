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

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import jsonlines
except ImportError:
    jsonlines = None


INDEX_KEYS = (
    'trajectory_data_dir',
    'trajectory_rgb_path',
    'trajectory_depth_path',
    'trajectory_afford_path',
)


def _empty_index() -> dict:
    return {key: [] for key in INDEX_KEYS}


def _read_jsonl_records(path: Path) -> list:
    if jsonlines is not None:
        with jsonlines.open(str(path), 'r') as reader:
            return list(reader)

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


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
    result = _empty_index()

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
    episode_info = _read_jsonl_records(meta_path)

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
                    episode_rgb_path = rgb_paths[image_start_index : image_end_index + 1]
                    episode_depth_path = depth_paths[image_start_index : image_end_index + 1]
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
    index_data = _empty_index()
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


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _selected_scene_dirs(group_dir: Path, scene_scale: float) -> list[Path]:
    all_scene_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])
    if scene_scale < 1.0:
        indices = np.arange(0, len(all_scene_dirs), 1 / scene_scale).astype(np.int32)
        return [all_scene_dirs[i] for i in indices if i < len(all_scene_dirs)]
    return all_scene_dirs


def _collect_scan_units(root_dir: str | Path, scene_scale: float = 1.0) -> list[dict]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {root_path}")

    units = []
    for group_dir in sorted(root_path.iterdir()):
        if not group_dir.is_dir():
            continue

        for scene_dir in _selected_scene_dirs(group_dir, scene_scale):
            subdirs = [d for d in scene_dir.iterdir() if d.is_dir()]
            trajectory_dirs = sorted([d for d in subdirs if d.name.startswith('trajectory_')])
            unit_dirs = trajectory_dirs if trajectory_dirs else [scene_dir]
            for unit_dir in unit_dirs:
                unit_rel = unit_dir.relative_to(root_path).as_posix()
                units.append(
                    {
                        'unit_rel': unit_rel,
                        'unit_path': unit_dir,
                        'group': group_dir.name,
                        'scene': scene_dir.name,
                    }
                )
    return units


def _shard_path(work_dir: Path, unit_rel: str) -> Path:
    digest = hashlib.sha1(unit_rel.encode('utf-8')).hexdigest()
    return work_dir / 'shards' / f'{digest}.json'


def _load_completed_shard(work_dir: Path, unit: dict) -> dict | None:
    path = _shard_path(work_dir, unit['unit_rel'])
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get('unit_rel') != unit['unit_rel']:
        return None
    index = data.get('index')
    if not isinstance(index, dict) or any(not isinstance(index.get(key), list) for key in INDEX_KEYS):
        return None
    lengths = [len(index[key]) for key in INDEX_KEYS]
    if len(set(lengths)) != 1:
        return None
    return data


def _summarize_work(units: list[dict], work_dir: Path) -> dict:
    completed_unit_rels, total_episodes, last_completed = _scan_completed_shards(units, work_dir)
    return _make_resume_summary(
        units,
        completed_unit_rels=completed_unit_rels,
        total_episodes=total_episodes,
        last_completed=last_completed,
    )


def _scan_completed_shards(units: list[dict], work_dir: Path) -> tuple[set[str], int, str | None]:
    completed_unit_rels: set[str] = set()
    completed_units = 0
    total_episodes = 0
    last_completed = None
    for unit in units:
        shard = _load_completed_shard(work_dir, unit)
        if shard is None:
            continue
        completed_unit_rels.add(unit['unit_rel'])
        completed_units += 1
        last_completed = unit['unit_rel']
        total_episodes += int(shard.get('episodes', len(shard['index']['trajectory_data_dir'])))
    return completed_unit_rels, total_episodes, last_completed


def _first_uncompleted_unit(units: list[dict], completed_unit_rels: set[str], start_index: int = 0) -> str | None:
    for unit in units[start_index:]:
        if unit['unit_rel'] not in completed_unit_rels:
            return unit['unit_rel']
    return None


def _make_resume_summary(
    units: list[dict],
    *,
    completed_unit_rels: set[str],
    total_episodes: int,
    last_completed: str | None,
    next_unit: str | None = None,
) -> dict:
    completed_units = len(completed_unit_rels)
    complete = completed_units == len(units)
    if next_unit is None and not complete:
        next_unit = _first_uncompleted_unit(units, completed_unit_rels)
    return {
        'complete': complete,
        'total_units': len(units),
        'completed_units': completed_units,
        'total_episodes': total_episodes,
        'last_completed': last_completed,
        'next_unit': None if complete else next_unit,
    }


def _write_progress(
    work_dir: Path,
    *,
    root_dir: Path,
    output_path: Path,
    scene_scale: float,
    summary: dict,
    status: str,
    current_unit: str | None = None,
) -> None:
    progress = {
        'status': status,
        'root_dir': str(root_dir),
        'output': str(output_path),
        'work_dir': str(work_dir),
        'scene_scale': scene_scale,
        'current_unit': current_unit,
        **summary,
    }
    _atomic_write_json(work_dir / 'progress.json', progress)


def _write_unit_shard(work_dir: Path, unit: dict, unit_result: dict) -> int:
    episode_count = len(unit_result['trajectory_data_dir'])
    shard = {
        'unit_rel': unit['unit_rel'],
        'unit_path': str(unit['unit_path']),
        'group': unit['group'],
        'scene': unit['scene'],
        'episodes': episode_count,
        'index': unit_result,
    }
    _atomic_write_json(_shard_path(work_dir, unit['unit_rel']), shard)
    return episode_count


def _write_final_index_from_shards(units: list[dict], work_dir: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + '.tmp')
    counts = {key: 0 for key in INDEX_KEYS}

    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        for key_idx, key in enumerate(INDEX_KEYS):
            f.write(f'  "{key}": [')
            first_value = True
            for unit in units:
                shard = _load_completed_shard(work_dir, unit)
                if shard is None:
                    raise RuntimeError(f"缺少可恢复索引 shard: {unit['unit_rel']}")
                for value in shard['index'][key]:
                    if first_value:
                        f.write('\n')
                        first_value = False
                    else:
                        f.write(',\n')
                    f.write('    ')
                    json.dump(value, f, ensure_ascii=False)
                    counts[key] += 1
            if first_value:
                f.write(']')
            else:
                f.write('\n  ]')
            f.write(',\n' if key_idx < len(INDEX_KEYS) - 1 else '\n')
        f.write('}\n')

    os.replace(tmp_path, output_path)
    return counts


def generate_preload_index_resumable(
    root_dir: str | Path,
    output_path: str | Path,
    scene_scale: float = 1.0,
    work_dir: str | Path | None = None,
    *,
    status_only: bool = False,
    max_units: int | None = None,
) -> dict:
    """Generate preload_index.json with per-unit shards for resumable full scans.

    The final JSON keeps the legacy Bridge-DP/NavDP preload format, but
    intermediate shards make the expensive scan resumable at scene/trajectory
    granularity and avoid keeping the entire index in memory while scanning.
    """
    root_path = Path(root_dir)
    output_path = Path(output_path)
    work_dir = Path(work_dir) if work_dir is not None else Path(str(output_path) + '.work')
    (work_dir / 'shards').mkdir(parents=True, exist_ok=True)

    units = _collect_scan_units(root_path, scene_scale)
    completed_unit_rels, total_episodes, last_completed = _scan_completed_shards(units, work_dir)
    summary = _make_resume_summary(
        units,
        completed_unit_rels=completed_unit_rels,
        total_episodes=total_episodes,
        last_completed=last_completed,
    )
    _write_progress(
        work_dir,
        root_dir=root_path,
        output_path=output_path,
        scene_scale=scene_scale,
        summary=summary,
        status='complete' if summary['complete'] else 'pending',
    )
    if status_only:
        return summary

    processed_this_run = 0
    for unit_index, unit in enumerate(units):
        if unit['unit_rel'] in completed_unit_rels:
            continue

        current_summary = _make_resume_summary(
            units,
            completed_unit_rels=completed_unit_rels,
            total_episodes=total_episodes,
            last_completed=last_completed,
            next_unit=unit['unit_rel'],
        )
        _write_progress(
            work_dir,
            root_dir=root_path,
            output_path=output_path,
            scene_scale=scene_scale,
            summary=current_summary,
            status='running',
            current_unit=unit['unit_rel'],
        )

        print(f"  扫描 unit: {unit['unit_rel']}", flush=True)
        unit_result = _process_data_unit(unit['unit_path'])
        episode_count = _write_unit_shard(work_dir, unit, unit_result)
        completed_unit_rels.add(unit['unit_rel'])
        total_episodes += episode_count
        last_completed = unit['unit_rel']
        print(f"    {unit['unit_rel']}: {episode_count} episodes", flush=True)
        processed_this_run += 1

        next_unit = _first_uncompleted_unit(units, completed_unit_rels, unit_index + 1)
        summary = _make_resume_summary(
            units,
            completed_unit_rels=completed_unit_rels,
            total_episodes=total_episodes,
            last_completed=last_completed,
            next_unit=next_unit,
        )
        _write_progress(
            work_dir,
            root_dir=root_path,
            output_path=output_path,
            scene_scale=scene_scale,
            summary=summary,
            status='partial' if not summary['complete'] else 'merging',
        )

        if max_units is not None and processed_this_run >= max_units:
            return summary

    summary = _make_resume_summary(
        units,
        completed_unit_rels=completed_unit_rels,
        total_episodes=total_episodes,
        last_completed=last_completed,
    )
    if not summary['complete']:
        return summary

    counts = _write_final_index_from_shards(units, work_dir, output_path)
    total = counts['trajectory_data_dir']
    if total == 0:
        raise RuntimeError('未找到任何有效的 episode 数据')

    summary = _make_resume_summary(
        units,
        completed_unit_rels=completed_unit_rels,
        total_episodes=total_episodes,
        last_completed=last_completed,
    )
    _write_progress(
        work_dir,
        root_dir=root_path,
        output_path=output_path,
        scene_scale=scene_scale,
        summary=summary,
        status='complete',
    )
    return summary


def _print_resumable_status(summary: dict, work_dir: Path) -> None:
    print("可恢复 preload 生成状态")
    print(f"  work_dir:        {work_dir}")
    print(f"  complete:        {summary['complete']}")
    print(f"  units:           {summary['completed_units']} / {summary['total_units']}")
    print(f"  episodes:        {summary['total_episodes']}")
    print(f"  last_completed:  {summary['last_completed']}")
    print(f"  next_unit:       {summary['next_unit']}")


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

  # 全量数据推荐：可恢复、低内存扫描
  python scripts/dataset/generate_preload_index.py \\
      --root_dir /path/to/data \\
      --output preload.json \\
      --resume
        """
    )

    parser.add_argument('--root_dir', type=str, required=True,
                        help='数据集根目录（包含 group 子目录的目录）')
    parser.add_argument('--output', type=str, default='navdp_dataset_lerobot.json',
                        help='输出索引文件路径（默认: navdp_dataset_lerobot.json）')
    parser.add_argument('--scene_scale', type=float, default=1.0,
                        help='场景采样比例，1.0=全部（默认: 1.0）')
    parser.add_argument('--resume', action='store_true',
                        help='使用可恢复的分片扫描模式，适合全量 InternData-N1')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='可恢复扫描的工作目录（默认: <output>.work）')
    parser.add_argument('--status', action='store_true',
                        help='只打印可恢复扫描状态，不继续扫描')

    args = parser.parse_args()

    if args.resume or args.status:
        output_path = Path(args.output)
        work_dir = Path(args.work_dir) if args.work_dir else Path(str(output_path) + '.work')
        summary = generate_preload_index_resumable(
            root_dir=args.root_dir,
            output_path=output_path,
            scene_scale=args.scene_scale,
            work_dir=work_dir,
            status_only=args.status,
        )
        _print_resumable_status(summary, work_dir)
        if args.status:
            return
        if not summary['complete']:
            print("\n扫描未完成，再次执行同一条 --resume 命令会从 next_unit 继续。")
            sys.exit(2)
        total = summary['total_episodes']
        print(f"\n{'='*60}")
        print(f"索引文件已生成: {output_path.absolute()}")
        print(f"总 episode 数: {total}")
        print(f"可恢复工作目录: {work_dir.absolute()}")
        print(f"{'='*60}")
        return

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
