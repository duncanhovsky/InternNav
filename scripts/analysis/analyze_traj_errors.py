#!/usr/bin/env python3
"""
分析 traj_batches.jsonl 中的轨迹预测错误样本。

错误模式定义（以 Batch 3130 第4个样本为参考）：
1. zigzag：预测轨迹连续航点方向交替反转，而真值平滑
2. 大跨度：预测平均步长远大于真值
3. 轨迹不短：排除 GT 本身很短的样本
"""

import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def load_jsonl(path: str):
    """逐行加载 jsonl 文件"""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def traj_to_np(traj):
    """将 [[x,y,z], ...] 转为 numpy (N,3)"""
    return np.array(traj, dtype=np.float64)


def traj_total_length(pts: np.ndarray) -> float:
    """轨迹总弧长"""
    diffs = np.diff(pts, axis=0)  # (N-1, 3)
    step_lens = np.linalg.norm(diffs, axis=1)
    return float(np.sum(step_lens))


def traj_avg_step(pts: np.ndarray) -> float:
    """平均步长"""
    diffs = np.diff(pts, axis=0)
    step_lens = np.linalg.norm(diffs, axis=1)
    return float(np.mean(step_lens)) if len(step_lens) > 0 else 0.0


def count_zigzag(pts: np.ndarray) -> int:
    """
    计算 zigzag 次数：连续位移向量的余弦 < 0 计为一次 zigzag。
    """
    diffs = np.diff(pts, axis=0)  # (N-1, 3)
    count = 0
    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_sim = np.dot(v1, v2) / (n1 * n2)
        if cos_sim < 0:
            count += 1
    return count


def compute_ade(gt: np.ndarray, pred: np.ndarray) -> float:
    """Average Displacement Error"""
    return float(np.mean(np.linalg.norm(gt - pred, axis=1)))


def traj_curvature(pts: np.ndarray) -> float:
    """
    平均曲率近似：相邻位移向量夹角的均值 (弧度)
    """
    diffs = np.diff(pts, axis=0)
    angles = []
    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(math.acos(cos_val))
    return float(np.mean(angles)) if angles else 0.0


def compute_sample_metrics(gt_traj_list, pred_traj_list):
    """计算单个样本的所有指标"""
    gt = traj_to_np(gt_traj_list)
    pred = traj_to_np(pred_traj_list)

    gt_len = traj_total_length(gt)
    pred_len = traj_total_length(pred)
    gt_avg = traj_avg_step(gt)
    pred_avg = traj_avg_step(pred)
    gt_zz = count_zigzag(gt)
    pred_zz = count_zigzag(pred)
    ade = compute_ade(gt, pred)
    gt_curv = traj_curvature(gt)
    pred_curv = traj_curvature(pred)

    return {
        "gt_total_length": gt_len,
        "pred_total_length": pred_len,
        "gt_avg_step": gt_avg,
        "pred_avg_step": pred_avg,
        "gt_zigzag": gt_zz,
        "pred_zigzag": pred_zz,
        "ade": ade,
        "gt_curvature": gt_curv,
        "pred_curvature": pred_curv,
        "step_ratio": pred_avg / gt_avg if gt_avg > 1e-9 else float("inf"),
    }


# ──────────────────────────────────────────────
# 主分析
# ──────────────────────────────────────────────

def main():
    data_path = "checkpoints/bridgedp_train/logs/traj_batches.jsonl"
    if not os.path.exists(data_path):
        print(f"[ERROR] 数据文件不存在: {data_path}")
        sys.exit(1)

    records = load_jsonl(data_path)
    print(f"共加载 {len(records)} 条记录")
    print(f"batch_idx 范围: {records[0]['batch_idx']} ~ {records[-1]['batch_idx']}")
    print()

    # ─────────────────────────────────
    # Part 1: 分析 Batch 3130 第4个样本
    # ─────────────────────────────────
    ref_record = None
    for r in records:
        if r["batch_idx"] == 3130:
            ref_record = r
            break

    if ref_record is None:
        print("[ERROR] 未找到 batch_idx=3130")
        sys.exit(1)

    ref_sample = ref_record["samples"][3]  # index=3, 第4个
    ref_gt = ref_sample["gt_traj"]
    ref_pred = ref_sample["pred_traj"]

    print("=" * 70)
    print("Part 1: Batch 3130, Sample #4 (index=3) 详细分析")
    print("=" * 70)

    print(f"\nGT 轨迹 ({len(ref_gt)} 个航点):")
    for i, wp in enumerate(ref_gt):
        print(f"  [{i:2d}] x={wp[0]:+.6f}, y={wp[1]:+.6f}, z={wp[2]:+.6f}")

    print(f"\nPred 轨迹 ({len(ref_pred)} 个航点):")
    for i, wp in enumerate(ref_pred):
        print(f"  [{i:2d}] x={wp[0]:+.6f}, y={wp[1]:+.6f}, z={wp[2]:+.6f}")

    ref_metrics = compute_sample_metrics(ref_gt, ref_pred)

    print("\n--- 指标 ---")
    print(f"  GT 轨迹总长度:   {ref_metrics['gt_total_length']:.6f}")
    print(f"  Pred 轨迹总长度: {ref_metrics['pred_total_length']:.6f}")
    print(f"  GT 平均步长:     {ref_metrics['gt_avg_step']:.6f}")
    print(f"  Pred 平均步长:   {ref_metrics['pred_avg_step']:.6f}")
    print(f"  步长比 (Pred/GT): {ref_metrics['step_ratio']:.4f}")
    print(f"  GT zigzag 次数:  {ref_metrics['gt_zigzag']}")
    print(f"  Pred zigzag 次数: {ref_metrics['pred_zigzag']}")
    print(f"  ADE:             {ref_metrics['ade']:.6f}")
    print(f"  GT 平均曲率:     {ref_metrics['gt_curvature']:.6f} rad")
    print(f"  Pred 平均曲率:   {ref_metrics['pred_curvature']:.6f} rad")

    # ─────────────────────────────────
    # Part 2: 扫描所有样本，计算指标
    # ─────────────────────────────────
    print("\n" + "=" * 70)
    print("Part 2: 全量扫描 — 计算所有样本指标")
    print("=" * 70)

    all_metrics = []
    for rec in records:
        batch_idx = rec["batch_idx"]
        step = rec["step"]
        for si, sample in enumerate(rec["samples"]):
            m = compute_sample_metrics(sample["gt_traj"], sample["pred_traj"])
            m["batch_idx"] = batch_idx
            m["step"] = step
            m["sample_idx"] = si
            all_metrics.append(m)

    total_samples = len(all_metrics)
    print(f"总样本数: {total_samples}")

    # 计算 GT 总长度的统计量
    gt_lengths = np.array([m["gt_total_length"] for m in all_metrics])
    gt_len_median = float(np.median(gt_lengths))
    gt_len_mean = float(np.mean(gt_lengths))
    print(f"GT 轨迹总长度 — 中位数: {gt_len_median:.4f}, 均值: {gt_len_mean:.4f}, "
          f"min: {gt_lengths.min():.4f}, max: {gt_lengths.max():.4f}")

    ade_arr = np.array([m["ade"] for m in all_metrics])
    print(f"ADE — 中位数: {np.median(ade_arr):.4f}, 均值: {np.mean(ade_arr):.4f}, "
          f"min: {ade_arr.min():.4f}, max: {ade_arr.max():.4f}")

    step_ratios = np.array([m["step_ratio"] for m in all_metrics])
    print(f"步长比 — 中位数: {np.median(step_ratios):.4f}, 均值: {np.mean(step_ratios):.4f}")

    pred_zz_arr = np.array([m["pred_zigzag"] for m in all_metrics])
    gt_zz_arr = np.array([m["gt_zigzag"] for m in all_metrics])
    print(f"Pred zigzag — 中位数: {np.median(pred_zz_arr):.1f}, 均值: {np.mean(pred_zz_arr):.2f}")
    print(f"GT zigzag — 中位数: {np.median(gt_zz_arr):.1f}, 均值: {np.mean(gt_zz_arr):.2f}")

    # ─────────────────────────────────
    # Part 3: 基于参考样本确定阈值并筛选
    # ─────────────────────────────────
    print("\n" + "=" * 70)
    print("Part 3: 筛选类似错误样本")
    print("=" * 70)

    # 根据参考样本的指标设定阈值
    gt_len_threshold = gt_len_median * 0.5
    zigzag_extra_threshold = 5
    step_ratio_threshold = 1.5
    ade_threshold = ref_metrics["ade"] * 0.5  # 取参考 ADE 的 50% 作为下限

    print(f"\n筛选阈值（基于 Batch 3130 #4 参考样本）:")
    print(f"  GT 轨迹总长度 > {gt_len_threshold:.4f}  (中位数 {gt_len_median:.4f} 的 50%)")
    print(f"  Pred zigzag - GT zigzag > {zigzag_extra_threshold}")
    print(f"  步长比 (Pred/GT) > {step_ratio_threshold}")
    print(f"  ADE > {ade_threshold:.4f}  (参考 ADE {ref_metrics['ade']:.4f} 的 50%)")

    error_samples = []
    for m in all_metrics:
        if (m["gt_total_length"] > gt_len_threshold
                and (m["pred_zigzag"] - m["gt_zigzag"]) > zigzag_extra_threshold
                and m["step_ratio"] > step_ratio_threshold
                and m["ade"] > ade_threshold):
            error_samples.append(m)

    print(f"\n筛选结果: {len(error_samples)} / {total_samples} 个错误样本 "
          f"({100.0 * len(error_samples) / total_samples:.2f}%)")

    if len(error_samples) == 0:
        print("未找到符合条件的错误样本，尝试放宽阈值...")
        # 放宽阈值
        zigzag_extra_threshold = 3
        step_ratio_threshold = 1.3
        ade_threshold = ref_metrics["ade"] * 0.3

        print(f"  放宽后阈值:")
        print(f"    Pred zigzag - GT zigzag > {zigzag_extra_threshold}")
        print(f"    步长比 > {step_ratio_threshold}")
        print(f"    ADE > {ade_threshold:.4f}")

        for m in all_metrics:
            if (m["gt_total_length"] > gt_len_threshold
                    and (m["pred_zigzag"] - m["gt_zigzag"]) > zigzag_extra_threshold
                    and m["step_ratio"] > step_ratio_threshold
                    and m["ade"] > ade_threshold):
                error_samples.append(m)

        print(f"  放宽后结果: {len(error_samples)} / {total_samples} 个错误样本")

    if len(error_samples) == 0:
        print("\n即使放宽阈值也未找到，跳过后续分析。")
        return

    # ─────────────────────────────────
    # 训练过程中的分布
    # ─────────────────────────────────
    print("\n--- 错误样本在训练过程中的分布 ---")

    batch_indices = sorted(set(m["batch_idx"] for m in all_metrics))
    min_batch = min(batch_indices)
    max_batch = max(batch_indices)
    mid_batch = (min_batch + max_batch) / 2

    early = [m for m in error_samples if m["batch_idx"] <= mid_batch]
    late = [m for m in error_samples if m["batch_idx"] > mid_batch]
    print(f"  训练前半段 (batch_idx <= {mid_batch:.0f}): {len(early)} 个")
    print(f"  训练后半段 (batch_idx > {mid_batch:.0f}):  {len(late)} 个")

    # 按 batch_idx 分桶统计
    n_bins = 10
    bin_edges = np.linspace(min_batch, max_batch + 1, n_bins + 1)
    bin_counts = [0] * n_bins
    for m in error_samples:
        for bi in range(n_bins):
            if bin_edges[bi] <= m["batch_idx"] < bin_edges[bi + 1]:
                bin_counts[bi] += 1
                break

    print(f"\n  按 {n_bins} 个区间统计:")
    for bi in range(n_bins):
        lo = int(bin_edges[bi])
        hi = int(bin_edges[bi + 1])
        bar = "█" * bin_counts[bi]
        print(f"    [{lo:5d}, {hi:5d}): {bin_counts[bi]:3d}  {bar}")

    # ─────────────────────────────────
    # 错误样本 GT 轨迹的共同特征
    # ─────────────────────────────────
    print("\n--- 错误样本的 GT 轨迹共同特征 ---")

    err_gt_lens = np.array([m["gt_total_length"] for m in error_samples])
    err_gt_curvs = np.array([m["gt_curvature"] for m in error_samples])
    err_gt_steps = np.array([m["gt_avg_step"] for m in error_samples])
    err_gt_zz = np.array([m["gt_zigzag"] for m in error_samples])
    err_ade = np.array([m["ade"] for m in error_samples])
    err_pred_zz = np.array([m["pred_zigzag"] for m in error_samples])
    err_step_ratio = np.array([m["step_ratio"] for m in error_samples])

    # 对比全量
    all_gt_curvs = np.array([m["gt_curvature"] for m in all_metrics])

    print(f"  GT 总长度:  均值={err_gt_lens.mean():.4f}, 中位={np.median(err_gt_lens):.4f}, "
          f"std={err_gt_lens.std():.4f}")
    print(f"             (全量: 均值={gt_len_mean:.4f}, 中位={gt_len_median:.4f})")

    print(f"  GT 平均曲率: 均值={err_gt_curvs.mean():.4f}, 中位={np.median(err_gt_curvs):.4f}")
    print(f"             (全量: 均值={all_gt_curvs.mean():.4f}, 中位={np.median(all_gt_curvs):.4f})")

    print(f"  GT 平均步长: 均值={err_gt_steps.mean():.6f}, 中位={np.median(err_gt_steps):.6f}")
    print(f"  GT zigzag:  均值={err_gt_zz.mean():.2f}, 中位={np.median(err_gt_zz):.1f}")
    print(f"  Pred zigzag: 均值={err_pred_zz.mean():.2f}, 中位={np.median(err_pred_zz):.1f}")
    print(f"  ADE:        均值={err_ade.mean():.4f}, 中位={np.median(err_ade):.4f}")
    print(f"  步长比:     均值={err_step_ratio.mean():.4f}, 中位={np.median(err_step_ratio):.4f}")

    # ─────────────────────────────────
    # 打印部分错误样本
    # ─────────────────────────────────
    print("\n--- 部分错误样本列表 (前 20 个) ---")
    print(f"{'batch_idx':>10} {'sample':>6} {'gt_len':>8} {'pred_len':>9} "
          f"{'step_ratio':>10} {'gt_zz':>5} {'pred_zz':>7} {'ADE':>8} {'gt_curv':>8}")
    print("-" * 85)
    for m in error_samples[:20]:
        print(f"{m['batch_idx']:>10} {m['sample_idx']:>6} {m['gt_total_length']:>8.4f} "
              f"{m['pred_total_length']:>9.4f} {m['step_ratio']:>10.4f} "
              f"{m['gt_zigzag']:>5} {m['pred_zigzag']:>7} {m['ade']:>8.4f} "
              f"{m['gt_curvature']:>8.4f}")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
