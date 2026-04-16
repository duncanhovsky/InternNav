import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from internnav.dataset.flownav_dyn_lerobot_dataset import FlowNav_Dyn_Lerobot_Dataset, flownav_dyn_collate_fn
from internnav.dataset.flownav_lerobot_dataset import FlowNav_Base_Datset, flownav_collate_fn
from internnav.model.encoder.dyn_module import DynModuleConfig, FlowNavDynamicsRuntime, PointCloudFrame, PoseFrame
from scripts.train.base_train.configs import flownav_dyn_exp_cfg, flownav_static_exp_cfg


@dataclass
class VisualizerArgs:
    model_name: str = "flownav_static"
    num_samples: int = 4
    batch_size: int = 2
    output_dir: str = "checkpoints/flownav_intermediate_vis"
    point_stride: int = 4
    max_points_plot: int = 30000
    device: str = "cuda:0"


def _safe_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _depth_to_point_cloud(depth_m: np.ndarray, intrinsic: np.ndarray, stride: int = 2) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    h, w = d.shape
    v = np.arange(0, h, stride, dtype=np.float32)
    u = np.arange(0, w, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    zz = d[::stride, ::stride]

    valid = np.isfinite(zz) & (zz > 0.05)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = zz[valid]
    x = (uu[valid] - cx) * z / max(fx, 1e-6)
    y = (vv[valid] - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _apply_depth_preprocess_to_intrinsic(intrinsic: np.ndarray, preprocess_meta: Optional[np.ndarray]) -> np.ndarray:
    k = np.asarray(intrinsic, dtype=np.float32).copy()
    if preprocess_meta is None:
        return k

    m = np.asarray(preprocess_meta, dtype=np.float32).reshape(-1)
    if m.shape[0] < 10:
        return k

    scale_x, scale_y = float(m[0]), float(m[1])
    pad_left, pad_top = float(m[2]), float(m[3])
    crop_x0, crop_y0 = float(m[4]), float(m[5])
    final_scale_x, final_scale_y = float(m[8]), float(m[9])

    k[0, 0] = k[0, 0] * scale_x
    k[1, 1] = k[1, 1] * scale_y
    k[0, 2] = k[0, 2] * scale_x + pad_left
    k[1, 2] = k[1, 2] * scale_y + pad_top

    k[0, 2] = k[0, 2] - crop_x0
    k[1, 2] = k[1, 2] - crop_y0

    k[0, 0] = k[0, 0] * final_scale_x
    k[1, 1] = k[1, 1] * final_scale_y
    k[0, 2] = k[0, 2] * final_scale_x
    k[1, 2] = k[1, 2] * final_scale_y
    return k


def _resolve_dynamic_voxels_for_batch(batch: Dict[str, torch.Tensor], cfg, device: torch.device) -> torch.Tensor:
    if "batch_dynamic_voxels" in batch:
        return batch["batch_dynamic_voxels"].to(device)

    required = [
        "batch_depth_raw_m_hist",
        "batch_pose_world_hist",
        "batch_timestamp_hist_s",
        "batch_camera_intrinsic",
    ]
    for key in required:
        if key not in batch:
            raise KeyError(f"Missing key '{key}' for dyn_module fallback visualization.")

    depth_hist = _safe_to_numpy(batch["batch_depth_raw_m_hist"])
    pose_hist = _safe_to_numpy(batch["batch_pose_world_hist"])
    ts_hist = _safe_to_numpy(batch["batch_timestamp_hist_s"])
    intr_hist = _safe_to_numpy(batch["batch_camera_intrinsic"])
    preprocess_meta_hist = None
    if "batch_depth_preprocess_meta" in batch:
        preprocess_meta_hist = _safe_to_numpy(batch["batch_depth_preprocess_meta"])

    dyn_cfg = DynModuleConfig()
    dyn_cfg.device = str(device)

    voxels = []
    for b in range(depth_hist.shape[0]):
        runtime = FlowNavDynamicsRuntime(dyn_cfg)
        for t in range(depth_hist.shape[1]):
            meta_bt = None if preprocess_meta_hist is None else preprocess_meta_hist[b, t]
            k_bt = _apply_depth_preprocess_to_intrinsic(intr_hist[b], meta_bt)
            pts = _depth_to_point_cloud(depth_hist[b, t], k_bt, stride=2)
            point_frame = PointCloudFrame(points_xyz=pts, stamp=float(ts_hist[b, t]))
            pose_frame = PoseFrame(t_world_ego=np.asarray(pose_hist[b, t], dtype=np.float32), stamp=float(ts_hist[b, t]))
            runtime.ingest(point_frame, pose_frame)
        voxels.append(runtime.get_dynamic_voxels(batch_size=1, device=device, dtype=torch.float32))

    return torch.cat(voxels, dim=0)


def _sample_from_batch(batch: Dict[str, torch.Tensor], sample_idx: int) -> Dict[str, np.ndarray]:
    sample = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] > sample_idx:
            sample[key] = value[sample_idx].detach().cpu().numpy()
    return sample


def _plot_ego_pose(sample: Dict[str, np.ndarray], out_file: Path) -> bool:
    pose = sample.get("batch_pose_world_hist", None)
    if pose is None and "batch_odom_pose" in sample and sample["batch_odom_pose"].ndim >= 3:
        pose = sample["batch_odom_pose"]
    if pose is None or pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        return False

    xy = pose[:, :2, 3]
    heading = pose[:, :2, 0]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.plot(xy[:, 0], xy[:, 1], "-o", markersize=3, linewidth=1.5)
    ax.quiver(xy[:, 0], xy[:, 1], heading[:, 0], heading[:, 1], angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ax.set_title("Ego Pose History (world frame)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return True


def _plot_history_pointcloud(sample: Dict[str, np.ndarray], out_file: Path, point_stride: int, max_points_plot: int) -> bool:
    if "batch_depth_raw_m_hist" not in sample or "batch_camera_intrinsic" not in sample:
        return False

    depth_hist = sample["batch_depth_raw_m_hist"]
    intrinsic = sample["batch_camera_intrinsic"]
    pose_hist = sample.get("batch_pose_world_hist", None)
    preprocess_meta_hist = sample.get("batch_depth_preprocess_meta", None)

    if depth_hist.ndim < 4:
        return False

    all_points = []
    all_t = []
    t_len = depth_hist.shape[0]

    for t in range(t_len):
        meta_t = None if preprocess_meta_hist is None else preprocess_meta_hist[t]
        k_t = _apply_depth_preprocess_to_intrinsic(intrinsic, meta_t)
        pts_cam = _depth_to_point_cloud(depth_hist[t], k_t, stride=point_stride)
        if pts_cam.shape[0] == 0:
            continue

        if pose_hist is not None and pose_hist.ndim == 3 and pose_hist.shape[-2:] == (4, 4):
            r = pose_hist[t, :3, :3]
            tr = pose_hist[t, :3, 3]
            pts_world = pts_cam @ r.T + tr[None, :]
        else:
            pts_world = pts_cam

        all_points.append(pts_world)
        all_t.append(np.full((pts_world.shape[0],), t, dtype=np.int32))

    if len(all_points) == 0:
        return False

    points = np.concatenate(all_points, axis=0)
    steps = np.concatenate(all_t, axis=0)

    if points.shape[0] > max_points_plot:
        idx = np.random.choice(points.shape[0], size=max_points_plot, replace=False)
        points = points[idx]
        steps = steps[idx]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    sc = ax.scatter(points[:, 0], points[:, 1], c=steps, s=1.5, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, ax=ax, label="history frame index")
    ax.set_title("History Point Cloud BEV Projection")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return True


def _figure_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def _build_dynamic_frames(dynamic_voxels_t_cxyz: np.ndarray):
    # 输入形状：(T, C=4, X, Y, Z)
    if dynamic_voxels_t_cxyz.ndim != 5 or dynamic_voxels_t_cxyz.shape[1] != 4:
        return []

    t_len = dynamic_voxels_t_cxyz.shape[0]
    frames = []
    for t in range(t_len):
        occ = dynamic_voxels_t_cxyz[t, 0]  # (X,Y,Z)
        vx = dynamic_voxels_t_cxyz[t, 1]
        vy = dynamic_voxels_t_cxyz[t, 2]

        occ_bev = occ.max(axis=-1)  # (X,Y)
        occ_sum = occ.sum(axis=-1) + 1e-6
        vx_bev = (vx * occ).sum(axis=-1) / occ_sum
        vy_bev = (vy * occ).sum(axis=-1) / occ_sum

        gx, gy = np.meshgrid(np.arange(occ_bev.shape[1]), np.arange(occ_bev.shape[0]))
        stride = max(1, int(max(occ_bev.shape[:2]) / 20))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=120)

        ax0 = axes[0]
        im = ax0.imshow(occ_bev.T, origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        ax0.set_title(f"Occ BEV @ t={t}")
        ax0.set_xlabel("x voxel")
        ax0.set_ylabel("y voxel")

        ax1 = axes[1]
        ax1.imshow(occ_bev.T, origin="lower", cmap="gray", alpha=0.35)
        ax1.quiver(
            gx[::stride, ::stride],
            gy[::stride, ::stride],
            vx_bev[::stride, ::stride].T,
            vy_bev[::stride, ::stride].T,
            color="cyan",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
        )
        ax1.set_title(f"Future Velocity Field (vx, vy) @ t={t}")
        ax1.set_xlabel("x voxel")
        ax1.set_ylabel("y voxel")

        fig.tight_layout()
        frames.append(_figure_to_rgb(fig))
        plt.close(fig)

    return frames


def _save_dynamic_voxel_vis(sample: Dict[str, np.ndarray], out_png: Path, out_gif: Path) -> bool:
    dynamic_voxels = sample.get("batch_dynamic_voxels", None)
    if dynamic_voxels is None:
        return False
    if dynamic_voxels.ndim != 6:
        return False

    vox_t = dynamic_voxels  # (T,C,X,Y,Z)
    occ0 = vox_t[0, 0].max(axis=-1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    im = ax.imshow(occ0.T, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Future 4D Motion Field: Occupancy BEV at t=0")
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    frames = _build_dynamic_frames(vox_t)
    if len(frames) > 0:
        imageio.mimsave(out_gif, frames, fps=2)
    return True


def _build_dataset_and_loader(args: VisualizerArgs):
    if args.model_name == "flownav_static":
        cfg = flownav_static_exp_cfg
        dataset = FlowNav_Base_Datset(
            root_dirs=cfg.il.root_dir,
            preload_path=cfg.il.dataset_flownav,
            memory_size=cfg.il.memory_size,
            history_frames=cfg.il.history_frames,
            predict_frames=cfg.il.predict_frames,
            predict_size=cfg.il.predict_size,
            batch_size=cfg.il.batch_size,
            image_size=cfg.il.image_size,
            scene_data_scale=cfg.il.scene_scale,
            pixel_channel=cfg.il.pixel_channel,
            action_dim=cfg.il.action_dim,
            fallback_fps=cfg.il.fallback_fps,
            preload=cfg.il.preload,
            random_digit=cfg.il.random_digit,
            prior_sample=cfg.il.prior_sample,
        )
        collate_fn = flownav_collate_fn
    elif args.model_name == "flownav_dyn":
        cfg = flownav_dyn_exp_cfg
        dataset = FlowNav_Dyn_Lerobot_Dataset(
            root_dirs=cfg.il.root_dir,
            preload_path=cfg.il.dataset_flownav,
            memory_size=cfg.il.memory_size,
            history_frames=cfg.il.history_frames,
            predict_frames=cfg.il.predict_frames,
            predict_size=cfg.il.predict_size,
            batch_size=cfg.il.batch_size,
            image_size=cfg.il.image_size,
            scene_data_scale=cfg.il.scene_scale,
            pixel_channel=cfg.il.pixel_channel,
            action_dim=cfg.il.action_dim,
            fallback_fps=cfg.il.fallback_fps,
            preload=cfg.il.preload,
            random_digit=cfg.il.random_digit,
            prior_sample=cfg.il.prior_sample,
            dynamic_time_tolerance_ns=cfg.il.dynamic_time_tolerance_ns,
            dynamic_grid_shape=cfg.il.dynamic_grid_shape,
            dynamic_grid_resolution=cfg.il.dynamic_grid_resolution,
            dynamic_grid_z_min=cfg.il.dynamic_grid_z_min,
            use_cached_dynamic_voxels=cfg.il.use_cached_dynamic_voxels,
            use_cached_est_dynamic_voxels=cfg.il.use_cached_est_dynamic_voxels,
            est_dynamic_voxel_subdir=cfg.il.est_dynamic_voxel_subdir,
            est_voxel_ratio=cfg.il.est_voxel_ratio,
            est_voxel_seed=cfg.il.est_voxel_seed,
            use_cached_pred_critic=cfg.il.use_cached_pred_critic,
            dynamic_weight=cfg.il.dynamic_weight,
            static_weight=cfg.il.static_weight,
            near_threshold=cfg.il.near_threshold,
        )
        collate_fn = flownav_dyn_collate_fn
    else:
        raise ValueError("model_name only supports flownav_static or flownav_dyn.")

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return cfg, dataset, loader


def run(args: VisualizerArgs):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if ("cuda" in args.device and torch.cuda.is_available()) else "cpu")
    cfg, dataset, loader = _build_dataset_and_loader(args)

    print("=" * 60)
    print("FlowNav 中间结果可视化预检")
    print(f"model_name: {args.model_name}")
    print(f"dataset_size: {len(dataset)}")
    print(f"num_samples_to_visualize: {args.num_samples}")
    print(f"output_dir: {out_dir}")
    print("=" * 60)

    saved = 0
    for batch_idx, batch in enumerate(loader):
        try:
            dynamic_voxels = _resolve_dynamic_voxels_for_batch(batch, cfg, device=device).detach().cpu()
            batch["batch_dynamic_voxels"] = dynamic_voxels
        except Exception as exc:
            print(f"[WARN] batch {batch_idx} dynamic voxel resolve failed: {exc}")
            continue

        bsz = int(batch["batch_pg"].shape[0])
        for i in range(bsz):
            if saved >= args.num_samples:
                break

            sample_dir = out_dir / f"sample_{saved:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            sample = _sample_from_batch(batch, i)

            pose_ok = _plot_ego_pose(sample, sample_dir / "ego_pose_history.png")
            pcd_ok = _plot_history_pointcloud(
                sample,
                sample_dir / "history_pointcloud_bev.png",
                point_stride=max(1, int(args.point_stride)),
                max_points_plot=max(1000, int(args.max_points_plot)),
            )
            dyn_ok = _save_dynamic_voxel_vis(
                sample,
                sample_dir / "future_4d_occ_t0.png",
                sample_dir / "future_4d_motion.gif",
            )

            # 保存原始中间张量，便于后续排查与自定义可视化
            npz_payload = {}
            for key in [
                "batch_pose_world_hist",
                "batch_timestamp_hist_s",
                "batch_camera_intrinsic",
                "batch_depth_raw_m_hist",
                "batch_depth_preprocess_meta",
                "batch_dynamic_voxels",
                "batch_odom_pose",
                "batch_odom_delta",
            ]:
                if key in sample:
                    npz_payload[key] = sample[key]
            np.savez_compressed(sample_dir / "intermediate_tensors.npz", **npz_payload)

            with open(sample_dir / "summary.txt", "w", encoding="utf-8") as f:
                f.write(f"model_name={args.model_name}\n")
                f.write(f"sample_global_index={saved}\n")
                f.write(f"source_batch_index={batch_idx}\n")
                f.write(f"pose_vis={pose_ok}\n")
                f.write(f"pointcloud_vis={pcd_ok}\n")
                f.write(f"dynamic_voxel_vis={dyn_ok}\n")
                if "batch_dynamic_voxels" in sample:
                    f.write(f"dynamic_voxels_shape={sample['batch_dynamic_voxels'].shape}\n")

            print(
                f"[OK] sample_{saved:03d} saved | "
                f"pose={pose_ok}, pointcloud={pcd_ok}, dyn_voxel={dyn_ok}"
            )
            saved += 1

        if saved >= args.num_samples:
            break

    if saved == 0:
        print("[ERROR] No sample visualization generated. Please check dataset paths/config.")
    else:
        print(f"[DONE] Generated {saved} sample visualizations at: {out_dir}")


def parse_args() -> VisualizerArgs:
    parser = argparse.ArgumentParser(description="FlowNav 中间结果可视化预检脚本")
    parser.add_argument("--model_name", type=str, default="flownav_static", choices=["flownav_static", "flownav_dyn"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="checkpoints/flownav_intermediate_vis")
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument("--max_points_plot", type=int, default=30000)
    parser.add_argument("--device", type=str, default="cuda:0")
    ns = parser.parse_args()
    return VisualizerArgs(
        model_name=ns.model_name,
        num_samples=ns.num_samples,
        batch_size=ns.batch_size,
        output_dir=ns.output_dir,
        point_stride=ns.point_stride,
        max_points_plot=ns.max_points_plot,
        device=ns.device,
    )


if __name__ == "__main__":
    run(parse_args())
