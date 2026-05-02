"""Bridge-DP LeRobot 数据集读取与预处理模块。

独立继承 ``torch.utils.data.Dataset``（与 NavDP_Base_Datset 使用相同基类），
仿照 ``NavDP_Base_Datset`` 的实现思路，全部数据加载方法独立实现。

与 NavDP 数据集的核心区别：
1. 动作空间：增量×4 → 绝对坐标 (x, y, θ)
2. 新增先验轨迹生成（含 30% 对抗训练）
3. 新增目标方位角 θ_g 计算
4. 返回字段：10 个 → 12 个（+prior_traj, theta_g）

参考：
    - internnav/dataset/navdp_lerobot_dataset.py（NavDP 数据集，不修改）
    - docs/Bridge-DP推导.md §4（先验轨迹与偏移补偿）
    - docs/Bridge-DP推导.md §5.4（对抗先验训练）
"""

import builtins
import json
import os
from datetime import datetime

import cv2
import jsonlines
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from tqdm import tqdm

original_print = builtins.print


def print(*args, **kwargs):
    """带时间戳的安全打印函数（仅 rank=0 输出）。"""
    try:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            original_print(f"[{timestamp}]", *args, **kwargs)
    except Exception:
        pass


builtins.print = print


class BridgeDP_Base_Dataset(Dataset):
    """Bridge-DP 基础数据集。

    独立继承 Dataset，仿照 NavDP_Base_Datset 实现全部数据加载逻辑。
    与 NavDP 的核心区别：动作空间为绝对坐标，新增先验轨迹和目标方位角。

    场景自检：
        1. 障碍物密集区域：rank_steps 优先采样困难片段 → 模型学会避障。
        2. 长直走廊：轨迹接近直线 → 绝对坐标直接表达 → 无增量累积误差。
        3. U-turn 场景：先验轨迹可能不准 → 30% 对抗训练保证鲁棒性。
    """

    def __init__(
        self,
        root_dirs,
        preload_path=False,
        memory_size=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        pixel_channel=7,
        action_dim=3,
        debug=False,
        preload=False,
        random_digit=False,
        prior_sample=False,
        sigma_base=1.0,
    ):
        """初始化数据集。仿照 NavDP_Base_Datset.__init__（L77-183）。

        Args:
            root_dirs: 轨迹数据根目录。
            preload_path: 预索引文件路径。
            memory_size: 历史记忆帧数。
            predict_size: 未来预测步数。
            batch_size: 批大小（用于日志）。
            image_size: 图像分辨率。
            scene_data_scale: 场景采样比例。
            trajectory_data_scale: 轨迹采样比例。
            pixel_channel: 像素目标通道数。
            action_dim: 动作维度。
            debug: 调试模式。
            preload: 是否使用预构建索引。
            random_digit: 是否随机采样时序步长。
            prior_sample: 是否启用障碍密度优先采样。
            sigma_base: Bridge-DP 基础方差常数。
        """
        self.dataset_dirs = np.array([p for p in os.listdir(root_dirs)])
        self.memory_size = memory_size
        self.image_size = image_size
        self.scene_scale_size = scene_data_scale
        self.trajectory_data_scale = trajectory_data_scale
        self.predict_size = predict_size
        self.action_dim = action_dim
        self.debug = debug
        self.sigma_base = sigma_base

        self.trajectory_data_dir = []
        self.trajectory_rgb_path = []
        self.trajectory_depth_path = []
        self.trajectory_afford_path = []
        self.random_digit = random_digit
        self.prior_sample = prior_sample
        self.pixel_channel = pixel_channel
        self.item_cnt = 0
        self.batch_size = batch_size
        self.batch_time_sum = 0.0
        self._last_time = None

        if preload is False:
            for group_dir in self.dataset_dirs:
                all_scene_dirs = np.array([p for p in os.listdir(os.path.join(root_dirs, group_dir))])
                select_scene_dirs = all_scene_dirs[
                    np.arange(0, all_scene_dirs.shape[0], 1 / self.scene_scale_size).astype(np.int32)
                ]
                for scene_dir in tqdm(select_scene_dirs):
                    chunk_name = os.listdir(os.path.join(root_dirs, group_dir, scene_dir, 'data'))[0]
                    data_dir = os.path.join(root_dirs, group_dir, scene_dir, f'data/{chunk_name}')
                    afford_dir = os.path.join(root_dirs, group_dir, scene_dir, 'meta/pointcloud.ply')
                    with jsonlines.open(
                        os.path.join(root_dirs, group_dir, scene_dir, 'meta/episodes_stats.jsonl'), 'r'
                    ) as reader:
                        episode_info = list(reader)
                    rgb_dir = os.path.join(
                        root_dirs, group_dir, scene_dir, f"videos/{chunk_name}/observation.images.rgb/"
                    )
                    rgb_paths = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
                    depth_dir = os.path.join(
                        root_dirs, group_dir, scene_dir, f"videos/{chunk_name}/observation.images.depth/"
                    )
                    depth_paths = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
                    data_paths = [os.path.join(data_dir, p) for p in sorted(os.listdir(data_dir))]
                    for episode_idx, episode in enumerate(episode_info):
                        image_start_index = episode['image_index']['min']
                        image_end_index = episode['image_index']['max']
                        episode_rgb_path = np.array(rgb_paths)[image_start_index : image_end_index + 1].tolist()
                        episode_depth_path = np.array(depth_paths)[image_start_index : image_end_index + 1].tolist()
                        try:
                            self.trajectory_data_dir.append(data_paths[episode_idx])
                            self.trajectory_rgb_path.append(episode_rgb_path)
                            self.trajectory_depth_path.append(episode_depth_path)
                            self.trajectory_afford_path.append(afford_dir)
                        except Exception as e:
                            print(f"Error processing episode {episode_idx}: {e}")

            save_dict = {
                'trajectory_data_dir': self.trajectory_data_dir,
                'trajectory_rgb_path': self.trajectory_rgb_path,
                'trajectory_depth_path': self.trajectory_depth_path,
                'trajectory_afford_path': self.trajectory_afford_path,
            }
            with open(preload_path, 'w') as f:
                json.dump(save_dict, f, indent=4)
            self.trajectory_data_dir = self.trajectory_data_dir * 50
            self.trajectory_rgb_path = self.trajectory_rgb_path * 50
            self.trajectory_depth_path = self.trajectory_depth_path * 50
            self.trajectory_afford_path = self.trajectory_afford_path * 50
        else:
            load_dict = json.load(open(preload_path, 'r'))
            self.trajectory_data_dir = load_dict['trajectory_data_dir'] * 50
            self.trajectory_rgb_path = load_dict['trajectory_rgb_path'] * 50
            self.trajectory_depth_path = load_dict['trajectory_depth_path'] * 50
            self.trajectory_afford_path = load_dict['trajectory_afford_path'] * 50

    def __len__(self):
        """返回数据集样本数。"""
        return len(self.trajectory_data_dir)

    def load_image(self, image_url):
        """读取 RGB 图像。仿照 NavDP L189-204。"""
        try:
            image = cv2.imread(image_url)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def load_depth(self, depth_url):
        """读取深度图。仿照 NavDP L206-221。"""
        try:
            depth = cv2.imread(depth_url, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
            return depth.astype(np.float32)
        except Exception as e:
            print(f"Error loading depth {depth_url}: {e}")
            return np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)

    def load_pointcloud(self, pcd_url):
        """读取点云。仿照 NavDP L223-233。"""
        pcd = o3d.io.read_point_cloud(pcd_url)
        points = np.asarray(pcd.points)
        return points

    def process_image(self, image_path):
        """图像增强：resize + normalize。仿照 NavDP L235-261。"""
        image = self.load_image(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        return image

    def process_depth(self, depth_path):
        """深度图增强。仿照 NavDP L263-287。"""
        depth = self.load_depth(depth_path)
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        if len(depth.shape) == 2:
            depth = depth[:, :, np.newaxis]
        max_depth = np.max(depth)
        if max_depth > 0:
            depth = depth / max_depth
        return depth.astype(np.float32)

    def process_data_parquet(self, index):
        """解析 Parquet 轨迹数据。仿照 NavDP L289-309。"""
        data_path = self.trajectory_data_dir[index]
        df = pd.read_parquet(data_path)
        camera_intrinsic = np.array(df['observation.camera_intrinsic'].iloc[0]).reshape(3, 3)
        base_extrinsic = np.array(df['observation.camera_extrinsic'].iloc[0]).reshape(4, 4)
        extrinsics = np.array([np.array(e).reshape(4, 4) for e in df['observation.camera_extrinsic']])
        trajectory_length = len(df)
        return camera_intrinsic, base_extrinsic, extrinsics, trajectory_length

    def process_obstacle_points(self, index):
        """障碍点处理。仿照 NavDP L311-332。"""
        afford_path = self.trajectory_afford_path[index]
        try:
            points = self.load_pointcloud(afford_path)
        except Exception:
            points = np.zeros((1, 3), dtype=np.float32)
        return points, afford_path

    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1):
        """构建历史帧记忆。仿照 NavDP L334-355。"""
        memory_images = []
        memory_index = []
        for i in range(self.memory_size):
            idx = max(0, start_step - (self.memory_size - 1 - i) * memory_digit)
            memory_images.append(self.process_image(rgb_paths[idx]))
            memory_index.append(idx)
        depth_image = self.process_depth(depth_paths[start_step])
        return np.array(memory_images), depth_image, memory_index

    def process_pixel_goal(self, image_url, target_point, camera_intrinsic, camera_extrinsic):
        """将局部目标点投影到图像平面。仿照 NavDP L357-412。"""
        try:
            image = self.process_image(image_url)
            target_3d = np.array([target_point[0], 0.0, target_point[1], 1.0])
            cam_point = camera_extrinsic @ target_3d
            if cam_point[2] <= 0:
                pixel_goal = np.zeros((self.image_size, self.image_size, 4), dtype=np.float32)
                return pixel_goal, 0.0
            pixel_coords = camera_intrinsic @ cam_point[:3]
            px = int(pixel_coords[0] / pixel_coords[2])
            py = int(pixel_coords[1] / pixel_coords[2])
            mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
            scale_x = self.image_size / image.shape[1] if image.shape[1] > 0 else 1.0
            scale_y = self.image_size / image.shape[0] if image.shape[0] > 0 else 1.0
            px_scaled = int(px * scale_x)
            py_scaled = int(py * scale_y)
            r = max(3, self.image_size // 30)
            if 0 <= px_scaled < self.image_size and 0 <= py_scaled < self.image_size:
                cv2.circle(mask[:, :, 0], (px_scaled, py_scaled), r, 1.0, -1)
                pixel_goal = np.concatenate([mask, image], axis=-1)
                return pixel_goal, 1.0
            else:
                pixel_goal = np.zeros((self.image_size, self.image_size, 4), dtype=np.float32)
                return pixel_goal, 0.0
        except Exception:
            pixel_goal = np.zeros((self.image_size, self.image_size, 4), dtype=np.float32)
            return pixel_goal, 0.0

    def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic):
        """计算相对位姿。仿照 NavDP L414-448。"""
        R_rel = R_base.T @ R_world
        T_rel = R_base.T @ (T_world - T_base)
        return R_rel, T_rel

    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic):
        """计算绝对位姿。仿照 NavDP L450-484。"""
        R_abs = R_base @ R_frame
        T_abs = R_base @ T_frame + T_base
        return R_abs, T_abs

    def xyz_to_xyt(self, xyz_actions, init_vector):
        """将 3D 坐标转为平面 (x, y, theta)。仿照 NavDP L486-505。"""
        xyt_actions = []
        for i in range(xyz_actions.shape[0]):
            x = xyz_actions[i, 0]
            y = xyz_actions[i, 2] if xyz_actions.shape[1] > 2 else 0.0
            if i == 0:
                theta = np.arctan2(init_vector[2], init_vector[0]) if np.linalg.norm(init_vector[[0, 2]]) > 1e-6 else 0.0
            else:
                dx = xyz_actions[i, 0] - xyz_actions[i - 1, 0]
                dz = xyz_actions[i, 2] - xyz_actions[i - 1, 2] if xyz_actions.shape[1] > 2 else 0.0
                theta = np.arctan2(dz, dx) if abs(dx) + abs(dz) > 1e-6 else (xyt_actions[-1][2] if xyt_actions else 0.0)
            xyt_actions.append([x, y, theta])
        return np.array(xyt_actions, dtype=np.float32)

    def process_actions(self, extrinsics, base_extrinsic, start_step, end_step, pred_digit=1):
        """处理动作轨迹（含旋转增强+样条插值）。仿照 NavDP L507-590。"""
        label_linear_pos = []
        for i in range(start_step, end_step + 1):
            f_ext = extrinsics[i]
            _, T = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3],
                f_ext[0:3, 0:3], f_ext[0:3, 3], base_extrinsic,
            )
            label_linear_pos.append(T)
        label_actions = np.array(label_linear_pos)

        # 轨迹增强：随机旋转 + 样条平滑
        rotate_yaw_angle = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotate_matrix = np.array([
            [np.cos(rotate_yaw_angle), -np.sin(rotate_yaw_angle)],
            [np.sin(rotate_yaw_angle), np.cos(rotate_yaw_angle)],
        ], np.float32)
        rotate_local_actions = np.matmul(rotate_matrix, label_actions[:, 0:2].T).T
        rotate_local_actions = np.stack(
            (rotate_local_actions[:, 0], rotate_local_actions[:, 1], np.zeros_like(rotate_local_actions[:, 0])),
            axis=-1
        )
        rotate_world_points = []
        for act in rotate_local_actions:
            _, w_act = self.absolute_pose(
                extrinsics[start_step, 0:3, 0:3], extrinsics[start_step, 0:3, 3],
                np.eye(3), act, base_extrinsic
            )
            rotate_world_points.append(w_act)
        rotate_world_points = np.array(rotate_world_points)
        origin_world_points = extrinsics[start_step:end_step + 1, 0:3, 3]
        mix_anchor_points = rotate_world_points

        t = np.linspace(0, 1, mix_anchor_points.shape[0])
        cs_x = CubicSpline(t, mix_anchor_points[:, 0])
        cs_y = CubicSpline(t, mix_anchor_points[:, 1])
        cs_z = CubicSpline(t, mix_anchor_points[:, 2])
        interpolate_nums = origin_world_points.shape[0]
        t_fine = np.linspace(0, 1, int(interpolate_nums))
        result_augment_points = np.stack((cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)), axis=-1)

        local_label_points = []
        local_augment_points = []
        for f_ext, g_ext in zip(origin_world_points, result_augment_points):
            _, Tf = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3],
                np.eye(3), f_ext, base_extrinsic
            )
            _, Tg = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3],
                np.eye(3), g_ext, base_extrinsic
            )
            local_label_points.append(Tf)
            local_augment_points.append(Tg)
        local_label_points = np.array(local_label_points)
        local_augment_points = np.array(local_augment_points)
        action_indexes = np.clip(
            np.arange(self.predict_size + 1) * pred_digit, 0, label_actions.shape[0] - 2
        )
        return local_label_points, local_augment_points, origin_world_points, result_augment_points, action_indexes

    def rank_steps(self, extrinsics, obstacle_points, pred_digit=4):
        """基于障碍物密度的概率采样。仿照 NavDP L592-633。"""
        points_score = []
        trajectory = extrinsics[:, 0:2, 3]
        for i in range(0, trajectory.shape[0] - 1):
            future_actions = trajectory[i:min(i + self.predict_size * pred_digit, trajectory.shape[0] - 1)]
            future_bound = [
                np.min(future_actions[:, 0]) - 1, np.min(future_actions[:, 1]) - 1,
                np.max(future_actions[:, 0]) + 1, np.max(future_actions[:, 1]) + 1,
            ]
            within = (
                (obstacle_points[:, 0] > future_bound[0]) & (obstacle_points[:, 1] > future_bound[1])
                & (obstacle_points[:, 0] < future_bound[2]) & (obstacle_points[:, 1] < future_bound[3])
            )
            points_score.append(np.sum(within))
        points_score = np.array(points_score) / (np.array(points_score).max() + 1e-8)
        probs = np.exp(points_score / 0.2) / np.sum(np.exp(points_score / 0.2))
        start_choice = np.random.choice(np.arange(probs.shape[0]), p=probs)
        target_candidates = np.arange(start_choice + 1, trajectory.shape[0])
        target_p = (target_candidates - start_choice) / ((target_candidates - start_choice).max() + 1e-8)
        target_p = np.exp(target_p / 0.2) / np.exp(target_p / 0.2).sum()
        target_choice = np.random.choice(target_candidates, p=target_p)
        return start_choice, target_choice

    def generate_prior_trajectory(self, pred_actions, point_goal):
        """生成先验轨迹（70%正确 + 30%对抗）。

        正确先验：从起点到目标的直线插值 + 小幅高斯噪声。
        对抗先验：随机旋转或缩放的错误轨迹。

        Args:
            pred_actions: 标签轨迹 (T, 3)，绝对坐标。
            point_goal: 目标位置 (3,)。

        Returns:
            prior_traj: 先验轨迹 (T, 3)。

        场景自检：
            1. 正确先验 + 静态场景 → VisualGate G高 → 先验有用。
            2. 对抗先验 (30%) → 网络学会不盲信先验。
            3. 长距离目标 → 直线插值偏差大 → 但方差允许偏离。
        """
        T = pred_actions.shape[0]
        if np.random.random() < 0.3:
            # 对抗先验：随机旋转标签轨迹 60-300 度
            angle = np.random.uniform(np.pi / 3, 5 * np.pi / 3)
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            prior = pred_actions.copy()
            prior[:, 0:2] = (rot @ pred_actions[:, 0:2].T).T
            # 随机缩放
            scale = np.random.uniform(0.3, 2.0)
            prior *= scale
        else:
            # 正确先验：起点到目标的直线插值 + 噪声
            start = np.zeros(3, dtype=np.float32)
            goal = point_goal.copy()
            t_interp = np.linspace(0, 1, T).reshape(-1, 1)
            prior = start * (1 - t_interp) + goal * t_interp
            noise_std = 0.05 * np.linalg.norm(goal - start)
            prior += np.random.randn(T, 3).astype(np.float32) * noise_std
        return prior.astype(np.float32)

    def __getitem__(self, index):
        """获取单条训练样本。

        与 NavDP __getitem__（L635-809）的核心区别：
        1. 不做差分×4，直接使用绝对 xyt 坐标
        2. 新增先验轨迹生成（含对抗训练）
        3. 新增目标方位角 theta_g 计算
        4. 返回 12 个字段（多出 prior_traj 和 theta_g）

        Returns:
            tuple: 12 个字段的元组。
        """
        import time

        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        (camera_intrinsic, trajectory_base_extrinsic,
         trajectory_extrinsics, trajectory_length) = self.process_data_parquet(index)

        trajectory_obstacle_points, _ = self.process_obstacle_points(index)

        if self.prior_sample:
            pixel_start_choice, target_choice = self.rank_steps(
                trajectory_extrinsics, trajectory_obstacle_points
            )
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        memory_images, depth_image, memory_index = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )
        (target_local_points, augment_local_points,
         target_world_points, augment_world_points,
         action_indexes) = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic,
            memory_start_choice, target_choice, pred_digit=pred_digit
        )

        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]

        # Critic 评分（与 NavDP 一致）
        if trajectory_obstacle_points.shape[0] != 0:
            pred_distance = (
                np.abs(target_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1).min(axis=-1)
            )
            augment_distance = (
                np.abs(augment_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1).min(axis=-1)
            )
            pred_critic = (
                -5.0 * (pred_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (pred_distance[action_indexes][1:] - pred_distance[action_indexes][:-1]).sum()
            )
            augment_critic = (
                -5.0 * (augment_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (augment_distance[action_indexes][1:] - augment_distance[action_indexes][:-1]).sum()
            )
        else:
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]
        image_goal = np.concatenate((
            self.process_image(self.trajectory_rgb_path[index][target_choice]),
            self.process_image(self.trajectory_rgb_path[index][memory_start_choice]),
        ), axis=-1)

        pixel_target_local_points, _, _, _, _ = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic,
            pixel_start_choice, target_choice, pred_digit=pred_digit
        )
        pixel_init_vector = pixel_target_local_points[1] - pixel_target_local_points[0]
        pixel_xyt_actions = self.xyz_to_xyt(pixel_target_local_points, pixel_init_vector)
        pixel_goal, pixel_flag = self.process_pixel_goal(
            self.trajectory_rgb_path[index][pixel_start_choice],
            pixel_xyt_actions[-1], camera_intrinsic, trajectory_base_extrinsic,
        )
        if self.pixel_channel == 7:
            pixel_goal = np.concatenate((pixel_goal, memory_images[-1]), axis=-1)

        # ====== Bridge-DP 核心差异点 ======
        # 1. 不做差分×4，直接使用绝对坐标（去掉起点保持 T 步）
        pred_actions = pred_actions[1:]   # (T_pred, 3) 绝对坐标
        augment_actions = augment_actions[1:]

        # 对齐动作维度
        pred_actions = np.pad(
            pred_actions, ((0, 0), (0, self.action_dim - pred_actions.shape[-1])),
            mode='constant', constant_values=0,
        )
        augment_actions = np.pad(
            augment_actions, ((0, 0), (0, self.action_dim - augment_actions.shape[-1])),
            mode='constant', constant_values=0,
        )

        # 2. 生成先验轨迹（含对抗训练）
        prior_traj = self.generate_prior_trajectory(pred_actions, point_goal)

        # 3. 计算目标方位角
        theta_g = np.arctan2(point_goal[1], point_goal[0]).astype(np.float32)

        # 日志
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(f'__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s')
            self.batch_time_sum = 0.0

        # 转换为 torch 张量
        point_goal = torch.tensor(point_goal, dtype=torch.float32)
        image_goal = torch.tensor(image_goal, dtype=torch.float32)
        pixel_goal = torch.tensor(pixel_goal, dtype=torch.float32)
        memory_images = torch.tensor(memory_images, dtype=torch.float32)
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        augment_actions = torch.tensor(augment_actions, dtype=torch.float32)
        pred_critic = torch.tensor(pred_critic, dtype=torch.float32)
        augment_critic = torch.tensor(augment_critic, dtype=torch.float32)
        prior_traj = torch.tensor(prior_traj, dtype=torch.float32)
        theta_g = torch.tensor(theta_g, dtype=torch.float32)

        return (
            point_goal,       # 0: (3,)
            image_goal,       # 1: (H, W, 6)
            pixel_goal,       # 2: (H, W, C)
            memory_images,    # 3: (mem, H, W, 3)
            depth_image,      # 4: (H, W, 1)
            pred_actions,     # 5: (T_pred, 3) 绝对坐标
            augment_actions,  # 6: (T_pred, 3) 绝对坐标
            pred_critic,      # 7: scalar
            augment_critic,   # 8: scalar
            float(pixel_flag),  # 9: float
            prior_traj,       # 10: (T_pred, 3) 先验轨迹
            theta_g,          # 11: scalar 目标方位角
        )


def bridgedp_collate_fn(batch):
    """Bridge-DP 数据集自定义拼接函数。

    将 __getitem__ 返回的 12 个字段拼接为 batch 字典。
    与 NavDP 的 navdp_collate_fn 对比：新增 batch_prior 和 batch_theta_g。
    """
    collated = {
        "batch_pg": torch.stack([item[0] for item in batch]),
        "batch_ig": torch.stack([item[1] for item in batch]),
        "batch_tg": torch.stack([item[2] for item in batch]),
        "batch_rgb": torch.stack([item[3] for item in batch]),
        "batch_depth": torch.stack([item[4] for item in batch]),
        "batch_labels": torch.stack([item[5] for item in batch]),
        "batch_augments": torch.stack([item[6] for item in batch]),
        "batch_label_critic": torch.stack([item[7] for item in batch]),
        "batch_augment_critic": torch.stack([item[8] for item in batch]),
        # pixel_flag (item[9]) 与 NavDP 一致，不进 collate
        "batch_prior": torch.stack([item[10] for item in batch]),
        "batch_theta_g": torch.stack([item[11] for item in batch]),
    }
    return collated
