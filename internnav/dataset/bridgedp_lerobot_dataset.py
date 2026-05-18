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

        # ── 动作空间归一化参数 ──────────────────────────────────────────
        # 将绝对坐标从 [0, ~10m] 归一化到 [-2, 2]，使训练目标尺度与 NavDP 的
        # 噪声预测目标 ε ~ N(0,1) 对齐。详见 TRAINING_CONVERGENCE_ANALYSIS.md。
        self.action_scale_xy = 5.0    # x,y 分量除以此值
        self.action_scale_theta = 3.14159  # θ 分量除以 π

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
                    scene_path = os.path.join(root_dirs, group_dir, scene_dir)
                    # 检查是否有 trajectory_XX 子目录
                    subdirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
                    trajectory_dirs = [d for d in subdirs if d.startswith('trajectory_')]
                    
                    if trajectory_dirs:
                        # 新格式：scene/trajectory_XX/data
                        # 每个 trajectory_XX 是独立的单 episode 数据单元
                        for traj_dir in trajectory_dirs:
                            traj_path = os.path.join(scene_path, traj_dir)
                            try:
                                data_subdir = os.path.join(traj_path, 'data')
                                if not os.path.isdir(data_subdir):
                                    continue
                                chunk_name = os.listdir(data_subdir)[0]
                                data_dir = os.path.join(traj_path, f'data/{chunk_name}')
                                # pointcloud.ply 可能在 meta/ 或 data/chunk-000/ 下
                                afford_ply = os.path.join(traj_path, 'meta/pointcloud.ply')
                                if not os.path.exists(afford_ply):
                                    afford_ply = os.path.join(data_dir, 'path.ply')
                                rgb_dir = os.path.join(traj_path, f"videos/{chunk_name}/observation.images.rgb/")
                                if not os.path.isdir(rgb_dir):
                                    continue
                                rgb_paths = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
                                depth_dir = os.path.join(traj_path, f"videos/{chunk_name}/observation.images.depth/")
                                depth_paths = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))] if os.path.isdir(depth_dir) else []
                                data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
                                data_paths = [os.path.join(data_dir, p) for p in data_files]
                                if not data_paths or not rgb_paths:
                                    continue
                                # 新格式每个 trajectory 只有 1 个 episode，直接使用全部帧
                                self.trajectory_data_dir.append(data_paths[0])
                                self.trajectory_rgb_path.append(rgb_paths)
                                self.trajectory_depth_path.append(depth_paths)
                                self.trajectory_afford_path.append(afford_ply)
                            except Exception as e:
                                print(f"Error processing {traj_dir}: {e}")
                    else:
                        # 旧格式：scene/data（向后兼容，使用 image_index 切分）
                        try:
                            chunk_name = os.listdir(os.path.join(scene_path, 'data'))[0]
                            data_dir = os.path.join(scene_path, f'data/{chunk_name}')
                            afford_dir = os.path.join(scene_path, 'meta/pointcloud.ply')
                            with jsonlines.open(os.path.join(scene_path, 'meta/episodes_stats.jsonl'), 'r') as reader:
                                episode_info = list(reader)
                            rgb_dir = os.path.join(scene_path, f"videos/{chunk_name}/observation.images.rgb/")
                            rgb_paths = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
                            depth_dir = os.path.join(scene_path, f"videos/{chunk_name}/observation.images.depth/")
                            depth_paths = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
                            data_paths = [os.path.join(data_dir, p) for p in sorted(os.listdir(data_dir))]
                            for episode_idx, episode in enumerate(episode_info):
                                if 'image_index' in episode:
                                    image_start_index = episode['image_index']['min']
                                    image_end_index = episode['image_index']['max']
                                    episode_rgb_path = np.array(rgb_paths)[image_start_index : image_end_index + 1].tolist()
                                    episode_depth_path = np.array(depth_paths)[image_start_index : image_end_index + 1].tolist()
                                else:
                                    # 没有 image_index 字段，使用全部帧
                                    episode_rgb_path = rgb_paths
                                    episode_depth_path = depth_paths
                                try:
                                    self.trajectory_data_dir.append(data_paths[episode_idx])
                                    self.trajectory_rgb_path.append(episode_rgb_path)
                                    self.trajectory_depth_path.append(episode_depth_path)
                                    self.trajectory_afford_path.append(afford_dir)
                                except Exception as e:
                                    print(f"Error processing episode {episode_idx}: {e}")
                        except Exception as e:
                            print(f"Error processing scene {scene_dir}: {e}")

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
        """读取 RGB 图像。与 NavDP 一致，使用 PIL 读取。"""
        try:
            image = Image.open(image_url)
            image = np.array(image, np.uint8)
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return image

    def load_depth(self, depth_url):
        """读取深度图（16-bit PNG）。与 NavDP 一致，使用 PIL 读取 uint16。"""
        try:
            depth = Image.open(depth_url)
            depth = np.array(depth, np.uint16)
        except Exception as e:
            print(f"Error loading depth {depth_url}: {e}")
            depth = np.zeros((self.image_size, self.image_size), dtype=np.uint16)
        return depth

    def load_pointcloud(self, pcd_url):
        """读取点云。与 NavDP 一致，返回 open3d PointCloud 对象。"""
        pcd = o3d.io.read_point_cloud(pcd_url)
        return pcd

    def process_image(self, image_path):
        """预处理 RGB 图像到统一分辨率并归一化。与 NavDP 一致。

        处理流程：
        1. 等比例缩放到最长边为 image_size。
        2. 居中零填充至正方形。
        3. 再次 resize 到精确尺寸。
        4. 转为 float32 并归一化到 [0, 1]。
        """
        image = self.load_image(image_path)
        H, W, C = image.shape
        prop = self.image_size / max(H, W)
        image = cv2.resize(image, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - image.shape[1]) // 2, 0)
        pad_height = max((self.image_size - image.shape[0]) // 2, 0)
        pad_image = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0
        )
        image = cv2.resize(pad_image, (self.image_size, self.image_size))
        image = np.array(image, np.float32) / 255.0
        return image

    def process_depth(self, depth_path):
        """预处理深度图到统一分辨率并过滤异常值。与 NavDP 一致。

        深度由原始单位转换为米（除以 10000），并将过近/过远值置零。
        输出 shape=(image_size, image_size, 1), dtype=float32, 单位=米。
        """
        depth = self.load_depth(depth_path) / 10000.0
        H, W = depth.shape
        prop = self.image_size / max(H, W)
        depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - depth.shape[1]) // 2, 0)
        pad_height = max((self.image_size - depth.shape[0]) // 2, 0)
        pad_depth = np.pad(
            depth, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0
        )
        pad_depth[pad_depth > 5.0] = 0
        pad_depth[pad_depth < 0.1] = 0
        depth = cv2.resize(pad_depth, (self.image_size, self.image_size))
        depth = np.array(depth, np.float32)
        return depth[:, :, np.newaxis]

    def process_data_parquet(self, index):
        """解析 Parquet 轨迹数据。与 NavDP process_data_parquet 保持一致。

        重要修复（2026-05-13）：
        旧代码从 `observation.camera_extrinsic` 读取位姿序列，但该列存储的是
        **固定的相机安装矩阵**（camera-to-body transform），每帧完全相同。
        正确的逐帧世界位姿存储在 `action` 列中，与 NavDP 一致。

        注意：Parquet 中 camera_intrinsic/camera_extrinsic 存储为嵌套 list，
        需要用 .tolist()[0] + np.vstack() 才能正确展开为 (3,3) / (4,4)。
        直接用 .iloc[0] 会导致 np.array() 只拿到外层元素数（如 3 或 4），
        reshape 时报 ValueError。
        """
        data_path = self.trajectory_data_dir[index]
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        df = pd.read_parquet(data_path)
        camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
        base_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
        # 修复：从 action 列读取逐帧世界位姿（4x4 矩阵），而非 camera_extrinsic（固定安装矩阵）
        # 与 NavDP navdp_lerobot_dataset.py:342 保持一致
        extrinsics = np.array(
            [np.stack(frame) for frame in df['action']], dtype=np.float64
        ).reshape(-1, 4, 4)
        trajectory_length = len(df)
        return camera_intrinsic, base_extrinsic, extrinsics, trajectory_length

    def process_obstacle_points(self, index):
        """从场景点云中提取障碍物点。与 NavDP navdp_lerobot_dataset.py:345-366 对齐。

        颜色约定：接近 [0, 0, 0.5] 的点视作障碍物。
        详见 docs/InternData-N1-数据集格式说明.md §五。

        Returns:
            tuple: (obstacle_points: np.ndarray, obstacle_pcd: open3d.PointCloud)
                与 NavDP 返回类型一致。
        """
        scene_pcd = self.load_pointcloud(self.trajectory_afford_path[index])
        scene_color = np.array(scene_pcd.colors)
        scene_points = np.array(scene_pcd.points)
        color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
        select_index = np.where(color_distance < 0.05)[0]
        scene_obstacle = o3d.geometry.PointCloud()
        scene_obstacle.points = o3d.utility.Vector3dVector(scene_points[select_index])
        scene_obstacle.colors = o3d.utility.Vector3dVector(scene_color[select_index])
        return np.array(scene_obstacle.points), scene_obstacle

    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1):
        """构建历史帧记忆。与 NavDP navdp_lerobot_dataset.py:368-389 对齐。

        越界帧用零填充（而非重复第 0 帧），保留时序位置语义。
        """
        memory_index = np.arange(start_step - (self.memory_size - 1) * memory_digit, start_step + 1, memory_digit)
        outrange_sum = (memory_index < 0).sum()
        memory_index = memory_index[outrange_sum:]
        context_image = np.zeros((self.memory_size, self.image_size, self.image_size, 3), np.float32)
        context_image[outrange_sum:] = np.array([self.process_image(rgb_paths[i]) for i in memory_index])
        context_depth = self.process_depth(depth_paths[start_step])
        return context_image, context_depth, memory_index

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
        """计算相对位姿，与 NavDP navdp_dataset.py L221-239 完全对齐。

        NavDP 在变换后做了坐标轴置换 [T[1], -T[0], T[2]]，
        原简化实现遗漏了此步骤，导致 x/y 轴方向错误。
        """
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        homo_RT = np.eye(4)
        homo_RT[0:3, 0:3] = R_base
        homo_RT[0:3, 3] = T_base
        if len(T_world.shape) == 1:
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(np.linalg.inv(homo_RT), np.array([*T_world, 1]).T)[0:3]
            T_frame = np.array([T_frame[1], -T_frame[0], T_frame[2]])
        else:
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(
                np.linalg.inv(homo_RT),
                np.concatenate((T_world, np.ones((T_world.shape[0], 1))), axis=-1).T,
            ).T[:, 0:3]
            T_frame = T_frame[:, [1, 0, 2]]
            T_frame[:, 1] = -T_frame[:, 1]
        return R_frame, T_frame

    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic):
        """计算绝对位姿，与 NavDP navdp_dataset.py L241-255 完全对齐。"""
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        homo_RT = np.eye(4)
        homo_RT[0:3, 0:3] = R_base
        homo_RT[0:3, 3] = T_base
        if len(T_frame.shape) == 1:
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(homo_RT, np.array([-T_frame[1], T_frame[0], T_frame[2], 1]).T)[0:3]
        else:
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(
                homo_RT,
                np.concatenate(
                    (np.stack((-T_frame[:, 1], T_frame[:, 0], T_frame[:, 2]), axis=-1),
                     np.ones((T_frame.shape[0], 1))),
                    axis=-1,
                ).T,
            ).T[:, 0:3]
        return R_world, T_world

    def xyz_to_xyt(self, xyz_actions, init_vector):
        """将局部三维轨迹点序列转换为二维平面动作序列 (x, y, theta)。

        与 NavDP navdp_lerobot_dataset.py:520-539 对齐：
        - 使用水平面分量 [0] 和 [1]（x, y），而非 [0] 和 [2]（x, z）
        - theta 由 init_vector 与当前位移向量的夹角计算
        - 输出 shape = (T-1, 3)，与 NavDP 一致

        增强：短轨迹末端重复点（dx≈dy≈0）时，theta 用线性插值避免阶梯跳变。
        """
        xyt_actions = []
        for i in range(0, xyz_actions.shape[0] - 1):
            current_vector = xyz_actions[i + 1] - xyz_actions[i]
            dot_product = np.dot(init_vector[0:2], current_vector[0:2])
            cross_product = np.cross(init_vector[0:2], current_vector[0:2])
            theta = np.arctan2(cross_product, dot_product)
            xyt_actions.append([xyz_actions[i][0], xyz_actions[i][1], theta])
        return np.array(xyt_actions)

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
        # 与 NavDP navdp_lerobot_dataset.py:623 对齐：
        # 上界用 label_actions.shape[0] - 2，预留差分缓冲，避免 critic 差分末端退化。
        action_indexes = np.clip(np.arange(self.predict_size + 1) * pred_digit, 0, label_actions.shape[0] - 2)
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

    def generate_prior_trajectory(self, pred_actions, is_task_start=False):
        """生成先验轨迹，三种情况：

        1. 任务开始（is_task_start=True）：全零，无先验。
        2. 任务中正确先验（70%）：起点到轨迹末端直线插值 + 0.5% 噪声。
        3. 任务中错误先验（30%）：随机旋转 60-300° 的错误轨迹。

        Args:
            pred_actions: 标签轨迹 (T, 3)，已归一化绝对坐标。
            is_task_start: 是否为任务开始帧（无先验）。
        """
        T = pred_actions.shape[0]
        if is_task_start:
            return np.zeros((T, 3), dtype=np.float32)

        if np.random.random() < 0.7:
            # 正确先验：起点到轨迹末端直线插值 + 极小噪声
            traj_end = pred_actions[-1].copy()
            t_interp = np.linspace(0, 1, T).reshape(-1, 1)
            prior = t_interp * traj_end
            noise_std = 0.005 * np.linalg.norm(traj_end)
            prior += np.random.randn(T, 3).astype(np.float32) * noise_std
        else:
            # 错误先验：随机旋转 60-300°
            angle = np.random.uniform(np.pi / 3, 5 * np.pi / 3)
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]], dtype=np.float32)
            prior = pred_actions.copy()
            prior[:, 0:2] = (rot @ pred_actions[:, 0:2].T).T
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
            # 方案A：强制最小帧间距 = predict_size/2 * 4，防止轨迹退化为静止样本
            min_gap = self.predict_size // 2 * 4  # = 48 帧
            pixel_start_choice = np.random.randint(0, max(1, trajectory_length // 2))
            target_min = pixel_start_choice + min_gap
            if target_min >= trajectory_length - 1:
                target_choice = min(pixel_start_choice + 1, trajectory_length - 1)
            else:
                target_choice = np.random.randint(target_min, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, max(pixel_start_choice + 1, target_choice))

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

        # 方案A：计算有效步数掩码（归一化前，用原始坐标判断是否为重复填充点）
        # 相邻步位移 > 阈值则为有效运动步；第 0 步（第 1 个航点）始终有效
        step_diffs = np.linalg.norm(pred_actions[1:, :2] - pred_actions[:-1, :2], axis=-1)  # (T-1,)
        valid_mask = np.concatenate([[1.0], (step_diffs > 1e-4).astype(np.float32)])         # (T,)

        # 2. 计算目标方位角（在归一化之前，使用原始水平面坐标 x, y）
        # point_goal 来自 xyz_to_xyt，[0]=x, [1]=y（水平面），[2]=theta
        theta_g = np.arctan2(point_goal[1], point_goal[0]).astype(np.float32)

        # 3. 动作空间归一化：将绝对坐标从 [0,~10m] 映射到 [-2,2]
        #    使训练目标尺度与 NavDP 的噪声 ε ~ N(0,1) 对齐。
        #    详见 TRAINING_CONVERGENCE_ANALYSIS.md §4 方案 A。
        pred_actions[:, 0:2] = pred_actions[:, 0:2] / self.action_scale_xy
        pred_actions[:, 2] = pred_actions[:, 2] / self.action_scale_theta
        augment_actions[:, 0:2] = augment_actions[:, 0:2] / self.action_scale_xy
        augment_actions[:, 2] = augment_actions[:, 2] / self.action_scale_theta
        point_goal[0:2] = point_goal[0:2] / self.action_scale_xy
        point_goal[2] = point_goal[2] / self.action_scale_theta

        # 4. 生成先验轨迹（三种情况：任务开始/正确先验/错误先验）
        is_task_start = (memory_start_choice == pixel_start_choice)
        prior_traj = self.generate_prior_trajectory(pred_actions, is_task_start=is_task_start)

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
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32)

        return (
            point_goal,       # 0: (3,) 已归一化
            image_goal,       # 1: (H, W, 6)
            pixel_goal,       # 2: (H, W, C)
            memory_images,    # 3: (mem, H, W, 3)
            depth_image,      # 4: (H, W, 1)
            pred_actions,     # 5: (T_pred, 3) 已归一化绝对坐标
            augment_actions,  # 6: (T_pred, 3) 已归一化绝对坐标
            pred_critic,      # 7: scalar
            augment_critic,   # 8: scalar
            float(pixel_flag),  # 9: float
            prior_traj,       # 10: (T_pred, 3) 已归一化先验轨迹
            theta_g,          # 11: scalar 目标方位角（原始值，未归一化）
            valid_mask,       # 12: (T_pred,) 有效步掩码（1=真实运动，0=填充静止）
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
        "batch_valid_mask": torch.stack([item[12] for item in batch]),
    }
    return collated
