"""NavDP LeRobot 数据集读取与预处理模块。

该模块实现了 NavDP 训练所需的数据管线，核心职责包括：
1. 扫描并索引多场景离线轨迹数据。
2. 读取 RGB、Depth、Parquet 轨迹、点云等多模态输入。
3. 构造历史记忆帧、点目标、图像目标、像素目标。
4. 生成原始动作与增强动作，并计算对应的 critic 分数。
5. 输出可直接用于 PyTorch 训练的张量格式样本。

说明：
- 本文件保留了原有类名 `NavDP_Base_Datset`（存在拼写习惯问题），
    为兼容既有代码，不在此处更名。
- 为了便于分布式训练日志排查，模块覆盖了内置 `print`，仅在 rank=0 打印。
"""

# 覆盖内置 print：只在主进程打印，并附带毫秒级时间戳。
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
    """带时间戳的安全打印函数。

    在分布式训练中，仅当环境变量 RANK 为 0 时输出日志，避免多进程重复打印。
    若发生任意异常（例如环境变量格式异常），函数静默失败以避免影响训练流程。
    """
    try:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            original_print(f"[{timestamp}]", *args, **kwargs)
    except Exception:  # Catch any exception to prevent crashes
        pass


builtins.print = print


class NavDP_Base_Datset(Dataset):
    """NavDP 基础数据集。

    该数据集从指定根目录递归收集轨迹文件与图像路径，并在 `__getitem__`
    中动态执行多模态预处理和动作标签构造。

    Args:
        root_dirs: 数据根目录，目录结构需符合项目约定。
        preload_path: 预加载索引文件路径；当 `preload=True` 时从该 JSON 加载。
        memory_size: 历史帧长度（时间记忆窗口大小）。
        predict_size: 未来动作预测长度（用于采样 action index）。
        batch_size: 仅用于打印性能统计时的窗口大小。
        image_size: RGB/Depth/Mask 统一缩放到的边长。
        scene_data_scale: 场景采样比例，`1.0` 表示不下采样。
        trajectory_data_scale: 轨迹采样比例（当前实现中仅保留参数）。
        pixel_channel: 像素目标通道模式，常见为 4 或 7。
        action_dim: 动作维度，若 xyt 维度不足会右侧补零。
        debug: 调试开关（当前实现中仅保留参数）。
        preload: 是否直接使用 `preload_path` 加载索引。
        random_digit: 是否随机采样时间步长（memory/pred digit）。
        prior_sample: 是否按障碍密度优先采样起终点。
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
    ):

        self.dataset_dirs = np.array([p for p in os.listdir(root_dirs)])
        self.memory_size = memory_size
        self.image_size = image_size
        self.scene_scale_size = scene_data_scale
        self.trajectory_data_scale = trajectory_data_scale
        self.predict_size = predict_size
        self.action_dim = action_dim
        self.debug = debug

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
            # 首次扫描数据目录，构建每个 episode 对应的路径索引。
            for group_dir in self.dataset_dirs:  # gibson_zed, 3dfront ...
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
                                    # 每个 episode 使用 image_index 对齐 RGB 与 Depth 帧范围。
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

            # 将扫描结果保存为索引文件，后续可跳过目录扫描加速启动。
            save_dict = {
                'trajectory_data_dir': self.trajectory_data_dir,
                'trajectory_rgb_path': self.trajectory_rgb_path,
                'trajectory_depth_path': self.trajectory_depth_path,
                'trajectory_afford_path': self.trajectory_afford_path,
            }
            with open(preload_path, 'w') as f:
                json.dump(save_dict, f, indent=4)

            # 简单数据重复，提升小数据集下的采样次数。
            self.trajectory_data_dir = self.trajectory_data_dir * 50
            self.trajectory_rgb_path = self.trajectory_rgb_path * 50
            self.trajectory_depth_path = self.trajectory_depth_path * 50
            self.trajectory_afford_path = self.trajectory_afford_path * 50
        else:
            # 直接加载预计算索引，避免每次初始化都遍历文件系统。
            load_dict = json.load(open(preload_path, 'r'))
            self.trajectory_data_dir = load_dict['trajectory_data_dir'] * 50
            self.trajectory_rgb_path = load_dict['trajectory_rgb_path'] * 50
            self.trajectory_depth_path = load_dict['trajectory_depth_path'] * 50
            self.trajectory_afford_path = load_dict['trajectory_afford_path'] * 50

    def __len__(self):
        """返回数据集样本数。"""
        return len(self.trajectory_data_dir)

    def load_image(self, image_url):
        """读取 RGB 图像。

        Args:
            image_url: 图像文件路径。

        Returns:
            np.ndarray: uint8 格式的 HWC 图像，读取失败时返回全零占位图。
        """
        try:
            image = Image.open(image_url)
            image = np.array(image, np.uint8)
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        return image

    def load_depth(self, depth_url):
        """读取深度图。

        Args:
            depth_url: 深度图文件路径。

        Returns:
            np.ndarray: uint16 格式的 HW 深度图，失败时返回全零图。
        """
        try:
            depth = Image.open(depth_url)
            depth = np.array(depth, np.uint16)
        except Exception as e:
            print(f"Error loading depth {depth_url}: {e}")
            depth = np.zeros((self.image_size, self.image_size), dtype=np.uint16)
        return depth

    def load_pointcloud(self, pcd_url):
        """读取场景点云。

        Args:
            pcd_url: 点云文件路径（.ply）。

        Returns:
            open3d.geometry.PointCloud: 读取到的点云对象。
        """
        pcd = o3d.io.read_point_cloud(pcd_url)
        return pcd

    def process_image(self, image_path):
        """预处理 RGB 图像到统一分辨率并归一化。

        处理流程：
        1. 等比例缩放到最长边为 `image_size`。
        2. 居中零填充至正方形。
        3. 再次 resize 到精确尺寸。
        4. 转为 float32 并归一化到 [0, 1]。

        Args:
            image_path: 原始图像路径。

        Returns:
            np.ndarray: shape=(image_size, image_size, 3) 的 float32 图像。
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
        """预处理深度图到统一分辨率并过滤异常值。

        深度由原始单位转换为米（除以 10000），并将过近/过远值置零。

        Args:
            depth_path: 深度图路径。

        Returns:
            np.ndarray: shape=(image_size, image_size, 1) 的 float32 深度图。
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
        """读取轨迹 parquet 并解析相机参数与动作序列。

        Args:
            index: 数据索引。

        Returns:
            tuple:
                camera_intrinsic: shape=(3, 3) 相机内参。
                camera_extrinsic: shape=(4, 4) 基准外参。
                camera_trajectory: shape=(T, 4, 4) 位姿序列。
                trajectory_length: 轨迹长度 T。
        """
        if not os.path.isfile(self.trajectory_data_dir[index]):
            raise FileNotFoundError(self.trajectory_data_dir[index])
        df = pd.read_parquet(self.trajectory_data_dir[index])
        camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
        camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
        trajectory_length = len(df['action'].tolist())
        camera_trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64).reshape(-1, 4, 4)
        return camera_intrinsic, camera_extrinsic, camera_trajectory, trajectory_length

    def process_obstacle_points(self, index):
        """从场景点云中提取障碍物点。

        当前实现按颜色阈值筛选（接近 [0, 0, 0.5] 的点视作障碍）。

        Args:
            index: 数据索引。

        Returns:
            tuple:
                np.ndarray: 过滤后的障碍点坐标，shape=(N, 3)。
                open3d.geometry.PointCloud: 障碍点云对象。
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
        """构造历史记忆帧与当前深度图。

        Args:
            rgb_paths: 当前 episode 的 RGB 路径列表。
            depth_paths: 当前 episode 的 Depth 路径列表。
            start_step: 当前记忆终点时刻（包含该帧）。
            memory_digit: 记忆采样间隔。

        Returns:
            tuple:
                context_image: shape=(memory_size, H, W, 3) 的历史图像序列。
                context_depth: shape=(H, W, 1) 的当前深度图。
                memory_index: 实际使用的时间索引（前部越界会被裁剪）。
        """
        memory_index = np.arange(start_step - (self.memory_size - 1) * memory_digit, start_step + 1, memory_digit)
        outrange_sum = (memory_index < 0).sum()
        memory_index = memory_index[outrange_sum:]
        context_image = np.zeros((self.memory_size, self.image_size, self.image_size, 3), np.float32)
        context_image[outrange_sum:] = np.array([self.process_image(rgb_paths[i]) for i in memory_index])
        context_depth = self.process_depth(depth_paths[start_step])
        return context_image, context_depth, memory_index

    def process_pixel_goal(self, image_url, target_point, camera_intrinsic, camera_extrinsic):
        """将局部目标点投影到图像平面，生成像素目标输入。

        Args:
            image_url: 起始帧图像路径。
            target_point: 局部目标点（x, y, theta，其中仅 x/y 用于投影）。
            camera_intrinsic: 相机内参矩阵，shape=(3, 3)。
            camera_extrinsic: 相机外参矩阵，shape=(4, 4)。

        Returns:
            tuple:
                np.ndarray: 拼接后的像素目标输入，通道为 [RGB + Mask]。
                bool: 目标点是否在图像可视范围内。
        """
        try:
            image = Image.open(image_url)
            image = np.array(image, np.uint8)
        except Exception as e:
            print(f"Error loading image {image_url}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        resize_image = self.process_image(image_url)

        coordinate = np.array([-target_point[1], target_point[0], camera_extrinsic[2, 3] * 0.8])
        camera_coordinate = np.matmul(camera_extrinsic[0:3, 0:3], coordinate[:, None])
        pixel_coord_x = camera_intrinsic[0, 2] + (camera_coordinate[0] / camera_coordinate[2]) * camera_intrinsic[0, 0]
        pixel_coord_y = camera_intrinsic[1, 2] + (-camera_coordinate[1] / camera_coordinate[2]) * camera_intrinsic[1, 1]
        pixel_mask = np.zeros_like(image)
        visible_flag = False

        if (
            pixel_coord_x > 0
            and pixel_coord_x < image.shape[1]
            and pixel_coord_y > 0
            and pixel_coord_y < image.shape[0]
        ):
            pixel_mask = cv2.rectangle(
                pixel_mask,
                (int(pixel_coord_x - np.random.randint(6, 12)), int(pixel_coord_y - np.random.randint(6, 12))),
                (int(pixel_coord_x + np.random.randint(6, 12)), int(pixel_coord_y + np.random.randint(6, 12))),
                (255, 255, 255),
                -1,
            )
            visible_flag = True

        H, W, C = pixel_mask.shape
        prop = self.image_size / max(H, W)
        pixel_mask = cv2.resize(pixel_mask, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - pixel_mask.shape[1]) // 2, 0)
        pad_height = max((self.image_size - pixel_mask.shape[0]) // 2, 0)
        pad_mask = np.pad(
            pixel_mask, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0
        )
        mask = cv2.resize(pad_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, np.float32) / 255.0
        mask = mask.mean(axis=-1)[:, :, None]
        return np.concatenate((resize_image, mask), axis=-1), visible_flag

    def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic):
        """将世界坐标系位姿转换为以当前帧为参考的相对位姿。

        支持单帧输入（1D 平移向量）与批量输入（2D 平移矩阵）。

        Args:
            R_base: 基准旋转矩阵。
            T_base: 基准平移向量。
            R_world: 世界旋转矩阵（单帧或批量）。
            T_world: 世界平移向量（单帧或批量）。
            base_extrinsic: 数据集中定义的基准外参。

        Returns:
            tuple: 相对旋转与相对平移。
        """
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_world.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(np.linalg.inv(homo_RT), np.array([*T_world, 1]).T)[0:3]
            T_frame = np.array([T_frame[1], -T_frame[0], T_frame[2]])  # [:T[1],-T[0],T[2]
            return R_frame, T_frame
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_frame = np.dot(R_world, R_base.T)
            T_frame = np.dot(
                np.linalg.inv(homo_RT), np.concatenate((T_world, np.ones((T_world.shape[0], 1))), axis=-1).T
            ).T[:, 0:3]
            T_frame = T_frame[:, [1, 0, 2]]
            T_frame[:, 1] = -T_frame[:, 1]
            return R_frame, T_frame

    def absolute_pose(self, R_base, T_base, R_frame, T_frame, base_extrinsic):
        """将相对位姿恢复到世界坐标系绝对位姿。

        该函数是 `relative_pose` 的逆向过程，同样支持单帧与批量输入。

        Args:
            R_base: 基准旋转矩阵。
            T_base: 基准平移向量。
            R_frame: 局部旋转矩阵。
            T_frame: 局部平移（单帧或批量）。
            base_extrinsic: 数据集基准外参。

        Returns:
            tuple: 世界系下旋转与平移。
        """
        R_base = np.matmul(R_base, np.linalg.inv(base_extrinsic[0:3, 0:3]))
        if len(T_frame.shape) == 1:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(homo_RT, np.array([-T_frame[1], T_frame[0], T_frame[2], 1]).T)[0:3]
        else:
            homo_RT = np.eye(4)
            homo_RT[0:3, 0:3] = R_base
            homo_RT[0:3, 3] = T_base
            R_world = np.dot(R_frame, R_base)
            T_world = np.dot(
                homo_RT,
                np.concatenate(
                    (np.stack((-T_frame[:, 1], T_frame[:, 0], T_frame[:, 2]), axis=-1), np.ones((T_frame.shape[0], 1))),
                    axis=-1,
                ).T,
            ).T[:, 0:3]
        return R_world, T_world

    def xyz_to_xyt(self, xyz_actions, init_vector):
        """将局部三维轨迹点序列转换为二维平面动作序列 (x, y, theta)。

        角度 `theta` 由初始方向向量与当前位移向量之间的夹角计算得到。

        Args:
            xyz_actions: shape=(T, 3) 的局部点轨迹。
            init_vector: 初始参考方向向量。

        Returns:
            np.ndarray: shape=(T-1, 3) 的 xyt 序列。
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
        """构造监督动作与增强动作轨迹。

        主要步骤：
        1. 将 `[start_step, end_step]` 的绝对轨迹转到局部坐标，得到标签轨迹。
        2. 在局部平面随机旋转未来轨迹，得到增强锚点。
        3. 使用三次样条在世界系插值，再转回局部系，形成增强轨迹。
        4. 依据 `predict_size` 与 `pred_digit` 生成采样索引。

        Args:
            extrinsics: shape=(T, 4, 4) 的位姿序列。
            base_extrinsic: 基准外参。
            start_step: 预测起始时刻。
            end_step: 预测终止时刻。
            pred_digit: 未来轨迹采样步长。

        Returns:
            tuple:
                local_label_points: 局部标签轨迹点。
                local_augment_points: 局部增强轨迹点。
                origin_world_points: 原始世界系轨迹点。
                result_augment_points: 增强后的世界系轨迹点。
                action_indexes: 用于截取预测步的索引。
        """
        label_linear_pos = []
        for f_ext in extrinsics[start_step : end_step + 1]:
            R, T = self.relative_pose(
                extrinsics[start_step][0:3, 0:3],
                extrinsics[start_step][0:3, 3],
                f_ext[0:3, 0:3],
                f_ext[0:3, 3],
                base_extrinsic,
            )
            label_linear_pos.append(T)
        label_actions = np.array(label_linear_pos)

        # 轨迹增强：先随机旋转，再用样条平滑，避免出现不连续转向。
        rotate_yaw_angle = np.random.uniform(-np.pi / 3, np.pi / 3)
        rotate_matrix = np.array(
            [
                [np.cos(rotate_yaw_angle), -np.sin(rotate_yaw_angle)],
                [np.sin(rotate_yaw_angle), np.cos(rotate_yaw_angle)],
            ],
            np.float32,
        )
        rotate_local_actions = np.matmul(rotate_matrix, label_actions[:, 0:2].T).T
        rotate_local_actions = np.stack(
            (rotate_local_actions[:, 0], rotate_local_actions[:, 1], np.zeros_like(rotate_local_actions[:, 0])), axis=-1
        )
        rotate_world_points = []
        for act in rotate_local_actions:
            w_rot, w_act = self.absolute_pose(
                extrinsics[start_step, 0:3, 0:3], extrinsics[start_step, 0:3, 3], np.eye(3), act, base_extrinsic
            )
            rotate_world_points.append(w_act)
        rotate_world_points = np.array(rotate_world_points)
        origin_world_points = extrinsics[start_step : end_step + 1, 0:3, 3]
        mix_anchor_points = rotate_world_points

        t = np.linspace(0, 1, mix_anchor_points.shape[0])
        cs_x = CubicSpline(t, mix_anchor_points[:, 0])
        cs_y = CubicSpline(t, mix_anchor_points[:, 1])
        cs_z = CubicSpline(t, mix_anchor_points[:, 2])
        interpolate_nums = origin_world_points.shape[0]
        t_fine = np.linspace(0, 1, int(interpolate_nums))
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        z_fine = cs_z(t_fine)
        result_augment_points = np.stack((x_fine, y_fine, z_fine), axis=-1)
        local_label_points = []
        local_augment_points = []
        for f_ext, g_ext in zip(origin_world_points, result_augment_points):
            Rf, Tf = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3], np.eye(3), f_ext, base_extrinsic
            )
            Rg, Tg = self.relative_pose(
                extrinsics[start_step][0:3, 0:3], extrinsics[start_step][0:3, 3], np.eye(3), g_ext, base_extrinsic
            )
            local_label_points.append(Tf)
            local_augment_points.append(Tg)
        local_label_points = np.array(local_label_points)
        local_augment_points = np.array(local_augment_points)
        action_indexes = np.clip(np.arange(self.predict_size + 1) * pred_digit, 0, label_actions.shape[0] - 2)
        return local_label_points, local_augment_points, origin_world_points, result_augment_points, action_indexes

    def rank_steps(self, extrinsics, obstacle_points, pred_digit=4):
        """基于障碍物密度对起终点进行概率采样。

        直觉上：未来局部区域障碍越密，越优先作为起点；终点倾向更远位置。
        目的：提高训练样本的多样性和挑战性，使模型能够更好地处理复杂的导航环境。

        Args:
            extrinsics: shape=(T, 4, 4) 位姿序列。
            obstacle_points: shape=(N, 3) 障碍点。
            pred_digit: 未来窗口采样步长。

        Returns:
            tuple[int, int]: (start_choice, target_choice)
        """
        points_score = []
        trajectory = extrinsics[:, 0:2, 3]
        # bev_points = obstacle_points[:, 0:2]
        for i in range(0, trajectory.shape[0] - 1):
            future_actions = trajectory[i : min(i + self.predict_size * pred_digit, trajectory.shape[0] - 1)]
            future_bound = [
                np.min(future_actions[:, 0]) - 1,
                np.min(future_actions[:, 1]) - 1,
                np.max(future_actions[:, 0]) + 1,
                np.max(future_actions[:, 1]) + 1,
            ]
            within_bound_points = (
                (obstacle_points[:, 0] > future_bound[0])
                & (obstacle_points[:, 1] > future_bound[1])
                & (obstacle_points[:, 0] < future_bound[2])
                & (obstacle_points[:, 1] < future_bound[3])
            )
            points_score.append(np.sum(within_bound_points))
        points_score = np.array(points_score) / (np.array(points_score).max() + 1e-8)
        probs = np.exp(points_score / 0.2) / np.sum(np.exp(points_score / 0.2))
        start_choice = np.random.choice(np.arange(probs.shape[0]), p=probs)
        target_choice_candidates = np.arange(start_choice + 1, trajectory.shape[0])
        target_choice_p = (target_choice_candidates - start_choice) / (
            (target_choice_candidates - start_choice).max() + 1e-8
        )
        target_choice_p = np.exp(target_choice_p / 0.2) / np.exp(target_choice_p / 0.2).sum()
        target_choice = np.random.choice(target_choice_candidates, p=target_choice_p)
        return start_choice, target_choice

    def __getitem__(self, index):
        """获取单条训练样本。

        输出内容涵盖点目标、图像目标、像素目标、历史图像、深度图、
        原始/增强动作序列及其 critic 分数。

        Args:
            index: 样本索引。

        Returns:
            tuple: 训练所需的 10 个字段（主要为 torch.float32 张量）。
        """
        import os
        import time

        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
        ) = self.process_data_parquet(index)

        trajectory_obstacle_points, trajectory_obstacle_pcd = self.process_obstacle_points(index)

        if self.prior_sample:
            # 注意：prior 采样依赖障碍分布，倾向采集更“困难”的导航片段。
            pixel_start_choice, target_choice = self.rank_steps()
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            # 默认均匀随机采样，覆盖更多轨迹阶段。
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        # 历史观测与未来动作共享同一个记忆起点，保证时序一致性。
        memory_images, depth_image, memory_index = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )
        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic, memory_start_choice, target_choice, pred_digit=pred_digit
        )

        # 将局部三维点轨迹转为平面动作表示 (x, y, theta)。
        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        # 按预测长度和步长采样固定数量的动作点。
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]
        if trajectory_obstacle_points.shape[0] != 0:
            # 使用与障碍点的最小 L1 距离估计轨迹“安全性/可行性”。
            pred_distance = (
                np.abs(target_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            augment_distance = (
                np.abs(augment_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
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
            # 无障碍点时给予中性分数，避免 critic 为 NaN 或极端值。
            pred_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            augment_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]
        image_goal = np.concatenate(
            (
                self.process_image(self.trajectory_rgb_path[index][target_choice]),
                self.process_image(self.trajectory_rgb_path[index][memory_start_choice]),
            ),
            axis=-1,
        )

        # 额外构造像素级目标监督（在起始图像中标出目标区域）。
        pixel_target_local_points, _, _, _, _ = self.process_actions(
            trajectory_extrinsics, trajectory_base_extrinsic, pixel_start_choice, target_choice, pred_digit=pred_digit
        )
        pixel_init_vector = pixel_target_local_points[1] - pixel_target_local_points[0]
        pixel_xyt_actions = self.xyz_to_xyt(pixel_target_local_points, pixel_init_vector)
        pixel_goal, pixel_flag = self.process_pixel_goal(
            self.trajectory_rgb_path[index][pixel_start_choice],
            pixel_xyt_actions[-1],
            camera_intrinsic,
            trajectory_base_extrinsic,
        )
        # pixel_channel=7: [pixel_mask(1) + 带标注历史帧(3) + 当前帧(3)]。
        # pixel_channel=4: [pixel_mask(1) + 当前帧(3)]。
        # 带标注历史帧: 既提供视觉上下文，又突出目标位置，帮助模型学习从历史信息中推断目标。
        if self.pixel_channel == 7:
            pixel_goal = np.concatenate((pixel_goal, memory_images[-1]), axis=-1)

        # 将绝对动作点差分为相邻步增量，并按经验系数放大。
        # 说人话就是：模型预测相对动作增量（而非绝对位置），更符合实际导航控制的闭环反馈机制。
        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        # 对齐动作维度到 `action_dim`，缺失维补零，便于统一 batch 张量。
        pred_actions = np.pad(
            pred_actions,
            ((0, 0), (0, self.action_dim - pred_actions.shape[-1])),
            mode='constant',
            constant_values=(0, 0),
        )
        augment_actions = np.pad(
            augment_actions,
            ((0, 0), (0, self.action_dim - augment_actions.shape[-1])),
            mode='constant',
            constant_values=(0, 0),
        )

        # 按 batch 窗口打印 __getitem__ 平均耗时，便于性能分析。
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(
                f'__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s, cnt={self.item_cnt}'
            )
            self.batch_time_sum = 0.0

        # 统一转换为 torch.float32，便于后续模型前向与 loss 计算。
        point_goal = torch.tensor(point_goal, dtype=torch.float32)
        image_goal = torch.tensor(image_goal, dtype=torch.float32)
        pixel_goal = torch.tensor(pixel_goal, dtype=torch.float32)
        memory_images = torch.tensor(memory_images, dtype=torch.float32)
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        augment_actions = torch.tensor(augment_actions, dtype=torch.float32)
        pred_critic = torch.tensor(pred_critic, dtype=torch.float32)
        augment_critic = torch.tensor(augment_critic, dtype=torch.float32)
        return (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_image,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            float(pixel_flag),
        )


def navdp_collate_fn(batch):
    """NavDP 数据集自定义拼接函数。
    用来将 `__getitem__` 返回的单条样本列表拼接成 batch 张量字典，便于 DataLoader 直接使用。
    Args:
        batch: `__getitem__` 返回元组组成的列表。

    Returns:
        dict: 以字段名组织的 batch 张量字典。
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
    }
    return collated


if __name__ == "__main__":
    # 简单的离线可视化自测：导出像素目标、图像目标等结果图。
    os.makedirs("./navdp_dataset_test/", exist_ok=True)
    dataset = NavDP_Base_Datset(
        "/mnt/data/liuyu/InternDate-N1-v05/vln-n1",
        "./navdp_dataset_test/dataset_lerobot_v05_with_interiorgs.json",
        8,
        24,
        224,
        trajectory_data_scale=1.0,
        scene_data_scale=1.0,
        preload=False,
    )

    for i in range(10):
        (
            point_goal,
            image_goal,
            pixel_goal,
            memory_images,
            depth_image,
            pred_actions,
            augment_actions,
            pred_critic,
            augment_critic,
            pixel_flag,
        ) = dataset.__getitem__(i)
        if pixel_flag == 1.0:
            pixel_obs = pixel_goal.numpy()[:, :, 0:3] * 255
            pixel_obs[pixel_goal[:, :, 3] == 1] = np.array([0, 0, 255])

            draw_current_image = cv2.cvtColor(image_goal[:, :, 3:6].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_current_image = cv2.putText(
                draw_current_image, "Current-Image", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_goal_image = cv2.cvtColor(image_goal[:, :, 0:3].numpy() * 255, cv2.COLOR_BGR2RGB)
            draw_goal_image = cv2.putText(
                draw_goal_image, "Image-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            draw_pixel_image = cv2.cvtColor(pixel_obs.copy(), cv2.COLOR_BGR2RGB)
            draw_pixel_image = cv2.putText(
                draw_pixel_image, "Pixel-Goal", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)
            )

            goal_info_image = np.concatenate((draw_current_image, draw_goal_image, draw_pixel_image), axis=1)
            goal_info_image = cv2.putText(
                goal_info_image,
                "PointGoal=[{:.3f}, {:.3f}, {:.3f}]".format(*point_goal),
                (190, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
            )
            cv2.imwrite("./navdp_dataset_test/goal_information_%d.png" % i, goal_info_image)
