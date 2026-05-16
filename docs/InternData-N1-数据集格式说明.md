# InternData-N1 数据集格式说明

> 本文档描述 InternData-N1（LeRobot 格式）数据集的目录结构、文件格式和各字段含义。
> 适用于 NavDP / Bridge-DP / FlowNav 等训练管线。

---

## 一、目录结构

### 新格式（每个 scene 下有多个 trajectory）

```
traj_data/                          ← root_dir（训练配置中的 root_dir）
├── group_A/                        ← group 目录（如 gibson_zed, 3dfront 等）
│   ├── scene_001/                  ← scene 目录
│   │   ├── trajectory_00/          ← 单个轨迹数据单元
│   │   │   ├── data/
│   │   │   │   └── chunk-000/
│   │   │   │       └── episode_000000.parquet    ← 轨迹数据（核心）
│   │   │   ├── meta/
│   │   │   │   ├── episodes_stats.jsonl          ← episode 统计信息
│   │   │   │   └── pointcloud.ply                ← 场景点云（含障碍物标注）
│   │   │   └── videos/
│   │   │       └── chunk-000/
│   │   │           ├── observation.images.rgb/    ← RGB 图像帧序列
│   │   │           │   ├── 000000.png
│   │   │           │   ├── 000001.png
│   │   │           │   └── ...
│   │   │           └── observation.images.depth/  ← 深度图帧序列
│   │   │               ├── 000000.png
│   │   │               ├── 000001.png
│   │   │               └── ...
│   │   ├── trajectory_01/
│   │   └── ...
│   ├── scene_002/
│   └── ...
└── group_B/
    └── ...
```

### 旧格式（scene 下直接包含 data/，多 episode 通过 image_index 切分）

```
traj_data/
├── group_A/
│   ├── scene_001/
│   │   ├── data/
│   │   │   └── chunk-000/
│   │   │       ├── episode_000000.parquet
│   │   │       ├── episode_000001.parquet
│   │   │       └── ...
│   │   ├── meta/
│   │   │   ├── episodes_stats.jsonl
│   │   │   └── pointcloud.ply
│   │   └── videos/
│   │       └── chunk-000/
│   │           ├── observation.images.rgb/
│   │           └── observation.images.depth/
│   └── ...
```

---

## 二、Parquet 文件字段说明

每个 `.parquet` 文件存储一条轨迹（episode）的所有帧数据。

| 列名 | 数据类型 | 形状（每行） | 说明 |
|------|---------|-------------|------|
| `action` | 嵌套 list (float64) | `[[4×4]]` → reshape 为 `(4, 4)` | **逐帧世界位姿矩阵**。这是机器人在世界坐标系下的绝对位姿（4×4 齐次变换矩阵）。NavDP/Bridge-DP 的轨迹标签从此列读取。 |
| `observation.camera_intrinsic` | 嵌套 list (float64) | `[[3×3]]` → reshape 为 `(3, 3)` | **相机内参矩阵**。每帧相同（固定相机）。包含焦距 fx/fy 和光心 cx/cy。用于将 3D 点投影到图像平面（pixel goal 计算）。 |
| `observation.camera_extrinsic` | 嵌套 list (float64) | `[[4×4]]` → reshape 为 `(4, 4)` | **相机安装外参矩阵（camera-to-body transform）**。每帧相同，描述相机在机器人体坐标系中的固定安装位置和朝向。**注意：这不是逐帧变化的世界位姿，不要与 `action` 混淆。** |

### 重要：字段读取方式

由于 Parquet 中嵌套 list 的存储方式，读取时需要特别注意：

```python
# ✅ 正确读取方式
camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64).reshape(-1, 4, 4)

# ❌ 错误读取方式（会导致 reshape 失败）
# camera_intrinsic = np.array(df['observation.camera_intrinsic'].iloc[0]).reshape(3, 3)
```

---

## 三、字段语义详解

### `action` — 世界位姿序列

```
action[i] = T_world_body^{(i)} ∈ SE(3)

    ┌                    ┐
    │ R_00  R_01  R_02  tx │
    │ R_10  R_11  R_12  ty │    4×4 齐次变换矩阵
    │ R_20  R_21  R_22  tz │
    │  0     0     0    1  │
    └                    ┘

其中：
- R (3×3): 机器人体坐标系在世界坐标系中的旋转
- t (3×1): 机器人体坐标系原点在世界坐标系中的位置 (tx, ty, tz)
- 单位：米（m）
```

**坐标系约定**：
- 世界坐标系：右手坐标系，Y 轴朝上（重力反方向）
- 机器人在地面上运动时：主要变化在 X-Z 平面（水平面），Y 基本不变
- `extrinsics[:, 0:2, 3]` 取 (X, Y) 用于 BEV 碰撞检测

### `observation.camera_extrinsic` — 相机安装矩阵

```
T_body_camera = observation.camera_extrinsic ∈ SE(3)

含义：将相机坐标系中的点变换到机器人体坐标系
用途：
  1. relative_pose() / absolute_pose() 中的坐标变换
  2. process_pixel_goal() 中将 3D 目标点投影到图像平面
  
注意：该矩阵每帧相同（相机固定安装在机器人上）
```

### `observation.camera_intrinsic` — 相机内参

```
K = observation.camera_intrinsic ∈ R^{3×3}

    ┌            ┐
    │ fx  0   cx │
    │ 0   fy  cy │
    │ 0   0   1  │
    └            ┘

用途：process_pixel_goal() 中将相机坐标系下的 3D 点投影为像素坐标
```

---

## 四、图像文件说明

### RGB 图像 (`observation.images.rgb/`)

- 格式：PNG
- 分辨率：原始分辨率（训练时由 Dataset 统一 resize）
- 色彩空间：RGB（PIL 读取）或 BGR（OpenCV 读取后需转换）
- 数据类型：uint8, 范围 [0, 255]
- 命名：`000000.png`, `000001.png`, ... 按时间顺序排列
- 与 Parquet 行对齐：第 i 帧 RGB 对应 `action[i]` 的位姿

### 深度图 (`observation.images.depth/`)

- 格式：PNG（16-bit 单通道）
- 分辨率：与 RGB 相同
- 数据类型：uint16
- **单位：0.1 毫米**（即 pixel_value / 10000.0 = 深度值（米））
- 有效范围约定：
  - 有效深度：0.1m ~ 5.0m（即 pixel_value ∈ [1000, 50000]）
  - pixel_value = 0：无效/无深度
  - pixel_value > 50000：超远距离（噪声）

### NavDP 的深度图预处理流程（标准参考）

```python
depth = np.array(Image.open(depth_path), np.uint16)  # 读取 uint16
depth = depth / 10000.0                               # 转为米制
# ... resize + padding ...
depth[depth > 5.0] = 0                                # 过远置零
depth[depth < 0.1] = 0                                # 过近置零
# 输出 shape=(H, W, 1), dtype=float32, 单位=米
```

---

## 五、场景点云文件说明

### `pointcloud.ply`

- 格式：PLY（Open3D 可读取）
- 坐标系：与 `action` 使用相同的世界坐标系
- 单位：米
- 包含字段：
  - `points` (N, 3)：点坐标 (x, y, z)
  - `colors` (N, 3)：点颜色 (r, g, b)，范围 [0, 1]

### 颜色编码约定

点云中的颜色用于标注语义类别：

| 颜色 (R, G, B) | 含义 | 过滤条件 |
|---|---|---|
| `(0, 0, 0.5)` ± 0.05 | **障碍物**（不可通行区域） | `abs(color - [0, 0, 0.5]).sum() < 0.05` |
| 其他颜色 | 可通行区域 / 地面 / 天花板等 | 不参与碰撞检测 |

### 障碍点过滤（NavDP 标准实现）

```python
pcd = o3d.io.read_point_cloud(ply_path)
colors = np.array(pcd.colors)           # (N, 3), [0, 1]
points = np.array(pcd.points)           # (N, 3), 米
color_distance = np.abs(colors - np.array([0, 0, 0.5])).sum(axis=-1)
obstacle_mask = color_distance < 0.05
obstacle_points = points[obstacle_mask]  # 仅障碍物点
```

---

## 六、episodes_stats.jsonl 格式

每行一个 JSON 对象，描述一个 episode 的统计信息。

### 新格式（单 episode per trajectory_XX）

```json
{"total_frames": 120}
```

仅包含帧总数，无 `image_index` 字段，表示该 trajectory 的所有帧属于同一个 episode。

### 旧格式（多 episode per scene）

```json
{"image_index": {"min": 0, "max": 59}, "total_frames": 60}
{"image_index": {"min": 60, "max": 119}, "total_frames": 60}
```

通过 `image_index.min` / `image_index.max` 指定该 episode 对应的 RGB/Depth 帧范围。

---

## 七、坐标系与变换关系图

```
World Frame (世界坐标系)
    │
    │  action[i] = T_world_body^{(i)}
    ▼
Body Frame (机器人体坐标系)
    │
    │  observation.camera_extrinsic = T_body_camera (固定)
    ▼
Camera Frame (相机坐标系)
    │
    │  observation.camera_intrinsic = K
    ▼
Image Plane (像素坐标)
```

### 从世界点到像素坐标的完整投影链

```python
# 1. 世界坐标 → 体坐标（以当前帧为参考）
p_body = inv(T_world_body_current) @ p_world_homo

# 2. 体坐标 → 相机坐标
p_camera = T_body_camera @ p_body

# 3. 相机坐标 → 像素坐标
p_pixel = K @ p_camera[:3]
u, v = p_pixel[0] / p_pixel[2], p_pixel[1] / p_pixel[2]
```

---

## 八、数据集在训练管线中的使用方式

| 训练模型 | 数据集类 | 轨迹表示 | 预处理索引文件 |
|---|---|---|---|
| NavDP | `NavDP_Base_Datset` | 相邻步增量 × 4 | `preload_index.json` |
| Bridge-DP | `BridgeDP_Base_Dataset` | 绝对坐标 (x, y, θ) / 5.0, π | 同上 |
| FlowNav | `FlowNav_Base_Datset` | 相邻步增量 × 4 | 同上 |

### 预加载索引文件格式

由 `scripts/dataset/generate_preload_index.py` 生成：

```json
{
    "trajectory_data_dir": ["/.../episode_000000.parquet", ...],
    "trajectory_rgb_path": [["/.../000000.png", "/.../000001.png", ...], ...],
    "trajectory_depth_path": [["/.../000000.png", "/.../000001.png", ...], ...],
    "trajectory_afford_path": ["/.../pointcloud.ply", ...]
}
```

每个 episode 对应四个数组中的一个元素（一一对应）。

---

## 九、深度图与轨迹坐标的尺度独立性说明

### 问题

Bridge-DP 对轨迹坐标做了归一化（`xy / 5.0`，`θ / π`），深度图保持米制（[0.1, 5.0]）。
这两种数据的"尺度不一致"会不会导致模型学习困难？

### 答案：不会。两者在架构中是尺度独立的。

### 详细解释

#### 1. 架构中的数据流隔离

```
深度图 (H,W,1) 米制             轨迹 (T,3) 归一化坐标
     │                                │
     ▼                                ▼
 DepthAnythingV2 ViT              input_embed (Linear 3→384)
     │                                │
     ▼                                ▼
 depth_token (B, 256, 384)        action_embed (B, T, 384)
     │                                │
     ├─── concat with rgb_token       │
     ▼                                ▼
 TransformerDecoder(memory)       TransformerDecoder(tgt)
     │                                │
     └──── Cross-Attention ──────────┘
                   │
                   ▼
            action_head (Linear 384→3)
                   │
                   ▼
            x̂₀ 预测 (B, T, 3) 归一化坐标
```

关键点：**深度图经过 ViT 编码后变成了 384 维的抽象特征 token**，其数值已经与原始的米制深度值无关。这些 token 通过 cross-attention 与轨迹 token 交互时，注意力权重是**学习出来的**，不是硬编码的数学运算。

#### 2. 为什么 ViT 特征与原始数值脱钩？

ViT（Vision Transformer）的处理流程：
1. **Patch Embedding**：将 (H,W) 的深度图切成 16×16 的 patch，每个 patch 通过线性投影变为 384 维向量。这个线性投影的权重是**可学习的**，会自适应输入的数值范围。
2. **Self-Attention**：patch 之间通过注意力机制交互。注意力分数 = `softmax(Q·K^T / √d)`，归一化后的分数 ∈ [0,1]，与输入值的绝对大小无关。
3. **LayerNorm**：每层输出都经过 LayerNorm，进一步消除尺度依赖。

因此，无论深度图输入是 [0.1, 5.0] 米制还是 [0, 1] 归一化，ViT 编码器都能通过学习适应，输出相同语义的特征 token。

#### 3. Cross-Attention 的尺度无关性

在 Transformer Decoder 中，深度/RGB token 作为 `memory`，轨迹 token 作为 `tgt`：

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

其中：
  Q = tgt_embed · W_Q     ← 来自轨迹 token（归一化坐标空间）
  K = memory_embed · W_K  ← 来自深度/RGB token（ViT 特征空间）
  V = memory_embed · W_V  ← 来自深度/RGB token（ViT 特征空间）
```

- Q 和 K 通过各自的**可学习权重矩阵** W_Q 和 W_K 投影到同一个注意力空间。
- softmax 归一化保证注意力权重与绝对数值无关。
- V 经过 W_V 投影后，其"原始尺度"信息已被线性变换吸收。

**结论**：Cross-Attention 不要求 Q 和 K/V 来自相同的数值范围。W_Q 和 W_K 会学习到合适的投影，使得语义相关的 token 获得高注意力分数。

#### 4. 类比理解

这类似于多模态模型（如 CLIP）中文本和图像的联合训练：
- 文本编码器输出的 token 值域可能在 [-10, 10]
- 图像编码器输出的 token 值域可能在 [-1, 1]
- 但通过可学习的投影层和注意力机制，模型能正确学习两种模态之间的对应关系

#### 5. 什么情况下才需要尺度对齐？

需要尺度对齐的场景是：**两种数据在同一个数学运算中直接进行数值比较或算术运算**。例如：
- 如果代码中有 `depth_value - trajectory_x`（直接相减）→ 需要对齐
- 如果代码中有 `if depth > trajectory_distance`（直接比较）→ 需要对齐

但在 Bridge-DP/NavDP 中，深度图和轨迹坐标之间**不存在任何直接的数值运算**——它们各自经过独立的编码器，在特征空间中通过注意力机制隐式关联。

#### 6. 为什么仍建议使用米制深度（而非逐帧 max 归一化）？

虽然尺度差异不影响正确性，但使用米制深度仍然更好，原因有二：

1. **帧间一致性**：米制深度 `/10000` 是全局固定缩放，不同帧的深度值在相同的数值空间中。而 `max_depth` 归一化是逐帧自适应的——如果一帧的最远距离是 3m，另一帧是 15m，同样的物理距离 1m 在两帧中的归一化值分别是 0.33 和 0.067，这会让 ViT 编码器更难学习一致的空间表征。

2. **与预训练权重对齐**：DepthAnythingV2 的预训练通常基于标准化的深度输入。保持米制单位更接近预训练数据分布，微调时收敛更快。

#### 7. 总结对照表

| 数据 | 进入模型前的值域 | 进入模型的方式 | 与其他数据的数值运算？ |
|---|---|---|---|
| RGB 图像 | [0, 1] float32 | ViT Patch Embed → 384d token | ❌ 无 |
| 深度图 | [0.1, 5.0] 米 (float32) | ViT Patch Embed → 384d token | ❌ 无 |
| 轨迹坐标 | [-2, 2] 归一化 | Linear(3→384) → 384d token | ❌ 无 |
| 目标点 | [-2, 2] 归一化 | Linear(3→384) → 384d token | ❌ 无 |

所有模态都先经过各自的编码器变成 384 维 token，然后在 Transformer 注意力空间中交互。
**没有任何位置需要不同模态的原始数值直接参与同一个算术运算。**
