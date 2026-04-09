1. 动态数据集组织规范（Isaac Sim 可采集）
目录沿用你现有读取习惯，保证兼容扫描逻辑（group/scene/data/chunk/videos/meta）：

root/group_name/scene_name/data/chunk_name/episode_x.parquet
root/group_name/scene_name/videos/chunk_name/observation.images.rgb/*.png 或 *.mp4 分段
root/group_name/scene_name/videos/chunk_name/observation.images.depth/*.png 或 *.mp4 分段
root/group_name/scene_name/meta/episodes_stats.jsonl
root/group_name/scene_name/meta/pointcloud_static.ply
root/group_name/scene_name/meta/dynamic_tracks.parquet
root/group_name/scene_name/cache/dyn_voxel/episode_x.npz（可选，推荐）
root/group_name/scene_name/cache/critic/episode_x.parquet（推荐，离线预计算）
为什么这样定：

与现有 Dataset 扫描结构一致，最小化改动，可参考 flownav_lerobot_dataset.py。
动态障碍采用“对象轨迹表”而不是逐帧动态点云，节省空间且后处理灵活。
dynamic_voxels 可选缓存，平衡存储和训练吞吐。
2. 必须包含的信息（都能在 Isaac Sim 获得）
每个 episode_x.parquet 至少包含：

episode_id
timestamps_ns（每步时间戳）
robot_pose_world（4x4 或 x,y,z,qx,qy,qz,qw）
action_pose_or_delta（与你当前 action 字段语义保持一致）
camera_intrinsic（首帧固定或逐帧）
camera_extrinsic_base（首帧固定或逐帧）
image_index（与 videos 对齐）
critic_base_pred（离线预计算，原始轨迹）
critic_base_augment（可空；增强分在线算时不强制）
dynamic_tracks.parquet 至少包含：

timestamp_ns
object_id
class_id（person/cart/forklift 等）
pose_world（x,y,z,qx,qy,qz,qw）
velocity_world（vx,vy,vz）
bbox_size（dx,dy,dz）
validity（可见/有效标记）
episodes_stats.jsonl 至少包含：

episode_id
image_index.min
image_index.max
start_time_ns
end_time_ns
num_frames
Isaac Sim 数据来源映射：

People Simulation：动态人体轨迹（object_id、pose、velocity）
Replicator：RGB/Depth、随机化配置、可选语义
NavMesh：可行走区域与路径可达性
你还需要一个全局时钟同步记录器（必须），否则 critic 时间对齐会漂
3. 时间对齐规则（核心）
统一主时钟为 timestamp_ns，所有表都按它对齐：

机器人每步动作时刻 t_k
图像帧时刻 t_img
动态物体状态时刻 t_obj
对齐策略：

默认最近邻匹配，阈值例如 20ms
超阈值则线性插值 object pose（仅动态轨迹）
若 object 连续缺失超过阈值窗口，该时刻不参与动态成本
记录每步对齐误差，便于过滤脏样本
4. critic 标签策略（与你要求一致）
你已确认“优先离线预计算，增强轨迹可在线算”，按这个落地：

原始轨迹 critic：离线预计算并写入 episode parquet 或 cache/critic
增强轨迹 critic：在 Dataset 的在线增强后按同样时间轴重算
总成本同时考虑静态和动态障碍距离
可选加入 TTC 项（动态场景建议）
5. 对接实现计划（flownav_dyn_lerobot_dataset）
实施顺序：

新建动态 Dataset（名称按你提议），以 flownav_lerobot_dataset.py 为骨架。
替换静态障碍读取逻辑（当前 static pointcloud-only）为 static + dynamic tracks 时间对齐读取。
增加 dynamic_voxels 构造路径（在线或读取 cache）。
保留现有 batch key，同时补充 dynamic_voxels 相关键，使 FlowNav 前向可用。
在训练入口中新增配置开关选择动态 Dataset，位置参考 train.py。
加入 schema 校验与时间一致性校验，避免训练时才炸。
6. 关键风险（提前规避）
目前 FlowNav 相关链路在你仓库里存在不完整/不一致迹象，需在接入时先做 batch key 与 forward 参数一致性检查，参考 flownav_trainer.py。
若动态轨迹采样频率低于相机帧率，必须插值，不然 critic 噪声会很大。
如果动态体 ID 不稳定（重生/切换），object-track 表会失真，建议在导出器里做 ID 稳定化。
已完成内容：

数据组织形式已定。
字段规范已定（均可从 Isaac Sim 获取）。
与动态 Dataset 的对接路径已定。
下一步你只要回复“按 v1 开始”，我就进入实现阶段，把动态 Dataset 与训练入口改造清单细化到逐文件逐函数。