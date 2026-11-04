# FoundationPose 视觉交接与问题分析

日期：2026-09-10

本文根据以下材料整理：

- `vision_handoff.md`
- `ObjectPose.msg`
- `vision_env_report.py`
- `vision_interface_acceptance.py`
- `/home/robot/FoundationPose` 当前 FoundationPose、YOLO-seg 和 D435i 入口代码

## 1. 文档中的要求与本文的工作范围

`vision_handoff.md` 是抓取/控制侧提出的接口约定和验收要求。它不是 FoundationPose 的代码修改指令，也不是要求视觉侧立即改环境的脚本。文档中的环境问卷和 `vision_env_report.py` 主要用于决定进程形态；`vision_interface_acceptance.py` 用于最终验证 `ObjectPose` 数据流。

本分析关注两件事：

1. FoundationPose 当前实际输出了什么，以及它和抓取侧 `ObjectPose` 合同的差异。
2. 把现有视觉代码接入抓取侧之前，必须补齐哪些数据、坐标和时间语义。

## 2. 抓取侧真正需要的输出

目标话题：`object_pose`

消息类型：`sb_communication/msg/ObjectPose`

关键字段如下：

| 字段 | 抓取侧要求 |
|---|---|
| `header.stamp` | 图像/观测发生的时间，不是推理完成时间 |
| `header.frame_id` | 相机光学坐标系，推荐 `head_d435i_optical_frame` |
| `object_id` | 稳定值，建议固定为 `bottle` |
| `pose.position` | 瓶子的几何中心，单位米 |
| `pose.orientation` | 物体系 `+Z` 必须是瓶子长轴；绕 Z 的 yaw 不要求稳定 |
| `covariance` | 6×6 行主序；不知道可以全零，但有位置标准差时应填入平移对角线 |
| `confidence` | 0 到 1 |
| `valid_until` | 可以填零，由接收侧使用新鲜度窗口 |
| `source` | 建议 `foundationpose` 或 `camera` |
| `sequence` | 每次发布严格递增，不能重新 register 时清零 |

控制侧会负责 TF 变换、跳变判定、滤波、目标锁定和超时处理。视觉侧应提供原始、可追溯的观测信息，不应再增加一层无法解释的输出滤波。

## 3. 当前 FoundationPose 代码结构

主入口是：

[`my_demo/d435i_autonomous_vision.py`](/home/robot/FoundationPose/my_demo/d435i_autonomous_vision.py)

其流程是：

```text
D435i/ROS2 RGB-D
    ↓
YOLOv8-seg find_targets()
    ↓
深度掩码过滤 + RGB-D 位置测量
    ↓
MultiObjectRGBDTracker / NearestTargetSelector
    ↓
稳定性判断
    ↓
异步 FoundationPose.register()
    ↓
姿态质量检查、IMU 竖直约束、中心位置约束
    ↓
runs/autonomous_vision/latest_state.json
```

FoundationPose 的核心实现位于 [`estimater.py`](/home/robot/FoundationPose/estimater.py)。模型初始化时会把 CAD 顶点平移到 AABB 中心附近（`reset_object()`），注册结束时又通过 `get_tf_to_centered_mesh()` 返回原 CAD 坐标系的位姿。因此 `pose_camera_object` 的平移列仍代表 CAD 原点，而不是自动代表瓶子的几何中心。

当前注册采用快速候选搜索：默认 `register_views=20`、`register_inplane_step=90`、`register_topk=32`。如果快速结果未通过几何质量检查，代码会请求完整搜索作为回退。`register()` 内部对深度执行腐蚀和双边滤波，并保存 `pose_last`。

当前实物尺寸已确认与 mesh 一致：直径约 65 mm，高度约 159.5 mm。`runs/mesh/bottle.obj` 的顶点范围约为 `X/Y = ±32.5 mm`、`Z = 0..159.5 mm`，因此可以确认使用米制、CAD 原点位于瓶底中心轴附近、CAD Z 轴为瓶子长轴。仍建议把这组尺寸同步到抓取侧的瓶子配置，避免继续使用旧的 `φ36 mm × 200 mm` 标称值。

## 4. 当前输出与交接要求的差异

### 4.1 当前没有直接发布 `ObjectPose`

自主入口目前写入 JSON 状态文件。另一个节点 [`my_demo/foundationpose_pose_ros_node.py`](/home/robot/FoundationPose/my_demo/foundationpose_pose_ros_node.py) 将 JSON 转成 `geometry_msgs/PoseStamped`、valid 和 status 话题，但它没有发布 `sb_communication/msg/ObjectPose`。

因此，当前链路不能直接通过 `vision_interface_acceptance.py` 的验收。需要增加一个真正的 `ObjectPose` 适配层，或者在 ROS 侧将现有状态桥接为 `ObjectPose`。

### 4.2 `pose_camera_object` 不是瓶子几何中心位姿

当前 JSON 字段：

```json
"pose_camera_object": [[...], [...], [...], [0, 0, 0, 1]]
```

它的语义已经在 ROS 桥代码中写明为 `T_camera_CAD_original`。这表示 CAD 原点在相机坐标系中的位姿。

抓取侧需要的是瓶子几何中心。当前入口已经计算了：

```python
center_pose = pose @ inverse_to_origin
```

其中 `inverse_to_origin` 来自 `trimesh.bounds.oriented_bounds(mesh)`。交接时应发布 `center_pose` 的平移和旋转，不能直接把 `pose_camera_object[:3, 3]` 当作瓶子中心。

如果只需要一个位置向量，也可以使用当前 JSON 中的：

```text
selected_target.position_camera_m
```

但该值来自 RGB-D Kalman tracker，是经过跟踪器处理的位置，不是严格意义上的单帧原始测量。若抓取侧要求原始输出，建议同时保存 `raw_position_camera_m` 和 `filtered_position_camera_m`，由桥接层选择发布哪个。

### 4.3 当前没有图像采集时间戳

ROS 桥当前使用 JSON 文件的 `mtime` 作为 `PoseStamped.header.stamp`，源码明确标记为：

```text
timestamp_semantics = state_file_mtime_unix_system_time
```

这不是图像拍摄时间。抓取侧要求 `header.stamp` 是观测时刻，否则无法正确计算 `observation_delay`。

需要从以下来源之一保存并透传时间：

- ROS2 图像消息的 `header.stamp`；
- RealSense 帧的硬件/系统时间，并统一转换为控制侧使用的 Unix 时间；
- 如果采用独立进程，协议中显式传递 `capture_stamp`。

不能用注册完成时间、JSON 写入时间或 ROS 发布时间替代观测时间。

### 4.4 当前没有 `ObjectPose` 所需的完整元数据

JSON 目前包含置信度、位置标准差和目标状态，但没有直接包含：

- `object_id`；
- 单调 `sequence`；
- 6×6 `covariance`；
- `valid_until`；
- 明确的 `source`；
- `capture_stamp`。

这些字段应在适配层补齐，且必须有明确的默认语义。

### 4.5 当前同时存在视觉侧跟踪器

`d435i_autonomous_vision.py` 使用 `ConstantVelocity3DTracker` 和 `MultiObjectRGBDTracker`。这与 `vision_handoff.md` 中“抓取侧负责滤波、视觉侧提供原始观测”的约定不一致。

这里有两种可行方案：

1. 保持现有 tracker，用它输出稳定的中心位置，并在交接文档中明确这是视觉侧滤波结果；
2. 增加原始测量字段，`ObjectPose` 发布原始位置和协方差，把视觉 tracker 只用于目标选择、注册触发和内部显示。

推荐方案 2。至少应记录：掩码内有效深度比例、深度中位数/MAD、原始位置、跟踪位置和位置标准差，方便抓取侧调门限。

## 5. 坐标和姿态约定

FoundationPose/D435i 使用相机光学坐标系：

```text
X：向右
Y：向下
Z：向前
```

当前入口默认使用 IMU 竖直约束。`pose_with_upright_axis()` 会把物体的 CAD Z 轴对齐到相机坐标系中的重力方向；这满足抓取侧要求的“物体系 +Z 是瓶子长轴”。

对于旋转对称瓶子，绕瓶轴 yaw 不可观测。FoundationPose 的完整旋转四元数可能发生 yaw 漂移。抓取侧验收脚本已经测量瓶轴角度和 yaw 漂移，因此不要用完整四元数角度直接否定所有视觉帧。需要比较瓶轴方向，或者在控制侧忽略绕瓶轴的 yaw。

相机到 `head_pitch_Link` 的手眼标定仍是硬性前置条件。当前代码中的相机 frame 名称可能是：

- RealSense 本地模式：`d435i_color_optical_frame`；
- ROS2 模式：来自 `CameraInfo.header.frame_id`。

需要最终统一到 TF 树中真实存在的光学 frame，例如 `head_d435i_optical_frame`，并给出 `head_pitch_Link → camera` 的平移、旋转和标定残差。

## 6. 当前 FoundationPose 参数和运行策略

当前自主入口的关键设置包括：

| 项目 | 当前代码行为 | 对交接的影响 |
|---|---|---|
| YOLO 输入 | 默认 848×480，相对推理尺寸 640 | 需要在报告中记录实际延迟 |
| YOLO 精度 | CUDA 时默认 FP16 | 适合实时推理，需记录显存峰值 |
| YOLO 类别 | `--target-class` | 必须确认 checkpoint 的类别映射 |
| YOLO 掩码 | `mask-source=auto`，默认不启用 retina mask | 需要记录分割掩码有效深度比例 |
| 全局注册 | `register()` 异步执行 | 注册期间结果可能是旧姿态，桥接必须检查状态和新鲜度 |
| 跟踪 | `pose_tracking_mode` 默认使用 upright-follow | 当前更多依赖 RGB-D tracker 和竖直约束 |
| 重注册 | 目标稳定、丢失或运动刷新时触发 | 需要把注册耗时和触发原因写入诊断 |
| 无目标 | 选择器没有有效目标时不应发布可抓取位姿 | 桥接必须发布 invalid 或停止发布 |

`confidence` 当前主要来自 YOLO 检测置信度；FoundationPose 的 scorer 分数没有直接写入 `ObjectPose.confidence`。这需要在交付清单中明确，不能让抓取侧误以为它是综合置信度。

## 7. 推荐的最终接口方案

建议采用“视觉进程 + ROS2 薄适配层”的形式。视觉核心继续使用当前 FoundationPose 环境，ROS 适配层负责发布 `ObjectPose`。

### 7.1 视觉状态增加的字段

建议在 `latest_state.json` 增加：

```json
{
  "capture_stamp_s": 0.0,
  "sequence": 123,
  "object_id": "bottle",
  "source": "foundationpose",
  "raw_position_camera_m": [0.0, 0.0, 0.0],
  "center_pose_camera": [[...], [...], [...], [0, 0, 0, 1]],
  "position_covariance_m2": [0.0, 0.0, 0.0],
  "observation_valid": true
}
```

`center_pose_camera` 应使用：

```python
center_pose_camera = pose_camera_object @ inverse_to_origin
```

如果当前 tracker 位置作为中心位置使用，则必须把这一语义写入 `source` 或状态元数据，避免把 CAD 原点误当成中心。

### 7.2 ObjectPose 字段映射

```text
header.stamp       <- capture_stamp_s
header.frame_id    <- camera optical frame
object_id          <- "bottle"
pose.position      <- center_pose_camera[:3, 3]
pose.orientation   <- center_pose_camera[:3, :3] 转 geometry_msgs Quaternion
covariance[0,0]    <- sigma_x^2
covariance[7,7]    <- sigma_y^2
covariance[14,14]  <- sigma_z^2
confidence         <- YOLO confidence（明确记录为检测置信度）
valid_until        <- 0
source             <- "foundationpose"
sequence           <- 单调递增计数器
```

没有有效目标、目标未在当前帧观测、姿态质量无效或 IMU 约束不可用时，应停止发布有效 `ObjectPose`，不能持续发送最后一帧。

### 7.3 ROS QoS

抓取侧使用默认 RELIABLE QoS。发布端建议：

```python
publisher = node.create_publisher(ObjectPose, "object_pose", 10)
```

不要使用 `qos_profile_sensor_data` 或 BEST_EFFORT，否则会出现订阅端完全收不到消息的静默不兼容。

## 8. 当前环境报告

已在当前 `pose` Python 3.10 环境运行 `vision_env_report.py --json`，结果摘要如下：

| 项目 | 当前结果 |
|---|---|
| Python | 3.10.0 |
| `rclpy` | 可导入 |
| `sb_communication/ObjectPose` | 不可导入 |
| NumPy | 1.26.4 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 build |
| 当前 shell GPU | `torch.cuda.is_available() = false`，`nvidia-smi` 无法连接驱动 |
| Ultralytics | 8.4.102 |
| nvdiffrast | 0.4.0 |
| PyTorch3D | 0.7.9 |
| Warp | 1.16.0 |
| 环境指纹 | `b51c6fd041a6` |
| hard_pass | false |

这份报告只能说明当前执行环境的可见状态。目标机器人实际运行前，需要在目标 GPU 机器和已 source 的 ROS workspace 中重新运行，并确认 `ObjectPose` 消息可导入、GPU 可见、Torch 架构与显卡匹配。

## 9. 联调前必须补齐的资料

### P0：没有这些不能判断抓取坐标是否正确

1. `head_pitch_Link → D435i optical frame` 手眼标定结果和残差。
2. FoundationPose 使用的 mesh 文件、模型单位、模型原点、瓶子长轴。
3. 相机内参、深度尺度、RGB-depth 对齐设置、分辨率和帧率。
4. 观测时间戳的来源和两台机器是否做时钟同步。

### P1：没有这些不能调验收门限

1. 60 秒静止录包：位置标准差、帧率、观测延迟、四元数变化、瓶轴变化、yaw 漂移。
2. 30 秒移动录包：跟踪滞后、重注册耗时、真实移动是否误判为跳变。
3. 首次 register、重注册、持续跟踪的实际耗时和显存峰值。
4. 置信度定义和 covariance 估计方法。

## 10. 建议的实现顺序

1. 先固定 `ObjectPose` 语义：发布瓶子几何中心和瓶轴姿态。
2. 在视觉状态中加入采集时间戳、sequence、object_id、covariance 和有效标志。
3. 发布端补齐默认 RELIABLE QoS 的 `ObjectPose`。
4. 用固定假数据完成通信验收，再接真实相机。
5. 录制静止和移动数据，运行 `vision_interface_acceptance.py`。
6. 完成手眼标定后再进行 `waist_yaw_Link` 几何误差验收。
7. 最后进行 20 轮真实抓取测试。

## 11. 验收命令

环境信息：

```bash
python3 /home/robot/list/vision_env_report.py --json
```

静止测试：

```bash
python3 /home/robot/list/vision_interface_acceptance.py --static --duration 60
```

移动测试：

```bash
python3 /home/robot/list/vision_interface_acceptance.py --moving --duration 30
```

验收重点：

- ObjectPose 是否收到且 QoS 兼容；
- `header.stamp` 是否为拍照时间；
- position 是否为瓶子几何中心；
- orientation 第三列是否为瓶子长轴；
- confidence、covariance、sequence 是否有正确语义；
- 丢目标时是否停止发布；
- 静止位置标准差是否满足验收门限；
- 移动时是否出现延迟、乱序或错误跳变拒绝。
