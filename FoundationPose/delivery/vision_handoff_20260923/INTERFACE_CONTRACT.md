# ObjectPose 接口与字段语义

## 进程边界

相机和视觉是两个独立进程：

- `my_demo/run_realsense_viewer.py` 打开 D435i 并发布 RGB-D、CameraInfo、IMU。
- `my_demo/scripts/run_sb_vision.sh` 只启动视觉订阅节点，不会打开相机。

## ROS 话题

| 方向 | 话题 | 内容 |
|---|---|---|
| 订阅 | `/head_camera/color/image_raw/compressed` | JPEG 彩色图 |
| 订阅 | `/head_camera/depth/image_rect_raw/compressedDepth` | 16UC1 无损 PNG 深度 |
| 订阅 | `/head_camera/color/camera_info` | 彩色相机内参 |
| 订阅 | `/head_camera/depth/camera_info` | RGB 对齐深度内参 |
| 订阅 | `/head_camera/capture_status` | 相机和时间诊断 |
| 发布 | `/object_pose` | `sb_communication/msg/ObjectPose`，RELIABLE |
| 发布 | `/vision/status` | 计数、时钟、性能和序号状态 |

## 字段

| 字段 | 当前语义 |
|---|---|
| `header.stamp` | RGB 曝光/采集时间，原样透传 |
| `header.frame_id` | `head_d435i_optical_frame`；X 右、Y 下、Z 前 |
| `object_id` | 固定 `bottle` |
| `pose.position` | 瓶子几何中心，米，相机光学系 |
| `pose.orientation` | FoundationPose 旋转；物体 Z 为瓶轴；绕瓶轴 yaw 对对称瓶不可观测 |
| `covariance` | 36 个零，表示未知，不冒充标定协方差 |
| `confidence` | `min(YOLO confidence, sqrt(IoU * depth_support * valid_depth_ratio))` |
| `valid_until` | 零，由控制侧新鲜度窗口处理 |
| `source` | `foundationpose` |
| `sequence` | 每次发布严格递增，持久化跨进程重启 |

质量门：投影框 IoU >= 0.35、有效深度比例 >= 0.30、CAD 深度支持率 >= 0.60（15 mm 容差）、confidence >= 0.2。过时、未来、重复时间戳和质量不合格结果不发布。

`ObjectPose.msg` 与 `/home/robot/list/ObjectPose.msg` 的 SHA-256 均为 `03463e448beaa0829223ed5afd90993ece5f307b9217562b8c163e8d853f0bda`。
