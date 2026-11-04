# FoundationPose 使用当前 ROS2 相机话题

以下两个演示入口共用 `ros_rgbd_input.py`：

- `d435i_autonomous_vision.py --ros2`：自主检测、跟踪和自动注册。
- `d435i_foundationpose.py --ros2`：交互式演示，按 R 注册，或使用 `--auto-register`。

默认订阅与 `run_realsense_viewer.py` 的压缩发布一致：

| 数据 | 话题 | 消息类型 |
|---|---|---|
| 彩色 | `/head_camera/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage`，JPEG |
| 对齐深度 | `/head_camera/depth/image_rect_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage`，16UC1 无损 PNG |
| 彩色内参 | `/head_camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` |
| 深度内参 | `/head_camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` |
| 必需 IMU | `/head_camera/imu` | `sensor_msgs/msg/Imu` |

接收端按采集时间匹配两张图及各自 CameraInfo；使用实际内参与光学坐标系。
默认从彩色 CameraInfo 读取 frame_id，也可用 `--camera-frame head_d435i_optical_frame` 固定校验。
JPEG 与压缩深度均进行真正解码；不会将压缩话题当成原始 Image。
深度单位按 ROS 编码解释：16UC1/PNG 是毫米，32FC1 是米；推理输入统一为米，保留原始采集时间。
`/head_camera/depth_scale` 无需等待或重复应用；旧 `--depth-scale-topic` 参数仍可解析，但会提示按标准编码解释单位。

## 必须使用 IMU 的启动方式

相机端现已默认开启 IMU，`--use-imu` 仅为兼容旧命令保留。旧进程不会热加载修改，请重新启动相机和视觉程序。

终端一：

```bash
conda activate pose
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=0 python my_demo/run_realsense_viewer.py
```

终端二：

```bash
conda activate pose
source /opt/ros/humble/setup.bash
python my_demo/d435i_autonomous_vision.py --ros2 --ros-domain-id 0 --trust-checkpoints
```

交互式入口使用 `python my_demo/d435i_foundationpose.py --ros2 --ros-domain-id 0 --trust-checkpoints --auto-register`。
两个入口均不再支持 `--upright-source off/reference`。IMU 缺失、过期、与图像时间差超限或坐标变换不可用时，不进行注册，也不提供有效位姿。IMU 恢复后继续原目标流程，不把 IMU 故障当成目标失败。

`ros2 topic list` 中的 IMU 名称也可能只来自订阅者。检查实际发布者与时间对齐：

```bash
ROS_DOMAIN_ID=0 ros2 topic info /head_camera/imu --verbose
ROS_DOMAIN_ID=0 ros2 topic echo /head_camera/imu --once --qos-reliability best_effort
python my_demo/scripts/check_imu_alignment.py --ros-domain-id 0
```

默认自动注册，无需按 R；调试按键须让检测图像窗口获得焦点。R 不能绕过 IMU 门控。
更多时序、坐标与输出字段见 [IMU 强制对齐说明](README_imu_required.md)。

## 自定义话题和原图模式

同一命名空间只需修改 `--camera-prefix`。原图模式必须收发一致：

```bash
ROS_DOMAIN_ID=0 python my_demo/run_realsense_viewer.py --image-transport raw --topic-prefix /wrist_camera
python my_demo/d435i_autonomous_vision.py --ros2 --ros-domain-id 0 --trust-checkpoints --image-transport raw --camera-prefix /wrist_camera
```

逐个覆盖参数：`--rgb-topic`、`--depth-topic`、`--camera-info-topic`、`--depth-camera-info-topic`、`--imu-topic`。
原图模式默认去掉 `/compressed`、`/compressedDepth` 后缀，CameraInfo 名称不变。
自定义图像名称不会自动改变类型，需同时选择正确的 `--image-transport`。

## 域与排查

`--ros-domain-id` 只设置当前视觉进程，不修改终端或其他节点的域；不传时遵循 `ROS_DOMAIN_ID`，未设置则为 0。
`source my_demo/ros2_handoff_env.sh` 默认设置 Domain 10；若使用它并与当前 Domain 0 相机通信，先设置 `SB_ROS_DOMAIN_ID=0`，或者在演示入口显式传 `--ros-domain-id 0`。
启动日志打印实际域、传输类型和四个已解析话题。收不到数据时显示每路消息计数以及最近一次解码/标定错误。
`ros2 topic list` 也需要与发布端使用同一个域；存在话题并不代表类型、QoS、两份 CameraInfo 与采集时间已经匹配。

生产抓取入口 `sb_vision_node.py` 原本已支持该压缩协议，本次没有改变其选目标或发布契约。
相机日志每 10 秒重复“第一帧已发布”的问题已修正：首帧通知与分段帧率计数分离。
