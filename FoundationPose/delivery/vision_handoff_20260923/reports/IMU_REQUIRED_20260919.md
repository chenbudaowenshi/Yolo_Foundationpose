# 强制 IMU 与图像时间对齐验收

本轮实现：相机默认发布 IMU；自主和交互演示入口不再提供无 IMU / reference 模式；注册前、异步结果验收时及有效位姿输出前都必须有匹配图像采集时间的 IMU。没有修改手眼标定、目标关联、遮挡宽限或自动换目标规则，没有发布机器人运动指令。

## 关键改动

- `run_realsense_viewer.py` 默认开启加速度计和陀螺仪，发布 `/head_camera/imu`，保留加速度真实采集时间。
- `RealSenseCamera.py` 后台持续读取 SDK 帧并保存有界运动样本；从历史中选取接近曝光时刻的真实加速度/陀螺仪，避免只读 ROS 15 Hz 发布时刻的最新样本。
- `imu_alignment.py` 按图像/IMU header 时间匹配，默认最大时差 30 ms、IMU 最大年龄 0.5 s。重复包不更新时间；超前时间戳不会排队等待变“新”。
- SDK 外参旋转后的 IMU 默认已在 RGB optical frame。若源坐标系不同，按 IMU 采集时刻查询 TF；无变换时停止有效输出，不使用最新 TF 或单位变换代替。
- `d435i_autonomous_vision.py` 与 `d435i_foundationpose.py` 仅允许 IMU 模式；无有效对齐 IMU 时不注册、不验收旧注册结果、不输出有效位姿。
- IMU 故障不记为目标注册失败，不因 IMU 故障主动排除目标。诊断读取端独立核验 IMU 时间、图像绑定及 optical frame，不能通过旧的关闭门控参数绕过 v2 协议。
- 输出 `imu_required`、`imu_alignment.ready/reason`、图像和 IMU 原始时间、时差与坐标系；退出清除 IMU fresh/ready 状态。

## 验证

| 检查 | 结果 |
|---|---|
| 单元/回归 | 126 项通过，`tests.log` |
| 真实 DDS 正常输入 | 19/19 对通过 |
| DDS 缺失 IMU、旧时间、错时、异常重力 | 各 19 对全部拒绝 |
| DDS IMU 恢复 | 20/20 对恢复通过 |
| D435i 默认启动，无显式 --use-imu | 真实 IMU 发布者 1、105 条 IMU，104/104 对图像匹配成功 |
| 真实采集时差 | 绝对值 P50 5.07 ms、P95 9.84 ms、最大 10.21 ms |
| 完整自主 ROS2 程序 | 25 秒、375 次新 RGB-D 检测，1,117 条循环状态中的 IMU 均有效；正常退出为 stopped |

真实场景当前仍未得到 YOLO 目标，因此没有有效物体位姿；本轮实机结论是 IMU 接收、时间/坐标对齐和门控链路通过，不代表完成物体位姿或抓取精度验收。

首次实机检测暴露了“只用最新运动样本”的问题：75 对中 19 对超过 30 ms。保留该失败记录在 `live/`；随后增加 SDK 连续采集与历史样本选择，最终 `live_buffered/` 达到 104/104。没有增大 30 ms 阈值，也没有把 IMU 时间戳改成图像时间凑齐。

启动初期出现过一帧约 1 秒旧的 SDK 图像，原有相机年龄门限将其拒绝；随后稳态发布约 15 Hz。测试相机和视觉进程均正常结束，没有留下本次测试实例。

## 文件与启动

- `dds_imu.json`、`dds_imu.log`：私有 Domain 84 / loopback 故障与恢复测试。
- `live_buffered/alignment.json`：每个真实对齐样本及分位数。
- `live_buffered/vision_summary.json`、`vision.log`、`camera.log`、`recordings/`：实际视觉入口测试。
- `my_demo/README_imu_required.md`、`my_demo/README_ros_camera_input.md`：规则和双终端命令。
- `my_demo/scripts/check_imu_alignment.py`：只订阅的在线检查工具。

旧进程需重启加载修改。在两个终端分别运行：

```bash
ROS_DOMAIN_ID=0 python my_demo/run_realsense_viewer.py
```

```bash
python my_demo/d435i_autonomous_vision.py --trust-checkpoints --ros2 --ros-domain-id 0
```

`--use-imu` 仍接受但已不是必需参数；`--upright-source off/reference` 会直接报错。话题存在可能只来自订阅者，可用 `ros2 topic info /head_camera/imu --verbose` 确认发布者，再运行 `check_imu_alignment.py --ros-domain-id 0` 检查真实消息与时间差。
