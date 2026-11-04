# IMU 强制参与注册和位姿有效性判断

适用入口：`run_realsense_viewer.py`、`d435i_autonomous_vision.py`、`d435i_foundationpose.py` 及自主状态的诊断读取端。

## 默认行为

- 相机默认启用加速度计和陀螺仪，发布 `/head_camera/imu`。`--use-imu` 保留为兼容参数，不再需要显式添加。
- 两个检测入口只允许 `--upright-source imu`，默认就是此模式。禁用 IMU 或用首次位姿代替 IMU 的模式已移除。
- 注册前、异步结果验收时和对外输出前必须有与本图像对齐的有效 IMU。缺失时状态为 `waiting_for_imu`，`pose_valid=false`。
- IMU 故障不累计为目标注册失败，不因 IMU 故障主动排除或改选瓶子。原有物体关联、遮挡预测和自动重新选择规则保留。
- 必须重新启动旧相机与视觉进程才会加载修改。话题列表出现 IMU 名称可能仅代表有人订阅，需要确认发布者和实际消息。

## 时间和坐标

`imu_alignment.py` 维护有界 IMU 缓冲，按原始 `Imu.header.stamp` 匹配图像 `Image/CompressedImage.header.stamp`，不使用消息接收时刻刷新旧数据。

默认门限：

- IMU 与图像采集时差绝对值不超过 **30 ms**，参数 `--max-imu-skew-ms`。
- IMU 采集时间距当前系统时间不超过 **0.5 s**，参数 `--imu-timeout`。
- 超前系统时间超过 5 ms 的 IMU 在入队时被拒绝，不会保留到将来伪装为当前数据。
- 重复消息不推进时间和重力滤波；非法时间戳、非有限加速度、异常重力模长或空坐标系被拒绝。

SDK 加速度和角速度在相机端通过 SDK 外参旋转到 RGB optical frame，并保留运动流自身时间戳。不能只把 header 改成 optical 名称而不旋转向量。
相机在后台持续读取 SDK 帧并保留有界 IMU 缓冲，即使 ROS 只发布 15 Hz，也不会丢掉上一原始帧中更接近曝光时刻的 IMU 样本。发布时选择接近图像时刻的真实加速度及陀螺仪样本，不修改时间戳凑齐。
默认 IMU 和图像均为 `head_d435i_optical_frame`，不需要额外 TF。
若输入 IMU 在另一个坐标系，视觉端查询 **IMU 采集时刻**的 `T_color_imu`；缺失 TF 时拒绝，不能用最新 TF 或假定单位矩阵代替。
本地直接使用 RealSense 的路径也检查实际 IMU/图像 SDK 时间，不能无限复用旧重力轴。

IMU 在此提供重力轴约束；不提供可观测的绝对 yaw，也不替代手眼标定。

## 状态字段

`latest_state.json` 增加 `imu_required=true` 和 `imu_alignment`：

| 字段 | 意义 |
|---|---|
| `ready`、`reason` | 本次图像是否具有有效对齐 IMU，以及拒绝原因 |
| `image_stamp_ns` | 本次匹配的图像采集时间 |
| `imu_stamp_ns` | 对应 IMU 采集时间 |
| `skew_ms` | IMU 时间减图像时间，毫秒 |
| `age_s` | 当前时间减 IMU 采集时间，秒 |
| `source_frame`、`camera_frame` | 原 IMU 坐标系及使用的相机光学系 |

`imu_fresh` 现在表示采集时间和坐标均通过检查，不再表示“刚刚收到一条消息”。
正常退出强制清除 IMU ready/fresh 标志。读取端对 v2 状态重新验证 IMU 时间、图像绑定和光学坐标，即使传入旧的 `require_imu_fresh=False` 也不能绕过。
诊断 ROS 节点的 `use_imu` 默认 true，设置 false 会报错；其 `maximum_imu_skew_ms` 默认 30，应与视觉端配置一致。

## 检查与回归

```bash
source /opt/ros/humble/setup.bash
python my_demo/scripts/check_imu_alignment.py --ros-domain-id 0 --seconds 5
```

成功时会给出发布者、实际 IMU 消息数、图像匹配数和时差分位数。

```bash
source my_demo/ros2_handoff_env.sh
PYTHONPYCACHEPREFIX=/tmp/imu-tests python -m unittest discover -s my_demo/tests -v
SB_ROS_DOMAIN_ID=84 SB_DDS_INTERFACE=lo source my_demo/ros2_handoff_env.sh
python my_demo/scripts/test_ros_imu_required.py
```

DDS 测试在私有话题上验证正常、断流、旧时间戳、错时、异常加速度及恢复，不发布机器人运动指令。
