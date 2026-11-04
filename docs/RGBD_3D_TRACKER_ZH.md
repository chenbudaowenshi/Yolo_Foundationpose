# D435i RGB-D 三维位置跟踪与直立瓶位姿约束

本项目在 YOLOv8-seg 与 FoundationPose 之间加入了三维常速度卡尔曼跟踪层，并针对百事、芬达、七喜这类近似圆柱、裸 CAD 模型增加了旋转防漂移机制。

默认使用 `upright-follow`：FoundationPose 只在全局注册时求一次完整 6D 位姿；因为饮料瓶已知始终竖直，注册结果的 CAD Z 轴会直接吸附到 D435i 的平滑重力轴。之后重力轴更新滚转/俯仰，KF3D 实时更新平移，不再因为物体移动或手持相机运动自动调用耗时的全局注册。绕瓶身轴的朝向沿用第一次注册结果。

## 数据流

```text
D435i RGB + 对齐深度 + 加速度计
        ↓
YOLOv8-seg 目标框和掩膜
        ↓
掩膜深度鲁棒过滤
        ↓
像素中心 + 深度 + K 反投影
        ↓
三维测量 [X,Y,Z]
        ↓
常速度卡尔曼滤波 [X,Y,Z,Vx,Vy,Vz]
        ↓
位置、速度、静止状态、异常测量门控（KF3D）
        ↓
FoundationPose 全局注册
        ↓
注册姿态吸附到已知重力直立轴
        ↓
IMU 更新直立轴 + KF3D 更新平移
        ↓
连续异常时保留流畅画面并提示手动按 R
```

跟踪器实现位于 `examples/rgbd_kalman_tracker.py`，重力与锁姿态工具位于 `examples/upright_pose_utils.py`，实时集成位于 `examples/d435i_foundationpose.py`。

## 首次启用 D435i IMU

当前主机缺少 librealsense 的 udev 规则，因此 RGB/深度可用，但加速度计会报 `scan_element ... Permission denied`。只需安装一次 Intel 官方规则：

```bash
curl -fsSL \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.3/config/99-realsense-libusb.rules \
  -o /tmp/99-realsense-libusb.rules

sudo install -m 644 /tmp/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

执行后拔掉并重新插入 D435i。以上命令不会修改本项目或相机固件。若暂时不安装，可在运行命令末尾添加 `--upright-source reference`；这仍会锁住第一次可信注册的旋转，但第一次注册无法利用重力判断瓶子是否真正竖直。

## 运行

```bash
conda activate fopose
cd ~/YOLOv8

python examples/d435i_foundationpose.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --debug 1 \
  --auto-register \
  --record-preview \
  --trust-checkpoints
```

类别：`0=fenda`、`1=cola`、`2=qixi`。

## 界面

- 紫色框和掩膜：YOLOv8-seg 二维测量。
- 黄色十字：卡尔曼滤波后的三维中心重新投影到图像的位置。
- 绿色框和坐标轴：FoundationPose 的 CAD 六维位姿。
- `UP-FOLLOW`：默认模式；瓶身轴跟随平滑重力方向，平移由 KF3D 更新。
- `ROT-HOLD`：完全锁定注册旋转，适合固定相机对照实验。
- `FP-TRACK`：兼容旧方案，FoundationPose 每帧调用 `track_one()` 更新旋转和平移。
- `UP=IMU`：直立轴来自 D435i 加速度计；`UP=REF` 表示来自第一次可信注册。
- `tilt=...deg`：CAD 的 Z 轴与重力/参考轴夹角；默认超过 15° 会拒绝位姿。
- `moved=no/yes`：本次注册后是否检测到物体发生明显移动。
- `KF3D tracking`：当前正在融合测量。
- `KF3D coasting`：当前帧没有可信测量，仅用运动模型短时预测。
- `meas=ok`：当前三维测量已接受。
- `meas=reject`：测量跳变超过马氏距离门限，已拒绝。
- `predict-only`：当前帧没有可用三维测量。

按键：

- `R`：FoundationPose 注册或重新注册。
- `C`：清除六维位姿，但保留三维位置轨迹。
- `T`：重置三维位置跟踪器。
- `S` 或空格：保存 RGB、深度、掩膜、位姿和跟踪器 JSON。
- `Q` 或 `Esc`：退出。

正常现象：在 `UP-FOLLOW` 模式中，移动瓶子或缓慢改变手持相机俯仰时，绿色 CAD 框连续运动，瓶身轴保持世界竖直。程序默认不会在停稳时卡住做全局注册；只有你按 `R` 时才会出现约一秒的注册停顿。

## 估计状态

卡尔曼状态为：

```text
[X, Y, Z, Vx, Vy, Vz]
```

坐标系为 D435i 彩色相机坐标系：

```text
+X 向图像右侧
+Y 向图像下方
+Z 向相机前方
```

掩膜深度通常落在瓶子可见前表面。程序默认沿像素射线增加 CAD 横向半径，把前表面深度近似修正到瓶子中心。可使用 `--tracker-center-depth-offset` 指定实测偏移。

## 旋转防漂移参数

```text
--pose-tracking-mode upright-follow
```

默认且推荐用于已知始终竖直的圆柱饮料瓶。它不会逐帧调用 FoundationPose，而是保留第一次注册的绕轴朝向，只根据IMU更新瓶身竖直轴。固定相机、希望旋转完全不动时可用 `upright-hold`。旧行为可用 `foundationpose` 恢复，同时会启用旋转角速度门控。

```text
--upright-source imu
--max-upright-tilt 15
--gravity-smoothing 0.10
--accel-fps 100
```

IMU 重力轴经过指数平滑。当前 D435i 实测支持 100/200/400 Hz，默认使用 100 Hz。因为相机处于手持状态，IMU仍有价值：它告诉程序“世界竖直方向在当前相机坐标系里指向哪里”。它不再触发反复注册。加速度计不能观测绕重力轴的 yaw，圆柱瓶的这个自由度继续使用第一次注册值。

```text
--pose-motion-speed 0.040
--pose-motion-distance 0.020
```

KF3D 速度达到 0.04 m/s，或相对注册位置移动达到 2 cm，就标记为已经移动，但默认只做诊断，不再触发全局注册。只有同时显式添加 `--reregister-after-motion` 时，目标停稳后才会重新注册。

为避免实时画面周期性冻结，姿态丢失后也默认等待你按 `R`。如确实需要旧的自动恢复行为，可添加 `--auto-reregister`。这两个自动注册选项都会造成约一秒的画面停顿，不建议在当前手持测试中启用。

## 三维位置参数

```text
--tracker-accel-std 0.10
```

运动模型允许的加速度噪声。增大后响应更快但位置和速度更抖；减小后更平滑但快速运动时滞后。

```text
--tracker-gate 16.27
```

三维测量的马氏距离门限。超过门限的突然跳变不会进入滤波器。

```text
--tracker-max-missed 8
```

最多允许连续多少帧只预测不更新，超过后轨迹进入 `lost`。

```text
--tracker-stationary-speed 0.030
--tracker-stationary-frames 5
```

速度低于 0.03 m/s 并持续5帧后认为目标静止。默认仅用于判断首次注册条件；显式启用 `--reregister-after-motion` 后才用于触发重新注册。

```text
--tracker-min-valid-depth-ratio 0.20
```

掩膜内过滤后有效深度少于 20% 时，不生成三维测量。

```text
--max-pose-3d-error 0.10
```

FoundationPose CAD中心与卡尔曼三维中心相差超过10厘米时，六维位姿质量检查失败。

如需做对照实验，可添加 `--no-3d-tracker`。

注意：`upright-follow` 和 `upright-hold` 必须使用 KF3D，因此 `--no-3d-tracker` 需要同时添加 `--pose-tracking-mode foundationpose`。

## 诊断文件

按 `S` 后，`runs/foundationpose_realtime/snapshots/` 中新增：

```text
<时间>_tracker.json
```

每次 FoundationPose 注册时，`runs/foundationpose_realtime/registrations/<时间>/metadata.json` 中新增：

```text
rgbd_measurement
tracker
pose_projection_quality.position_error_m
pose_projection_quality.upright_tilt_degrees
gravity
pose_tracking_mode
motion_since_registration
```

其中 `position_std_m` 是滤波器对位置标准差的估计，不是通过真值测得的实际误差。

## 17 秒漂移为什么会发生

旧问题视频中，17 秒附近 YOLO 置信度约0.95，二维框和KF3D位置都稳定，但FoundationPose的旋转从约 `(39.6, -17.2, -2.8)°` 连续漂到 `(-0.6, -42.4, 72.0)°`。原来的质量门控只检查投影外接框IoU和三维位置；圆柱倾斜或绕轴旋转后，外接框仍可能与二维框重叠，所以错误姿态能够逐帧通过。

在 `20260724_113333_399_annotated.mp4` 中，150秒内触发了约40次全局注册，产生39个超过0.5秒的帧间断，最大约1.34秒；姿态有约55%的时间处于缺失状态。这说明视觉卡顿来自同步执行FoundationPose注册，而不是IMU读取或GPU速度。新版默认关闭移动后和丢失后的自动注册，并把注册姿态直接吸附到已知竖直轴。

根因是裸 CAD 缺少纹理、瓶体接近旋转对称、七喜透明表面对深度相机不友好。3D 卡尔曼滤波只约束位置，无法自动约束旋转。新方案把“可稳定观测的位置”和“难以稳定观测的旋转”分开处理，因此不会再把优化噪声误认为真实旋转。

## 限制

- 这是单类别、单实例的三维位置跟踪层，还没有同类别多目标 ID 和匈牙利匹配。
- 卡尔曼滤波能减少随机抖动，不能修复持续错误的透明瓶背景深度。
- 黄色中心是由二维框中心和鲁棒深度估算的近似几何中心，精确抓取前仍需用真值标定其系统偏差。
- `upright-follow` 和 `upright-hold` 会用黄色 KF3D 中心更新 CAD 平移；因此 CAD 尺寸、从可见前表面到几何中心的偏移必须正确。
- 对圆柱瓶而言，绕瓶身 Z 轴的 yaw 在几何和裸模型上不可观测。锁定能阻止虚假漂移，但不会凭空获得真实标签朝向；若抓取必须区分正反面，需要带正确纹理的模型、清晰标签特征或额外姿态标记。
- 机器人执行前仍需手眼标定、抓取变换、时序同步和安全门控。
