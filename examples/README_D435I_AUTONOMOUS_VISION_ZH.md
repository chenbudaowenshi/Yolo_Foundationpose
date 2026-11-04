# D435i 自主多目标视觉运行说明

本文说明如何运行 [`d435i_autonomous_vision.py`](./d435i_autonomous_vision.py)。该程序将 Intel RealSense D435i、YOLOv8-seg、RGB-D 三维跟踪和 FoundationPose 组合起来，检测指定饮料类别的所有实例，锁定距离相机最近的有效目标，并持续输出机器人可读取的目标位置和 6D 位姿。

程序默认全自动运行，不需要按键触发首次注册或丢失恢复。

## 1. 运行前准备

### 1.1 硬件

- Intel RealSense D435i，建议直接连接 USB 3.x 接口。
- 支持 CUDA 的 NVIDIA GPU。FoundationPose 不能在当前流程中使用 CPU 运行。
- 与训练模型对应的真实物体，以及尺寸单位为米的 CAD 网格。

先确认 GPU 和相机能够被系统识别：

```bash
nvidia-smi
rs-enumerate-devices
```

### 1.2 默认文件

脚本默认使用以下路径：

| 用途 | 默认路径 |
|---|---|
| YOLO 分割模型 | `runs/segment/drink_seg/weights/best.pt` |
| CAD 网格 | `mesh/yinliao_edit_fixed.obj` |
| FoundationPose | `/home/hssw/FoundationPose` |
| 视觉状态 | `runs/autonomous_vision/latest_state.json` |
| 预览录像 | `runs/autonomous_vision/recordings/` |

在项目根目录检查关键文件：

```bash
cd /home/hssw/YOLOv8
ls -l runs/segment/drink_seg/weights/best.pt
ls -l mesh/yinliao_edit_fixed.obj
ls -l /home/hssw/FoundationPose/estimater.py
```

如果路径不同，运行时分别使用 `--yolo-model`、`--mesh` 和 `--foundationpose-root` 指定。

### 1.3 Python 环境

推荐使用已经安装好 FoundationPose、PyTorch、CUDA 和 RealSense Python 绑定的 `fopose` 环境：

```bash
conda activate fopose
cd /home/hssw/YOLOv8

python -c "import cv2, numpy, pyrealsense2, torch, trimesh; import nvdiffrast.torch; print('CUDA:', torch.cuda.is_available())"
```

最后必须输出 `CUDA: True`。若项目使用其他环境，可将其 Python 路径写入 `FOPOSE_PYTHON`；当默认的 `/home/hssw/miniconda3/envs/fopose/bin/python` 存在时，脚本也会自动切换到该解释器。

FoundationPose 本体及其 CUDA 扩展应按照 `/home/hssw/FoundationPose` 仓库的安装说明完成编译。仅安装本项目根目录的 `requirements.txt` 并不足以运行 FoundationPose。

## 2. 快速启动

以七喜为例：

```bash
conda activate fopose
cd /home/hssw/YOLOv8

python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --trust-checkpoints
```

当前饮料模型的类别为：

| `--target-class` | 类别名 |
|---:|---|
| `0` | `fenda` |
| `1` | `cola` |
| `2` | `qixi` |

`--trust-checkpoints` 是必需参数，表示你确认加载的 PyTorch/YOLO 权重来源可信。不要对未知来源的 checkpoint 使用此选项。

模型加载完成后，终端应出现类似输出：

```text
Loading FoundationPose scorer/refiner weights...
Autonomous target: 2=qixi
Vision state: /home/hssw/YOLOv8/runs/autonomous_vision/latest_state.json
```

把目标放在相机有效深度范围内并短暂停稳。轨迹确认且稳定约 1 秒后，程序会在后台自动执行 FoundationPose 注册；注册期间实时画面继续更新，YOLO 会暂时暂停。

## 3. 常用运行方式

### 无显示运行

适合机器人服务或没有桌面环境的终端。此模式使用 `Ctrl+C` 停止：

```bash
python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --no-display \
  --trust-checkpoints
```

### 保存标注预览

```bash
python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --record-preview \
  --trust-checkpoints
```

默认生成两个同名文件：

```text
runs/autonomous_vision/recordings/<时间>_annotated.mp4
runs/autonomous_vision/recordings/<时间>_frames.jsonl
```

### 同时录制 RealSense 原始数据

```bash
python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --record-preview \
  --record-bag runs/autonomous_vision/recordings/test_01.bag \
  --trust-checkpoints
```

`.bag` 输出路径必须不存在，程序不会覆盖已有的 RealSense 录像。

### 多台 RealSense 相机

先用 `rs-enumerate-devices` 查询序列号，再添加：

```text
--serial <D435i序列号>
```

### 使用自定义模型、网格或 FoundationPose 路径

```bash
python examples/d435i_autonomous_vision.py \
  --yolo-model /path/to/best.pt \
  --mesh /path/to/object.obj \
  --foundationpose-root /path/to/FoundationPose \
  --target-class 0 \
  --mask-source seg \
  --trust-checkpoints
```

CAD 必须与实物等比例，脚本按“米”解释网格尺寸。模型类别编号也必须与 `--target-class` 一致。

### 15 秒冒烟测试

```bash
python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --no-display \
  --max-runtime-seconds 15 \
  --trust-checkpoints
```

该测试仍然需要真实 D435i 和可用的 CUDA GPU。

## 4. 界面和按键

每个目标框上会显示类似信息：

```text
ID=3 OK   0.72m 0.95
ID=7 LOCK 0.58m 0.93
```

- `NEW`：新轨迹，尚未达到连续确认帧数。
- `OK`：已确认且可参与选择。
- `LOCK`：当前已锁定的最近目标。
- `0.72m`：目标三维中心到相机光心的距离。
- `0.95`：YOLO 置信度。

显示窗口可使用：

| 按键 | 功能 |
|---|---|
| `Q` 或 `Esc` | 安全退出 |
| `R` | 清除当前可信位姿并请求自动恢复，仅用于调试 |
| `V` | 开始或停止标注预览录像 |

正常生产流程不依赖任何按键。

## 5. 状态文件

程序默认以 10 Hz 原子更新：

```text
runs/autonomous_vision/latest_state.json
```

可通过以下参数修改位置和频率：

```text
--vision-state-file /path/to/latest_state.json
--vision-output-rate 10
```

如果只修改 `--autonomous-output-dir`，状态文件不会自动跟随到新目录；需要同时设置 `--vision-state-file`。

简化后的数据格式如下：

```json
{
  "schema_version": 1,
  "wall_time": "2026-08-04T12:00:00.000",
  "camera_frame": "d435i_color_optical_frame",
  "requested_class_id": 2,
  "requested_class_name": "qixi",
  "selected_track_id": 7,
  "selected_target": {
    "track_id": 7,
    "position_camera_m": [0.10, -0.02, 0.65],
    "position_std_m": [0.004, 0.004, 0.009],
    "distance_camera_m": 0.658,
    "track_status": "tracking",
    "observed_this_frame": true,
    "depth_valid_ratio": 0.72
  },
  "status": "pose_valid",
  "registration_busy": false,
  "pose_camera_object": [
    [1, 0, 0, 0.10],
    [0, 1, 0, -0.02],
    [0, 0, 1, 0.65],
    [0, 0, 0, 1]
  ],
  "pose_quality": {"valid": true},
  "yaw_observable": false,
  "tracks": []
}
```

`pose_camera_object` 是物体坐标系到 D435i 彩色相机光学坐标系的 `4×4` 齐次变换矩阵，平移单位为米。光学坐标系通常为 X 向右、Y 向下、Z 向前。

机器人使用位姿前，至少应同时检查：

```text
status == "pose_valid"
selected_track_id != null
pose_camera_object != null
selected_target.track_status == "tracking"
selected_target.observed_this_frame == true
pose_quality.valid == true
```

还应检查 `wall_time` 是否足够新，并由机器人侧设置超时。`yaw_observable=false` 表示当前圆柱物体的外观/几何不能可靠观测绕瓶身轴的旋转，抓取规划应按旋转对称目标处理。

常见状态包括：

| 状态 | 含义 |
|---|---|
| `searching` | 尚无可用目标 |
| `stabilizing` | 已选中目标，等待稳定 |
| `registering_recovery` | 正在后台注册或恢复位姿 |
| `pose_valid` | 位姿通过质量检查，可供下游进一步校验使用 |
| `pose_degraded` | 位姿质量暂时下降 |
| `reacquiring` | 位姿已丢失，正在自动重新获取 |
| `registration_retry_N` | 注册失败，等待第 N 次退避重试 |
| `candidate_failed` | 当前候选连续失败，暂时排除并选择其他实例 |

## 6. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--target-class` | `1` | 只搜索指定类别 |
| `--conf` | `0.35` | YOLO 置信度阈值 |
| `--iou` | `0.45` | YOLO NMS IoU 阈值 |
| `--imgsz` | `640` | YOLO 输入尺寸 |
| `--device` | `0` | CUDA 设备 |
| `--half` | 关闭 | 使用 FP16 YOLO 推理，要求 CUDA |
| `--max-det` | `100` | 单帧最大检测实例数 |
| `--track-confirm-hits` | `3` | 轨迹被确认所需命中次数 |
| `--track-max-missed` | `90` | 删除轨迹前允许的漏检帧数 |
| `--selection-distance` | `euclidean` | 按欧氏距离或相机 Z 深度选择最近目标 |
| `--registration-stable-seconds` | `1.0` | 目标稳定多久后开始注册 |
| `--registration-cooldown-seconds` | `5.0` | 两次刷新注册的最短间隔 |
| `--registration-max-attempts` | `3` | 单个候选连续注册尝试上限 |
| `--vision-output-rate` | `10.0` | 状态 JSON 更新频率（Hz） |
| `--upright-source` | `imu` | 直立轴来自 D435i IMU、首次可信位姿或关闭 |
| `--pose-tracking-mode` | `upright-follow` | 使用直立约束更新姿态并用 KF3D 更新平移 |

查看全部参数：

```bash
python examples/d435i_autonomous_vision.py --help
```

自主多实例入口不支持 `--pose-tracking-mode foundationpose`，应使用 `upright-follow` 或 `upright-hold`。自动注册、丢失恢复和移动后异步刷新默认已经开启，无需额外添加 `--auto-register`。

## 7. 常见问题

### `confirm the YOLO checkpoint is trusted`

确认权重确实来自可信训练流程后，在命令末尾添加 `--trust-checkpoints`。

### `No RealSense camera detected`

关闭 RealSense Viewer 等可能占用相机的程序，重新插拔相机，并确认 `rs-enumerate-devices` 能够列出设备。多相机环境应通过 `--serial` 明确指定。

### `D435i IMU permission denied`

安装并重新加载 Intel librealsense 官方 udev 规则，然后重新插拔相机。暂时无法配置 IMU 时可使用：

```text
--upright-source reference
```

该模式使用第一次可信注册的方向作为后续参考，无法像 IMU 模式一样跟随相机姿态变化更新世界竖直轴。

### `FoundationPose requires a visible CUDA GPU`

检查 `nvidia-smi`、CUDA 版 PyTorch 和当前环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

### `FoundationPose repository not found`

确认目录内存在 `estimater.py`，或者添加：

```text
--foundationpose-root /实际路径/FoundationPose
```

### `target class ... is absent`

`--target-class` 超出了当前 YOLO 模型类别范围。检查训练数据的类别顺序，或在可信权重上查看：

```bash
python -c "from ultralytics import YOLO; print(YOLO('runs/segment/drink_seg/weights/best.pt').names)"
```

### 有检测框但无法得到有效位姿

依次检查：

1. 目标是否在 D435i 的有效深度范围内，深度是否有大面积空洞。
2. YOLO 分割掩码是否完整，必要时调整 `--conf`。
3. CAD 是否与实物对应、比例是否为米、Z 轴是否为物体直立轴。
4. 目标是否保持稳定超过 `--registration-stable-seconds`。
5. `pose_quality.reason` 指出的失败原因，以及 `box_iou`、`position_error_m` 等字段。

透明或高反光瓶体的 D435i 深度可能不稳定。程序会拒绝深度质量不足的候选；不建议仅通过大幅降低位姿质量阈值来绕过问题。

### 服务器上出现 OpenCV/显示错误

添加 `--no-display`。如果仍需录像，`--record-preview` 可以与 `--no-display` 同时使用，但系统中的 OpenCV 必须支持 MP4 编码。

## 8. 安全退出与补充文档

有显示窗口时按 `Q` 或 `Esc`；无显示模式按 `Ctrl+C`。程序退出时会停止录像写入、后台注册任务和 RealSense 数据流。

更详细的算法状态、多实例选择逻辑和现有数据验证结果见：

- [`../docs/AUTONOMOUS_VISION_ZH.md`](../docs/AUTONOMOUS_VISION_ZH.md)
- [`../docs/RGBD_3D_TRACKER_ZH.md`](../docs/RGBD_3D_TRACKER_ZH.md)
- [`../docs/TEST_RECORDING_ZH.md`](../docs/TEST_RECORDING_ZH.md)
