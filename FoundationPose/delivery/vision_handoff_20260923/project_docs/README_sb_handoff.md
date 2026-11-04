# 双臂抓取视觉接口（2026-09-18）

独立自主检测演示的观测时效、预测状态与退出协议见 [自主检测输出说明](README_autonomous_state.md)。该修复保留自主选瓶规则，不替代下述正式抓取接口。

正式对接入口是 `sb_vision_node.py`，相机入口是 `run_realsense_viewer.py`。时钟检查默认使用 `required` 模式；当前同机调试需显式加 `--clock-sync-mode local`，跨机必须配置共同授时源，见[时钟同步说明](README_clock_sync.md)。本机 RTX 4070 运行 YOLO/FoundationPose；机器人仅运行相机采集、TF 和控制，不需要 NVIDIA 驱动。

当前采用用户确认的 `runs/mesh/bottle.obj`：外包围尺寸 65.068 × 65.122 × 159.5 mm。`config/bottle_current.json` 固定模型 SHA256、米单位、几何中心 `(0, 0, 0.07975)` 和物体 Z 长轴。这里的中心是已确认模型的包围中心。模型不作缩放，输出在测得的旋转下转换中心。下游原有 φ36 × 200 mm、质量 0.45 kg 及手部抓握偏移不能沿用；真实质量与新瓶径的 TCP/抓握中心仍待实测。

## 实现与 P0 对应

| P0 | 本仓库完成内容 | 仍需现场/下游完成 |
|---|---|---|
| 1–2 | 统一 `head_d435i_optical_frame`；`config/head_camera_calibration_20260918.*` 保存只读标定结果的米单位副本及 optical URDF 参考 | 部署真实 TF；核对相机安装、URDF 和实测 q_ours；额外零偏补偿为 0 |
| 3 | 直接通过内存发布 `sb_communication/ObjectPose`，RELIABLE，几何中心、归一化四元数、递增 sequence | 控制侧同步消息及配置 |
| 4 | SDK global/system clock；有元数据时转换为曝光中点；RGB/depth 各自原始时刻；异步推理任务保留原始时间；过期结果不发布 | 核实真机元数据、跨机时钟及延迟 |
| 5 | 模型指纹、尺寸、中心和长轴固定于配置 | 新瓶径抓取配置及碰撞体 |
| 6 | 注册后逐帧 `track_one`，发布 FP 原始测量旋转/位置；无 Kalman 覆盖或竖直硬覆盖 | 控制侧滤波与规划系倾角门 |
| 7 | YOLO 检测、投影/深度质量检查、无观测停发；锁定框关联、歧义停发、失踪 1 s 后重新选最近目标；reset 服务 | APPROACH 锁定后的异常策略与实物多瓶测试 |
| 8 | 交接包校验器按无方向瓶轴夹角判跳变；验收器同步 | 控制侧必须同步该版本；非对称物体选择 quaternion 模式 |
| 9 | Domain 10/CycloneDDS；JPEG 彩色与无损 PNG 深度收发；按时间匹配四个图像/内参话题；原始 SDK K/D；RGB 对齐深度；必要时同网格去畸变 | 机器人网卡/Peers 配置、带宽测试 |
| 10–11 | 本机协议、GPU 合成几何测试；记录运行时测量信息 | 实际 34.4° 待命角及工作距离定位、机械臂/TCP 和实物抓取验收 |

`hand_eyes_calibration` 源码和结果均未修改。URDF 副本**没有部署**，不能与现有同名 optical frame 同时发布，也不能直接写入实体 camera_link 的安装关节。旧 `d435i_autonomous_vision.py` / `foundationpose_pose_ros_node.py` 保留用于旧 demo/诊断；正式对接请启动本文的新入口。

## 环境与启动

首次启用持久化序号时，启动命令需额外带 `--sequence-start-after N`：新接收端用0，旧接收端填其实际最后接受的序号；后续启动不再需要该参数。不要删除 `.state/sb_vision`。详见[进程重启与序号持久化](README_sequence_state.md)。

本机使用 `/home/robot/anaconda3/envs/pose/bin/python`（Python 3.10、torch 2.6.0+cu124）；Python 3.11 的 `foundationpose` 环境不能直接加载本机 Humble 的 rclpy。

```bash
cd /home/robot/FoundationPose
bash my_demo/scripts/build_sb_interfaces.sh
# 仅本机测试时使用 lo；正式联网改为实际机器人网络接口。
export SB_DDS_INTERFACE=lo
source my_demo/ros2_handoff_env.sh
bash my_demo/scripts/run_sb_vision.sh --image-transport compressed --clock-sync-mode local --metrics-jsonl runs/sb_vision/live_metrics.jsonl
```

系统中缺少 CycloneDDS 时，标准安装命令是 `sudo apt install ros-humble-rmw-cyclonedds-cpp`。本次因 sudo 需要密码，已从 ROS 软件源下载二进制包并解压至 `.local_ros/root`，环境脚本会加载本项目的运行库。其他机器需要单独安装依赖。`ros2_domain91_env.sh` 作为兼容入口也已切换为该配置，默认 Domain **10**。

正式联网时，两端配置实际网卡，例如：

```bash
export SB_DDS_INTERFACE=enp3s0       # 替换为本机实际网卡
export SB_DDS_PEERS='192.168.112.104 192.168.112.252'  # 按现场拓扑填写
source my_demo/ros2_handoff_env.sh
```

机器人相机入口（source ROS 且安装 numpy、OpenCV、pyrealsense2、CycloneDDS 后）：

```bash
python3 run_realsense_viewer.py --image-transport compressed --jpeg-quality 90 --png-level 3 --width 1280 --height 720 --camera-fps 30 --publish-fps 15
```

默认使用 `compressed`：彩色为 JPEG（默认质量90），深度为16位 PNG（默认压缩级别3）。PNG 不做逆深度量化、深度截断或8位转换，完整保留原本要发布的16UC1毫米数值，包括0无效值。JPEG 是有损彩色压缩，分辨率、K/D和像素网格不变。

相机支持 `--image-transport raw|compressed|both`，视觉端支持 `--image-transport raw|compressed`；两端选择对应模式。`both` 同时发两套图像，仅供对比调试，CameraInfo仍只发一套。`raw` 模式下1280×720×15Hz的RGB+depth载荷约553Mbps；压缩后载荷以 `/head_camera/capture_status` 和实测报告为准，跨机网络仍需实测。

CameraInfo 的话题名及内容不随传输模式改变。RGB和深度各自携带原始采集时间，压缩、解压均不重新盖章；同步器在匹配到相应时间的CameraInfo后解码，并核对解码尺寸与内参尺寸。分辨率调整时由实际SDK stream读取K，不能复制另一分辨率的K。

| 话题 | 约定 |
|---|---|
| `/head_camera/color/image_raw/compressed` | 默认，CompressedImage，`rgb8; jpeg compressed bgr8`，RGB原始采集时间 |
| `/head_camera/depth/image_rect_raw/compressedDepth` | 默认，CompressedImage，`16UC1; compressedDepth png`，深度原始采集时间 |
| `/head_camera/color/image_raw` | raw/both模式，rgb8，RGB采集时间，RGB optical |
| `/head_camera/depth/image_rect_raw` | raw/both模式，16UC1毫米，深度采集时间，已对齐RGB网格 |
| `/head_camera/color/camera_info` | 对应 RGB 时间与尺寸，实际 K/D |
| `/head_camera/depth/camera_info` | 对应深度时间，使用对齐后 RGB 网格 K/D |
| `/head_camera/depth_scale` | 发布编码尺度恒为 0.001 m；传感器原始尺度见诊断 |
| `/head_camera/capture_status` | 相机序列号、时间来源、RGB-D时间差、SDK畸变、K/D、原始深度尺度、压缩耗时和图像载荷字节数 |
| `/object_pose` | `ObjectPose`，bottle，采集时间、光学系中心位姿 |
| `/vision/status` | 拒绝计数、最近模式/耗时/质量、延迟 P95 |

`image_rect_raw` 名称沿用交接约定：像素与原始 RGB 对齐；若 RGB 存在非零畸变，CameraInfo 保留真实 D，视觉入口对 RGB 与深度一起去畸变后推理。未知或无法正确映射的 SDK 畸变模型会显式报错，不伪装为零畸变。

相机现已默认开启 IMU，SDK 外参将加速度与角速度旋转到 RGB optical frame，使用原始运动流时间。自主/交互演示入口强制 IMU 对齐，见 [说明](README_imu_required.md)；本节的生产 `sb_vision_node.py` 仍采用原始 FP 6D 测量契约，不在此次演示入口修改范围内。

## 输出与状态规则

- 发布位置：`T_camera_model @ T_model_object_center`；物体 Z 是模型长轴。所有位置单位为米。
- `confidence = min(YOLO_confidence, sqrt(投影框IoU × 深度支持比例 × 掩膜有效深度比例))`。这是质量启发式，不是标定后的概率。
- 质量门：投影框 IoU ≥ 0.35，有效深度比例 ≥ 0.30，变换到模型包围体后的深度支持比例 ≥ 0.60（容差 15 mm），confidence ≥ 0.2。
- `covariance` 全零表示未知；没有把 tracker 平滑协方差冒充当前 FP 测量误差。
- 图像队列有界，每个 RGB/depth 帧只用一次，配对时间差默认 ≤20 ms，CameraInfo 必须匹配各自采集时间。过时输入、过期/未来结果、重复时间戳不发布。
- 注册可能超过 300 ms。旧注册结果可以初始化内部跟踪器，但不会重盖时间戳发布；下一张新图经跟踪和质量检查后才能发布。
- 丢目标即停止有效输出；生产节点默认以 10 Hz 刷新 YOLO，FoundationPose 仍处理每个 RGB-D 帧。检测刷新间隔内复用当前目标关联进行跟踪；检测漏失不超过 300 ms 时保留内部初值，重新检测后再用新图跟踪。相机短暂断帧（≤1 s）恢复后优先使用已有位姿种子跟踪，质量门失败才回退到重新注册。单帧跟踪质量失败时恢复上一有效内部初值，连续 3 帧失败才重新注册。多瓶歧义时不选；连续丢失超过 1 s 才允许重新选择。目标切换、注册和重注册都不重置 sequence。
- `ros2 service call /sb_foundationpose/reset_target std_srvs/srv/Trigger '{}'` 重新选目标并丢弃 reset 前在途结果，保留 sequence。进程重启通过持久化预留序号段继续递增，无需每次重置下游；首次升级需明确接收端序号下界，见[序号恢复说明](README_sequence_state.md)。
- `valid_until=0` 使用控制侧新鲜度窗口。控制侧仍须按采集时刻 TF 转换到 `waist_yaw_Link`，并处理目标锁定后的失踪/超时。

推理默认目标类别为 `0=fenda`，YOLO 输入长边 960、检测门限 0.30。FP 输入长边限制为 640，并同步按像素中心约定缩放 K，完整分辨率仍用于发布、质量验证。该限制避免全量重注册时显存随“候选数×原图尺寸”膨胀。现场需根据独立验证数据继续评估召回率及误检。

RTX 4070 部署时 `run_sb_vision.sh` 默认预加载 YOLO 和 FoundationPose，启动阶段会先花约 3～4 s 加载模型，首帧不再承担这次初始化。可以用以下参数调节速度与目标切换策略：`--detector-hz 5`～`10`、`--miss-grace-seconds 0.30`、`--reselect-after-seconds 1.0`。降低检测频率只减少重复 YOLO 计算，不降低 FoundationPose 跟踪频率；提高精度时保留 `--detector-image-size 960`、`--register-iterations 3` 和 `--track-iterations 2`。实际运行中的 `detection_fresh`、`detection_age_s`、YOLO/注册/跟踪耗时都写入 metrics JSONL。

首次注册、重注册、跟踪事件和实际运行显存使用统一记录与汇总，见 [注册性能统计说明](README_registration_metrics.md)。2026-09-19 的快速/全量实机结果见 `runs/registration_metrics_20260919/REPORT.md`。

## 本机测试

```bash
PYTHONPYCACHEPREFIX=/tmp/sb_test_pycache /home/robot/anaconda3/envs/pose/bin/python -m unittest discover -s my_demo/tests -v

export SB_DDS_INTERFACE=lo SB_ROS_DOMAIN_ID=71
source my_demo/ros2_handoff_env.sh
export ROS_LOG_DIR=/home/robot/FoundationPose/runs/sb_vision/ros_logs
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_ros_handoff.py
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/gpu_smoke.py
```

已完成 30 项回归测试、真实 CycloneDDS 上的合成相机/位姿集成测试，以及本机 RTX 4070 的真实 CUDA 推理测试。报告见 `runs/sb_vision/ros_smoke.json`、`gpu_smoke.json`。GPU 几何测试使用当前 mesh 渲染的已知 RGB-D 与 mask；另用保存的真实照片测试 YOLO。没有使用缺失内参的历史照片声称毫米级实测精度。

首次测试：注册 251 ms；跟踪 14.6～29.2 ms；8 帧合成图像的中心误差约 0.46～0.59 mm。测试期间 CUDA 分配峰值约 962 MiB。以上不包含相机、网络及完整持续 YOLO 开销，不等于实物抓取精度/帧率。

## 2026-09-18 原图模式的真实本机 D435i 静止验收

已使用本机 D435i（327122073383）和 RTX 4070 对用户摆放的瓶子完成真实 RGB-D → YOLO/FP → ObjectPose 测试。相机 1280×720、原始采集30Hz、发布15Hz；SDK 原始模型 inverse_brown_conrady，D实际为五个零。

`runs/sb_vision/live_acceptance_static.txt`：暖机后静止60秒、899条稳态观测，15.0Hz、最大空档106ms、延迟中位100ms/P95 123ms/最大156ms；位置σ=(0.52,0.42,0.95)mm，逐帧最大位移5.83mm，校验器接受率100%。光学系倾角项按脚本约定未判定；15项适用检查全部通过。抖动不是相对外部真值的绝对定位误差。

完整记录：`live_local.json`、`live_object_pose.json`、`live_metrics.jsonl`。相机单独测试报告是 `camera_local.json`。该机真实 K 与9月18日标定报告里的 K 不同；本次未使用头部标定外参进行基座定位，部署时仍需核对相机身份和安装关系。

相机发布已使用 `array('B')` 赋值，修复 Humble 对 bytes 逐元素校验导致的5.3Hz瓶颈。初期失败报告保留用于追溯，不覆盖成通过结果。重新运行完整静止验收：

```bash
# 先 source 上文环境并关闭占用同一相机的程序
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_live_vision.py --acceptance
```

## 2026-09-19 压缩传输验证

无相机的编解码/同步器/回归测试共47项，包含全部65536个uint16值的无损往返、彩色通道顺序、损坏数据拒绝，以及RGB/depth原始时间与CameraInfo匹配。

深度数据遵循ROS标准 `ConfigHeader`（小端int32 INV_DEPTH=0、两个float32参数）加16位灰度PNG。两个float参数对16UC1无作用。接收端兼容Humble的旧格式串 `16UC1; compressedDepth` 和显式PNG格式；不接受RVL或有量化的32FC1压缩格式。原图模式仍支持32FC1米单位深度。

已与官方 `compressed_depth_image_transport` 2.5.5 C++实现双向交叉验证全部65536个深度值。实现依据：[ROS codec.cpp](https://github.com/ros-perception/image_transport_plugins/blob/humble/compressed_depth_image_transport/src/codec.cpp)、[ConfigHeader](https://github.com/ros-perception/image_transport_plugins/blob/humble/compressed_depth_image_transport/include/compressed_depth_image_transport/compression_common.h)。Python运行端仅需要已有OpenCV、NumPy和ROS消息；官方插件用于互通验证，不是Python编码端的新增运行依赖。

```bash
# source前述ROS环境后，使用回环网卡及独立测试Domain
export SB_DDS_INTERFACE=lo SB_ROS_DOMAIN_ID=72
source my_demo/ros2_handoff_env.sh
export ROS_LOG_DIR=/home/robot/FoundationPose/runs/sb_vision/compression_20260919/ros_logs
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_ros_handoff.py --image-transport compressed
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_ros_handoff.py --image-transport both
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_compressed_depth_reference.py
# 真实相机测试，自动保存频率、延迟、压缩耗时、接收图像载荷与K/D
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_camera_stream.py --image-transport compressed --duration 20 --report runs/sb_vision/compression_20260919/camera_compressed.json
# 完整相机+GPU测试，报告默认存到runs/sb_vision/live_compressed/，可用--output-dir指定
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_live_vision.py --image-transport compressed --acceptance
```

原图与压缩图不能通过仅修改话题名互换，必须匹配消息类型及编解码。测试 `both` 模式会比较同一时间戳的原图深度与解压深度是否逐值相等。

本次验证结果保存在 `runs/sb_vision/compression_20260919/`：

- `ros_raw.json`、`ros_compressed.json`、`ros_both.json`：三种模式的真实CycloneDDS收发全部通过，覆盖时间戳、CameraInfo、深度数值、缺失停发及reset后的序号；使用合成相机/位姿后端。
- `standard_codec_interop.json`：与ROS官方C++插件双向验证65536个深度值，全部一致。
- `camera_compressed.json`：真实D435i压缩链路20秒测试通过，292帧，约15.00Hz，最大空档96.5ms，采集到解码接收延迟P95约97.2ms。
- 该次场景接收图像载荷约21.7Mbps，压缩比随纹理和有效深度变化，且不包含DDS/IP开销。`payload_benchmark.json` 的另一组保存的真实RGB-D样本为4.608MB→0.507MB/帧，减少约89%，按15Hz折算约60.9Mbps；这是离线编码测量。
- `restored_camera.json` 和 `camera_restored_compressed.log`：记录相机暂停测试后以项目配置恢复压缩发布的信息。

这些结果验证了压缩与相机收发；不能替代压缩模式下的完整GPU姿态验收或跨机网络验收。2026-09-18的静止位姿报告仍对应原图模式。


## 机器人开机后的部署与实测

当前机器人未开机，尚未部署或进行真机测试。部署脚本只上传独立、带版本号的相机程序，不覆盖手眼标定目录，不自动停止其他相机进程：

```bash
bash my_demo/scripts/deploy_camera.sh 192.168.112.104
```

脚本输出机器人上的 release 目录。在该目录下 source ROS、配置 Domain 10/CycloneDDS 后运行：

```bash
python3 scripts/test_camera_stream.py --duration 20 --report camera_test_report.json
```

该测试实际启动相机并通过 ROS 订阅校验 RGB-D/CameraInfo，要求 ≥10Hz、空档 <250ms、采集年龄 P95 ≤250ms。若相机已被其他进程占用，先明确现场运行程序的归属再切换。

相机链路通过后，在本机运行视觉入口，保存静止 60 s、移动 30 s 的真实观测流，运行交接包 `vision_interface_acceptance.py`。再独立验收头部待命角、工作区位置误差和 TCP/抓取成功率。控制侧原有文档中的“16 项通过”不涵盖真实 TF、TCP 与实物精度。
