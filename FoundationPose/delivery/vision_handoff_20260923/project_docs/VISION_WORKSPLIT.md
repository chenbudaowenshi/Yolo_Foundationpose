# 视觉接入：分工与交付清单

日期：2026-09-10
参与方：抓取/控制侧（我）、视觉侧（同事，YOLOv8-seg + FoundationPose）

配套文档：[vision_handoff.md](vision_handoff.md)（接口约定的完整版，本文只讲谁做什么）

---

## 0. 架构：三台机器，为什么这么分

```
┌─ 机器人本体 ────────────┐   ┌─ 同事电脑 ──────────┐   ┌─ 我的电脑 ─────────────┐
│ i7-12700H / Iris Xe     │   │ NVIDIA GPU          │   │ RTX 5070 Ti            │
│ 没有 NVIDIA 显卡        │   │ torch 2.6+cu124     │   │                        │
│                         │   │ Python 3.10         │   │                        │
│ · D435i 相机节点        │──▶│ · YOLOv8-seg        │──▶│ · target_manager       │
│   RGB-D（压缩）         │   │ · FoundationPose    │   │ · grasp_planner        │
│ · shuangbi_demo 50 Hz   │◀──│ · 发 ObjectPose     │   │ · task_manager / 面板  │
│ · robot_state_publisher │   │                     │   │                        │
└─────────────────────────┘   └─────────────────────┘   └────────────────────────┘
        盖 header.stamp                                      量 observation_delay
```

**为什么视觉必须离机**：机器人只有 Intel Iris Xe 核显，FoundationPose 依赖 nvdiffrast + PyTorch3D，都是 CUDA 的，**没有任何迁移到 Intel 的路径**。这不是选择题。

**这么分带来的最大好处**：同事的 `torch 2.6.0+cu124` 跑在他自己的显卡上，**Blackwell / sm_120 的问题直接消失**——不用换 torch，也就不用重编 nvdiffrast 和 PyTorch3D。那本来是整件事里最费时间的一项。

**代价**：多两跳网络、时钟必须同步、同事的电脑成了系统的一部分。最后一条对测试没问题，但**这不是可交付的形态**——真要脱机，最终得给机器人配算力（Jetson Orin 之类），或者退回轻量方案（YOLO-seg → 掩膜点云 → RANSAC 圆柱拟合，CPU 能扛，见 README 阶段 8 的"首选轻方案"）。现在不用管，但别让它变成默认架构。

---

## 1. 我要做什么

> **相机链路整条不归我。** 相机节点在机器人上跑、图像发给同事、他订阅——这条链的两端都是他，
> 我的机器**根本不在里面**。我不需要装 `realsense2_camera`，也不需要装压缩插件。
> （唯一例外见 1.7：想在面板上看实时画面才需要，纯可选。）

### P0 — 测试阶段必须的

| # | 事项 | 说明 |
|---|---|---|
| 1.1 | **跨机 DDS 配置** | 现在 `CYCLONEDDS_URI` 是 `lo`（回环），**永远看不到别的机器**。完整命令见 §6 |
| 1.2 | **打包 `sb_communication` 给他** | 见 §3 |
| 1.3 | **改跳变门：按瓶轴判，不按完整四元数** | 这是我的 bug，不是他的，理由见 §5.3 |

### P1 — 联调阶段

| # | 事项 | 说明 |
|---|---|---|
| 1.4 | **时钟同步** | NTP（systemd-timesyncd/chrony），**机器人、视觉机、控制机三台都需同步**（见 §6.8）。机器人那端要写权限，可能得他或管机器人的人配合 |
| 1.5 | **机器人 CPU 亲和性** | 12700H 是 6 P-core + 8 E-core。`shuangbi_demo` 的 50 Hz 控制环钉 P 核，相机+压缩赶 E 核。代码里已有 `CpuAffinityPlan` / `bootstrap_process_scheduling`。**这是我的节点和他的相机节点抢 CPU 的问题，配置我出，部署要你上机器人** |
| 1.6 | 拿静止/移动录包定 `filter_tau` 和跳变门限 | 等 4.6 / 4.7 |
| 1.7 | *（可选）* 面板实时画面 | 只有想在面板上看真机相机才需要装压缩插件订阅图像。和抓取链路无关 |

### P2 — 标定完成后

| # | 事项 |
|---|---|
| 1.8 | 把他给的手眼标定结果回填 URDF 的 `head_d435i_joint`（现在那组是从 STL 量的估计值） |
| 1.9 | 按他给的实测瓶子尺寸更新 `grasp_profile_bottle.yaml`（现标称 φ36 × 200 mm × 0.45 kg，手的抓握中心偏移是**按 φ36 实测标定的**，换瓶子要重新查表） |
| 1.10 | *（可选）* 用他的真实内参更新仿真相机（现在是理想针孔无畸变）。只影响仿真保真度，不影响真机——真机内参从 `CameraInfo` 话题自动流过来 |

---

## 2. 他要做什么

### P0

| # | 事项 | 依据 |
|---|---|---|
| 2.1 | **直接发 `ObjectPose`，去掉 JSON 文件中转** | 他环境是 Python 3.10 + `rclpy` 可导入，**可以同进程发**。文件 IPC 有三个问题：可能读到写了一半的 JSON、30 Hz 写文件的抖动直接进 `observation_delay`、时间戳要跨这一跳透传（他现在踩的就是这个）。JSON 保留做诊断/回放没问题，**发布链路走内存** |
| 2.2 | **`header.stamp` = 相机采集时刻** | 他文档 §4.3 自认现在用的是 JSON 文件 mtime。要从 ROS 图像消息的 `header.stamp` 一路透传下来 |
| 2.3 | **位置发几何中心，不是 CAD 原点** | 他文档 §4.2 已经找到了：用 `center_pose = pose @ inverse_to_origin`，不能用 `pose_camera_object[:3,3]` |
| 2.4 | **竖直约束改成"校验"，不要"硬覆盖"** | 见 §5，这条最容易被忽略 |
| 2.5 | **构建 `sb_communication`，校验指纹** | 见 §3 和 §6 的 type hash 陷阱 |
| 2.6 | **订阅机器人的压缩相机话题** | 话题名见 §3 |

### P1

| # | 事项 |
|---|---|
| 2.7 | `sequence` 严格递增，重新 register 时**不清零** |
| 2.8 | **没检测到就停止发布**（或 `confidence` 压到 0.2 以下）。不要继续发漂移的 tracking 结果 |
| 2.9 | `confidence` 要包含**位姿质量**，不能只是 YOLO 检测置信度（他文档 §6 自己指出了）。见 §5 |
| 2.10 | 发**原始观测** + covariance；内部 tracker 只用于选目标/触发注册（他文档 §4.5 的方案 2，我同意） |
| 2.11 | 默认 QoS（RELIABLE），**不要** `qos_profile_sensor_data` |
| 2.12 | `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`ROS_DOMAIN_ID=10`、`CYCLONEDDS_URI` 用真网卡（见 §6） |

### 整条相机链路都是他的

| # | 事项 | 说明 |
|---|---|---|
| 2.13 | 机器人上的 D435i 相机节点 | 话题名、编码、`frame_id` **保持和仿真一致**（见 §3.6），否则我这边的 TF 查找会失败 |
| 2.14 | **图像压缩传输** | 见下面的带宽账，不做链路跑不动 |
| 2.15 | **手眼标定** ★ | 需要机器人动头 + 标定板 + 他的求解器。**他需要我给的东西见 §3.11** |
| 2.16 | 相机内参标定 | 通过 `CameraInfo` 话题流给下游，我不用手动填 |

**带宽账（决定相机链路能不能跑）**

| 分辨率 / 帧率 | 原始码率 |
|---|---|
| 640×480 @ 15 Hz | **176 Mbps**（实测：彩色 104 + 深度 72） |
| 848×480 @ 30 Hz（他 FP 的默认） | **466 Mbps**（推算） |

> ⚠️ **我这台是 WiFi**（`wlo1`，协商 130 Mbit/s，实际吞吐约 65–90 Mbps）。
> 如果机器人和他的电脑也走同一个 WiFi，**原始图像连 640×480@15 都塞不下**（176 > 130）。
>
> - **首选：机器人 ↔ 他的电脑走有线千兆。** 我这端可以留 WiFi——我只收 `ObjectPose`，
>   约 200 字节 × 30 Hz ≈ 48 kbps，无关紧要。
> - **退而求其次：压缩 + 640×480 @ 15 Hz。** JPEG 彩色 ~80 KB + `compressedDepth` PNG ~250 KB
>   ≈ 330 KB/帧 × 15 Hz = **40 Mbps**，WiFi 也能扛。代价是机器人端每帧多花 15–35 ms 编码，
>   直接吃我 300 ms 的延迟预算。
>
> 插件：`ros-humble-compressed-image-transport`、`ros-humble-compressed-depth-image-transport`。

---

## 3. 我要给他什么

| # | 东西 | 位置 / 说明 |
|---|---|---|
| 3.1 | **`sb_communication` 包** | `ws/src/sb_ros2_pkg/sb_communication/`。只依赖 `std_msgs / geometry_msgs / builtin_interfaces / sensor_msgs`，全是标准包，丢进任何 Humble 工作空间 `colcon build` 就行。共 2 个 msg + 1 个 srv |
| 3.2 | **`ObjectPose.msg` 指纹** | **`03463e448beaa082`**，见下 |
| 3.3 | 接口约定文档 | [vision_handoff.md](vision_handoff.md) |
| 3.4 | 环境体检脚本 | [vision_env_report.py](../ws/src/shuangbi_grasp/scripts/vision_env_report.py) |
| 3.5 | 接口验收脚本 | [vision_interface_acceptance.py](../ws/src/shuangbi_grasp/scripts/vision_interface_acceptance.py) |
| 3.6 | **相机话题名与格式** | 见下 |
| 3.7 | **网络参数** | `ROS_DOMAIN_ID` 的值、`CYCLONEDDS_URI` 模板、机器人/我方 IP |
| 3.8 | 瓶子几何与抓取约定 | `grasp_profile_bottle.yaml`：φ36 × 200 mm × 0.45 kg；**倾角 > 15° 我这边直接拒绝** |
| 3.9 | 相机 frame 的标定前估计值 | URDF `head_d435i_joint`：`xyz="0.044 0.0375 -0.0355" rpy="0 0 0"`（从 STL 量的，**不准**，等标定） |
| 3.10 | 参考实现 | [sim_object_pose.py](../ws/src/shuangbi_grasp/shuangbi_grasp/sources/sim_object_pose.py) — 字段填法、时间戳处理都是现成的 |
| 3.11 | **做手眼标定要的东西** | 见下 |

**3.11 手眼标定他需要我给的**（标定归他做，但他动不了机器人）：

| 东西 | 说明 |
|---|---|
| URDF | `ws/src/shuangbi_description/urdf/shuangbi20260803_righthand.urdf` |
| frame 名 | 待标定的是 `head_pitch_Link → head_d435i_link`（眼在外，相机固连在头上） |
| **怎么让头动** | 标定要采多组姿态。头的两个关节是 `head_yaw_joint` / `head_pitch_joint`。得给他一个能改这两个角的口子——最简单是加个话题/服务，或者他用面板手动摆 |
| `joint_states` / TF | 每组姿态要配一个对应时刻的关节角，这个从机器人的 `joint_states` 出 |
| 待命位的头角 | `READY_HEAD_PITCH = 0.6 rad`（低头 0.6 rad 是按自遮挡实测选的，**标定采样最好覆盖这个角附近**，因为抓取时头就在这个姿态） |

> 我要回收的是：`head_pitch_Link → head_d435i_link` 的 **平移 + 旋转** 和**标定残差**（重投影误差 / 标定板 RMS）。

**相机话题（仿真已是这个布局，真机保持一致）：**

```
/head_camera/color/image_raw          sensor_msgs/Image   rgb8
/head_camera/color/camera_info        sensor_msgs/CameraInfo
/head_camera/depth/image_rect_raw     sensor_msgs/Image   16UC1
/head_camera/depth/camera_info        sensor_msgs/CameraInfo
frame_id = head_d435i_optical_frame          ← REP-103 光学系：Z 前 / X 右 / Y 下
```

压缩话题上线后追加 `/compressed` 和 `/compressedDepth` 后缀。

**3.2 指纹是什么、谁来核**

就是 `ObjectPose.msg` 这个**文本文件**的 sha256 前 16 位。之所以要人肉核，是因为 **Humble 没有类型哈希**
（详见 §6 第 4 条）：两边的 `.msg` 只要类型名一样就认，字段布局不同**不报错**，直接按错的布局反序列化——
位姿变成随机数。这比"收不到"难查得多。

**做法**：我把整个 `sb_communication` 包给他，他拷过去后跑一次核对，然后**不要动这个文件**。

```bash
sha256sum sb_communication/msg/ObjectPose.msg | cut -c1-16
# 期望：03463e448beaa082
```

对得上就再也不用管；只有我这边改了 `.msg`（目前没这个计划）才需要重新同步 + 双方重新 `colcon build`。

---

## 4. 他要给我什么

### P0 — 没有这些判断不了抓取坐标对不对

| # | 东西 | 为什么卡 |
|---|---|---|
| 4.1 | **`head_pitch_Link → D435i optical frame` 手眼标定结果 + 残差** ★ | 我现在 URDF 里那组是从 STL 量的估计值。**这一项不给，抓取精度无从谈起** |
| 4.2 | FoundationPose 用的 **mesh 文件、单位、原点位置、哪个轴是长轴** | 我要对齐坐标约定，也可能用于碰撞体 |
| 4.3 | 相机内参、深度尺度、RGB-depth 对齐设置、分辨率、帧率 | 真机要替换掉仿真的理想针孔模型 |
| 4.4 | 观测时间戳的**来源** | 决定 `observation_delay` 算不算得对 |
| 4.5 | **register / 重注册 / 持续跟踪的实际耗时** ★ | **从 P1 升上来的**：现在有两跳网络，这个数直接决定 300 ms 预算够不够。他文档 §6 说 register 是异步的、完整搜索可能几百 ms |

### P1 — 没有这些调不了验收门限

| # | 东西 |
|---|---|
| 4.6 | **静止 60 s 录包**：位置 σ、帧率、观测延迟分布、四元数逐帧变化、**瓶轴变化**、**yaw 漂移** |
| 4.7 | **移动 30 s 录包**：跟踪滞后、重注册耗时、真实移动会不会误判跳变 |
| 4.8 | `confidence` 的定义、covariance 的估计方法 |
| 4.9 | 显存峰值、GPU 型号 |
| 4.10 | `vision_env_report.py` 在**目标 GPU 机器上、source 过 `sb_communication` 之后**重跑的输出（他上次那份 `nvidia-smi` 都连不上驱动，`hard_pass=false` 说明不了什么） |
| 4.11 | 他的 ROS 包 / launch / 依赖清单（`environment.yml` 或 `pip freeze`）、FoundationPose 的 fork 与 commit、权重来源 |

**4.6 / 4.7 两段录包决定我把 `filter_tau` 和跳变门限设成多少，没有它们我只能瞎猜。** 录 `object_pose` 一个话题就够，能同时录 RGB/depth 更好。

---

## 5. 三个需要说清楚的设计点

### 5.1 竖直约束：用**校验**，不要**硬覆盖**

他文档 §5 说 `pose_with_upright_axis()` 把 CAD Z 轴对齐到重力方向。这满足了"+Z 是瓶轴"的字面要求，但它是**强制**而非**测量**。

**硬覆盖买到的**：轴向噪声更小。值多少？我这边

```
contact = centre + grasp_height × axis      (grasp_height = 0.020 m)
```

轴偏 5° → 接触点偏 `0.020 × sin5°` = **1.7 mm**。而且这只手是**卷曲抓握**（power grasp），手指包住圆柱，几度腕部错位由手指吸收。**这 1–2 mm 我不需要。**

**硬覆盖赔掉的**：我这道门变成死代码——

```python
if tilt > self.profile.object.max_tilt:      # max_tilt_deg: 15.0
    raise GraspError(f"bottle is tilted {...} deg")
```

Z 被无条件掰成重力方向 → 倾角恒为 0 → 瓶子倒了照报"立着" → 我会对一个躺着的瓶子做侧向卷曲抓握。

**而这个场景在本项目里是已知会发生的**：手在接近过程中会把瓶子碰倒（2026-09-09 实测过）。

**结论：代价远大于收益。** 具体做法：

- **发布测量值**
- 把"测量轴与重力的夹角"当**质量指标**——压进 `confidence`，或进 covariance 的旋转对角线，我这边有门能接住
- 竖直先验**只在内部用**（给 register 剪枝加速），不改输出

### 5.2 `confidence` 要反映位姿质量

他文档 §6 自己指出：`confidence` 目前只是 YOLO 检测置信度，FoundationPose 的 scorer 分数没进去。

那我这道 `min_confidence = 0.2` 就只是在问"YOLO 有没有 20% 把握这是个瓶子"，**对位姿质量一无所知**。一个高置信度检测配一个垃圾位姿，比没有还危险——因为我们真的会伸手过去。

建议 `confidence = YOLO_conf × 归一化 scorer`，或者把位姿质量折进 covariance（位姿差 → sigma 大，我有 `max_position_sigma = 0.05` 的门接住）。无论选哪个，**在交付清单里写清楚它的定义**。

### 5.3 跳变门为什么是我改，不是他改

这条容易觉得"yaw 乱跳是视觉的毛病，该他修"。不是。

**我的门在判一个我随后就扔掉的量。** 跳变判据现在是：

```python
turned = quaternion_angle(sample.orientation, reference.orientation)   # 完整四元数，阈值 30°
```

而抓取侧真正消费姿态的地方只有一行：

```python
axis = orientation[:, 2]      # grasp_pose_generator.py：只取第三列
```

绕瓶轴的 yaw 我**根本不用**——圆柱的 yaw 不可观测，我这边本来就是自己采样 9 个 yaw 候选（±60°）
再按 IK 质量挑。所以这是**我拿一个自己不用的量去否决整帧数据**，属于我的 bug，跟视觉侧发什么无关。

**让他改的话，代价是长期的。** 旋转对称是物体的属性，不是估计器的缺陷——**没有任何算法能观测圆柱的 yaw**。
要他"修"就是要他凭空发明一个 yaw 约定并保证逐帧稳定，这套机器只为了满足一个随后就把值丢掉的门。

**改我这边是一处改动，改他那边是持续维护。** 所以我改。

（等 4.6 的静止录包到手再定阈值：万一瓶轴本身也在抖，只换判据还不够。）

---

## 6. 跨机通信：只设 `ROS_DOMAIN_ID` 不够

要六样，第 3、4 条最容易踩：

1. **`ROS_DOMAIN_ID` 一致** — 必要但不充分
2. **`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` 一致** — FastDDS ↔ Cyclone 理论上 RTPS 能互通，实际很不可靠，别赌
3. **他必须有 `sb_communication` 的构建** — 不用 source 我整个 ws，但那个包必须有（依赖全是标准包，见 §3.1）
4. **⚠️ Humble 没有 type hash：定义对不上不报错，会静默乱码**

   实测（2026-09-10）Humble 的 `ros2 topic info -v` 只显示：

   ```
   Type: sb_communication/msg/ObjectPose        ← 只有类型名，没有 hash
   ```

   `RIHS01_` 类型哈希是 Iron 之后才有的。**Humble 里只要类型名一样就认**，字段布局不同直接按错误布局反序列化——表现是位姿变成随机数，而不是"收不到"。这比收不到难查得多。**所以要核指纹（§3.2）。**

5. **`CYCLONEDDS_URI` 换真网卡** — 现在是 `lo`，回环，看不到任何别的机器。见 §6.6 的完整命令

6. **防火墙 + 同网段**。DDS 的 UDP 端口段与 `ROS_DOMAIN_ID` 相关，`ufw` 开着要放行

### 6.5 本项目的实际参数

| | 值 |
|---|---|
| `ROS_DOMAIN_ID` | **10** |
| 我的电脑 | `192.168.112.252`，网卡 **`wlo1`（WiFi）** |
| 机器人 | `192.168.112.104` |
| 同事电脑 | **待填** |
| 网段 | `192.168.112.0/24`，都在同一个局域网 |

> **测试阶段只需要"我 ↔ 同事"这一条链。** 我只订阅 `object_pose`，图像链路（机器人 → 同事）
> 跟我无关。到了 M4 几何验收才需要连机器人（要它的 `joint_states` 做 TF）。
>
> 实测到机器人 RTT：**平均 26 ms / 最大 80 ms / 抖动 23 ms**（WiFi）。只走 `ObjectPose` 完全够用，
> 但这个抖动会进 `observation_delay`，所以 300 ms 预算不算宽裕。

### 6.6 完整命令

**我这边**（`192.168.112.252`，每个开 ROS 的终端都要）：

```bash
source /opt/ros/humble/setup.bash
source /home/hssw/unitree_ros2/cyclonedds_ws/install/setup.bash
source /home/hssw/GYH/ws/install/setup.bash

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="wlo1" priority="default" multicast="default"/></Interfaces></General><Discovery><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex><Peers><Peer address="192.168.112.104"/><Peer address="同事IP"/></Peers></Discovery></Domain></CycloneDDS>'
```

**同事那边**（把 `他的网卡` 和 `他的IP` 换掉）：

```bash
source /opt/ros/humble/setup.bash
source ~/sb_ws/install/setup.bash            # 装了 sb_communication 的工作空间

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="他的网卡" priority="default" multicast="default"/></Interfaces></General><Discovery><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex><Peers><Peer address="192.168.112.252"/><Peer address="192.168.112.104"/></Peers></Discovery></Domain></CycloneDDS>'
```

网卡名用 `ip -o -4 addr show` 查（找 `192.168.112.x` 那一行的第二列）。

> **为什么用显式 `<Peers>` 而不是靠组播**：我这边是 WiFi，很多 AP 会限速或直接丢组播帧，
> 表现是"节点时有时无"。列 peer 走单播，稳。加了 `<Peers>` 之后组播开不开都无所谓。
>
> `MaxAutoParticipantIndex 60`：单播发现时每个域默认最多 10 个参与者，我们仿真栈就有 8 个节点，
> 不提高上限会报 `rcl node's rmw handle is invalid`。

**防火墙**（`ufw` 开着才需要）：

```bash
sudo ufw allow from 192.168.112.0/24 proto udp
```

### 6.7 验证（照顺序来）

```bash
# 1. 网络通不通
ping -c 3 同事IP

# 2. 互相看得见节点吗（他先起一个节点）
ros2 node list

# 3. 话题在不在、类型对不对、QoS 匹不匹配
ros2 topic info -v /object_pose
#   Type: sb_communication/msg/ObjectPose
#   Publisher count: 1            ← 0 说明发现没通，回到第 2 步
#   Reliability: RELIABLE         ← BEST_EFFORT 的话订不上，一条都收不到

# 4. 数据本身
ros2 topic hz /object_pose
ros2 topic echo /object_pose --once
```

第 4 步看到位姿数字明显不合理（比如量级差几个数量级、或者随机跳），**先怀疑 `.msg` 指纹对不上**（§3.2）。

### 6.8 时钟同步：三台机器使用同一授时源

**2026-09-19 按实际代码修正。** 视觉节点除透传采集时间戳，还会用视觉机系统时间检查输入帧年龄及输出延迟，因此视觉机也必须同步。机器人负责采集时间，视觉机和控制机都使用本机时间与它作差。

- 机器人、视觉机、控制机使用同一个明确的NTP授时源。
- OS服务负责调整系统时钟；视觉节点检查服务、实际选中源和主动NTP测量结果。
- 默认误差预算为每机相对共同源 `abs(offset)+RTT/2 ≤ 2ms`；检查失败时暂停位姿发布，恢复后使用新帧。
- 不修改原始`header.stamp`，不通过扩大300ms延迟门限掩盖时钟差异。
- 当前按用户要求仅搭建本机回环测试，未连接机器人，也未更改系统授时配置。共同服务器待正式联调时确定。

实现、三种运行模式和配置步骤见 [时钟同步说明](../../my_demo/README_clock_sync.md)。同机验证使用显式`--clock-sync-mode local`；跨机使用默认`required`并配置`--clock-server`。本机模拟授时模式仅允许回环IP。

### 6.9 延迟预算

我的门是 `max_observation_delay = 0.30 s`。两跳网络下：

| 环节 | 估计 |
|---|---|
| 采集 → 机器人发出 | 5–15 ms |
| 压缩（JPEG ~3 ms + 深度 PNG 10–30 ms） | 15–35 ms |
| 网络 640×480 | 10–20 ms |
| YOLO | 10–20 ms |
| FoundationPose **track** | 20–30 ms |
| ObjectPose 回传 | < 5 ms |
| **合计** | **~90–150 ms** ✓ |

跟踪模式够用。**`register` 是问号**——所以 4.5 升成了 P0。

---

## 7. 还没定的

| # | 问题 | 谁答 |
|---|---|---|
| 7.1 | `register` / 重注册 / 跟踪各耗时多久？重注册的触发条件？ | 他 |
| 7.2 | 竖直约束改成校验之后，倒瓶子由谁发现？他的"姿态质量检查"能不能兜住？ | 他 |
| 7.3 | 手眼标定谁来做、什么时候做？（需要机器人动头 + 标定板 + 他的求解器，**大概率要一起做**） | 共同 ★ |
| 7.4 | 场景里出现两个瓶子怎么办？我这边一次只跟一个 `object_id`，**别发多个**（会按到达顺序互相覆盖，表现是目标在两瓶之间来回跳） | 他 |
| 7.5 | 机器人和同事电脑之间是有线还是 WiFi？ | 我 |

---

## 8. 顺序

每一步都能单独判定成败，**不要跳步**。

| 里程碑 | 内容 | 判据 |
|---|---|---|
| **M0 通信** | 他发写死的假数据（stamp/sequence/frame_id 填对），我 `ros2 topic echo /target/status` | `accepted` 在涨、`rejected` 为 0。**十分钟能做完，跳过它后面全是玄学** |
| **M1 相机链路** | 我把真机相机 + 压缩传输跑起来，他能订阅到 RGB-D | 他那边收到图，帧率 ≥ 10 Hz |
| **M2 时序** | 他接真实相机跑通，还不用标定 | `/target/status` 的 `reject_counts`：`delayed` 涨 → 流水线慢或时钟不同步；`jump` 涨 → 对称性 yaw（该我改）；`out_of_order` 涨 → sequence 没递增；收不到 → QoS 或指纹对不上 |
| **M3 验收脚本** | 静止 60 s + 移动 30 s | `vision_interface_acceptance.py` 16 项全 PASS |
| **M4 几何** | 手眼标定填进 URDF，瓶子放固定位置，卷尺量真值对比 `target.position` | **误差 < 2 cm**（头部 D435i 在 0.6–0.8 m 上深度误差本来就 1–2 cm，这是方案的物理上限） |
| **M5 抓取** | 接进任务状态机，跑 20 轮 | 成功率。仿真里用真值当"视觉"已跑通 15/15、16.2 s/轮，换真视觉就是替换数据源 |

---

## 附：一句话版本

**给他：** `sb_communication` 包（核指纹 `03463e448beaa082`）+ 三个文档/脚本 + 相机话题名 + 网络参数。

**要他：** 手眼标定 ★ + mesh 与原点 + register 耗时 ★ + 两段录包 + 环境报告（在目标机器上重跑）。

**他改：** 直接发 ObjectPose（去掉 JSON 中转）+ stamp 用采集时刻 + 位置用几何中心 + 竖直约束改校验 + confidence 含位姿质量 + **整条相机链路（节点/压缩/内参/手眼标定）**。

**我改：** 跨机 DDS + 跳变门按瓶轴判 + 时钟同步 + CPU 亲和性配置。**相机链路整条不归我。**
