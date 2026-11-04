# 视觉模块对接需求单

**给**：视觉侧（YOLOv8-seg + FoundationPose）
**来自**：抓取/控制侧（shuangbi20260803 双臂轮式机器人）
**日期**：2026-09-10

---

## 发什么给他（第一轮）

**四个文件：**

| 发什么 | 干什么用 |
|---|---|
| 本文档 `vision_handoff.md` | 全部约定与问卷 |
| [`ObjectPose.msg`](../ws/src/sb_ros2_pkg/sb_communication/msg/ObjectPose.msg) | 消息定义，一眼看懂字段 |
| [`vision_env_report.py`](../ws/src/shuangbi_grasp/scripts/vision_env_report.py) | 他在自己环境跑一次，把输出发回来 |
| [`vision_interface_acceptance.py`](../ws/src/shuangbi_grasp/scripts/vision_interface_acceptance.py) | 交付前的自检，先给他心里有数 |

**先别发** `setup_vision_env.sh` —— 那是按我这台机器写的，属于我这边的事。等问卷回来定了形态再说。

**第一轮我只要两样东西回来：** §8.3 的九个问题的回答，和 `vision_env_report.py` 的输出。
**代码一行都不用先写。**

<details>
<summary>可以直接复制的一段话</summary>

> 这是抓取那边的对接文档，麻烦先看两块：
>
> **一、§1 坐标约定** —— 只有两条，但错了机器人会稳定地抓错地方：
> 位置要给**瓶子的几何中心**（不是瓶底、不是掩膜质心），姿态只要求**物体系 +Z 是瓶子的长轴**
> （Z 的正负、绕 Z 的 yaw 我都不管，随便）。
>
> **二、§8.3 的九个问题** —— 关于你的环境。**你不用改任何东西**，环境按你顺手的来，
> 桥接是我的活；我只是要知道现状好定方案。顺便在你的环境里跑一下 `vision_env_report.py`，
> 把输出发我。
>
> 其它章节是时间戳、频率、丢目标怎么办这些工程细节，等定了方案再细看不迟。
> 滤波你**不用做**，我这边已经有一整套，而且需要看到你的原始输出才能调门限。

</details>

---

## 0. 一句话

**你只需要往一个 ROS 2 话题上发一个消息。** 话题 `object_pose`，类型 `sb_communication/msg/ObjectPose`，
里面装瓶子的 6D 位姿。滤波、跳变判定、坐标变换、锁定、超时，全部在我这边（`TargetManager`）做，
**你不用做滤波，也请不要做**——我需要看到原始观测，否则我这边的门限没法调。

下面的内容按重要性排序：**第 1、2 节不对齐的话，机器人会稳定地抓错地方**；第 3 节以后是工程细节。

---

## 1. 坐标约定（最容易出错的地方，请重点看）

### 1.1 位置：必须是瓶子的**几何中心**

不是掩膜质心，不是瓶底，不是点云重心，是**圆柱体的几何中心**（瓶身高度的一半处，轴线上）。

我这边算抓取点是这么算的：

```
抓取接触点 = 物体中心 + grasp_height × 瓶轴  (grasp_height = 0.020 m)
```

`grasp_height` 是相对**中心**标定的。如果你给的是瓶底，抓取点会整体偏低 10 cm（瓶高 0.20 m），
手会去抓桌面。

> 如果你的 CAD 模型原点不在几何中心，**不要改 CAD**，在发布前把原点平移过来，并把这个偏移量
> 写进交付清单第 6 条告诉我。

### 1.2 姿态：**+Z 轴必须是瓶子的长轴**，其它随便

我的代码只取旋转矩阵的第三列当瓶轴：

```python
axis = orientation[:, 2]          # 物体系的 +Z
if axis[2] < 0.0: axis = -axis    # 朝上朝下我自己翻，你不用管
```

所以你只需要保证一件事：**物体系的 Z 轴 = 瓶子的对称轴**。

好消息（对 FoundationPose 特别重要）：

- **Z 的正负无所谓**，我会自己翻正。瓶子倒着装 CAD 也行。
- **绕 Z 的 yaw 完全无所谓**。圆柱绕自身轴旋转不可观测，我这边本来就是采样 9 个 yaw 候选
  （±60°）再按 IK 质量挑，你给什么 yaw 我都不用。

坏消息，**这条请务必确认**：

> ⚠️ 瓶子是旋转对称体，FoundationPose 对对称物体的 yaw 输出通常是**随机游走**的。
> 而我这边的"跳变门"目前比较的是**完整四元数**，阈值 30°。yaw 乱跳会让每一帧都被判成"跳变"而丢弃，
> 表现是"视觉在发，机器人说没有目标"。
>
> **两个解法，选一个**：
> (a) 你侧把 yaw 规范化（每帧输出一个确定的 yaw，比如始终让物体系 X 轴投影到某个固定方向）；
> (b) 我侧把跳变判据从"四元数夹角"改成"瓶轴夹角"（改动很小，在 `target_manager.py::_is_jump`）。
>
> 我倾向 (b)，因为对称性是物体的属性，不该让你去补。**但我需要先知道你的 yaw 到底稳不稳**——
> 见交付清单第 4 条的静止录包。

### 1.3 倾斜超过 15° 我会直接拒绝

```yaml
max_tilt_deg: 15.0   # 瓶轴与竖直方向的夹角
```

超过就报 `ungraspable`，因为立着的瓶子突然被估成躺着的，绝大多数情况是位姿估计错了而不是瓶子真倒了。
如果你们的场景里瓶子真的会躺着，提前告诉我，这是另一套抓取策略（现在的手是侧向卷曲抓握，只抓立着的）。

### 1.4 发在哪个坐标系：**建议直接发相机光学系**

`header.frame_id` 填什么都行，只要那个 frame 在我的 TF 树里。我会用 tf2 在**观测时刻**把它变换到
规划系 `waist_yaw_Link`。

**推荐**：`head_d435i_optical_frame`
（REP-103 光学系约定：**Z 向前、X 向右、Y 向下**，不是 x 向前的那套）

我 URDF 里已经有这两个 frame 了：

```xml
<!-- head_pitch_Link 上，D435i 模块前表面 -->
<joint name="head_d435i_joint">        <origin xyz="0.044 0.0375 -0.0355" rpy="0 0 0"/>
<joint name="head_d435i_optical_joint"><origin xyz="0 0 0" rpy="-1.5708 0 -1.5708"/>
```

> ⚠️ **这组数是从 STL 里量的，是标定前的估计值，不准。** 手眼标定（眼在外，相机固连在
> `head_pitch_Link` 上）的结果要回填到 `head_d435i_joint`。见交付清单第 2 条。

**不要**自己在你的节点里把位姿变换到 `base_link` 再发给我。原因：头部有 pitch 关节会动，
变换必须用**拍照那一刻**的 TF；我这边已经实现了按 `header.stamp` 查 TF（`tf_timeout` 50 ms），
你自己变换的话大概率会用"当前"的 TF，头一动就错。

---

## 2. 消息字段：逐个说明

话题名 `object_pose`（可改，参数 `object_pose_topic`），类型 `sb_communication/msg/ObjectPose`：

| 字段 | 类型 | 要求 | 不满足会怎样 |
|---|---|---|---|
| `header.stamp` | Time | **拍照时刻**，不是发布时刻 | 见 §3 |
| `header.frame_id` | string | 见 §1.4 | 空 → 当成已在规划系，位姿全错 |
| `object_id` | string | 稳定的 id，固定用 `"bottle"` | 每帧变 → `wrong_object` 丢弃 |
| `pose.position` | Point | 瓶子几何中心，**米** | 见 §1.1 |
| `pose.orientation` | Quaternion | +Z = 瓶轴，见 §1.2 | 见 §1.2 |
| `covariance` | float64[36] | 6×6 行主序 (x y z rx ry rz)，**全零 = 未知，可接受** | 见下 |
| `confidence` | float32 | 0..1 | < 0.2 丢弃 |
| `valid_until` | Time | **填 0 就行**（用我的新鲜度窗口） | — |
| `source` | string | 填 `"camera"` 或 `"foundationpose"`，只用于诊断 | — |
| `sequence` | uint32 | **每次发布严格递增** | `<=` 上一帧 → `out_of_order` 丢弃 |

**关于 `covariance`**：全零是合法的，表示"不知道"，我不会因此拒绝。但如果你能给，
我会用**平移部分对角线的最大标准差**做门限，默认 `max_position_sigma = 0.05 m`——
超过 5 cm 的位置不确定度会被丢弃。给了比不给好：一个远处的、掩膜只有几十个像素的检测，
靠 covariance 能被自动过滤掉，靠 confidence 往往过滤不掉。

**关于 `sequence`**：这是防重放/乱序的，请用一个单调递增的计数器，**不要**用帧号取模，
**不要**在重新 register 的时候清零。uint32 在 30 Hz 下能跑 4.5 年，不用担心溢出。

**关于 QoS**（这一条会导致"完全收不到数据"的静默故障）：

> 我的订阅是**默认 QoS**（RELIABLE / VOLATILE / depth 20）。
> 如果你用 `rclpy.qos.qos_profile_sensor_data` 或者手写 BEST_EFFORT 发布，
> **QoS 不兼容，一个消息都收不到，而且不报错**。
> 请用默认 QoS 发布（`create_publisher(ObjectPose, "object_pose", 10)`）。

---

## 3. 时间要求

### 3.1 `header.stamp` 必须是**拍照时刻**

不是推理完成时刻，不是发布时刻。我这边有两个独立的时间检查：

```
max_age                  = 0.25 s   # 最近一帧"到达"至今多久 → 判断视觉流是否还活着
max_observation_delay    = 0.30 s   # 到达时刻 − header.stamp → 判断这一帧描述的瞬间有多旧
```

这两个是**不同的问题**：一条 30 Hz 稳定输出、但每帧都描述 500 ms 之前的流水线，
"活着"没问题，但"位姿够新"不合格——机器人会朝一个 0.5 s 前的位置伸手。

所以：**请把 RealSense 驱动给的图像 header.stamp 一路透传下来**，YOLO 和 FoundationPose 的耗时
就自动体现在 `observation_delay` 里，我能看到、能报警、能调门限。如果你填发布时刻，这个信息就永久丢失了。

### 3.2 频率

| | 值 | 说明 |
|---|---|---|
| 绝对下限 | **4 Hz** | 低于这个 `max_age` 就会持续超时 |
| 希望 | **10–30 Hz** | 抓取过程中我会在线重规划（目标移动 > 3 cm 或 > 10° 触发） |
| 上限 | 无所谓 | 我按 20 Hz 转发下游，发多了我自己抽 |

### 3.3 时钟

整个栈现在**不用 sim_time，全是系统墙钟**。

- 如果视觉和控制在**同一台机器**上：没问题，忽略这条。
- 如果视觉跑在**另一台机器/另一块显卡的独立主机**上：**两台机器必须做时钟同步**（chrony/NTP，
  局域网内做到 ms 级）。不同步的话 `observation_delay` 会是个随机数——可能是负的，也可能常年超过 0.30 s
  被全量丢弃。这个坑很隐蔽，请提前确认。

### 3.4 丢失目标时：**请停止发布**，不要发最后一帧

FoundationPose 的 tracking 模式在物体被遮挡/出画后往往还会继续输出一个漂移的位姿。
**这种情况请不要发**，或者把 `confidence` 压到 0.2 以下。

我这边 0.25 s 收不到就会报 `target timeout`，状态机会停下来等——这是正确行为。
而如果你持续发一个漂移的位姿，机器人会朝一个不存在的瓶子伸手，这是危险行为。

**判据建议**：YOLO-seg 这一帧没有检测到瓶子 → 不发。这比让 FoundationPose 自己判断 tracking 是否有效可靠。

---

## 4. 关于"没加滤波"——这是对的，别加

我这边 `TargetManager` 已经有完整的一套：

```
原始位姿 → 校验器（新鲜度/时序/置信度/协方差/跳变） → TF 变换到规划系 → 一阶低通 → 快照
                                                                    (filter_tau, 默认 0 = 直通)
```

| 门 | 默认值 | 作用 |
|---|---|---|
| `min_confidence` | 0.2 | |
| `max_position_sigma` | 0.05 m | 从 covariance 来 |
| `jump_position` | 0.05 m | 相邻帧位置突变 |
| `jump_orientation` | 30° | 相邻帧姿态突变（**见 §1.2 的对称性问题**） |
| `jump_confirm_samples` | 3 | 连续 3 帧一致才认可跳变——单帧野值不会移动目标，真的被人挪走了会被接受 |
| `filter_tau` | 0.0 | 一阶低通，0 = 不滤。EKF 的位置留好了（`TargetManager._filter` 一个方法） |

**你加滤波的坏处**：滤波会把野值抹平，我就看不出你的原始噪声有多大，跳变门限没法调；
而且两级滤波串起来相位滞后翻倍，在线重规划会追不上。

**我需要你给的是原始输出 + 一次噪声测量**（交付清单第 4 条），我据此定 `filter_tau` 和跳变门限。

---

## 5. 我这边已经准备好的东西（你可以直接对着测）

不用等我，这些现在就能跑：

```bash
# 消息定义
ros2 interface show sb_communication/msg/ObjectPose

# 我的接收端（校验 + 变换 + 转发），你发了它就会有诊断输出
ros2 run shuangbi_grasp target_manager --ros-args -p planning_frame:=waist_yaw_Link

# 看它为什么拒绝你的帧——每一条拒绝理由都有计数
ros2 topic echo /target/status
```

`/target/status` 是 JSON，长这样：

```json
{"has_target": true, "accepted": 412, "rejected": 7, "last_reject": null,
 "reject_counts": {"jump": 5, "delayed": 2}, "pending_jump": 0, "age": 0.031,
 "tf_fallbacks": 0, "target": {"position": [...], "orientation": [...]}}
```

**这就是联调时唯一需要看的东西。** `reject_counts` 里哪一项在涨，就对应上面哪一条没对齐。

还有一个**参考实现**可以直接抄：
[sim_object_pose.py](../ws/src/shuangbi_grasp/shuangbi_grasp/sources/sim_object_pose.py) ——
它从仿真真值发布 ObjectPose，字段填法、时间戳处理、噪声注入都是现成的，
你的节点就是把"读仿真真值"换成"跑 YOLO + FoundationPose"。

---

## 6. 交付清单

请按这几条给我，每一条都会直接变成我这边的一个配置项或一次改动：

### 1) 代码本身 —— 具体是什么形状，看 §8.4 定下来的形态

- **形态 B（推荐）**：一个纯 Python 类 / 模块，接口见 §8.5。你不碰 ROS。
- **形态 A**：一个 ROS 2 包（包名、可执行名、`.launch.py`、订阅什么发布什么）。
- **形态 C**：一个能长期跑的进程 + 启动命令，IPC 协议我来定。

不管哪种，都要有：

- **依赖清单**：`environment.yml` 或 `pip freeze`；Python / CUDA / torch / ultralytics 版本；
  FoundationPose 的具体 commit / fork；权重文件从哪来。
- **显存占用峰值**（我这台还要同时跑 MuJoCo 仿真 + 渲染，16 GB 要分着用）
- **路径可配置**，别写死 `/home/<你的用户名>/`

### 2) 手眼标定结果 ★最高优先级

相机固连在 `head_pitch_Link` 上（眼在外）。我需要 `head_pitch_Link → head_d435i_link` 的
**平移 + 旋转**（xyz + rpy 或四元数都行），以及标定的**残差**（重投影误差 / 标定板 RMS）。

我现在 URDF 里那组数是从 STL 量的估计值，肯定不准。**这一项不给，抓取精度无从谈起。**

顺带：相机内参、`align_depth` 是否开启、深度分辨率和帧率也给一下。

### 3) CAD 模型与原点约定

- FoundationPose 用的**那个** CAD/mesh 文件（我需要它来对齐坐标约定，也可能用于碰撞体）
- **模型原点在哪**（几何中心？瓶底？）、**哪个轴是瓶子长轴**
- 实测尺寸：直径、高度、质量

  > 我现在的标称值是 **φ36 mm × 200 mm × 0.45 kg**（`grasp_profile_bottle.yaml`）。
  > 手的抓握中心偏移是**按这个直径实测标定的**，换瓶子我要重新查表。
  > 如果你们用的是另一个瓶子，尽早告诉我。

### 4) 两段录包（rosbag2），用来测噪声 ★

这两个包决定我把跳变门限和 `filter_tau` 设成多少，**没有它们我只能瞎猜**：

| 录包 | 内容 | 我要从里面测什么 |
|---|---|---|
| **A. 静止** | 瓶子放桌上不动、机器人不动，录 **60 s** | 位置 σ（三轴分别）、**yaw 的漂移幅度**（决定 §1.2 走 (a) 还是 (b)）、逐帧最大跳变、实际帧率、`observation_delay` 分布 |
| **B. 移动** | 手动把瓶子在桌面上慢慢挪 20–30 cm，录 **30 s** | 跟踪滞后、真实移动会不会误触发跳变门、re-register 的时候会不会输出野值 |

录 `object_pose` 一个话题就够。如果能同时录 RGB/depth 更好（我可以离线重放）。

### 5) FoundationPose 的运行模式

- **register 还是 track？** 如果是"首帧 register + 后续 track"：
  - register 一次要多久？track 一帧多久？（这两个数直接决定 §3 的延迟预算）
  - **什么条件下会重新 register？** 重 register 的那一帧输出可不可信？
  - 我会不会在 register 期间收到一个"旧的"位姿？
- tracking 漂移了怎么办、你怎么检测、检测到之后 `confidence` 会不会掉

### 6) 字段语义确认

- `confidence` 是怎么算的？（YOLO 的分割置信度？FoundationPose 的匹配分？两者的积？）
  我需要知道 0.2 这个门限对你的数值意味着什么——如果你的 confidence 常年 0.9+，这个门就是废的。
- `covariance` 你打算填吗？填的话是怎么估的？
- §1.1 说的原点偏移，如果有，具体是多少米、沿哪个轴？

### 7) 多目标 / 误检的处理

场景里出现两个瓶子怎么办？我这边 `TargetManager` 一次只跟一个 `object_id`。

建议你侧就定一个策略（比如"取掩膜面积最大的"或"取离机器人最近的"）并告诉我，
**别发多个**——发多个的话我这边会按到达顺序互相覆盖，表现是目标在两个瓶子之间来回跳。

---

## 7. 联调怎么走（建议顺序）

分四步，每一步都能单独判定成败，不要跳步：

**第 1 步 — 通信**（不需要机器人、不需要标定）
你发假数据（写死一个位姿，正确填 stamp/sequence/frame_id），我 `ros2 topic echo /target/status`
看到 `accepted` 在涨、`rejected` 为 0。
> 这一步专治 QoS 不兼容和消息字段填错，**十分钟能做完，但跳过它后面全是玄学**。

**第 2 步 — 时序**
你接上真实相机跑起来，还是不用标定。看 `/target/status` 的 `reject_counts`：
- `delayed` 在涨 → 流水线太慢或时钟不同步（§3.1、§3.3）
- `jump` 在涨 → 大概率是对称性 yaw 问题（§1.2）
- `out_of_order` 在涨 → sequence 没递增
- `stale` 在涨 → 帧率不够
- 什么都没收到 → QoS（§2）

**第 3 步 — 几何**
标定填进 URDF 之后，瓶子放桌上固定位置，用卷尺量出它在 `base_link` 里的真实坐标，
和 `/target/status` 里的 `target.position` 比。
> **验收线：误差 < 2 cm。** 头部 D435i 在 0.6–0.8 m 距离上深度误差本来就有 1–2 cm，
> 这是这套方案的物理上限（本期不用腕部 D405 做接近段精修）。

**第 4 步 — 抓**
接进抓取状态机，跑批量循环（面板上"开始自动抓取"，设 20 轮），看成功率。
这一步我这边已经跑通了（用仿真真值当"视觉"，15/15 成功，16 s/轮），换成真视觉就是替换数据源。

---

## 8. 环境：这一节是**问你**，不是要求你

### 8.1 先说结论：环境的事我来适配

**你不需要为了对接改你的环境。** 你是视觉那边的人，环境按你顺手的来；我这边是 ROS 那边的人，
桥接是我的活。

我这边只有一个绕不开的**事实**（不是对你的要求）：

```
/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/_rclpy_pybind11.cpython-310-x86_64-linux-gnu.so
                                                                          ^^^^^^^^^^^
```

ROS 2 Humble 的 `rclpy` 是 **cp310** 的二进制扩展，只能在 Python 3.10 的进程里 import。

这个事实**不决定你用什么 Python**，只决定**我们用哪种对接形态**。所以第一步不是我给你提要求，
是你告诉我你那边是什么（§8.3 的问卷），我来选形态。

### 8.2 我这边的事实（供你判断用）

| | |
|---|---|
| ROS | ROS 2 Humble，`rclpy` = cp310 |
| 系统 Python | 3.10.12，numpy 1.24.4 |
| 目标机 GPU | RTX 5070 Ti 16 GB，Blackwell / **算力 12.0 (sm_120)**，驱动 580.173.02 |
| OS | Ubuntu 22.04 |
| 我这台已有的环境 | conda 里有 `yolov8`（py3.10 / torch 2.7.1+cu128）和 `fopose`（py3.11 / torch 2.7.0+cu128 / nvdiffrast 0.4.0 / pytorch3d 0.7.9） |

> **关于 sm_120，只在"代码最终要在我这台跑"时才相关**：cu118 / cu121 的 torch 轮子里没有 sm_120 的核，
> `torch.cuda.is_available()` 仍返回 True、`get_device_name()` 仍正常打印 5070 Ti，然后在第一次
> kernel launch 时炸。如果你在 3090 / 4090 上开发、用 FoundationPose 官方那套 torch 2.0 + cu118，
> 搬过来就是这个下场。**解决办法很便宜（换 cu128 的轮子重装 torch），但要提前知道。**

### 8.3 环境问卷 ★ 这是我现在最需要你回的

回这九条，我就能定形态、定延迟门限、定要不要动我这边的代码。**不用改任何东西，只要告诉我现状。**

1. **Python 版本？**（决定形态，见 §8.4）
2. **torch / CUDA 版本？在什么显卡上开发的？**
3. 环境是 conda / venv / docker / 系统 pip？能导出 `environment.yml` 或 `pip freeze` 最好。
4. **视觉最终跑在哪台机器？** 我这台（和控制同机），还是你那台通过网络？
   —— 这决定要不要做时钟同步（§3.3），也决定 sm_120 相不相关。
5. **你愿意写 ROS 2 节点吗？** 还是更想只交一个纯 Python 的类 / 函数，ROS 那层我来包？
   （没有对错，我推荐后者，见 §8.4 形态 B）
6. **你的推理是"单帧函数调用"还是"有内部状态的持续跟踪"？**
   —— FoundationPose 的 track 模式有状态，这决定进程边界能不能切在哪。
7. FoundationPose 首帧 register 多久、track 一帧多久？显存峰值多少？
8. **依赖里有哪些是必须现场编译的 CUDA 扩展？**（nvdiffrast、pytorch3d 之类）
   —— 这决定换 Python 版本的代价有多大。
9. 用的是 FoundationPose 哪个 fork / commit？权重从哪来？

### 8.4 三种对接形态，看问卷结果选一个

| | 你要做的 | 前提 | 代价 |
|---|---|---|---|
| **A** 你直接写 ROS 节点 | 学一点 rclpy，按 §1–3 发消息 | 你的环境是 py3.10 | 你要碰 ROS |
| **B** 你给纯 Python 类，我包 ROS ★推荐 | 一个类，不碰 ROS | 你的依赖装得进 py3.10 | 换 Python 版本可能要重编 CUDA 扩展 |
| **C** 独立进程 + IPC | 一个能长期跑的进程 + 简单协议 | **无** —— 你的环境完全自由 | 多一层序列化，时间戳要跨进程透传 |

**我推荐 B**：分工最合理——你不用学 ROS，我不用碰视觉，中间是一个普通的 Python 调用，没有网络、没有序列化、没有额外延迟。

**但如果你的依赖迁不到 py3.10（第 8 问的答案很重），就走 C，不要硬迁。**
C 是完全可行的，我这边多写一个桥接节点而已。它的代价我清楚（见 §8.6），我来承担。

**这个选择不需要你现在做决定** —— 先回问卷，我看完给建议。

### 8.5 如果走 B：接口长这样

你交一个类，大概是这个形状（细节可以商量，以你现有代码顺手为准）：

```python
class BottlePoseEstimator:
    def __init__(self, config: dict): ...

    def process(self, rgb, depth, K, stamp: float) -> dict | None:
        """返回 None 表示这一帧没看到瓶子（很重要，见 §3.4）。

        返回 dict 时：
            position    (3,)  瓶子几何中心，米，在相机光学系里     ← §1.1
            orientation (4,)  四元数 (w,x,y,z)，+Z 是瓶轴          ← §1.2
            confidence  float 0..1
            covariance  (6,6) 可选，不知道就别给
        """
```

**你完全不碰 ROS**：相机取流、时间戳、TF、话题、QoS 全是我的事。
你只管"给我图，我告诉你瓶子在哪"。§1–3 里跟你有关的就只剩**坐标约定**和**返回 None**两件事。

### 8.6 如果走 C：我需要额外知道的

进程间协议我来定（大概率 ZeroMQ PUB/SUB 或 Unix socket + msgpack），但有两件事要先说好：

- **时间戳必须跨进程透传**。这是最容易做错的一环，一旦丢了 §3.1 那套新鲜度判据全部失效。
  协议里会有一个显式的 `capture_stamp` 字段，请**不要**用发送时刻填它。
- **谁负责取流？** 你的进程自己开相机，还是我把图传给你？前者简单，后者省一次拷贝。看你现有代码。

### 8.7 包放哪（这条跟形态无关）

```
~/GYH/ws/src/sb_vision/          ← ROS 包，进 git。薄封装：节点 + launch + 配置 + 标定 yaml
~/third_party/FoundationPose/    ← 第三方仓库，不进 git
~/vision_assets/                 ← 权重、CAD 模型，不进 git
```

**第三方仓库和权重不要进我的 git**：这个仓库的 `.git` 已经 146 MB，FoundationPose 仓库加权重是 GB 级，
一旦提交进去就永久留在历史里，之后谁 clone 都要付这个代价，而且删不掉（除非改写历史让所有已有 clone 失效）。

请给一个 `README` 或 `setup` 脚本说明：FoundationPose 从哪个 commit clone、权重从哪下、放哪个路径、
用什么环境变量指过去。**路径要能配置，别写死 `/home/<你的用户名>/`。**

---

## 9. 验收指标（达到再交给我）

**目的很直接：避免来回传代码。** 下面的判据全部是**关于数据**的，不是关于你的环境的——
不管走 §8.4 的哪种形态，最终都要满足同一套。可以自己跑、自己判，不需要我参与。

### 9.1 环境报告（信息收集，不是门禁）

在**你自己的环境**里跑一次，把输出发我。它不判你对错，只是把 §8.3 问卷的前几条自动填了：

```bash
python3 vision_env_report.py            # 加 --json 便于 diff
```

它会打印 Python / torch / CUDA / GPU / 各依赖版本，加一条**环境指纹**。

脚本里那几项标"同进程前提"的检查（Python 3.10、rclpy 可导入、numpy 1.x、sm_120），
**是用来判断能不能走形态 A/B 的，不是对你的要求**。全绿 → 走 B，最省事；
有红 → 走 C，我多写个桥接，也完全可行。**两种都不是你的问题。**

### 9.2 门禁二：接口一致性

视觉节点跑起来，瓶子放桌上不动，机器人不动：

```bash
python3 ws/src/shuangbi_grasp/scripts/vision_interface_acceptance.py --static --duration 60
```

再来一次，手动把瓶子慢慢挪 20–30 cm：

```bash
python3 ws/src/shuangbi_grasp/scripts/vision_interface_acceptance.py --moving --duration 30
```

| # | 硬性项 | 门限 | 不过说明什么 |
|---|---|---|---|
| 1 | 收到消息 | > 0 条 | 话题名错，或用了 BEST_EFFORT（§2） |
| 2 | 频率 | ≥ 10 Hz | 流水线太慢 |
| 3 | 最大空档 | < 0.25 s | 有卡顿，机器人那一刻就是"没目标" |
| 4 | 观测延迟 p95 | ≤ 0.30 s | 流水线太慢，或 stamp 填的是发布时刻 |
| 5 | 观测延迟非负 | 全部 ≥ 0 | 两台机器时钟没同步（§3.3） |
| 6 | `header.stamp` 不倒退 | 单调 | 会被判 `out_of_order` |
| 7 | `sequence` 严格递增 | 单调递增 | 会被判 `out_of_order` |
| 8 | `frame_id` 非空且唯一 | — | 空 frame 会让位姿全错（§1.4） |
| 9 | `object_id` 非空且唯一 | — | 多目标要先定策略（清单第 7 条） |
| 10 | `confidence` | ∈ [0.2, 1] | 低于 0.2 全被丢 |
| 11 | 四元数已归一化 | \|q\| = 1 ± 1e-3 | — |
| 12 | 没有畸形消息 | 0 条 | — |
| 13 | 瓶轴倾角 | < 15° | 抓取侧硬门限（§1.3）。位姿在相机系里时此项自动跳过 |
| 14 | 静止位置标准差 | < 10 mm 逐轴 | 源本身太抖（**仅 `--static`**） |
| 15 | 静止逐帧位置跳变 | < 50 mm | 会撞上跳变门（**仅 `--static`**） |
| 16 | **生产校验器接受率** | **≥ 95%** | 见下 |

> **第 16 条是最重要的一项。** 它不是我另写的一套判据——脚本会把抓到的整条流
> **回放进机器人真正在跑的那个 `TargetValidator`**（生产默认配置）。
> 这个脚本说流是干净的，机器人就一定同意，因为跑的是同一份代码。
>
> 阈值是 95% 不是 100%：偶发一两帧超时任何流水线都有，机器人那一刻只是保持上一个目标，没有危害。
> 持续性的问题会被第 3、4 条抓住。脚本会自动丢弃前 2 秒暖机（CUDA 首次 JIT、FoundationPose 首帧
> register 天然慢，拿它们判稳态不公平）。

### 9.3 脚本还会输出一段"调参用测量值"

这段**不判定**,但请连同结果一起发我，它决定我这边 `filter_tau` 和跳变门怎么设：

```
  位置标准差 σ         [2.96, 3.16, 2.90] mm
  逐帧位置变化         中位 6.40 mm，最大 17.39 mm（跳变门 50 mm）
  逐帧姿态变化(四元数)  中位 0.00°，最大 0.00°（跳变门 30°）
  逐帧绕瓶轴 yaw 变化   中位 0.00°，最大 0.00°
  yaw 总漂移            0.0°
```

**特别是 yaw 那两行**：如果出现「位置几乎不动、姿态却在跳」，脚本会直接点名说这是 §1.2 的
旋转对称体 yaw 随机游走，并且告诉你**该改的是我这边的跳变门，不是视觉侧**。
这正是这份文档想避免的那类来回扯皮——把判据写成可执行的，谁该改就一目了然。

### 9.4 交付清单

**第一轮（现在就能给，不用写一行代码）：**

- [ ] §8.3 九个问题的回答
- [ ] `vision_env_report.py` 在**你自己环境**里的输出
- [ ] 手眼标定结果 + 残差（§6 第 2 条）★ 这一项不给，抓取精度无从谈起
- [ ] CAD 模型 + 原点约定 + 实测尺寸（§6 第 3 条）

**第二轮（代码能跑之后）：**

- [ ] `vision_interface_acceptance.py --static --duration 60` 全 PASS，输出已附上
- [ ] `vision_interface_acceptance.py --moving --duration 30` 全 PASS，输出已附上
- [ ] 两段 rosbag（静止 60 s / 移动 30 s）
- [ ] FoundationPose 运行模式说明（§6 第 5 条）
- [ ] 代码 + 依赖清单 + 第三方仓库/权重获取方式（§6 第 1 条，形状按 §8.4 定的形态）
- [ ] 目标丢失时**停止发布**已实现并验证过（§3.4，脚本测不了这条，需要你手动确认）

> 走形态 B / C 时，`vision_interface_acceptance.py` 测的是**我把 ROS 包好之后**的输出，
> 那时候我们已经在一起调了。所以第二轮的时机以实际形态为准，不必卡死。
> 但**坐标约定和"丢了就别发"这两条无论哪种形态都要满足** —— 它们是关于数据的，不是关于 ROS 的。

---

## 10. 一句话总结

**关于数据（无论用哪种对接形态，这几条都要满足）：**

> **位置 = 瓶子几何中心（米）**；姿态**只要求 +Z 是瓶轴**（Z 的正负、绕 Z 的 yaw 都无所谓）；
> 时间戳填**拍照时刻**不是发布时刻；**没检测到就别发**；**别做滤波**。

**关于环境：**

> 按你顺手的来，我来适配。只要回 §8.3 的九个问题 + 跑一次 `vision_env_report.py` 把输出发我。

**如果最后走形态 A（你直接发 ROS 话题）**，再加上：

> 发 `sb_communication/msg/ObjectPose` 到 `object_pose`，**默认 QoS**（用 BEST_EFFORT 会一条都收不到），
> 10–30 Hz，`frame_id` 填 `head_d435i_optical_frame`，`sequence` 严格递增。

---

## 附：相关代码位置

### 验收与环境脚本（交给同事的就是这三个）

| 脚本 | 干什么 |
|---|---|
| [vision_env_report.py](../ws/src/shuangbi_grasp/scripts/vision_env_report.py) | 环境体检 + 指纹，6 项硬性检查，退出码 0 才算过 |
| [vision_interface_acceptance.py](../ws/src/shuangbi_grasp/scripts/vision_interface_acceptance.py) | 对着活话题跑 16 项接口验收，含生产校验器回放 |
| [setup_vision_env.sh](../ws/src/shuangbi_grasp/scripts/setup_vision_env.sh) | 一键建环境（克隆 yolov8 → sb_vision） |

### 我方代码

| 内容 | 文件 |
|---|---|
| 消息定义 | [ObjectPose.msg](../ws/src/sb_ros2_pkg/sb_communication/msg/ObjectPose.msg) |
| 接收/校验/滤波 | [target_manager.py](../ws/src/shuangbi_grasp/shuangbi_grasp/target_manager.py) |
| ROS 包装 + TF 变换 | [target_manager_node.py](../ws/src/shuangbi_grasp/shuangbi_grasp/target_manager_node.py) |
| 数据契约与默认门限 | [contracts.py](../ws/src/shuangbi_grasp/shuangbi_grasp/contracts.py) |
| **参考实现**（照着抄） | [sim_object_pose.py](../ws/src/shuangbi_grasp/shuangbi_grasp/sources/sim_object_pose.py) |
| 瓶子几何 / 抓取约定 | [grasp_profile_bottle.yaml](../ws/src/shuangbi_grasp/config/grasp_profile_bottle.yaml) |
| 姿态怎么被用（`orientation[:, 2]`） | [grasp_pose_generator.py](../ws/src/shuangbi_grasp/shuangbi_grasp/grasp_pose_generator.py) |
| 相机 frame 定义 | [shuangbi20260803_righthand.urdf](../ws/src/shuangbi_description/urdf/shuangbi20260803_righthand.urdf) 第 1000 行起 |
| 整体设计（阶段 8 = 视觉接入） | [README.md](../README.md) 第 916 行起 |
