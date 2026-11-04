# 视觉对接包（抓取/控制侧 → 视觉侧）

2026-09-18 实现更新：当前运行入口、模型配置和本机实测结果见
[FoundationPose 对接说明](../my_demo/README_sb_handoff.md)。本包校验器已改为按瓶轴判跳变；
控制侧生产代码需要同步。以下 2026-09-10 文档保留作原始接口约定。

打包日期：2026-09-10

这个包里是你接入抓取侧需要的全部东西：消息定义、接口约定、两个自检脚本。
**先看 `docs/vision_worksplit.md`（分工），再看 `docs/vision_handoff.md`（接口细节）。**

---

## 里面是什么

```
sb_communication/       ROS 2 消息包 —— 放进你的工作空间 src/ 编译
shuangbi_grasp/         验收脚本的最小依赖（纯 numpy，3 个文件，不是完整包）
scripts/
  vision_env_report.py             环境体检 + 指纹，跑一次把输出发回来
  vision_interface_acceptance.py   接口验收，16 项，交付前自己跑
docs/
  vision_worksplit.md              分工：谁做什么、谁给谁什么 ← 先看这个
  vision_handoff.md                接口约定的完整版
  sim_object_pose.py.reference     参考实现：字段怎么填、时间戳怎么处理，可以直接抄
```

---

## 第一步：编译消息包

```bash
cp -r sb_communication ~/你的工作空间/src/
cd ~/你的工作空间
colcon build --packages-select sb_communication
source install/setup.bash

# 验证
ros2 interface show sb_communication/msg/ObjectPose
```

### ⚠️ 核一次指纹，然后别再动这个文件

```bash
sha256sum sb_communication/msg/ObjectPose.msg | cut -c1-16
# 必须是：03463e448beaa082
```

**为什么要人肉核**：ROS 2 Humble **没有类型哈希**（`RIHS01_` 是 Iron 之后才有的）。
两边的 `.msg` 只要**类型名**一样就认，字段布局不同**不会报错**，直接按错误的布局反序列化——
位姿变成随机数。这比"收不到"难查得多。

对上了就再也不用管。只有抓取侧改了 `.msg` 才需要重新同步（目前没这个计划）。

---

## 第二步：网络

三台机器都要一致。`ROS_DOMAIN_ID = 10`。

```bash
source /opt/ros/humble/setup.bash
source ~/你的工作空间/install/setup.bash

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="你的网卡" priority="default" multicast="default"/></Interfaces></General><Discovery><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex><Peers><Peer address="192.168.112.252"/><Peer address="192.168.112.104"/></Peers></Discovery></Domain></CycloneDDS>'
```

- 网卡名用 `ip -o -4 addr show` 查（找 `192.168.112.x` 那一行的第二列）
- `192.168.112.252` = 抓取侧，`192.168.112.104` = 机器人
- **`RMW_IMPLEMENTATION` 必须是 cyclonedds**：FastDDS ↔ Cyclone 理论上能互通，实际很不可靠
- **用显式 `<Peers>` 而不是靠组播**：抓取侧那台走 WiFi，很多 AP 会限速或丢组播帧，
  表现是"节点时有时无"。列 peer 走单播才稳
- `MaxAutoParticipantIndex 60`：单播发现时每域默认最多 10 个参与者，仿真栈就有 8 个节点，
  不提高上限会报 `rcl node's rmw handle is invalid`

---

## 第三步：环境体检

在**你实际要跑推理的那台机器**上、source 过工作空间之后跑：

```bash
python3 scripts/vision_env_report.py            # 加 --json 便于对比
```

**这不是考试。** 它检查的是"能不能把视觉代码和 rclpy 放进同一个进程"，用来决定对接形态：

- 全绿 → 你交一个纯 Python 类，ROS 那层抓取侧来包，你完全不用碰 ROS
- 有红 → 你的进程独立跑，中间走 IPC，抓取侧多写一个桥接节点

**两种都不是你的问题，也都不需要你改环境。** 把输出发回来就行。

---

## 第四步：接口验收（交付前自己跑）

视觉节点发 `ObjectPose` 之后：

```bash
# 瓶子放桌上不动、机器人不动
python3 scripts/vision_interface_acceptance.py --static --duration 60

# 手动把瓶子慢慢挪 20-30 cm
python3 scripts/vision_interface_acceptance.py --moving --duration 30
```

脚本会自己找到同级的 `shuangbi_grasp/`，不用设 `PYTHONPATH`（设了反而会把 ROS 的路径覆盖掉、丢失 rclpy）。

16 项全 PASS 再交付。**第 16 项最重要**：它把抓到的整条流**回放进机器人真正在跑的那个校验器**
（`shuangbi_grasp/target_manager.py`，就是同一份代码），所以这个脚本说流是干净的，机器人就一定同意。

脚本还会打一段"调参用测量值"（位置 σ、逐帧变化、yaw 漂移）——**不判定，但请连同结果一起发回来**，
抓取侧要靠它定滤波和跳变门限。

---

## 最要紧的两条约定（其余看 `docs/vision_handoff.md`）

**位置 = 瓶子的几何中心**（米），不是瓶底、不是掩膜质心、不是 CAD 原点。

**姿态只要求物体系 +Z 是瓶子的长轴。** Z 的正负、绕 Z 的 yaw 都无所谓——
圆柱的 yaw 不可观测，抓取侧本来就自己采样候选。

另外：**时间戳填拍照时刻**（不是发布时刻）、**没检测到就别发**、**别做滤波**（滤波在抓取侧，
而且需要看到你的原始输出才能调门限）。

---

## 第一轮要回什么

1. `docs/vision_worksplit.md` §7「还没定的」里点你名的几条
2. `scripts/vision_env_report.py` 的输出
3. **手眼标定结果 + 残差** ★ —— 这一项不给，抓取精度无从谈起
4. **register / 重注册 / 持续跟踪各耗时多久** ★ —— 现在有两跳网络，这个数直接决定延迟预算够不够
5. FoundationPose 用的 mesh 文件、单位、原点位置、哪个轴是长轴

代码可以之后再说，这五条不用写一行代码就能给。
