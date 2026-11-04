# 自主检测输出与观测时效（schema v2）

入口仍为 `python my_demo/d435i_autonomous_vision.py --trust-checkpoints`。
使用 ROS2 相机时，压缩话题、域与 IMU 配置见 [ROS2 相机接入说明](README_ros_camera_input.md)。
多瓶关联、最近已确认目标选择、短时遮挡预测、丢失宽限和自动重新注册规则保持原样。
新增检查只决定对外位姿是否有效；内部跟踪和预览可继续保留预测位姿。
正式抓取的 `sb_communication/ObjectPose` 入口仍是 `sb_vision_node.py`，本说明不表示任务锁定接口已完成。

## 输出契约

`runs/autonomous_vision/latest_state.json` 使用原子替换，默认 10 Hz。

| 字段 | 定义 |
|---|---|
| `schema_version` | `2` |
| `session_id` | 此次进程的 UUID；轨迹号仅在该会话内有意义 |
| `status` | `pose_valid`、`pose_predicted`、`pose_stale`、`pose_degraded` 或注册/搜索/生命周期状态 |
| `processing_state` | 原有内部跟踪/注册状态，用于诊断，不作为有效位姿标志 |
| `pose_valid` | 当前有已接受的 RGB-D 测量和对齐 IMU，时效和几何检查通过且位姿非空 |
| `observation_state` | `measured`、`predicted`、`stale` 或 `unavailable` |
| `selected_target.observed_this_frame` | 本轮真正运行了检测且检测到该目标；后台暂停和重复帧为 false |
| `selected_target.measurement_accepted_this_frame` | 本轮该目标深度测量被跟踪器接受 |
| `observation_stamp_ns` | 该目标最后一次已接受 RGB-D 观测的原始彩色采集时刻，整数 Unix 纳秒 |
| `observation_sequence` | 对应的本进程相机帧序号；复用观测不更新 |
| `observation_age_s` | 当前处理时间减观测采集时刻，不是文件写入间隔 |
| `pose_age_s` | v2 中是 `observation_age_s` 的兼容别名，见 `pose_age_semantics` |
| `registration_age_s` | 距本目标上次 FoundationPose 注册成功的单调时钟时间；切换目标清空 |
| `pose_quality.depth_surface` | 原始目标深度点到 CAD 表面的独立一致性检查 |

`pose_camera_object` 仍表示 `T_camera_CAD_original`。为了继续显示预测框，预测或降级状态也可能包含矩阵，消费者不能以“矩阵非空”判定有效。

本机 SDK 开启 global time，优先使用曝光中点元数据；硬件时钟或无法匹配的 RGB-D 时间不提供有效输出。
ROS 输入保留原始 `Image.header.stamp`；RGB、depth 都有新消息才处理新的一对。
默认 RGB-D 时间差不超过 50 ms，观测年龄不超过 0.5 s，可分别用 `--max-rgbd-skew-ms`、`--max-observation-age-seconds` 调整。
跨机输入依然需要实际时钟同步；这些参数不修正系统时钟。

IMU 已改为强制使用；具体规则见 [IMU 强制对齐说明](README_imu_required.md)。

## 有效性与退出

后台注册暂停 YOLO 时，不推进观测时刻/序号，不把缓存标为本轮观测。
异步注册结果仅在真正完成新一轮检测之后处理，避免工作线程恰好结束时使用旧检测验收。
正常退出、Ctrl+C 和相机运行异常都会先强制写入 `stopped`，清空位姿、质量和目标，再等待线程结束；该写入绕过 10 Hz 限频。
SIGKILL、断电无法执行清理，因此读取端仍须检查过期时间。

`foundationpose_pose_ros_node.py` 是旧诊断桥接：v2 同时检查文件年龄、真实观测年龄、有效状态、当前观测及独立深度质量，以及必需 IMU 的采集时间绑定。
`PoseStamped.header.stamp` 原样使用整数纳秒观测时间；重写文件无法延长观测寿命。
默认文件/观测年龄上限均为 0.5 s。v1 历史文件仍可诊断回放，但明确标为文件 mtime 时间语义，不用于运动相机的动态 TF 对接。

## 三维质量含义

对最多 1,024 个原始有效目标深度点反投影，转换到 CAD 坐标系，计算其到采样模型表面的最近距离。
默认要求有效深度比例至少 20%，至少 60% 的采样点在模型表面 15 mm 内；记录距离中位数、P90 和支持比例。
表面采样使用独立固定随机种子，不改变模型推理的随机状态。坏深度或不一致几何使对外状态降级，不反馈改变选瓶规则。

`position_error_m` 改为滤波中心与当前原始 RGB-D 中心测量的差异，不再与刚赋值的同一个位置自比较。
这些指标是几何一致性约束，不是绝对定位误差；模型尺寸、完整视野和实物参考精度仍需现场核验。

## 回归

```bash
cd /home/robot/FoundationPose
source my_demo/ros2_handoff_env.sh
PYTHONPYCACHEPREFIX=/tmp/autonomous-tests /home/robot/anaconda3/envs/pose/bin/python -m unittest discover -s my_demo/tests -v

# 占用本机相机；真实 RGB-D/GPU + 测试脚本注入漏检和检测暂停，最长 60 秒自动停止。
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_autonomous_camera.py runs/autonomous_camera_regression
```

故障脚本只在自身进程注入 0.3 秒漏检、1.2 秒检测暂停、2.5 秒漏检，然后验证恢复。
它不模拟用户真实手部图像，不代表再次完成了物理遮挡或机器人抓取验收。
