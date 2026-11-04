# 自主检测状态与质量修复验收

日期：2026-09-19。本轮修改 `my_demo/d435i_autonomous_vision.py`、新增 `my_demo/autonomous_state.py`，同步修改旧诊断读取端 `my_demo/foundationpose_pose_ros_node.py`。

## 已完成的五项修复

1. 遮挡/漏检期间保留内部预测及预览矩阵，对外明确输出 `pose_predicted`、`pose_valid=false`，不再把预测当成当前有效观测。
2. 输出真实 SDK 曝光/采集时间或 ROS 图像原始时间戳；检测暂停时不更新观测时刻、序号，当前观测标志为 false。读取端同时检查捕获时间与文件更新时间，`PoseStamped` 使用整数纳秒原样传递采集时刻。异步结果只能在当前检测确实更新后验收。
3. 正常退出、Ctrl+C 和相机循环异常先强制写入 `stopped`，清空位姿、目标和质量，再等待工作线程，绕过常规发布限频。崩溃/强制终止仍依赖读取端超时保护。
4. 单独输出 `registration_age_s` 和 `observation_age_s`；v2 的 `pose_age_s` 是观测年龄别名，附带明确语义。切换目标清除注册年龄和旧观测时刻。
5. 新增原始深度点到 CAD 表面的几何一致性门控；`position_error_m` 改为滤波中心与原始 RGB-D 测量的差异。质量不通过时对外降级，不更改选瓶、短时预测与重注册规则。

`multi_object_rgbd_tracker.py`、`rgbd_kalman_tracker.py`、`d435i_foundationpose.py` 和 `async_pose_worker.py` 与修复前审计记录的哈希相同。没有修改标定或生产 `sb_vision_node.py` 的选择逻辑。

## 验证

| 检查 | 结果 |
|---|---|
| 单元/回归测试 | 99 项通过，日志 `unit_tests.log` |
| D435i + RTX 4070 实机静态测试 | 35 秒，926 帧状态，735 个有效测量；2 帧短时漏检明确标为预测；退出后为 stopped |
| 真实相机 + 受控故障注入 | 701 帧状态，172 个有效测量；0.3 s 漏检、1.2 s 检测暂停、2.5 s 漏检及恢复均通过 |
| 暂停核心区间 | 30 帧全部无有效输出，采集时间不推进，观测年龄增长超过 0.5 s；保留内部预测 |
| 真实输出的读取端离线回放 | 静态与故障共 1,627 帧，0 帧有效性不一致，0 个无效状态被接受 |
| 纳秒时间戳 | 共 907 个接受样本的 ROS 时间转换与原始观测时间逐纳秒一致 |
| 静态有效观测时间 | 采集至输出判断 P50 63.86 ms / P95 78.98 ms / 最大 80.85 ms；不包含下游网络和控制延迟 |
| 独立三维检查 | 静态样本的深度表面距离中位数约 7.96 mm；单测将深度或位姿错开 120 mm 后拒绝，非原有恒零自比较 |
| 相机收尾 | 两轮验收正常退出，相机已释放，最新状态清空位姿 |

静态场景中，初选候选一度未通过原有投影 IoU 门限，程序按原规则重试并切换至另一候选后取得有效位姿；没有为通过测试放宽原有投影阈值或新的深度门槛。

深度一致性门槛为 15 mm、支持比例至少 60%，它不是绝对定位精度。测试没有实物位置真值，也未驱动机器人。
故障测试使用真实 RGB-D 与 GPU，但漏检/暂停由测试脚本注入；不能等同于再次完成真实手部遮挡验收。

## 文件

- `static/live.log`、`static/recordings/`：35 秒原始实机记录。
- `faults_verified/regression_report.json`、`faults_verified/states.jsonl`、`faults_verified/recordings/`：故障回归证据，`failures=[]`。
- `reader_replay.json`：1,627 帧实际输出的读取端回放统计。
- `source_after.sha256`：修改后关键文件哈希。
- `my_demo/README_autonomous_state.md`：新旧字段、时效、质量门控与复测命令。
- `my_demo/tests/test_autonomous_state.py`：状态、超时、退出、精确时间戳、错误深度回归。
- `my_demo/scripts/test_autonomous_camera.py`：可复跑的真实相机故障测试。

保留了 `faults/` 第一轮测试记录：测试脚本最初按发布时刻划分故障阶段，把阶段开始前刚完成真实检测的边界帧误算为暂停帧。已修正测试时间窗口并重跑；所有边界帧仍参加全局“有效输出必须有当前测量”的断言，没有放宽业务规则。
