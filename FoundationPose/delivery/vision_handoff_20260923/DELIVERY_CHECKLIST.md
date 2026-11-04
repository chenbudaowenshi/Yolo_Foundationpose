# 视觉交付总清单

需求基准：`requirements/vision_handoff.md` §6、§9.4。

状态：`[x]` 已实现且有证据；`[~]` 有实现或旧证据但需复验；`[ ]` 未完成。

## 第一轮交付

- [x] §8.3 九个环境问题已回答，见 [环境与依赖](ENVIRONMENT_AND_DEPENDENCIES.md)。
- [x] 环境报告已生成：`/home/robot/FoundationPose/runs/sb_vision/environment_report.json`。
- [~] 手眼矩阵和残差已有，见 [标定与模型](CALIBRATION_AND_MODEL.md)；尚未部署到机器人 TF，也未完成外部绝对精度验证。
- [~] CAD、原点、长轴和尺寸已明确；瓶子质量未测，控制侧旧瓶子参数必须更新。

## 第二轮交付

- [~] 历史静止 60 s 验收全 PASS，但早于 2026-09-20 的检测降频/缓存修改，且对应原图链路；当前代码和压缩链路必须重跑。
- [ ] 当前代码静止 60 s 验收未完成。
- [ ] 移动 30 s 验收未完成。
- [ ] 静止 60 s `/object_pose` rosbag2 未提供。
- [ ] 移动 30 s `/object_pose` rosbag2 未提供。
- [x] FoundationPose 运行模式、注册/跟踪耗时和显存说明已完成。
- [~] ROS 节点、启动器、配置和测试已存在；实际 Python 3.10 锁定环境、权重获取方式和 CUDA 扩展重编说明仍需补齐。
- [~] 目标丢失停发与原合同存在差异：当前允许最多 0.30 s 检测缓冲。控制侧必须确认接受，或视觉侧恢复严格 miss 即停发并复验。

## 接口合同

- [x] 发布 `/object_pose`，类型 `sb_communication/msg/ObjectPose`，RELIABLE。
- [x] `header.stamp` 是原始图像采集时间，不是推理完成时间。
- [x] `frame_id=head_d435i_optical_frame`，`object_id=bottle`。
- [x] position 是瓶子 mesh 几何中心、单位米；物体 `+Z` 是瓶轴。
- [x] confidence 包含 YOLO 和几何/深度质量；covariance 全零表示未知。
- [x] sequence 持久化，reset、重注册和进程重启不清零。
- [x] 彩色 JPEG、深度 16UC1 无损 PNG、CameraInfo 和原时间戳收发已实现。
- [~] 对外 `object_id` 固定为 `bottle`；抓取任务显式 lock/unlock 尚未联调。

## 部署和最终门禁

- [ ] 机器人、视觉机和控制机配置同一 NTP/chrony 源并保存检查结果。
- [ ] 跨机 DDS 频率、延迟、丢包和稳定性验收完成。
- [ ] 手眼结果按 optical/physical frame 关系部署并保存 URDF/TF 证据。
- [ ] 实际工作区完整变换链位置误差 < 20 mm。
- [ ] 更新瓶子质量、碰撞体、抓握中心和夹爪参数。
- [ ] 抓取中目标锁定、遮挡和重新选择策略联调通过。
- [ ] 至少 20 轮真实抓取完成并记录成功率与失败原因。

当前结论：本机视觉接口和核心功能已经具备，但还不能声明真实抓取完整交付完成。详细证据和命令见 [验收结果与计划](ACCEPTANCE_STATUS.md)。
