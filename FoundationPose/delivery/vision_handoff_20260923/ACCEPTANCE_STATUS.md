# 验收结果与后续计划

## 已有证据

历史静止 60 s 验收：

- 稳态 899 条，15.0 Hz。
- 最大空档 106 ms。
- 延迟中位/P95/最大 100/123/156 ms。
- 位置标准差 `[0.52, 0.42, 0.95] mm`。
- 最大逐帧位置变化 5.83 mm。
- confidence 0.41 至 0.58。
- 生产 `TargetValidator` 接受率 899/899，100%。

证据：`/home/robot/FoundationPose/runs/sb_vision/live_acceptance_static.txt`。

限制：这次验收早于 2026-09-20 的检测降频/缓存修改，并对应原图链路，因此只能作为历史证据，不能替代当前版本最终验收。

其他证据：

- 性能：`/home/robot/FoundationPose/runs/optimization_20260920/REPORT.md`。
- 注册统计：`/home/robot/FoundationPose/runs/registration_metrics_20260919/REPORT.md`。
- 压缩传输：`/home/robot/FoundationPose/runs/sb_vision/compression_20260919/`。
- 序号重启：`/home/robot/FoundationPose/runs/sb_vision/sequence_restart_20260919/`。
- 时钟模拟：`/home/robot/FoundationPose/runs/sb_vision/clock_sync_20260919/`。

## 必须补做

1. 当前代码、压缩话题、静止瓶子运行 60 s：

```bash
python /home/robot/list/vision_interface_acceptance.py --static --duration 60
```

2. 手动移动瓶子 20 至 30 cm，运行 30 s：

```bash
python /home/robot/list/vision_interface_acceptance.py --moving --duration 30
```

3. 同时录制两段 rosbag2：静止 60 s、移动 30 s。至少包含 `/object_pose`；建议同时包含 RGB、depth、两路 CameraInfo、IMU、TF 和 `/vision/status`。
4. 机器人、视觉机和控制机统一 NTP/chrony 源，验证跨机时间、DDS 延迟和稳定性。
5. 部署手眼 TF 后，在工作区用外部真值验证完整坐标链，位置误差要求 < 20 mm。
6. 更新瓶子质量、碰撞体和抓握参数。
7. 完成遮挡时任务锁定、解锁/重选联调。
8. 完成至少 20 轮真实抓取并记录成功率、失败阶段、TCP 误差和恢复行为。

## 判定

当前状态：可继续本机和跨机联调；尚未达到真实抓取最终交付条件。
