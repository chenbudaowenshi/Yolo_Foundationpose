# 调整相机位置后的实机复测

结论：ROS2 压缩图像、深度与 CameraInfo 接入正常；当前摆放下的瓶子检测未通过，因此没有触发 FoundationPose 注册，也没有有效位姿输出。

## 用户确认重新摆放后的主测试

- 直接订阅用户自己启动的 `python my_demo/run_realsense_viewer.py`，Domain 0、默认 JPEG + compressedDepth、1280×720。
- 用户的发布端未启用 IMU，本轮视觉端显式使用 `--upright-source off`。未停止或重启用户的相机进程。
- 实际记录约 59.950 秒、2,172 条循环状态，完成 **900 次新 RGB-D 检测循环**，约 15 Hz。
- 所有循环状态均为 `searching`；目标轨迹数最大值 0，有效位姿数 0，未发生位姿注册。
- 测试视觉进程正常退出，输出文件最终为 `stopped`。
- 使用原默认目标 `0=fenda`、置信度 0.70、YOLO 输入 640；没有修改模型权重、选瓶逻辑或正式门限。

## 定位证据

独立订阅器从同一相机话题再次读取 12 对完整 RGB-D，保存原始解码图、米单位深度、K 与采集时间，见 `centered/probe/rgbd.npz`、`centered/probe/color.png`。

对该真实输入直接运行检测器：

| 项目 | 结果 |
|---|---|
| 默认门限 0.70 下的 YOLO 原始框 | 0 |
| 深度掩膜过滤后的实例 | 0（上游已无框） |
| 仅诊断用门限 0.05、输入 640 | 完整画面无候选 |
| 仅诊断用门限 0.05、输入 960 | 完整画面无候选 |
| FP16 / FP32 | 完整画面均无候选 |
| 瓶子局部裁剪、门限 0.05 | `fenda` 最高置信度约 0.0603 / 0.0602，仍远低于正式门限 |

这表明当前阻塞发生在 YOLO 候选产生阶段，而不是深度掩膜过滤或位姿质量门控之后。话题接通并不等于该检测模型能识别当前场景；下一步需核对该权重的训练类别、当前实物瓶型以及视角/背景覆盖。尚不能仅凭这一轮确定是某个单独的背景因素造成。

另外在用户确认重新摆放之前，进行过原图/JPEG90与1280/848分辨率诊断，当时都未检出目标。这些数据在 `input_comparison/`，不可作为重新摆放后同一场景的对照。
`model_checks.json` 还保留了对旧有带彩色掩膜预览图的诊断；由于它不是原始 RGB，不能用于判断模型是否变化或比较真实识别准确率。

## 文件

- `test_summary.json`：两轮 60 秒实测统计；以 `after_confirmation` 为用户确认重新摆放后的结果。
- `centered/vision.log`、`centered/recordings/`：主测试日志、录像、逐帧状态。
- `centered/probe/color.png`、`centered/probe/rgbd.npz`：独立订阅取得的真实场景与 RGB-D。
- `centered/probe/diagnostic.json`：默认检测、低阈值/多分辨率定位结果。
- `centered/probe/model_checks.json`：FP16/FP32与局部裁剪诊断。

主测试命令（相机发布端已经运行）：

```bash
python my_demo/d435i_autonomous_vision.py --ros2 --ros-domain-id 0 --trust-checkpoints --upright-source off --record-preview --max-runtime-seconds 60
```

本报告没有将“收到图像”记为“位姿验收通过”，也没有通过降低正式置信度来生成假通过结果。
