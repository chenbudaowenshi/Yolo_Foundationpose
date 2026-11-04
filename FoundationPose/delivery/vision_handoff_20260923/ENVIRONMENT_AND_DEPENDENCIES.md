# 环境与依赖回答

对应原始需求 §8.3 九个问题。

1. Python：3.10.0，解释器 `/home/robot/anaconda3/envs/pose/bin/python`。
2. torch/CUDA/GPU：torch 2.6.0+cu124，PyTorch CUDA 12.4；RTX 4070 12 GB，算力 8.9；驱动 595.84。
3. 环境：Conda `pose`。现有 `environment.yml` 写 Python 3.11，只描述编译依赖，与实际运行环境不一致，需重新导出锁定环境。
4. 最终机器：相机程序在机器人；FoundationPose 在 RTX 4070 视觉机；控制端跨 ROS 2 网络接收。因此正式部署需要时钟同步。
5. ROS：已实现 rclpy 节点，直接发布 `ObjectPose`。
6. 推理：有状态持续跟踪；首次 register，随后 track，异常时重注册。
7. 性能：初次注册 233 ms；重注册 P50/P95 130/136 ms；跟踪 P50/P95 19/25 ms；显存峰值约 1.43 GiB；模型预加载约 3.18 s。
8. 本地/CUDA 扩展：nvdiffrast 0.4.0、FoundationPose `mycpp`、Warp 1.16 JIT；pytorch3d 0.7.9。迁移 Python/torch/CUDA 后需重编和验证。
9. FoundationPose：上游 `NVlabs/FoundationPose`，当前 git HEAD `a1b694b83e633c2cb6115b9063d940a687759392`，含本地修改。权重文件存在，但训练/下载来源说明尚未补齐。

环境报告：`/home/robot/FoundationPose/runs/sb_vision/environment_report.json`，指纹 `b51c6fd041a6`，`hard_pass=true`。

资产标识：

| 资产 | 路径 | SHA-256 |
|---|---|---|
| YOLOv8-seg | `runs/yolov8/best.pt` | `1d66903956f9fe8913f0d75bc71392af351893f4e5ff555ba9d34cb9680f0910` |
| FP refiner | `weights/2023-10-28-18-33-37/model_best.pth` | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |
| FP scorer | `weights/2024-01-11-20-02-45/model_best.pth` | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
| bottle mesh | `runs/mesh/bottle.obj` | `566773b675e80bda7a76b0715336c2b144663492c0cd06cc0b4e6e5292862bc2` |

仍需交付：当前 Python 3.10 Conda 锁定文件、扩展编译步骤、YOLO 类别/训练配置和所有权重的可靠获取地址。
