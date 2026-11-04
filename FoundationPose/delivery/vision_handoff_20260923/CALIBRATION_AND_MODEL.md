# 手眼标定、相机和物体模型

## 手眼结果

来源：`/home/robot/FoundationPose/my_demo/config/head_camera_calibration_20260918.json`。

- 变换：`head_pitch_Link -> head_d435i_optical_frame`。
- 平移：`[0.0482865848, 0.0715464494, -0.0277481987] m`。
- RPY：`[-1.5388904723, 0.0000060869, -1.5610775346] rad`。
- 留出姿态数量：20。
- 板原点一致性 RMS/P95/Max：3.424/5.224/5.238 mm。
- 全角点 RMS/P95/Max：3.153/4.552/5.450 mm。
- 重投影 RMS/P95/Max：2.055/3.834/5.403 px。

限制：该结果标记为 `deployed=false`、`absolute_accuracy_verified=false`、`depth_pointcloud_verified=false`。它是 optical frame 变换，不能直接作为物理 `head_d435i_link` 安装关节；部署时必须保留正确的 optical 固定变换关系。

## 相机

- D435i 序列号：327122073383。
- 采集：1280x720@30；ROS 发布约 15 Hz。
- 深度与 RGB 对齐；线上深度编码为 16UC1 毫米。
- SDK 畸变模型：inverse_brown_conrady；本机实测 D 为五个零。
- CameraInfo 和图像保留各自采集时间戳。

正式部署后必须核对相机序列号和运行时 K/D；当前实机 K 与部分历史标定报告中的 K 不同。

## 瓶子模型

- mesh：`/home/robot/FoundationPose/runs/mesh/bottle.obj`，单位米。
- 尺寸：`65.068 x 65.122 x 159.5 mm`。
- CAD 原点：瓶底中心轴附近。
- 长轴：物体 `+Z`。
- 几何中心偏移：从 CAD 原点沿 Z `+0.07975 m`。
- 质量：尚未测量。

控制侧旧标称 `36 mm x 200 mm x 0.45 kg` 与当前瓶子不一致。必须更新碰撞体、抓握中心和夹爪直径参数后再做抓取验收。
