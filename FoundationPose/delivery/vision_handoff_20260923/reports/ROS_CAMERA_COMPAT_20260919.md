# ROS2 相机话题与传输适配验收

修复对象：`my_demo/d435i_autonomous_vision.py` 和 `my_demo/d435i_foundationpose.py` 的 ROS2 输入。
两个入口原本仍订阅 `/camera/...` 原始 Image，现在共用 `my_demo/ros_rgbd_input.py`，默认匹配发布端的 `/head_camera` JPEG + 16UC1 compressedDepth。

## 变更

- 默认接收 `/head_camera/color/image_raw/compressed`、`/head_camera/depth/image_rect_raw/compressedDepth`。
- 匹配 `/head_camera/color/camera_info` 与 `/head_camera/depth/camera_info`；检查采集时间、对齐内参、尺寸与坐标系，再解码最新的一对。
- 共用已有生产压缩编解码与 RGB-D 同步实现，保留 JPEG 色彩顺序和 PNG 无损深度。
- 16UC1 按毫米转米，32FC1 已是米；不再将编码单位与相机传感器原始 scale 混用。
- `--image-transport raw` 保留原图接入；支持 `--camera-prefix`、各图像/CameraInfo 话题覆盖及 `--camera-frame`。
- `--ros-domain-id 0` 可显式对齐当前相机域；启动日志打印实际域与最终话题。
- 默认 IMU 话题改为 `/head_camera/imu`。发布端不开 IMU 时，提示显式选择纯图像模式，或在相机端加 `--use-imu`；不自动关闭原有竖直约束。
- 修复相机每次帧率计数清零后重复打印“First synchronized RGB-D frame published”的日志问题。
- 没有修改目标选择、遮挡宽限、自动重注册规则或手眼标定。

## 验证结果

- 107 项单元/回归测试通过，见 `tests.log`。
- 独立 Domain 81、loopback、私有 `/rgbd_compat_test` 话题上的真实 DDS 测试：raw 和 compressed 各 99 对，原始时间戳、实际 K、米单位深度全部正确，见 `dds_transport.json`。
- 第一次 Domain 0 测试没有收到数据；检查时原相机进程已退出。对应日志保留在 `live_ros.log`，没有将这次空运行记作通过。
- 随后临时启动实体 D435i 压缩发布，在 Domain 0 / FastDDS 上分别运行纯图像、IMU 两种接收模式，各 18 秒。
- 纯图像模式处理 271 对真实 RGB-D；IMU 模式处理 270 对，807 条循环状态均记录新鲜 IMU。实际分辨率 1280×720，坐标系 `head_d435i_optical_frame`。
- 相机稳态发布约 14.9—15.0 Hz，整段日志仅打印一次首帧通知。
- 测试结束后，本次启动的相机与视觉进程均正常退出，输出状态为 `stopped`。

本轮实时画面中，瓶身被图像下边缘截断，默认分类/置信度下没有选中目标。因此实机验证覆盖了压缩图像、真实内参、时间匹配、GPU 检测循环和 IMU 输入，不代表完成了有效物体位姿或抓取精度验收。没有放宽检测阈值来制造通过结果。

## 证据和使用

- `summary.json`：汇总计数。
- `live_with_camera/camera.log`：实体相机发布日志。
- `live_with_camera/images_only/`、`live_with_camera/with_imu/`：视觉日志、预览录像、逐帧状态、最终 stopped 状态。
- `my_demo/README_ros_camera_input.md`：可直接复制的双终端启动方式、域说明、IMU 与自定义话题参数。
- `my_demo/scripts/test_ros_rgbd_input.py`：原图/压缩图 DDS 回归脚本。
