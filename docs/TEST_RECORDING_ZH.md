# D435i 测试录制说明

## 录制带标注的预览视频

正常启动程序后，按 `V` 开始录制，再按一次 `V` 停止。退出程序时会自动结束并封装当前视频。

也可以用命令行自动开始：

```bash
conda activate fopose
cd ~/YOLOv8

python examples/d435i_foundationpose.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --debug 1 \
  --record-preview \
  --record-fps 12 \
  --trust-checkpoints
```

默认输出目录：

```text
runs/foundationpose_realtime/recordings/
```

每段记录包含：

```text
<时间>_annotated.mp4
<时间>_frames.jsonl
```

MP4保存窗口中看到的紫色YOLO掩膜、黄色3D卡尔曼中心、绿色CAD框、位姿和状态文字。JSONL按处理帧保存类别、框、RGB-D测量、卡尔曼状态、6D矩阵和位姿质量，`video_frame_index`用于和MP4对齐。

程序按真实经过时间补帧，因此FoundationPose注册期间的停顿也会反映在视频时间线上。

## 同时录制RealSense原始RGB-D

如需以后离线重放原始彩色和深度流，添加一个尚不存在的 `.bag` 文件名：

```bash
python examples/d435i_foundationpose.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --debug 1 \
  --record-preview \
  --record-bag runs/foundationpose_realtime/recordings/qixi_test_01.bag \
  --trust-checkpoints
```

`.bag`包含相机原始彩色和深度流，不包含YOLO或CAD叠加。MP4用于直观看结果，`.bag`用于重新运行算法，两者用途不同。

程序拒绝覆盖已有 `.bag`，每次实验应更换文件名，例如：

```text
qixi_static_01.bag
qixi_slow_motion_01.bag
qixi_occlusion_01.bag
```

录制前可检查磁盘：

```bash
df -h ~/YOLOv8
```

原始RGB-D录包体积远大于MP4。测试结束后按 `Q` 正常退出，让RealSense完整关闭并写完文件。

## 回放 `.bag`

打开：

```bash
realsense-viewer
```

然后在RealSense Viewer中加载录制文件。回放时不要同时让其他程序占用D435i。

当前实时脚本已经支持录包，但尚未增加“直接把 `.bag` 作为算法输入”的离线回放模式；现阶段可先用Viewer检查彩色、深度、时间戳和深度空洞。

## 推荐测试命名

每个实验只改变一个条件：

```text
fenda_static_center_01
cola_slow_translation_01
qixi_fast_translation_01
qixi_partial_occlusion_01
qixi_multi_object_01
```

同时记录目标距离、光照、是否遮挡、是否按R重新注册以及最终是否发生漂移，后续才能对不同方案做可重复比较。
