# D435i 自主视觉交付包说明

这个目录用于制作和交付 `d435i_autonomous_vision.py`。推荐发送“应用包”，由接收方在自己的机器上安装 FoundationPose；若双方均为非商业研究/评估用途，也可用打包脚本把 FoundationPose 源码和权重一并放入归档。

## 交付内容

应用包约十余 MB，包含：

- 自主视觉主程序及 6 个本地运行依赖模块；
- 当前项目的完整 `ultralytics` 源码，避免接收方安装到不兼容版本；
- `best.pt` YOLO 权重；
- 米制 CAD 网格；
- 中文运行文档、安装校验器、启动器和许可证。

FoundationPose 约 272 MB，必须具有两组推理权重。接收方还需要 NVIDIA GPU、兼容的驱动/CUDA 工具链、D435i 和 Python 3.11 环境。

完整文件清单见 `MANIFEST.txt`。

## 在发送方生成归档

只生成应用包，推荐：

```bash
cd /home/hssw/YOLOv8
bash deployment/d435i_autonomous_vision/build_bundle.sh
```

输出位于 `dist/`，同时生成 `.sha256` 校验文件。

若确认符合 FoundationPose 的非商业许可证，并希望连同源码和权重发送：

```bash
bash deployment/d435i_autonomous_vision/build_bundle.sh \
  --with-foundationpose /home/hssw/FoundationPose
```

脚本不会打包数据集、训练结果、录像、本机 Conda 环境、缓存或发送方编译出的 `mycpp` 二进制。

## 接收方安装

### 1. 解压与校验

```bash
sha256sum -c d435i-autonomous-vision-*.tar.gz.sha256
tar -xzf d435i-autonomous-vision-*.tar.gz
cd d435i_autonomous_vision_bundle
```

如果发送的是应用包，在本目录旁边克隆或放置官方 FoundationPose，并下载其 scorer/refiner 权重。也可以通过 `FOUNDATIONPOSE_ROOT` 指向其他位置。

### 2. 创建 FoundationPose 环境

以 FoundationPose 自带的 `environment.yml`、`requirements.txt` 和 README 为准。大致流程是：

```bash
conda env create -f FoundationPose/environment.yml
conda activate foundationpose
```

接着根据接收方 NVIDIA 驱动和 CUDA 工具链安装匹配的 PyTorch、torchvision、PyTorch3D 和 nvdiffrast。不要直接复制发送方的 Conda 环境或 `.so` 文件。

安装 FoundationPose 与应用依赖：

```bash
python -m pip install -r FoundationPose/requirements.txt
python -m pip install -r requirements-app.txt

cd FoundationPose
bash build_all_conda.sh
cd ..
```

`mycpp` 含本机 ABI 相关的 C++/pybind11 扩展，必须在接收方重新编译。

### 3. 校验安装

FoundationPose 在包内时：

```bash
python verify_install.py --require-camera
```

FoundationPose 位于其他目录时：

```bash
python verify_install.py \
  --foundationpose-root /path/to/FoundationPose \
  --require-camera
```

### 4. 运行

包内 FoundationPose：

```bash
./run_d435i.sh \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --trust-checkpoints
```

外置 FoundationPose：

```bash
FOUNDATIONPOSE_ROOT=/path/to/FoundationPose \
./run_d435i.sh \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --trust-checkpoints
```

完整运行参数和常见问题见 `docs/README_D435I_AUTONOMOUS_VISION_ZH.md`。

## 不要发送的内容

- `datasets/`、CVAT 工具、训练缓存和无关训练结果；
- `last.pt`，运行只需要 `best.pt`；
- `.git/`、`.vscode/`、`__pycache__/`、`.pyc`；
- `runs/autonomous_vision/` 下的录像、状态和调试输出；
- 本机 Conda 环境或 FoundationPose 的 `mycpp/build/*.so`；
- `.blend`、无关 CAD 和中间网格。

发送前还应确认 `best.pt` 和 CAD 文件是否允许提供给接收方。

