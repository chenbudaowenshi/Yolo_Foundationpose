# RTX 5070 Ti vision installation

This installs the vision computer side of the ROS 2 Humble pipeline. The robot remains the D435i camera publisher; its CPU does not run FoundationPose. No robot connection or motion is part of installation.

The target must have Ubuntu 22.04 x86_64, an NVIDIA driver that sees the RTX 5070 Ti, ROS 2 Humble with `colcon` and CycloneDDS, Conda, and internet access. The script installs CUDA 12.8 toolkit inside a new Conda environment and checks an actual `sm_120` CUDA kernel. It does not change the NVIDIA driver or system ROS installation.

## Transfer the exact project snapshot

The project contains local, untracked code and private model weights, so a plain `git clone` is insufficient. On the source computer:

```bash
bash my_demo/scripts/package_vision_5070ti.sh
```

Transfer `runs/vision_5070ti_bundle.tar.gz` and its `.sha256` file to the target computer using the site's approved file transfer method. Verify and extract it into an empty directory such as `~/FoundationPose`:

```bash
sha256sum -c vision_5070ti_bundle.tar.gz.sha256
mkdir -p ~/FoundationPose
tar -xzf vision_5070ti_bundle.tar.gz -C ~/FoundationPose
```

## Install on the RTX 5070 Ti computer

```bash
cd ~/FoundationPose
bash my_demo/scripts/install_vision_5070ti.sh
```

The script creates `.envs/sb_vision_5070ti` in the extracted tree. It checks the exact SHA-256 of the bottle mesh, YOLO checkpoint, FoundationPose scorer/refiner checkpoints, and `ObjectPose.msg` before installing anything. It installs PyTorch 2.7.1/cu128 ([official PyTorch command](https://pytorch.org/get-started/previous-versions/)), builds PyTorch3D v0.7.9 and nvdiffrast v0.4.0 from their official tagged sources, builds `mycpp` and `sb_communication`, then runs unit tests, real CUDA kernels, and a local compressed RGB-D/ObjectPose ROS test. Progress and errors are in `runs/install_5070ti/install.log`.

It is safe to rerun after an interrupted installation. Build parallelism defaults to two jobs; set `SB_BUILD_JOBS=1` on a memory constrained machine. `SB_5070TI_ENV` can select a different absolute Conda environment path.

## Start the installed vision process

For operation with the robot camera, source the same ROS Domain and CycloneDDS network settings as the camera publisher, then run:

```bash
bash my_demo/scripts/run_sb_vision_5070ti.sh \
  --clock-sync-mode required --clock-server YOUR_NTP_SERVER_IP \
  --target-class 0
```

The robot, vision computer and controller must already use that same NTP/chrony server. `--clock-sync-mode local` is valid only when the camera and vision process share one host. Clock configuration is described in `my_demo/README_clock_sync.md`. The target computer does not have to open the RealSense device.

The receiver sequence file is separate for each ROS Domain and topic. On a new receiver, the first run needs `--sequence-start-after 0`; if the receiver has already accepted messages, use its last accepted sequence instead. Later runs reuse `.state/sb_vision` without the initialization flag.

This installer validates GPU kernels on synthetic RGB-D. It does not establish real camera latency, hand-eye TF correctness or grasp accuracy.
