# ROS2 MuJoCo 抓取流水线

`run_grasp_pipeline.py` 把仿真、头部扫描、FoundationPose 6D 位姿和手臂 IK
连接成一个流程。它只启动一个 ROS2 MuJoCo 窗口；头控、位姿和 IK 都作为子进程
连接到这个窗口。

## 一键运行

当前机器使用 `ROS_DOMAIN_ID=91` 时，在 FoundationPose 根目录执行：

```bash
python /home/robot/FoundationPose/my_sim/run_grasp_pipeline.py \
  --ros-domain-id 91 \
  --ros-navigate \
  --ros-hold-final
```

流程会依次执行：

1. 启动 `ros2 launch bxi_example_py_elf3 example_launch_demo.py enable_arm_ik:=true`；
2. 启动 `head_control_mujoco.py`，扫描头部并在检测到 bottle 后保持角度；
3. 启动 `get_pose_mujoco.py`，等待同步 RGB-D、头部反馈和 odom，写入
   `runs/foundationpose_mujoco/pose.json`；
4. 启动 `arm_ik.py --live-ros`，导航到底盘 stance 后执行 pregrasp、approach、
   grasp 三段手臂轨迹。

`--ros-hold-final` 会一直保持最后手臂姿态，按 `Ctrl-C` 结束流程。默认会在
成功后保留 MuJoCo 窗口；需要自动关闭时加 `--no-keep-sim`。

上面的命令会打开 FoundationPose 的 OpenCV 识别窗口，显示 RGB、分割框和位姿
投影效果。无桌面环境或不需要预览时才加 `--no-show`：

```bash
python /home/robot/FoundationPose/my_sim/run_grasp_pipeline.py \
  --ros-domain-id 91 --no-show --ros-navigate --ros-hold-final
```

如果机器人已经由 ROS2 MuJoCo 启动，使用 `--reuse-sim`，不要再次创建第二个
窗口：

```bash
python /home/robot/FoundationPose/my_sim/run_grasp_pipeline.py \
  --reuse-sim --ros-domain-id 91 --ros-navigate --ros-hold-final
```

## 重要条件

- 仿真必须是 ROS2 launch 启动的 MuJoCo。`python get_sim.py` 或
  `arm_ik.py --viewer` 创建的独立 viewer 没有 ROS IPC，不能被这个流程接管。
- controller 必须由 `enable_arm_ik:=true` 启动，并在日志中出现
  `SIMULATION ARM IK OVERRIDE ENABLED`。编译过 controller 后要重新 source：

  ```bash
  source /opt/ros/humble/setup.bash
  source /opt/bxi/bxi_ros2_pkg/setup.bash
  source /home/robot/FoundationPose/bxi_rl_controller_ros2_example/install/setup.bash
  ```

- 所有 ROS 节点必须使用同一个 domain。若不希望底盘移动，可使用
  `--no-ros-navigate`；这要求机器人已经在报告计算出的 stance 附近。

先查看命令而不启动进程：

```bash
python /home/robot/FoundationPose/my_sim/run_grasp_pipeline.py --dry-run
```
