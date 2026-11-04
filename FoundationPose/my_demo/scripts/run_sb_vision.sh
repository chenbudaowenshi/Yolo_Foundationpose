#!/usr/bin/env bash
set -eo pipefail
_sb_launch_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$_sb_launch_root/my_demo/ros2_handoff_env.sh"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$_sb_launch_root/runs/sb_vision/ros_logs}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$_sb_launch_root/.cache/torch_extensions}"
_sb_launch_python="${SB_VISION_PYTHON:-/home/robot/anaconda3/envs/pose/bin/python}"
cd "$_sb_launch_root"
exec "$_sb_launch_python" "$_sb_launch_root/my_demo/sb_vision_node.py" "$@"
