#!/usr/bin/env bash
set -eo pipefail
_sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
mkdir -p "$_sb_root/ros2_sb_ws"
cd "$_sb_root/ros2_sb_ws"
colcon build --base-paths "$_sb_root/sb_vision_handoff_20260910/sb_communication" --packages-select sb_communication --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
