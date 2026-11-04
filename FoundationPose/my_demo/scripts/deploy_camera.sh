#!/usr/bin/env bash
# Stage a versioned camera-only release. Run only after the robot is online.
set -eo pipefail
_sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_sb_host="${1:-192.168.112.104}"
_sb_release="foundationpose_camera_$(date -u +%Y%m%dT%H%M%SZ)"
_sb_archive="$_sb_root/runs/sb_vision/$_sb_release.tar.gz"
mkdir -p "$_sb_root/runs/sb_vision"
tar -czf "$_sb_archive" -C "$_sb_root/my_demo" RealSenseCamera.py run_realsense_viewer.py camera_contract.py rgbd_transport.py sb_pose_contract.py scripts/test_camera_stream.py
ssh -o BatchMode=yes -o ConnectTimeout=8 "$_sb_host" "mkdir -p '$_sb_release' && tar -xzf - -C '$_sb_release'" < "$_sb_archive"
printf 'Camera release staged on %s: ~/%s\n' "$_sb_host" "$_sb_release"
printf 'Source ROS and configure CycloneDDS on the robot, then run:\n'
printf '  cd ~/%s\n  python3 scripts/test_camera_stream.py --duration 20 --report camera_test_report.json\n' "$_sb_release"
