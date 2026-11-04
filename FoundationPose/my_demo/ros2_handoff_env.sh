#!/usr/bin/env bash
# source my_demo/ros2_handoff_env.sh
# SB_DDS_INTERFACE=lo is for local tests; set the robot-facing interface on deployment.
# Keep temporary variables local when this file is sourced by another script.
_sb_handoff_configure() {
  local _sb_root _sb_local_ros _sb_peers _sb_peer
  _sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
  source /opt/ros/humble/setup.bash || return 1
  if [ -f "$_sb_root/ros2_sb_ws/install/setup.bash" ]; then
    source "$_sb_root/ros2_sb_ws/install/setup.bash" || return 1
  fi
  _sb_local_ros="$_sb_root/.local_ros/root/opt/ros/humble"
  if [ -d "$_sb_local_ros" ]; then
    export AMENT_PREFIX_PATH="$_sb_local_ros:${AMENT_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$_sb_local_ros/lib:$_sb_local_ros/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  fi
  export ROS_DOMAIN_ID="${SB_ROS_DOMAIN_ID:-100}"
  export ROS_LOCALHOST_ONLY=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export PYTHONPYCACHEPREFIX="$_sb_root/.cache/python"
  if [ -n "${SB_DDS_INTERFACE:-}" ]; then
    case "$SB_DDS_INTERFACE" in *[!a-zA-Z0-9_.:-]*) echo 'Invalid SB_DDS_INTERFACE' >&2; return 1 ;; esac
    _sb_peers=''
    for _sb_peer in ${SB_DDS_PEERS:-}; do
      case "$_sb_peer" in *[!a-zA-Z0-9_.:-]*) echo 'Invalid SB_DDS_PEERS' >&2; return 1 ;; esac
      _sb_peers="$_sb_peers<Peer address=\"$_sb_peer\"/>"
    done
    export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$SB_DDS_INTERFACE\"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex><Peers>$_sb_peers</Peers></Discovery></Domain></CycloneDDS>"
  fi
}

if _sb_handoff_configure; then
  unset -f _sb_handoff_configure
else
  unset -f _sb_handoff_configure
  return 1
fi
