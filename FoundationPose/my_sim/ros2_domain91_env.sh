#!/usr/bin/env bash
# Source this file in every terminal that participates in the camera ROS graph.
# Usage: source my_demo/ros2_domain91_env.sh
export ROS_DOMAIN_ID=91
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
