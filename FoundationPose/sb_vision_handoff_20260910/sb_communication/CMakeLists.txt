cmake_minimum_required(VERSION 3.8)
project(sb_communication)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotionCommands.msg"
  "msg/ObjectPose.msg"
  "srv/SimReset.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
