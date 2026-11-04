"""Shared camera transport for the interactive and autonomous ROS2 demos."""
from __future__ import annotations

from collections import Counter
import math
import os
import time

from rgbd_transport import image_topics
from sb_pose_contract import RGBDSynchronizer
from imu_alignment import RosImuInput


def add_camera_arguments(parser):
    parser.add_argument("--camera-prefix", default="/head_camera", help="camera topic namespace")
    parser.add_argument("--image-transport", choices=("compressed", "raw"), default="compressed")
    parser.add_argument("--rgb-topic", help="override color topic (type follows --image-transport)")
    parser.add_argument("--depth-topic", help="override aligned depth topic (type follows --image-transport)")
    parser.add_argument("--camera-info-topic", help="override color CameraInfo topic")
    parser.add_argument("--depth-camera-info-topic", help="override aligned depth CameraInfo topic")
    parser.add_argument("--camera-frame", help="expected optical frame; default: read color CameraInfo")
    parser.add_argument("--max-rgbd-skew-ms", type=float, default=50.)
    parser.add_argument("--ros-domain-id", type=int, help="explicit ROS domain; default: inherit ROS_DOMAIN_ID or 0")
    parser.add_argument("--depth-scale-topic", help="legacy option; units now follow encoding: 16UC1 mm, 32FC1 m")
    parser.add_argument("--imu-topic", help="ROS2 IMU topic; default: <camera-prefix>/imu")
    parser.add_argument("--max-imu-skew-ms", type=float, default=30., help="maximum IMU/image capture-time difference")


def resolve_camera_arguments(args):
    if not args.camera_prefix.strip("/"):
        raise ValueError("--camera-prefix must name a nonempty camera namespace")
    prefix = "/" + args.camera_prefix.strip("/")
    rgb, depth = image_topics(prefix, args.image_transport)
    defaults = {"rgb_topic": rgb, "depth_topic": depth,
                "camera_info_topic": prefix + "/color/camera_info",
                "depth_camera_info_topic": prefix + "/depth/camera_info", "imu_topic": prefix + "/imu"}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not math.isfinite(args.max_rgbd_skew_ms) or args.max_rgbd_skew_ms <= 0:
        raise ValueError("--max-rgbd-skew-ms must be positive and finite")
    if args.ros_domain_id is not None and not 0 <= args.ros_domain_id <= 232:
        raise ValueError("--ros-domain-id must be between 0 and 232")
    if not math.isfinite(args.max_imu_skew_ms) or args.max_imu_skew_ms <= 0:
        raise ValueError("--max-imu-skew-ms must be positive and finite")
    return args


class RosRGBDInput:
    """Bounded, stamp-matched four-topic input; decode only the newest pair.

    Uses the same codecs/calibration checks as the production grasp node.
    ``RGBDFrame.depth_m`` is always metres, regardless of the wire encoding.
    """

    def __init__(self, node, args):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image
        self.node = node
        self.args = resolve_camera_arguments(args)
        self.sync = RGBDSynchronizer(frame_id=args.camera_frame, max_skew_ms=args.max_rgbd_skew_ms)
        self.counts = Counter()
        self.errors = Counter()
        self.last_error = None
        self.color_info = None
        self.last_frame = None
        self._last_notice = float("-inf")
        self._last_pair_at = float("-inf")
        self._received_first = False
        image_type = CompressedImage if args.image_transport == "compressed" else Image
        topics = (("rgb", args.rgb_topic, image_type), ("depth", args.depth_topic, image_type),
                  ("rgb_info", args.camera_info_topic, CameraInfo),
                  ("depth_info", args.depth_camera_info_topic, CameraInfo))
        self.subscriptions = [node.create_subscription(kind, topic,
                              lambda msg, key=key: self.receive(key, msg), qos_profile_sensor_data)
                              for key, topic, kind in topics]
        self.imu = RosImuInput(node, args)
        node.get_logger().info(
            f"ROS2 RGB-D: domain={node.context.get_domain_id()}, rmw={os.environ.get('RMW_IMPLEMENTATION', 'default')}, "
            f"transport={args.image_transport}; rgb={args.rgb_topic}; depth={args.depth_topic}; "
            f"color_info={args.camera_info_topic}; depth_info={args.depth_camera_info_topic}; "
            "units: 16UC1=mm, 32FC1=m; source timestamps preserved")
        if args.depth_scale_topic:
            node.get_logger().warning("--depth-scale-topic is retained for CLI compatibility; "
                                      "standard ROS depth units are taken from image encoding, not sensor scale")
        node.get_logger().info(f"IMU REQUIRED: {args.imu_topic}; capture skew <= {args.max_imu_skew_ms:g} ms, "
                               f"age <= {args.imu_timeout:g} s; no registration/output without aligned IMU")

    def receive(self, key, msg):
        self.counts[key] += 1
        # Learn only from calibration, not arbitrary image traffic. A user may
        # pin --camera-frame to require a particular calibrated optical frame.
        if self.sync.frame_id is None:
            if key != "rgb_info" or not msg.header.frame_id:
                return
            self.sync.frame_id = msg.header.frame_id
        try:
            self.sync.add(key, msg)
            if key == "rgb_info":
                self.color_info = msg
        except ValueError as error:
            self.errors["metadata"] += 1
            self.last_error = str(error)

    def take_latest(self):
        try:
            frame = self.sync.pop_latest()
        except (ValueError, TypeError) as error:
            self.errors["decode_or_calibration"] += 1
            self.last_error = str(error)
            frame = None
        if frame is not None:
            self.last_frame = frame
            self.last_error = None
            self._last_pair_at = time.monotonic()
            self.counts["pairs"] += 1
            if not self._received_first:
                h, w = frame.depth_m.shape
                self.node.get_logger().info(f"First synchronized ROS2 RGB-D: {w}x{h}, frame={frame.frame_id}")
                self._received_first = True
            return frame
        now = time.monotonic()
        if now - self._last_notice >= 2 and (now - self._last_pair_at >= 2 or self.last_error):
            self.node.get_logger().warning(
                f"Waiting for synchronized RGB-D: counts={dict(self.counts)}, last_error={self.last_error}; "
                "check publisher domain, transport and both stamped CameraInfo topics")
            self._last_notice = now
        return None
