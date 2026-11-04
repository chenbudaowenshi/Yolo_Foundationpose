#!/usr/bin/env python3
"""Autonomous multi-instance drink perception for D435i and FoundationPose.

The requested class is tracked as multiple persistent RGB-D instances. The
nearest confirmed instance is locked, FoundationPose registration is scheduled
automatically after stability checks, and no runtime key press is required.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import d435i_foundationpose as base

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from async_pose_worker import LatestPoseRegistrationWorker
from multi_object_rgbd_tracker import (
    Detection3D,
    MultiObjectRGBDTracker,
    NearestTargetSelector,
    TrackView,
    box_iou,
)
from rgbd_kalman_tracker import TrackerConfig
from upright_pose_utils import GravityAxisFilter, pose_with_center_position, pose_with_upright_axis


DEFAULT_OUTPUT = base.REPO_ROOT / "runs/autonomous_vision"


@dataclass(frozen=True)
class InstancePayload:
    observation: base.TargetObservation
    depth_filter: base.PoseDepthFilterResult
    measurement: base.RGBDPositionMeasurement | None


@dataclass(frozen=True)
class RegistrationRequest:
    request_id: int
    track_id: int
    reason: str
    submitted_s: float
    K: np.ndarray
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    box: np.ndarray
    gravity_axis: np.ndarray | None
    gravity_sample_s: float | None
    tracker_position_m: np.ndarray


class VisionStatePublisher:
    """Publish the latest robot-readable state with an atomic file replace."""

    def __init__(self, path: Path, rate_hz: float) -> None:
        if rate_hz <= 0:
            raise ValueError("vision output rate must be positive")
        self.path = path.expanduser().resolve()
        self.period_s = 1.0 / float(rate_hz)
        self.last_publish_s = float("-inf")

    def publish(self, state: dict[str, object], now_s: float) -> None:
        if now_s - self.last_publish_s < self.period_s:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, self.path)
        self.last_publish_s = now_s


def read_key_from_stdin(timeout_s: float = 0.0) -> int:
    if not sys.stdin.isatty():
        return 255
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not readable:
            return 255
        char = sys.stdin.read(1)
        return ord(char) if char else 255
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--use-imu",
        dest="upright_source",
        action="store_const",
        const="imu",
        help=(
            "use IMU acceleration as the upright constraint; disabled by default "
            "(equivalent to --upright-source imu)"
        ),
    )
    parser.add_argument("--max-det", type=int, default=100, help="maximum YOLO instances per frame")
    parser.add_argument("--track-confirm-hits", type=int, default=3)
    parser.add_argument("--track-max-missed", type=int, default=90)
    parser.add_argument("--track-association-distance", type=float, default=0.15)
    parser.add_argument("--selection-lost-grace", type=int, default=45)
    parser.add_argument("--selection-distance", choices=("euclidean", "z"), default="euclidean")
    parser.add_argument("--registration-stable-seconds", type=float, default=1.0)
    parser.add_argument("--registration-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--registration-retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--registration-max-attempts", type=int, default=3)
    parser.add_argument("--failed-track-exclusion-seconds", type=float, default=8.0)
    parser.add_argument("--registration-stale-distance", type=float, default=0.05)
    parser.add_argument("--registration-stale-iou", type=float, default=0.20)
    parser.add_argument(
        "--refresh-after-motion",
        dest="refresh_after_motion",
        action="store_true",
        help="refresh the locked target pose asynchronously after it moves (default)",
    )
    parser.add_argument(
        "--no-refresh-after-motion",
        dest="refresh_after_motion",
        action="store_false",
    )
    parser.add_argument(
        "--autonomous-registration",
        dest="autonomous_registration",
        action="store_true",
        help="automatically register/recover the selected target (default)",
    )
    parser.add_argument(
        "--no-autonomous-registration",
        dest="autonomous_registration",
        action="store_false",
    )
    parser.add_argument("--autonomous-output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--vision-state-file",
        type=Path,
        default=DEFAULT_OUTPUT / "latest_state.json",
    )
    parser.add_argument("--vision-output-rate", type=float, default=10.0)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        help="optional automatic stop for smoke tests; omit for production",
    )
    parser.set_defaults(
        auto_register=True,
        auto_reregister=True,
        reregister_stable_frames=30,
        autonomous_registration=True,
        refresh_after_motion=True,
        display=True,
        record_dir=DEFAULT_OUTPUT / "recordings",
        upright_source="off",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    base.validate_args(args, parser)
    positive = (
        args.max_det,
        args.track_confirm_hits,
        args.track_max_missed,
        args.track_association_distance,
        args.selection_lost_grace,
        args.registration_stable_seconds,
        args.registration_cooldown_seconds,
        args.registration_retry_base_seconds,
        args.registration_max_attempts,
        args.failed_track_exclusion_seconds,
        args.registration_stale_distance,
        args.registration_stale_iou,
        args.vision_output_rate,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("autonomous tracking/registration values must be positive and finite")
    if not 0.0 <= args.registration_stale_iou <= 1.0:
        parser.error("--registration-stale-iou must be in [0, 1]")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.pose_tracking_mode == "foundationpose":
        parser.error("autonomous multi-instance mode requires upright-follow or upright-hold")


def annotate_tracks(
    image: np.ndarray,
    K: np.ndarray,
    views: list[TrackView],
    selected_track_id: int | None,
) -> np.ndarray:
    output = image.copy()
    for view in views:
        selected = view.track_id == selected_track_id
        color = (0, 255, 255) if selected else (255, 0, 255)
        payload = view.payload if isinstance(view.payload, InstancePayload) else None
        if payload is not None:
            output = base.overlay_mask(output, payload.observation.mask, color, alpha=0.20)
        x1, y1, x2, y2 = (int(value) for value in view.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3 if selected else 2)
        distance = "depth?" if view.distance_m is None else f"{view.distance_m:.3f}m"
        state = "LOCK" if selected else ("OK" if view.confirmed else "NEW")
        cv2.putText(
            output,
            f"ID={view.track_id} {state} {distance} {view.confidence:.2f}",
            (max(0, x1), max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
        position = view.tracker.position_m
        if position is not None and position[2] > 0.01:
            projected = K @ position
            uv = np.rint(projected[:2] / projected[2]).astype(int)
            if 0 <= uv[0] < output.shape[1] and 0 <= uv[1] < output.shape[0]:
                cv2.drawMarker(output, tuple(uv), color, cv2.MARKER_CROSS, 15, 2)
    return output


def track_metadata(views: list[TrackView], selected_track_id: int | None) -> list[dict[str, object]]:
    return [view.metadata(selected=view.track_id == selected_track_id) for view in views]


def run(args: argparse.Namespace) -> int:
    yolo_path = args.yolo_model.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    foundationpose_root = args.foundationpose_root.expanduser().resolve()
    output_dir = args.autonomous_output_dir.expanduser().resolve()
    record_dir = args.record_dir.expanduser().resolve()
    for path, label in ((yolo_path, "YOLO checkpoint"), (mesh_path, "mesh")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose repository not found: {foundationpose_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(base.REPO_ROOT))
    sys.path.insert(0, str(foundationpose_root))
    sys.path.insert(0, str(foundationpose_root / "mycpp/build"))

    from ultralytics import YOLO
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from learning.training import predict_pose_refine, predict_score
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose requires a visible CUDA GPU")
    set_logging_format()
    set_seed(0)
    base.install_foundationpose_opencv_compat(predict_pose_refine, predict_score)

    model = YOLO(str(yolo_path))
    if args.target_class not in model.names:
        raise ValueError(f"target class {args.target_class} is absent; classes={model.names}")
    mesh = base.load_mesh(mesh_path, trimesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inverse_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack((-extents / 2.0, extents / 2.0), axis=0).reshape(2, 3)
    center_depth_offset = (
        float(args.tracker_center_depth_offset)
        if args.tracker_center_depth_offset is not None
        else float(np.mean(mesh.extents[:2]) * 0.5)
    )
    tracker_config = TrackerConfig(
        acceleration_std_mps2=args.tracker_accel_std,
        measurement_xy_std_m=args.tracker_measurement_xy_std,
        measurement_z_std_m=args.tracker_measurement_z_std,
        mahalanobis_gate=args.tracker_gate,
        max_missed_frames=max(args.tracker_max_missed, args.track_max_missed),
        stationary_speed_mps=args.tracker_stationary_speed,
        stationary_frames=args.tracker_stationary_frames,
    )
    tracks = MultiObjectRGBDTracker(
        tracker_config,
        confirmation_hits=args.track_confirm_hits,
        maximum_detection_misses=args.track_max_missed,
        maximum_3d_distance_m=args.track_association_distance,
        stable_box_motion=args.reregister_box_motion,
    )
    selector = NearestTargetSelector(
        lost_grace_frames=args.selection_lost_grace,
        distance_mode=args.selection_distance,
    )
    gravity_filter = (
        GravityAxisFilter(smoothing=args.gravity_smoothing)
        if args.upright_source == "imu"
        else None
    )

    print("Loading FoundationPose scorer/refiner weights...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(output_dir / "foundationpose_debug"),
        debug=args.debug,
        glctx=glctx,
    )

    def register_pose(request: RegistrationRequest) -> np.ndarray:
        return estimator.register(
            K=request.K,
            rgb=request.rgb,
            depth=request.depth_m,
            ob_mask=request.mask,
            iteration=args.est_refine_iter,
        )

    registration_worker = LatestPoseRegistrationWorker(register_pose)
    publisher = VisionStatePublisher(args.vision_state_file, args.vision_output_rate)
    preview_recorder = base.PreviewVideoRecorder(record_dir, args.record_fps)
    preview_auto_start_pending = bool(args.record_preview)

    ros_node = None
    ros_tf_buffer = None
    ros_helpers = []
    latest = {
        "rgb": None,
        "depth": None,
        "info": None,
        "depth_scale": None,
        "imu": None,
        "imu_sequence": 0,
        "message_counts": {"rgb": 0, "depth": 0, "info": 0, "depth_scale": 0, "imu": 0},
    }
    depth_scale = None
    started = False
    gravity_snapshot = None
    accel_to_color = None
    reference_upright_axis: np.ndarray | None = None
    track_views: list[TrackView] = []
    selected_track_id: int | None = None
    selected_stable_since: float | None = None
    pose: np.ndarray | None = None
    trusted_pose: np.ndarray | None = None
    pose_quality: base.PoseProjectionQuality | None = None
    pose_status = "searching"
    invalid_pose_frames = 0
    registration_anchor: np.ndarray | None = None
    motion_since_registration = False
    request_counter = 0
    registration_attempts = 0
    next_registration_s = 0.0
    last_registration_submit_s = float("-inf")
    last_pose_accepted_s: float | None = None
    excluded_until: dict[int, float] = {}
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    last_ros_imu_sequence = 0
    last_ros_imu_accepted_time: float | None = None
    last_ros_wait_log_s = float("-inf")
    ros_imu_status = (
        "waiting for first message"
        if args.ros2 and gravity_filter is not None
        else "disabled"
    )
    ros_imu_state = ros_imu_status

    def set_ros_imu_status(state: str, description: str) -> None:
        nonlocal ros_imu_state, ros_imu_status
        ros_imu_status = description
        if state != ros_imu_state:
            print(f"ROS2 IMU: {description}", flush=True)
            ros_imu_state = state

    def update_gravity_from_latest_ros_imu(camera_info) -> None:
        """Consume each ROS IMU message once and express acceleration in color axes."""
        nonlocal gravity_snapshot, last_ros_imu_accepted_time, last_ros_imu_sequence
        if not args.ros2 or gravity_filter is None:
            return
        imu_sequence = int(latest["imu_sequence"])
        if imu_sequence <= last_ros_imu_sequence:
            if last_ros_imu_accepted_time is not None:
                sample_age = time.perf_counter() - last_ros_imu_accepted_time
                if sample_age > args.imu_timeout:
                    set_ros_imu_status(
                        "stale",
                        f"last accepted sample is stale ({sample_age:.2f}s > {args.imu_timeout:.2f}s)",
                    )
            return
        last_ros_imu_sequence = imu_sequence
        imu_msg = latest["imu"]
        if imu_msg is None:
            return

        acceleration = np.array(
            [
                float(imu_msg.linear_acceleration.x),
                float(imu_msg.linear_acceleration.y),
                float(imu_msg.linear_acceleration.z),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(acceleration)):
            set_ros_imu_status("invalid", "rejected non-finite linear acceleration")
            return

        message_frame = str(imu_msg.header.frame_id).strip().lstrip("/")
        source_frame = (
            args.imu_frame.strip().lstrip("/")
            if args.imu_frame is not None
            else message_frame
        )
        if not source_frame:
            set_ros_imu_status(
                "missing-frame",
                "rejected empty Imu.header.frame_id; set --imu-frame to the IMU TF frame",
            )
            return
        target_frame = ""
        if camera_info is not None:
            target_frame = str(camera_info.header.frame_id).strip().lstrip("/")
        if not target_frame:
            set_ros_imu_status(
                "waiting-camera-frame",
                f"received {source_frame}; waiting for CameraInfo.header.frame_id",
            )
            return

        frame_description = source_frame
        if source_frame != target_frame:
            assert ros_tf_buffer is not None
            try:
                transform = ros_tf_buffer.lookup_transform(target_frame, source_frame, Time())
            except TransformException as error:
                set_ros_imu_status(
                    f"waiting-tf:{source_frame}:{target_frame}",
                    f"waiting for TF {source_frame} -> {target_frame}: {error}",
                )
                return
            rotation = transform.transform.rotation
            try:
                acceleration = base.rotate_vector_by_quaternion_xyzw(
                    acceleration,
                    np.array([rotation.x, rotation.y, rotation.z, rotation.w]),
                )
            except ValueError as error:
                set_ros_imu_status("invalid-tf", f"rejected invalid IMU TF rotation: {error}")
                return
            frame_description = f"{source_frame} -> {target_frame}"

        accepted_before = gravity_filter.snapshot().accepted_samples
        gravity_snapshot = gravity_filter.update(acceleration)
        acceleration_norm = gravity_snapshot.acceleration_norm_mps2
        norm_text = "unknown" if acceleration_norm is None else f"{acceleration_norm:.3f} m/s^2"
        if gravity_snapshot.accepted_samples > accepted_before:
            last_ros_imu_accepted_time = time.perf_counter()
            set_ros_imu_status(
                f"ok:{source_frame}:{target_frame}",
                f"receiving {frame_description}; acceleration norm={norm_text}",
            )
        else:
            set_ros_imu_status(
                "rejected-norm",
                f"rejected {frame_description}; acceleration norm={norm_text}, expected 7.0-12.5 m/s^2",
            )

    try:
        if args.ros2:
            try:
                import rclpy
                from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
                from sensor_msgs.msg import CameraInfo, Image
                from std_msgs.msg import Float32
                if gravity_filter is not None:
                    from rclpy.time import Time
                    from sensor_msgs.msg import Imu
                    from tf2_ros import Buffer, TransformException, TransformListener
            except Exception as error:
                raise RuntimeError(f"ROS2 python packages missing: {error}") from error

            def image_msg_to_cv2(msg, desired_encoding):
                if msg.height <= 0 or msg.width <= 0:
                    raise ValueError("invalid image dimensions")
                if msg.encoding == "bgr8" and desired_encoding == "bgr8":
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    return arr.reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8" and desired_encoding == "bgr8":
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    return arr[:, :, ::-1]
                if desired_encoding == "passthrough" and msg.encoding in ("16UC1", "mono16", "8UC1", "32FC1"):
                    dtype = {
                        "16UC1": np.uint16,
                        "mono16": np.uint16,
                        "8UC1": np.uint8,
                        "32FC1": np.float32,
                    }[msg.encoding]
                    arr = np.frombuffer(msg.data, dtype=dtype)
                    if msg.is_bigendian:
                        arr = arr.byteswap().newbyteorder()
                    return arr.reshape((msg.height, msg.width))
                raise ValueError(f"unsupported ROS image encoding: {msg.encoding} for desired {desired_encoding}")

            def make_image_callback(key):
                def cb(msg):
                    try:
                        encoding = "bgr8" if key == "rgb" else "passthrough"
                        latest[key] = image_msg_to_cv2(msg, desired_encoding=encoding)
                        latest["message_counts"][key] += 1
                    except Exception as error:
                        print(f"ROS2 callback error ({key}): {error}", flush=True)

                return cb

            def camera_info_cb(msg):
                latest["info"] = msg
                latest["message_counts"]["info"] += 1

            def depth_scale_cb(msg):
                try:
                    latest["depth_scale"] = float(msg.data)
                    latest["message_counts"]["depth_scale"] += 1
                except Exception:
                    latest["depth_scale"] = None

            def imu_cb(msg):
                latest["imu"] = msg
                latest["imu_sequence"] += 1
                latest["message_counts"]["imu"] += 1

            rclpy.init()
            ros_node = rclpy.create_node("d435i_autonomous_vision")
            if gravity_filter is not None:
                ros_tf_buffer = Buffer(node=ros_node)
                ros_helpers.append(
                    TransformListener(ros_tf_buffer, ros_node, spin_thread=False)
                )
            sensor_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
            )
            imu_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
            )
            ros_node.create_subscription(Image, args.rgb_topic, make_image_callback("rgb"), sensor_qos)
            ros_node.create_subscription(Image, args.depth_topic, make_image_callback("depth"), sensor_qos)
            ros_node.create_subscription(CameraInfo, args.camera_info_topic, camera_info_cb, sensor_qos)
            ros_node.create_subscription(Float32, args.depth_scale_topic, depth_scale_cb, sensor_qos)
            if gravity_filter is not None:
                ros_node.create_subscription(Imu, args.imu_topic, imu_cb, imu_qos)
            print(
                "ROS2 mode: subscribed to",
                f"rgb={args.rgb_topic},",
                f"depth={args.depth_topic},",
                f"camera_info={args.camera_info_topic},",
                f"depth_scale={args.depth_scale_topic},",
                (
                    f"imu={args.imu_topic}"
                    if gravity_filter is not None
                    else "imu=disabled (enable with --use-imu)"
                ),
            )
            print(
                "ROS2 environment: "
                f"domain={os.environ.get('ROS_DOMAIN_ID', '0')}, "
                f"localhost_only={os.environ.get('ROS_LOCALHOST_ONLY', '0')}, "
                f"rmw={os.environ.get('RMW_IMPLEMENTATION', 'default')}, "
                f"transport={os.environ.get('FASTDDS_BUILTIN_TRANSPORTS', 'default')}",
                flush=True,
            )
            for _ in range(max(1, args.warmup // 5)):
                rclpy.spin_once(ros_node, timeout_sec=0.05)
                for _ in range(8):
                    rclpy.spin_once(ros_node, timeout_sec=0.0)
                update_gravity_from_latest_ros_imu(latest.get("info"))
        else:
            if rs is None:
                raise RuntimeError("pyrealsense2 not available; use --ros2 to subscribe to ROS2 camera topics")
            context = rs.context()
            if not list(context.query_devices()):
                raise RuntimeError("No RealSense camera detected")
            pipeline = rs.pipeline()
            config = rs.config()
            if args.serial:
                config.enable_device(args.serial)
            config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            if args.upright_source == "imu":
                config.enable_stream(rs.stream.accel, base.realsense_motion_xyz_format(), args.accel_fps)
            if args.record_bag is not None:
                record_bag = args.record_bag.expanduser().resolve()
                if record_bag.suffix.lower() != ".bag":
                    raise ValueError("--record-bag output must use the .bag extension")
                if record_bag.exists():
                    raise FileExistsError(f"refusing to overwrite RealSense recording: {record_bag}")
                record_bag.parent.mkdir(parents=True, exist_ok=True)
                config.enable_record_to_file(str(record_bag))

            try:
                profile = pipeline.start(config)
            except RuntimeError as error:
                if args.upright_source == "imu" and "Permission denied" in str(error):
                    raise RuntimeError(
                        "D435i IMU permission denied; install the official librealsense udev rules"
                    ) from error
                raise
            started = True
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            align_to_color = rs.align(rs.stream.color)
            if args.upright_source == "imu":
                accel_profile = profile.get_stream(rs.stream.accel)
                color_profile = profile.get_stream(rs.stream.color)
                accel_to_color = accel_profile.get_extrinsics_to(color_profile)
            for _ in range(args.warmup):
                frames = pipeline.wait_for_frames()
                if gravity_filter is not None:
                    gravity_snapshot = gravity_filter.update(
                        base.accelerometer_vector_in_color_frame(frames, accel_to_color)
                    )

        print(f"Autonomous target: {args.target_class}={model.names[args.target_class]}")
        print(f"Vision state: {publisher.path}")
        if args.upright_source == "imu":
            imu_source = f"ROS2 {args.imu_topic}" if args.ros2 else "local D435i accelerometer"
            print(f"Upright constraint: {imu_source}, max tilt={args.max_upright_tilt:.1f}deg")
        print("No key press is required. R only forces a debug recovery; Q/Esc quits.")
        runtime_started_s = time.perf_counter()

        while True:
            if (
                args.max_runtime_seconds is not None
                and time.perf_counter() - runtime_started_s >= args.max_runtime_seconds
            ):
                print(f"Reached smoke-test runtime limit: {args.max_runtime_seconds:.1f}s")
                break
            if args.ros2:
                rclpy.spin_once(ros_node, timeout_sec=0.02)
                for _ in range(8):
                    rclpy.spin_once(ros_node, timeout_sec=0.0)
                update_gravity_from_latest_ros_imu(latest.get("info"))
                rgb_msg = latest.get("rgb")
                depth_msg = latest.get("depth")
                info_msg = latest.get("info")
                depth_scale_val = latest.get("depth_scale")
                if rgb_msg is None or depth_msg is None or info_msg is None:
                    missing = ", ".join(
                        name
                        for name, value in (
                            ("rgb", rgb_msg),
                            ("depth", depth_msg),
                            ("camera_info", info_msg),
                        )
                        if value is None
                    )
                    wait_now = time.perf_counter()
                    if wait_now - last_ros_wait_log_s >= 2.0:
                        counts = latest["message_counts"]
                        print(
                            "ROS2 waiting for camera data: "
                            f"missing={missing}; counts="
                            f"rgb={counts['rgb']},depth={counts['depth']},"
                            f"camera_info={counts['info']},depth_scale={counts['depth_scale']}",
                            flush=True,
                        )
                        last_ros_wait_log_s = wait_now
                    continue
                color_bgr = rgb_msg.copy()
                depth_raw = depth_msg.copy()
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                depth_scale = float(depth_scale_val) if depth_scale_val is not None else 0.001
                depth_m = depth_raw.astype(np.float32) * depth_scale
                depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
                class DummyIntrinsics:
                    pass

                intr = DummyIntrinsics()
                intr.fx = info_msg.k[0]
                intr.ppx = info_msg.k[2]
                intr.fy = info_msg.k[4]
                intr.ppy = info_msg.k[5]
                K = base.camera_matrix(intr)
                camera_frame_name = str(info_msg.header.frame_id).strip().lstrip("/")
                now = time.perf_counter()
            else:
                raw_frames = pipeline.wait_for_frames()
                if gravity_filter is not None:
                    gravity_snapshot = gravity_filter.update(
                        base.accelerometer_vector_in_color_frame(raw_frames, accel_to_color)
                    )
                frames = align_to_color.process(raw_frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth_raw = np.asanyarray(depth_frame.get_data())
                color_bgr = np.asanyarray(color_frame.get_data())
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                depth_m = depth_raw.astype(np.float32) * depth_scale
                depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                K = base.camera_matrix(intrinsics)
                camera_frame_name = "d435i_color_optical_frame"
                now = time.perf_counter()

            # During a GPU registration, keep acquiring/displaying live camera
            # frames but pause YOLO. This prevents the main inference call from
            # waiting behind FoundationPose kernels and avoids visible freezes.
            if not registration_worker.busy:
                observations, _ = base.find_targets(model, color_bgr, depth_raw, depth_scale, args)
                detections: list[Detection3D] = []
                for observation in observations:
                    depth_filter = base.filter_pose_depth(
                        depth_m,
                        observation.mask,
                        enabled=args.pose_depth_filter,
                        min_depth_m=args.min_depth,
                        max_depth_m=args.max_depth,
                        minimum_band_m=args.pose_depth_band,
                        maximum_band_m=args.pose_depth_max_band,
                        erode_kernel_size=args.pose_depth_erode,
                    )
                    measurement = base.make_rgbd_position_measurement(
                        K,
                        observation,
                        depth_filter,
                        center_depth_offset_m=center_depth_offset,
                        minimum_valid_depth_ratio=args.tracker_min_valid_depth_ratio,
                        tracker_config=tracker_config,
                    )
                    payload = InstancePayload(observation, depth_filter, measurement)
                    detections.append(
                        Detection3D(
                            box=observation.box,
                            confidence=observation.confidence,
                            position_m=None if measurement is None else measurement.position_m,
                            covariance=None if measurement is None else measurement.covariance,
                            payload=payload,
                        )
                    )
                track_views = tracks.update(detections, now)

            excluded_until = {
                track_id: deadline for track_id, deadline in excluded_until.items() if deadline > now
            }
            selected = selector.select(track_views, excluded_track_ids=set(excluded_until))
            new_selected_id = None if selected is None else selected.track_id
            if new_selected_id != selected_track_id:
                print(f"Selected track changed: {selected_track_id} -> {new_selected_id}")
                selected_track_id = new_selected_id
                selected_stable_since = None
                pose = None
                trusted_pose = None
                pose_quality = None
                pose_status = "stabilizing" if selected is not None else "searching"
                invalid_pose_frames = 0
                registration_anchor = None
                motion_since_registration = False
                registration_attempts = 0
                next_registration_s = now

            payload = (
                selected.payload
                if selected is not None and isinstance(selected.payload, InstancePayload)
                else None
            )
            position = None if selected is None else selected.tracker.position_m
            speed = (
                float("inf")
                if selected is None or selected.tracker.velocity_mps is None
                else float(np.linalg.norm(selected.tracker.velocity_mps))
            )
            stable_now = bool(
                selected is not None
                and payload is not None
                and position is not None
                and selected.confirmed
                and selected.stable_box_frames >= 3
                and selected.tracker.measurement_accepted is True
                and speed <= tracker_config.stationary_speed_mps
            )
            if stable_now:
                if selected_stable_since is None:
                    selected_stable_since = now
            else:
                selected_stable_since = None
            stable_seconds = 0.0 if selected_stable_since is None else now - selected_stable_since

            if args.upright_source == "imu":
                ros_imu_is_fresh = (
                    not args.ros2
                    or (
                        last_ros_imu_accepted_time is not None
                        and now - last_ros_imu_accepted_time <= args.imu_timeout
                    )
                )
                upright_axis = (
                    None
                    if gravity_snapshot is None or not ros_imu_is_fresh
                    else gravity_snapshot.axis_camera
                )
            elif args.upright_source == "reference":
                upright_axis = reference_upright_axis
            else:
                upright_axis = None

            if trusted_pose is not None and selected is not None and position is not None:
                pose = trusted_pose.copy()
                if args.pose_tracking_mode == "upright-follow" and upright_axis is not None:
                    pose = pose_with_upright_axis(pose, upright_axis, inverse_to_origin)
                pose = pose_with_center_position(pose, position, inverse_to_origin)
                if payload is not None:
                    upright_tilt = (
                        None
                        if upright_axis is None
                        else base.axis_tilt_degrees(pose[:3, 2], upright_axis)
                    )
                    pose_quality = base.assess_pose_projection(
                        K,
                        pose @ inverse_to_origin,
                        bbox,
                        payload.observation.box,
                        minimum_iou=args.min_pose_box_iou,
                        maximum_center_error_ratio=args.max_pose_center_error,
                        reference_position_m=position,
                        maximum_position_error_m=args.max_pose_3d_error,
                        upright_tilt_degrees=upright_tilt,
                        maximum_upright_tilt_degrees=args.max_upright_tilt,
                    )
                    invalid_pose_frames = 0 if pose_quality.valid else invalid_pose_frames + 1
                    if not registration_worker.busy:
                        pose_status = "pose_valid" if pose_quality.valid else "pose_degraded"
                elif not registration_worker.busy:
                    pose_quality = None
                    invalid_pose_frames += 1

                displacement = (
                    0.0
                    if registration_anchor is None
                    else float(np.linalg.norm(position - registration_anchor))
                )
                if speed >= args.pose_motion_speed or displacement >= args.pose_motion_distance:
                    motion_since_registration = True

                if invalid_pose_frames >= args.pose_lost_frames:
                    reason = "target missing" if pose_quality is None else pose_quality.reason
                    print(f"Track {selected.track_id} pose lost: {reason}; automatic recovery armed")
                    pose = None
                    trusted_pose = None
                    pose_status = "reacquiring"
                    invalid_pose_frames = 0
                    next_registration_s = max(next_registration_s, now)

            # Consume completed registrations only after current-frame YOLO has
            # updated the locked track; stale results are never applied blindly.
            for async_result in registration_worker.poll():
                request: RegistrationRequest = async_result.request
                if async_result.error is not None:
                    print(f"Registration worker failed for track {request.track_id}:\n{async_result.error}")
                    accepted = False
                    failure_reason = "worker exception"
                elif selected is None or selected.track_id != request.track_id:
                    print(f"Discarded stale registration for unselected track {request.track_id}")
                    continue
                elif payload is None or position is None:
                    accepted = False
                    failure_reason = "selected target not currently observed"
                else:
                    moved = float(np.linalg.norm(position - request.tracker_position_m))
                    current_iou = box_iou(payload.observation.box, request.box)
                    if moved > args.registration_stale_distance or current_iou < args.registration_stale_iou:
                        accepted = False
                        failure_reason = f"stale input: moved={moved:.3f}m IoU={current_iou:.3f}"
                    else:
                        candidate = np.asarray(async_result.value, dtype=np.float64).reshape(4, 4)
                        axis = upright_axis
                        request_axis_is_fresh = (
                            not args.ros2
                            or (
                                request.gravity_sample_s is not None
                                and now - request.gravity_sample_s <= args.imu_timeout
                            )
                        )
                        if (
                            axis is None
                            and request.gravity_axis is not None
                            and request_axis_is_fresh
                        ):
                            axis = request.gravity_axis
                        if args.snap_upright_registration and axis is not None:
                            candidate = pose_with_upright_axis(candidate, axis, inverse_to_origin)
                        candidate = pose_with_center_position(candidate, position, inverse_to_origin)
                        upright_tilt = (
                            None if axis is None else base.axis_tilt_degrees(candidate[:3, 2], axis)
                        )
                        quality = base.assess_pose_projection(
                            K,
                            candidate @ inverse_to_origin,
                            bbox,
                            payload.observation.box,
                            minimum_iou=args.min_pose_box_iou,
                            maximum_center_error_ratio=args.max_pose_center_error,
                            reference_position_m=position,
                            maximum_position_error_m=args.max_pose_3d_error,
                            upright_tilt_degrees=upright_tilt,
                            maximum_upright_tilt_degrees=args.max_upright_tilt,
                        )
                        imu_axis_ready = args.upright_source != "imu" or axis is not None
                        accepted = quality.valid and imu_axis_ready
                        failure_reason = (
                            quality.reason
                            if imu_axis_ready
                            else "IMU gravity axis unavailable or stale"
                        )
                if accepted:
                    trusted_pose = candidate.copy()
                    pose = candidate.copy()
                    pose_quality = quality
                    registration_anchor = position.copy()
                    motion_since_registration = False
                    registration_attempts = 0
                    invalid_pose_frames = 0
                    pose_status = "pose_valid"
                    last_pose_accepted_s = now
                    if args.upright_source == "reference" and reference_upright_axis is None:
                        reference_upright_axis = candidate[:3, 2].copy()
                    print(
                        f"Accepted async registration track={request.track_id} "
                        f"reason={request.reason} duration={async_result.duration_s:.3f}s "
                        f"IoU={quality.box_iou:.3f}"
                    )
                else:
                    registration_attempts += 1
                    retry_delay = args.registration_retry_base_seconds * 2 ** min(
                        max(0, registration_attempts - 1),
                        max(0, args.registration_max_attempts - 1),
                    )
                    next_registration_s = now + retry_delay
                    pose_status = f"registration_retry_{registration_attempts}"
                    print(
                        f"Rejected async registration track={request.track_id}: {failure_reason}; "
                        f"retry in {retry_delay:.1f}s"
                    )
                    if trusted_pose is None and registration_attempts >= args.registration_max_attempts:
                        excluded_until[request.track_id] = now + args.failed_track_exclusion_seconds
                        selector.release()
                        pose_status = "candidate_failed"
                    elif trusted_pose is not None and registration_attempts >= args.registration_max_attempts:
                        # A failed optional refresh must never discard an
                        # otherwise valid constrained pose or retry forever.
                        registration_attempts = 0
                        motion_since_registration = False
                        next_registration_s = now + args.failed_track_exclusion_seconds
                        pose_status = "pose_valid"

            registration_needed = bool(
                selected is not None
                and payload is not None
                and position is not None
                and args.autonomous_registration
                and (
                    trusted_pose is None
                    or (args.refresh_after_motion and motion_since_registration)
                )
            )
            cooldown_ready = (
                trusted_pose is None
                or now - last_registration_submit_s >= args.registration_cooldown_seconds
            )
            upright_ready = args.upright_source != "imu" or upright_axis is not None
            if (
                registration_needed
                and trusted_pose is None
                and stable_seconds >= args.registration_stable_seconds
                and not upright_ready
            ):
                if pose_status != "waiting_for_imu":
                    imu_label = "ROS2 IMU" if args.ros2 else "D435i IMU"
                    detail = ros_imu_status if args.ros2 else "no accepted acceleration sample"
                    print(f"Registration waiting for {imu_label}: {detail}")
                pose_status = "waiting_for_imu"
            if (
                registration_needed
                and not registration_worker.busy
                and stable_seconds >= args.registration_stable_seconds
                and now >= next_registration_s
                and cooldown_ready
                and upright_ready
            ):
                request_counter += 1
                reason = "recovery" if trusted_pose is None else "motion-refresh"
                request = RegistrationRequest(
                    request_id=request_counter,
                    track_id=selected.track_id,
                    reason=reason,
                    submitted_s=now,
                    K=K.copy(),
                    rgb=color_rgb.copy(),
                    depth_m=payload.depth_filter.depth_m.copy(),
                    mask=payload.observation.mask.copy(),
                    box=payload.observation.box.copy(),
                    gravity_axis=None if upright_axis is None else upright_axis.copy(),
                    gravity_sample_s=(
                        last_ros_imu_accepted_time if args.ros2 else None
                    ),
                    tracker_position_m=position.copy(),
                )
                if registration_worker.submit(request):
                    last_registration_submit_s = now
                    pose_status = f"registering_{reason}"
                    print(
                        f"Submitted async registration track={selected.track_id} "
                        f"reason={reason} stable={stable_seconds:.2f}s"
                    )

            display = annotate_tracks(color_bgr, K, track_views, selected_track_id)
            if pose is not None:
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                center_pose = pose @ inverse_to_origin
                display_rgb = draw_posed_3d_box(K, img=display_rgb, ob_in_cam=center_pose, bbox=bbox)
                display_rgb = draw_xyz_axis(
                    display_rgb,
                    ob_in_cam=center_pose,
                    scale=max(0.05, float(np.max(extents)) * 0.7),
                    K=K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                display = display_rgb[..., ::-1].copy()

            frame_end = time.perf_counter()
            instant_fps = 1.0 / max(frame_end - previous_time, 1e-9)
            previous_time = frame_end
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            selected_text = "none" if selected_track_id is None else str(selected_track_id)
            cv2.putText(
                display,
                f"AUTO class={model.names[args.target_class]} selected={selected_text} "
                f"state={pose_status} stable={stable_seconds:.1f}s {smoothed_fps:.1f}FPS",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if registration_worker.busy:
                cv2.putText(
                    display,
                    "BACKGROUND REGISTRATION - live preview, YOLO temporarily paused",
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            pose_age = None if last_pose_accepted_s is None else now - last_pose_accepted_s
            selected_target = None
            if selected is not None:
                selected_target = {
                    "track_id": selected.track_id,
                    "class_id": args.target_class,
                    "class_name": str(model.names[args.target_class]),
                    "confidence": selected.confidence,
                    "box_xyxy": [float(value) for value in selected.box],
                    "position_camera_m": (
                        None
                        if selected.tracker.position_m is None
                        else [float(value) for value in selected.tracker.position_m]
                    ),
                    "position_std_m": (
                        None
                        if selected.tracker.position_std_m is None
                        else [float(value) for value in selected.tracker.position_std_m]
                    ),
                    "distance_camera_m": selected.distance_m,
                    "track_status": selected.tracker.status,
                    "observed_this_frame": selected.observed_this_frame,
                    "depth_valid_ratio": (
                        None
                        if payload is None or payload.measurement is None
                        else payload.measurement.valid_depth_ratio
                    ),
                }
            ros_imu_age = (
                None
                if last_ros_imu_accepted_time is None
                else max(0.0, now - last_ros_imu_accepted_time)
            )
            ros_imu_fresh = bool(
                args.ros2
                and gravity_filter is not None
                and ros_imu_age is not None
                and ros_imu_age <= args.imu_timeout
            )
            gravity_is_valid = gravity_snapshot is not None and (
                not args.ros2 or ros_imu_fresh
            )
            imu_enabled = args.upright_source == "imu"
            imu_fresh = bool(imu_enabled and gravity_is_valid)
            state = {
                "schema_version": 1,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "camera_frame": camera_frame_name,
                "requested_class_id": args.target_class,
                "requested_class_name": str(model.names[args.target_class]),
                "selected_track_id": selected_track_id,
                "selected_target": selected_target,
                "status": pose_status,
                "registration_busy": registration_worker.busy,
                "selected_stable_seconds": stable_seconds,
                "pose_camera_object": None if pose is None else pose.tolist(),
                "pose_age_s": pose_age,
                "yaw_observable": False,
                "imu_enabled": imu_enabled,
                "imu_fresh": imu_fresh,
                "gravity": (
                    gravity_snapshot.metadata() if gravity_is_valid else None
                ),
                "ros2_imu_status": ros_imu_status if args.ros2 else None,
                "ros2_imu_age_s": ros_imu_age if args.ros2 else None,
                "ros2_imu_fresh": ros_imu_fresh if args.ros2 else None,
                "pose_quality": None if pose_quality is None else pose_quality.metadata(),
                "tracks": track_metadata(track_views, selected_track_id),
            }
            publisher.publish(state, now)

            if preview_auto_start_pending:
                preview_recorder.start(display.shape)
                preview_auto_start_pending = False
            preview_recorder.write(display, state)
            if args.display:
                cv2.imshow("D435i Autonomous Multi-Drink Vision", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = read_key_from_stdin(0.01)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                pose = None
                trusted_pose = None
                pose_quality = None
                invalid_pose_frames = 0
                registration_attempts = 0
                next_registration_s = now
                pose_status = "debug_recovery_requested"
                if (
                    selected is not None
                    and payload is not None
                    and position is not None
                    and not registration_worker.busy
                    and (args.upright_source != "imu" or upright_axis is not None)
                ):
                    reason = "recovery"
                    request_counter += 1
                    request = RegistrationRequest(
                        request_id=request_counter,
                        track_id=selected.track_id,
                        reason=reason,
                        submitted_s=now,
                        K=K.copy(),
                        rgb=color_rgb.copy(),
                        depth_m=payload.depth_filter.depth_m.copy(),
                        mask=payload.observation.mask.copy(),
                        box=payload.observation.box.copy(),
                        gravity_axis=None if upright_axis is None else upright_axis.copy(),
                        gravity_sample_s=(
                            last_ros_imu_accepted_time if args.ros2 else None
                        ),
                        tracker_position_m=position.copy(),
                    )
                    if registration_worker.submit(request):
                        last_registration_submit_s = now
                        pose_status = f"registering_{reason}"
                        print(
                            f"Forced async registration track={selected.track_id} "
                            f"reason={reason} stable={stable_seconds:.2f}s"
                        )
                    else:
                        print(
                            f"Forced registration skipped for track={selected.track_id}: "
                            "registration worker busy"
                        )
                else:
                    if args.upright_source == "imu" and upright_axis is None:
                        imu_label = "ROS2 IMU" if args.ros2 else "D435i IMU"
                        print(f"Forced registration skipped: waiting for {imu_label} ({ros_imu_status})")
                    else:
                        print("Forced registration skipped: no valid selected track or payload")
            elif key in (ord("v"), ord("V")):
                if preview_recorder.active:
                    preview_recorder.stop()
                else:
                    preview_recorder.start(display.shape)
        return 0
    finally:
        preview_recorder.stop()
        registration_worker.close(wait=True)
        if started and not args.ros2:
            pipeline.stop()
        if args.ros2 and ros_node is not None:
            try:
                import rclpy
                ros_helpers.clear()
                ros_node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
