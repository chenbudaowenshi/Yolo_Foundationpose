#!/usr/bin/env python3
"""Autonomous multi-instance drink perception for D435i and FoundationPose.

The requested class is tracked as multiple persistent RGB-D instances. The
nearest confirmed instance is locked, FoundationPose registration is scheduled
automatically after stability checks, and no runtime key press is required.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from autonomous_state import (VisionStatePublisher, ObservationHistory,
                              DepthSurfaceValidator, output_status)
from camera_contract import capture_stamp_ns
from imu_alignment import ImuAlignment, add_sdk_acceleration
from ros_rgbd_input import RosRGBDInput
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
    force_full: bool
    submitted_s: float
    K: np.ndarray
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    box: np.ndarray
    gravity_axis: np.ndarray | None
    imu_stamp_ns: int
    tracker_position_m: np.ndarray


def depth_to_meters(
    depth_raw: np.ndarray,
    depth_scale: float,
    *,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Convert one depth frame without retaining an avoidable temporary.

    The autonomous loop only needs metric depth when a detector result exists.
    Keeping this conversion in a small helper makes it possible to defer it
    until then, while preserving the exact clipping semantics used before.
    """
    depth_m = np.array(depth_raw, dtype=np.float32, copy=True, order="C")
    depth_m *= np.float32(depth_scale)
    invalid = (depth_m < min_depth) | (depth_m > max_depth)
    depth_m[invalid] = 0.0
    return depth_m


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
            "compatibility flag: IMU acceleration is required "
            "(equivalent to --upright-source imu)"
        ),
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=20,
        help="maximum YOLO instances per frame (default: 20; raise for crowded scenes)",
    )
    parser.add_argument("--track-confirm-hits", type=int, default=3)
    parser.add_argument("--track-max-missed", type=int, default=90)
    parser.add_argument("--track-association-distance", type=float, default=0.15)
    parser.add_argument("--selection-lost-grace", type=int, default=45)
    parser.add_argument("--selection-distance", choices=("euclidean", "z"), default="euclidean")
    parser.add_argument(
        "--registration-stable-seconds",
        type=float,
        default=0.5,
        help="seconds of stable target data before registration (default: 0.5)",
    )
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
    parser.add_argument("--max-observation-age-seconds", type=float, default=0.5)
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
        # Bottles are modeled upright.  Use the D435i gravity axis by default
        # so a registration cannot settle on a tilted rotational hypothesis.
        # A missing or invalid IMU blocks registration and valid output.
        upright_source="imu",
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
        args.max_observation_age_seconds,
        args.max_rgbd_skew_ms,
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
    if args.upright_source != "imu":
        raise ValueError("IMU is required for registration and valid pose output")
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
    publisher = VisionStatePublisher(args.vision_state_file, args.vision_output_rate)
    publisher.invalidate("starting")

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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_logging_format()
    set_seed(0)
    base.install_foundationpose_opencv_compat(predict_pose_refine, predict_score)

    model = YOLO(str(yolo_path))
    base.check_target_class(model, args)
    if args.yolo_warmup:
        # Build the Ultralytics predictor and autotune the fixed camera shape
        # before the first live frame. This moves one-time setup and CUDA
        # kernel selection out of the detection latency seen by the operator.
        warmup_image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        for _ in range(args.yolo_warmup):
            base.find_targets(
                model,
                warmup_image,
                np.zeros((args.height, args.width), dtype=np.uint16),
                0.001,
                args,
            )
    mesh = base.load_mesh(mesh_path, trimesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inverse_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack((-extents / 2.0, extents / 2.0), axis=0).reshape(2, 3)
    surface_validator = DepthSurfaceValidator(mesh.vertices, mesh.faces)
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
    imu_alignment = ImuAlignment(maximum_skew_ms=args.max_imu_skew_ms,
                                 timeout_s=args.imu_timeout, smoothing=args.gravity_smoothing)

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
        registration_views=args.register_views,
        registration_inplane_step=args.register_inplane_step,
        registration_topk=args.register_topk,
    )
    # FoundationPose emits detailed INFO messages inside every render/refine
    # iteration.  Keep them available for ``--debug >= 2`` (where visual
    # diagnostics are requested), but avoid formatting/locking log records in
    # the real-time hot path for normal operation.
    if args.debug < 2:
        logging.getLogger().setLevel(logging.WARNING)

    def register_pose(request: RegistrationRequest) -> np.ndarray:
        # ``estimator.register`` returns a NumPy pose, so its final device copy
        # already waits for the required CUDA work.  Avoid extra explicit
        # synchronizations around the hot path.
        return estimator.register(
            K=request.K,
            rgb=request.rgb,
            depth=request.depth_m,
            ob_mask=request.mask,
            iteration=args.est_refine_iter,
            force_full=request.force_full,
        )

    registration_worker = LatestPoseRegistrationWorker(register_pose)
    preview_recorder = base.PreviewVideoRecorder(record_dir, args.record_fps)
    preview_auto_start_pending = bool(args.record_preview)

    ros_node = None
    ros_tf_buffer = None
    ros_helpers = []
    depth_scale = None
    started = False
    gravity_snapshot = None
    accel_to_color = None
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
    force_full_registration_next = False
    next_registration_s = 0.0
    last_registration_submit_s = float("-inf")
    observation_history = ObservationHistory()
    camera_sequence = 0
    capture_stamp = None
    capture_semantics = "unavailable"
    surface_quality = None
    excluded_until: dict[int, float] = {}
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    last_imu_reason = None

    def current_imu():
        if args.ros2:
            return ros_rgbd.imu.match(ros_rgbd.last_frame)
        return imu_alignment.match(capture_stamp, camera_frame_name)

    try:
        if args.ros2:
            import rclpy
            rclpy.init(domain_id=args.ros_domain_id)
            ros_node = rclpy.create_node("d435i_autonomous_vision")
            ros_rgbd = RosRGBDInput(ros_node, args)
            for _ in range(max(1, args.warmup // 5)):
                rclpy.spin_once(ros_node, timeout_sec=0.05)
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

            # Export SDK epoch/exposure timestamps, never a new timestamp at inference end.
            resolved_device = config.resolve(rs.pipeline_wrapper(pipeline)).get_device()
            for sensor in resolved_device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
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
                add_sdk_acceleration(imu_alignment, frames, accel_to_color, "d435i_color_optical_frame")

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
            new_camera_frame = True
            if args.ros2:
                rclpy.spin_once(ros_node, timeout_sec=0.02)
                for _ in range(8):
                    rclpy.spin_once(ros_node, timeout_sec=0.0)
                frame = ros_rgbd.take_latest()
                if frame is None and ros_rgbd.last_frame is None:
                    if args.display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue
                new_camera_frame = frame is not None
                if new_camera_frame:
                    color_bgr, depth_raw = frame.color_bgr, frame.depth_m
                    depth_scale = 1.0  # receiver already normalized depth to metres
                    K = frame.K
                    camera_frame_name = frame.frame_id
                    capture_stamp = frame.stamp_ns
                    capture_semantics = "ros_image_header_capture_time"
                now = time.perf_counter()
            else:
                raw_frames = pipeline.wait_for_frames()
                add_sdk_acceleration(imu_alignment, raw_frames, accel_to_color, "d435i_color_optical_frame")
                frames = align_to_color.process(raw_frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth_raw = np.asanyarray(depth_frame.get_data())
                color_bgr = np.asanyarray(color_frame.get_data())
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                K = base.camera_matrix(intrinsics)
                camera_frame_name = "d435i_color_optical_frame"
                try:
                    stamps, bases = [], []
                    for source_frame in (raw_frames.get_color_frame(), raw_frames.get_depth_frame()):
                        metadata = {}
                        for name in ("frame_timestamp", "sensor_timestamp"):
                            key = getattr(rs.frame_metadata_value, name)
                            if source_frame.supports_frame_metadata(key):
                                metadata[name] = source_frame.get_frame_metadata(key)
                        stamp, basis = capture_stamp_ns(source_frame.get_timestamp(),
                                                       source_frame.get_frame_timestamp_domain(), metadata)
                        stamps.append(stamp)
                        bases.append(basis)
                    capture_stamp = stamps[0] if abs(stamps[0] - stamps[1]) <= args.max_rgbd_skew_ms * 1e6 else None
                    capture_semantics = "realsense_" + bases[0]
                except ValueError:
                    # SDK may still be switching from its hardware clock during warmup.
                    capture_stamp = None
                    capture_semantics = "unavailable"
                now = time.perf_counter()

            if new_camera_frame:
                camera_sequence += 1

            # These conversions are only needed when at least one target was
            # detected.  Avoiding them on empty frames saves two full-frame
            # passes at camera rate while retaining identical depth semantics.
            depth_m: np.ndarray | None = None
            # During a GPU registration, keep acquiring/displaying live camera
            # frames but pause YOLO. This prevents the main inference call from
            # waiting behind FoundationPose kernels and avoids visible freezes.
            detection_ran = bool(new_camera_frame and not registration_worker.busy)
            if detection_ran:
                observations, _ = base.find_targets(model, color_bgr, depth_raw, depth_scale, args)
                detections: list[Detection3D] = []
                if observations:
                    depth_m = depth_to_meters(
                        depth_raw,
                        depth_scale,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
                for observation in observations:
                    assert depth_m is not None
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
                force_full_registration_next = False
                next_registration_s = now
                observation_history.select(selected_track_id)
                surface_quality = None

            payload = (
                selected.payload
                if selected is not None and isinstance(selected.payload, InstancePayload)
                else None
            )
            position = None if selected is None else selected.tracker.position_m
            observed_this_frame = bool(detection_ran and selected is not None and selected.observed_this_frame)
            measured_this_frame = bool(observed_this_frame and payload is not None and
                                       payload.measurement is not None and
                                       selected.tracker.measurement_accepted is True)
            if measured_this_frame:
                observation_history.observe(stamp_ns=capture_stamp, sequence=camera_sequence,
                                            semantics=capture_semantics)
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

            imu_match = current_imu()
            gravity_snapshot = imu_match.gravity
            upright_axis = imu_match.axis_camera
            if imu_match.reason != last_imu_reason:
                print(f"IMU alignment: {imu_match.reason}; skew_ms={imu_match.skew_ms}; age_s={imu_match.age_s}", flush=True)
                if args.ros2 and not imu_match.ready:
                    print(f"IMU topic {args.imu_topic}: publishers={ros_node.count_publishers(args.imu_topic)}, "
                          f"received={ros_rgbd.imu.count}; topic existence alone is not live IMU data", flush=True)
                last_imu_reason = imu_match.reason

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
                        # Independent raw-depth checks are applied to the output below.
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
            # Completion can race with the busy check above. Keep the result queued
            # until YOLO has really refreshed the selected track for this frame.
            for async_result in (registration_worker.poll() if detection_ran else ()):
                request: RegistrationRequest = async_result.request
                if not imu_match.ready:
                    print(f"Discarded registration: aligned IMU unavailable ({imu_match.reason})")
                    pose_status = "waiting_for_imu"
                    continue  # An IMU outage is not a failed target and must not exclude it.
                retry_with_full_search = False
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
                            upright_tilt_degrees=upright_tilt,
                            maximum_upright_tilt_degrees=args.max_upright_tilt,
                        )
                        imu_axis_ready = imu_match.ready
                        accepted = quality.valid and imu_axis_ready
                        retry_with_full_search = bool(imu_axis_ready and not quality.valid)
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
                    force_full_registration_next = False
                    invalid_pose_frames = 0
                    pose_status = "pose_valid"
                    observation_history.registered_s = now
                    print(
                        f"Accepted async registration track={request.track_id} "
                        f"reason={request.reason} duration={async_result.duration_s:.3f}s "
                        f"IoU={quality.box_iou:.3f}"
                    )
                else:
                    registration_attempts += 1
                    # A fast coarse search is sufficient for most views.  If
                    # its candidate fails the geometric quality checks, the
                    # next attempt uses the original-density grid and all
                    # hypotheses so a difficult view gets the
                    # accuracy-preserving fallback. Operational/stale-input
                    # failures stay on the fast path.
                    force_full_registration_next = retry_with_full_search
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
            upright_ready = imu_match.ready
            if (
                registration_needed
                and trusted_pose is None
                and stable_seconds >= args.registration_stable_seconds
                and not upright_ready
            ):
                if pose_status != "waiting_for_imu":
                    imu_label = "ROS2 IMU" if args.ros2 else "D435i IMU"
                    detail = imu_match.reason
                    print(f"Registration waiting for {imu_label}: {detail}")
                    if args.ros2:
                        print(f"IMU is mandatory: verify a live publisher on {args.imu_topic} and matching capture time / optical frame.")
                pose_status = "waiting_for_imu"
            if (
                registration_needed
                and detection_ran
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
                    force_full=force_full_registration_next,
                    submitted_s=now,
                    K=K.copy(),
                    # RGB conversion is needed only by FoundationPose.  Keep
                    # it off the camera-rate hot path until a registration is
                    # actually submitted.
                    rgb=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
                    depth_m=payload.depth_filter.depth_m.copy(),
                    mask=payload.observation.mask.copy(),
                    box=payload.observation.box.copy(),
                    gravity_axis=None if upright_axis is None else upright_axis.copy(),
                    imu_stamp_ns=imu_match.imu_stamp_ns,
                    tracker_position_m=position.copy(),
                )
                if registration_worker.submit(request):
                    last_registration_submit_s = now
                    pose_status = f"registering_{reason}"
                    print(
                        f"Submitted async registration track={selected.track_id} "
                        f"reason={reason} stable={stable_seconds:.2f}s"
                    )

            # Validate output without changing selection, prediction or recovery rules.
            quality_metadata = None if pose_quality is None else pose_quality.metadata()
            if measured_this_frame and pose is not None and depth_m is not None:
                surface_quality = surface_validator.assess(K, depth_m, payload.observation.mask, pose)
            if quality_metadata is not None:
                raw_error = (None if not measured_this_frame or pose is None else
                             float(np.linalg.norm((pose @ inverse_to_origin)[:3, 3] - payload.measurement.position_m)))
                quality_metadata.update(position_error_m=raw_error,
                                        position_error_semantics="filtered_center_to_raw_rgbd_measurement",
                                        depth_surface=surface_quality)
                geometry_valid = bool(surface_quality and surface_quality["valid"] and
                                      raw_error is not None and raw_error <= args.max_pose_3d_error)
                quality_metadata["valid"] = bool(quality_metadata["valid"] and measured_this_frame and geometry_valid)
                if not quality_metadata["valid"] and quality_metadata["reason"] == "ok":
                    quality_metadata["reason"] = ("no current RGB-D measurement" if not measured_this_frame else
                                                  "raw RGB-D / CAD consistency failed")
                # JSON has no Infinity value; unavailable error metrics are null.
                quality_metadata = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                    for k, v in quality_metadata.items()}
            imu_match = current_imu()  # Recheck capture age after inference, before output.
            if quality_metadata is not None:
                quality_metadata["imu_aligned"] = imu_match.ready
                if not imu_match.ready:
                    quality_metadata["valid"] = False
                    quality_metadata["reason"] = imu_match.reason
            frame_end = time.perf_counter()
            time_metadata = observation_history.metadata(time.time_ns(), frame_end)
            # A missing/unsynchronized timestamp in this frame must not reuse an old one.
            current_measurement = measured_this_frame and capture_stamp is not None
            reported_status, observation_state, pose_valid = output_status(
                pose_status, has_pose=pose is not None, measured=current_measurement,
                observation_age_s=time_metadata["observation_age_s"],
                quality_valid=bool(quality_metadata and quality_metadata["valid"]),
                maximum_age_s=args.max_observation_age_seconds, imu_ready=imu_match.ready)
            instant_fps = 1.0 / max(frame_end - previous_time, 1e-9)
            previous_time = frame_end
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            # Rendering/annotation is independent of recognition.  In headless
            # deployments skip the full-resolution copies and OpenCV drawing;
            # this leaves GPU/CPU time for YOLO and the tracker.
            render_preview = bool(
                args.display or preview_recorder.active or preview_auto_start_pending
            )
            display = None
            if render_preview:
                display = annotate_tracks(color_bgr, K, track_views, selected_track_id)
                if pose is not None:
                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    center_pose = pose @ inverse_to_origin
                    display_rgb = draw_posed_3d_box(
                        K, img=display_rgb, ob_in_cam=center_pose, bbox=bbox
                    )
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
                selected_text = "none" if selected_track_id is None else str(selected_track_id)
                cv2.putText(
                    display,
                    f"AUTO class={model.names[args.target_class]} selected={selected_text} "
                    f"state={reported_status} stable={stable_seconds:.1f}s {smoothed_fps:.1f}FPS",
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
                    "observed_this_frame": observed_this_frame,
                    "measurement_accepted_this_frame": measured_this_frame,
                    "depth_valid_ratio": (
                        None
                        if payload is None or payload.measurement is None
                        else payload.measurement.valid_depth_ratio
                    ),
                }
            state = {
                "schema_version": 2,
                "session_id": publisher.session_id,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "camera_frame": camera_frame_name,
                "requested_class_id": args.target_class,
                "requested_class_name": str(model.names[args.target_class]),
                "selected_track_id": selected_track_id,
                "selected_target": selected_target,
                "status": reported_status,
                "processing_state": pose_status,
                "pose_valid": pose_valid,
                "observation_state": observation_state,
                "detection_ran": detection_ran,
                "camera_sequence": camera_sequence,
                "registration_busy": registration_worker.busy,
                "selected_stable_seconds": stable_seconds,
                "pose_camera_object": None if pose is None else pose.tolist(),
                **time_metadata,
                "yaw_observable": False,
                "imu_enabled": True,
                "imu_required": True,
                "imu_alignment": imu_match.metadata(),
                "imu_fresh": imu_match.ready,
                "gravity": (
                    imu_match.gravity.metadata() if imu_match.ready else None
                ),
                "ros2_imu_status": imu_match.reason if args.ros2 else None,
                "ros2_imu_age_s": imu_match.age_s if args.ros2 else None,
                "ros2_imu_fresh": imu_match.ready if args.ros2 else None,
                "pose_quality": quality_metadata,
                "tracks": [dict(item, observed_this_frame=detection_ran and item["observed_this_frame"])
                           for item in track_metadata(track_views, selected_track_id)],
            }
            publisher.publish(state, now)

            if preview_auto_start_pending:
                assert display is not None
                preview_recorder.start(display.shape)
                preview_auto_start_pending = False
            if display is not None:
                preview_recorder.write(display, state)
            if args.display:
                assert display is not None
                cv2.imshow("D435i Autonomous Multi-Drink Vision", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                # Headless mode should not spend up to 10 ms polling stdin on
                # every camera frame; terminal input remains available with a
                # non-blocking check.
                key = read_key_from_stdin(0.0)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                pose = None
                trusted_pose = None
                pose_quality = None
                observation_history.registered_s = None
                invalid_pose_frames = 0
                registration_attempts = 0
                next_registration_s = now
                pose_status = "debug_recovery_requested"
                if (
                    selected is not None
                    and payload is not None
                    and position is not None
                    and detection_ran
                    and not registration_worker.busy
                    and imu_match.ready
                ):
                    reason = "recovery"
                    request_counter += 1
                    request = RegistrationRequest(
                        request_id=request_counter,
                        track_id=selected.track_id,
                        reason=reason,
                        force_full=True,
                        submitted_s=now,
                        K=K.copy(),
                        rgb=cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB),
                        depth_m=payload.depth_filter.depth_m.copy(),
                        mask=payload.observation.mask.copy(),
                        box=payload.observation.box.copy(),
                        gravity_axis=None if upright_axis is None else upright_axis.copy(),
                        imu_stamp_ns=imu_match.imu_stamp_ns,
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
                        print(f"Forced registration skipped: waiting for {imu_label} ({imu_match.reason})")
                    else:
                        print("Forced registration skipped: no valid selected track or payload")
            elif key in (ord("v"), ord("V")):
                if preview_recorder.active:
                    preview_recorder.stop()
                else:
                    preview_recorder.start(color_bgr.shape)
        return 0
    finally:
        try:
            # Do this before waiting for a slow GPU job, including on Ctrl+C/errors.
            publisher.invalidate("stopped", reason="shutdown" if sys.exc_info()[0] is None else sys.exc_info()[0].__name__)
        finally:
            try:
                preview_recorder.stop()
            finally:
                try:
                    registration_worker.close(wait=True)
                finally:
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
