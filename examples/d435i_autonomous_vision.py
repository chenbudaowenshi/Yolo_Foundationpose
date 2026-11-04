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
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import d435i_foundationpose as base

import cv2
import numpy as np
import pyrealsense2 as rs

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


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
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

    try:
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
        print("No key press is required. R only forces a debug recovery; Q/Esc quits.")
        runtime_started_s = time.perf_counter()

        while True:
            if (
                args.max_runtime_seconds is not None
                and time.perf_counter() - runtime_started_s >= args.max_runtime_seconds
            ):
                print(f"Reached smoke-test runtime limit: {args.max_runtime_seconds:.1f}s")
                break
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
                upright_axis = None if gravity_snapshot is None else gravity_snapshot.axis_camera
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
                        axis = upright_axis if upright_axis is not None else request.gravity_axis
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
                        accepted = quality.valid
                        failure_reason = quality.reason
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
            state = {
                "schema_version": 1,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "camera_frame": "d435i_color_optical_frame",
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
                key = 255
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
            elif key in (ord("v"), ord("V")):
                if preview_recorder.active:
                    preview_recorder.stop()
                else:
                    preview_recorder.start(display.shape)
        return 0
    finally:
        preview_recorder.stop()
        registration_worker.close(wait=True)
        if started:
            pipeline.stop()
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
