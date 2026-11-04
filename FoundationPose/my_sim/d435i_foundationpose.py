#!/usr/bin/env python3
"""Real-time D435i + YOLO + FoundationPose model-based 6D pose demo.

The YOLO model may be either the existing detector or a YOLOv8-seg model.
For a detector, an aligned-depth/GrabCut mask is generated only for pose
registration. A segmentation checkpoint is used directly when available.
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


def ensure_fopose_python() -> None:
    """Re-execute with the verified fopose interpreter when Shell activation is stale."""
    if os.environ.get("D435I_FOPOSE_REEXEC") == "1":
        return
    expected = Path(
        os.environ.get("FOPOSE_PYTHON", "/home/hssw/miniconda3/envs/fopose/bin/python")
    ).expanduser()
    if not expected.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        target = expected.resolve()
    except OSError:
        return
    if current == target:
        return
    environment = os.environ.copy()
    environment["D435I_FOPOSE_REEXEC"] = "1"
    # PYTHONHOME can make a Conda interpreter search another environment's
    # standard library/site-packages. The repository itself is found from the
    # script directory and does not need PYTHONPATH here.
    environment.pop("PYTHONHOME", None)
    print(f"Switching Python interpreter: {current} -> {target}", flush=True)
    os.execve(str(target), [str(target), *sys.argv], environment)


ensure_fopose_python()

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Activate fopose and install numpy and opencv-python.\n"
        f"Original error: {exc}"
    ) from exc

# pyrealsense2 is optional when running against remote ROS2 camera topics.
try:
    import pyrealsense2 as rs
except Exception:
    rs = None

from depth_mask_utils import DepthMaskResult, depth_box_mask, overlay_mask
from rgbd_kalman_tracker import ConstantVelocity3DTracker, TrackerConfig, TrackerSnapshot
# from robot_grasp_controller import RobotGraspController
from upright_pose_utils import (
    GravityAxisFilter,
    GravitySnapshot,
    axis_tilt_degrees,
    pose_with_center_position,
    pose_with_upright_axis,
    relative_rotation_degrees,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_ROOT = Path("/home/robot/FoundationPose/")
DEFAULT_YOLO_MODEL = REPO_ROOT / "runs/yolov8/best.pt"
DEFAULT_MESH = REPO_ROOT / "runs/mesh/bottle.obj"
DEFAULT_OUTPUT = REPO_ROOT / "runs/foundationpose"
DEFAULT_RECORDINGS = DEFAULT_OUTPUT / "recordings"


@dataclass
class TargetObservation:
    class_id: int
    class_name: str
    confidence: float
    box: np.ndarray
    mask: np.ndarray
    mask_source: str
    depth_result: DepthMaskResult | None = None
    detection_index: int = -1


@dataclass(frozen=True)
class PoseDepthFilterResult:
    depth_m: np.ndarray
    enabled: bool
    mask_pixels: int
    valid_before: int
    valid_after: int
    removed_pixels: int
    reference_depth_m: float | None
    median_absolute_deviation_m: float | None
    effective_band_m: float | None

    def metadata(self) -> dict[str, bool | int | float | None]:
        return {
            "enabled": self.enabled,
            "mask_pixels": self.mask_pixels,
            "valid_before": self.valid_before,
            "valid_after": self.valid_after,
            "valid_ratio_before": self.valid_before / max(1, self.mask_pixels),
            "valid_ratio_after": self.valid_after / max(1, self.mask_pixels),
            "removed_pixels": self.removed_pixels,
            "reference_depth_m": self.reference_depth_m,
            "median_absolute_deviation_m": self.median_absolute_deviation_m,
            "effective_band_m": self.effective_band_m,
        }


@dataclass(frozen=True)
class RGBDPositionMeasurement:
    position_m: np.ndarray
    covariance: np.ndarray
    pixel_center: tuple[float, float]
    surface_depth_m: float
    center_depth_m: float
    center_depth_offset_m: float
    valid_depth_ratio: float
    depth_mad_m: float

    def metadata(self) -> dict[str, float | list[float]]:
        return {
            "position_m": [float(value) for value in self.position_m],
            "position_std_m": [float(value) for value in np.sqrt(np.diag(self.covariance))],
            "pixel_center_uv": [float(value) for value in self.pixel_center],
            "surface_depth_m": self.surface_depth_m,
            "center_depth_m": self.center_depth_m,
            "center_depth_offset_m": self.center_depth_offset_m,
            "valid_depth_ratio": self.valid_depth_ratio,
            "depth_mad_m": self.depth_mad_m,
        }


@dataclass(frozen=True)
class PoseProjectionQuality:
    valid: bool
    projected_box: np.ndarray | None
    box_iou: float
    center_error_ratio: float
    projected_to_observed_area_ratio: float
    position_error_m: float | None
    upright_tilt_degrees: float | None
    angular_speed_degrees_s: float | None
    reason: str

    def metadata(self) -> dict[str, bool | float | str | list[float] | None]:
        return {
            "valid": self.valid,
            "projected_box_xyxy": (
                None if self.projected_box is None else [float(value) for value in self.projected_box]
            ),
            "box_iou": self.box_iou,
            "center_error_ratio": self.center_error_ratio,
            "projected_to_observed_area_ratio": self.projected_to_observed_area_ratio,
            "position_error_m": self.position_error_m,
            "upright_tilt_degrees": self.upright_tilt_degrees,
            "angular_speed_degrees_s": self.angular_speed_degrees_s,
            "reason": self.reason,
        }


class PreviewVideoRecorder:
    """Record the annotated BGR preview and synchronized per-frame JSONL."""

    def __init__(self, output_dir: Path, fps: float) -> None:
        self.output_dir = output_dir
        self.fps = float(fps)
        self.writer: cv2.VideoWriter | None = None
        self.metadata_file = None
        self.video_path: Path | None = None
        self.metadata_path: Path | None = None
        self.frame_size: tuple[int, int] | None = None
        self.started_at: float | None = None
        self.frames_written = 0
        self.last_frame: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, frame_shape: tuple[int, ...]) -> Path:
        if self.active:
            assert self.video_path is not None
            return self.video_path
        height, width = frame_shape[:2]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.video_path = self.output_dir / f"{timestamp}_annotated.mp4"
        self.metadata_path = self.output_dir / f"{timestamp}_frames.jsonl"
        if self.video_path.exists() or self.metadata_path.exists():
            raise FileExistsError(f"recording output already exists for timestamp {timestamp}")
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise OSError("failed to open MP4 video writer; check OpenCV codec support")
        try:
            metadata_file = self.metadata_path.open("w", encoding="utf-8")
        except Exception:
            writer.release()
            raise
        self.writer = writer
        self.metadata_file = metadata_file
        self.frame_size = (width, height)
        self.started_at = time.perf_counter()
        self.frames_written = 0
        self.last_frame = None
        print(f"Preview recording started: {self.video_path}")
        print(f"Frame metadata: {self.metadata_path}")
        return self.video_path

    def write(self, frame_bgr: np.ndarray, metadata: dict[str, object]) -> None:
        if not self.active:
            return
        assert self.writer is not None
        assert self.metadata_file is not None
        assert self.frame_size is not None
        assert self.started_at is not None
        if frame_bgr.shape[1::-1] != self.frame_size:
            raise ValueError(
                f"recording frame size changed from {self.frame_size} to {frame_bgr.shape[1::-1]}"
            )

        now = time.perf_counter()
        elapsed = now - self.started_at
        target_total = max(1, int(np.floor(elapsed * self.fps)) + 1)
        frames_due = target_total - self.frames_written
        if frames_due > 0:
            if self.last_frame is not None:
                for _ in range(max(0, frames_due - 1)):
                    self.writer.write(self.last_frame)
                    self.frames_written += 1
            self.writer.write(frame_bgr)
            self.frames_written += 1
        self.last_frame = frame_bgr.copy()

        record = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": elapsed,
            "video_frame_index": max(0, self.frames_written - 1),
            **metadata,
        }
        self.metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stop(self) -> Path | None:
        if not self.active:
            return self.video_path
        assert self.writer is not None
        self.writer.release()
        self.writer = None
        if self.metadata_file is not None:
            self.metadata_file.close()
            self.metadata_file = None
        path = self.video_path
        print(f"Preview recording stopped: {path}; frames={self.frames_written}")
        self.last_frame = None
        return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH, help="meter-scale OBJ/PLY/STL mesh")
    parser.add_argument("--target-class", type=int, default=1, help="0=coca-cola, 1=fanta, 2=sprite for current dataset")
    parser.add_argument("--mask-source", choices=("auto", "seg", "depth"), default="auto")
    parser.add_argument("--foundationpose-root", type=Path, default=FOUNDATIONPOSE_ROOT)
    parser.add_argument("--serial", help="D435i serial number when multiple cameras are connected")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--quantize",
        type=str,
        choices=("8", "16", "32", "int8", "fp16", "fp32", "w8a8", "w16a16", "w8a16", "w8a32"),
        help="Use Ultralytics quantization mode for YOLO inference",
    )
    parser.add_argument("--auto-register", action="store_true", help="register as soon as a valid target mask is found")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--depth-tolerance", type=float, default=0.045)
    parser.add_argument("--max-depth-tolerance", type=float, default=0.12)
    parser.add_argument("--seed-ratio", type=float, default=0.35)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--pose-depth-band",
        type=float,
        default=0.055,
        help="minimum half-width in meters retained around robust object depth",
    )
    parser.add_argument(
        "--pose-depth-max-band",
        type=float,
        default=0.080,
        help="maximum adaptive half-width in meters for pose depth filtering",
    )
    parser.add_argument(
        "--pose-depth-erode",
        type=int,
        default=7,
        help="odd kernel size used to estimate depth away from the mask boundary",
    )
    parser.add_argument(
        "--no-pose-depth-filter",
        dest="pose_depth_filter",
        action="store_false",
        help="disable robust rejection of depth outliers inside the YOLO mask",
    )
    parser.add_argument(
        "--no-2d-overlay",
        dest="show_2d_overlay",
        action="store_false",
        help="do not keep running YOLO while tracking the 6D pose",
    )
    parser.add_argument(
        "--min-pose-box-iou",
        type=float,
        default=0.40,
        help="minimum IoU between projected CAD box and YOLO box",
    )
    parser.add_argument(
        "--max-pose-center-error",
        type=float,
        default=0.20,
        help="maximum CAD/YOLO center error divided by YOLO box diagonal",
    )
    parser.add_argument(
        "--pose-lost-frames",
        type=int,
        default=15,
        help="consecutive invalid tracking frames before declaring pose loss",
    )
    parser.add_argument(
        "--reregister-stable-frames",
        type=int,
        default=4,
        help="stable YOLO frames required before automatic re-registration",
    )
    parser.add_argument(
        "--reregister-box-motion",
        type=float,
        default=0.025,
        help="maximum normalized box motion considered stable",
    )
    parser.add_argument(
        "--auto-reregister",
        dest="auto_reregister",
        action="store_true",
        help="automatically run blocking global registration after pose loss",
    )
    parser.add_argument(
        "--no-auto-reregister",
        dest="auto_reregister",
        action="store_false",
        help="wait for R instead of automatically registering after pose loss (default)",
    )
    parser.add_argument(
        "--no-3d-tracker",
        dest="tracker_3d",
        action="store_false",
        help="disable RGB-D constant-velocity Kalman tracking",
    )
    parser.add_argument("--tracker-accel-std", type=float, default=0.10, help="motion noise in m/s^2")
    parser.add_argument("--tracker-gate", type=float, default=16.27, help="3D Mahalanobis outlier gate")
    parser.add_argument("--tracker-max-missed", type=int, default=8, help="frames predicted without measurement")
    parser.add_argument(
        "--tracker-stationary-speed",
        type=float,
        default=0.030,
        help="speed below which a target is stationary, in m/s",
    )
    parser.add_argument("--tracker-stationary-frames", type=int, default=5)
    parser.add_argument("--tracker-measurement-xy-std", type=float, default=0.008, help="base XY noise in meters")
    parser.add_argument("--tracker-measurement-z-std", type=float, default=0.015, help="base Z noise in meters")
    parser.add_argument(
        "--tracker-min-valid-depth-ratio",
        type=float,
        default=0.20,
        help="minimum usable depth fraction inside the YOLO mask",
    )
    parser.add_argument(
        "--tracker-center-depth-offset",
        type=float,
        help="front-surface to object-center offset in meters; default uses the CAD radius",
    )
    parser.add_argument(
        "--max-pose-3d-error",
        type=float,
        default=0.10,
        help="maximum distance between CAD center and filtered RGB-D track",
    )
    parser.add_argument(
        "--pose-tracking-mode",
        choices=("upright-follow", "upright-hold", "foundationpose"),
        default="upright-follow",
        help="follow gravity, hold registered rotation, or use FoundationPose track_one",
    )
    parser.add_argument(
        "--upright-source",
        choices=("imu", "reference", "off"),
        default="imu",
        help="expected bottle axis from local/ROS2 acceleration, first valid pose, or disabled",
    )
    parser.add_argument(
        "--max-upright-tilt",
        type=float,
        default=15.0,
        help="maximum angle in degrees between CAD Z axis and the gravity axis",
    )
    parser.add_argument("--gravity-smoothing", type=float, default=0.10)
    parser.add_argument(
        "--accel-fps",
        type=int,
        default=100,
        help="D435i accelerometer rate; this camera reports 100/200/400 Hz",
    )
    parser.add_argument(
        "--pose-motion-speed",
        type=float,
        default=0.040,
        help="KF3D speed in m/s that marks motion since registration",
    )
    parser.add_argument(
        "--pose-motion-distance",
        type=float,
        default=0.020,
        help="KF3D displacement in meters that marks motion since registration",
    )
    parser.add_argument(
        "--max-pose-angular-speed",
        type=float,
        default=90.0,
        help="maximum track_one angular speed in degrees/s",
    )
    parser.add_argument(
        "--no-snap-upright-registration",
        dest="snap_upright_registration",
        action="store_false",
        help="do not align a newly registered CAD Z axis to the known upright axis",
    )
    parser.add_argument(
        "--reregister-after-motion",
        dest="reregister_after_motion",
        action="store_true",
        help="run blocking global registration after a moved target becomes stationary",
    )
    parser.add_argument(
        "--no-reregister-after-motion",
        dest="reregister_after_motion",
        action="store_false",
        help="keep the constrained pose after motion instead of globally registering (default)",
    )
    parser.set_defaults(
        pose_depth_filter=True,
        show_2d_overlay=True,
        auto_reregister=False,
        tracker_3d=True,
        reregister_after_motion=False,
        snap_upright_registration=True,
    )
    parser.add_argument("--min-mask-pixels", type=int, default=300)
    parser.add_argument("--grabcut-iterations", type=int, default=3)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--record-preview",
        action="store_true",
        help="start annotated MP4 recording automatically; V toggles at runtime",
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument(
        "--record-fps",
        type=float,
        default=12.0,
        help="annotated MP4 playback frame rate",
    )
    parser.add_argument(
        "--record-bag",
        type=Path,
        help="optional RealSense .bag path containing the raw color/depth streams",
    )
    parser.add_argument(
        "--enable-grasp",
        action="store_true",
        help="solve IK for each accepted pose and publish a grasp command",
    )
    parser.add_argument("--grasp-side", choices=("left", "right"), default="left")
    parser.add_argument("--grasp-approach-offset", type=float, default=0.12)
    parser.add_argument("--grasp-offset", type=float, default=0.03)
    parser.add_argument("--trust-checkpoints", action="store_true")
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Subscribe to remote ROS2 camera topics instead of a local RealSense device",
    )
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw", help="ROS2 RGB image topic")
    parser.add_argument("--depth-topic", default="/camera/depth/image_rect_raw", help="ROS2 depth image topic")
    parser.add_argument("--camera-info-topic", default="/camera/camera_info", help="ROS2 CameraInfo topic")
    parser.add_argument("--depth-scale-topic", default="/camera/depth_scale", help="ROS2 depth scale (std_msgs/Float32) topic")
    parser.add_argument(
        "--imu-topic",
        default="/camera/imu",
        help="ROS2 IMU topic (sensor_msgs/Imu); linear acceleration supplies gravity",
    )
    parser.add_argument(
        "--imu-frame",
        help="override the ROS2 IMU source frame; required when Imu.header.frame_id is empty",
    )
    parser.add_argument(
        "--imu-timeout",
        type=float,
        default=0.5,
        help="maximum age in seconds of the last accepted ROS2 IMU sample",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.trust_checkpoints:
        parser.error("confirm the YOLO checkpoint is trusted, then add --trust-checkpoints")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.imgsz <= 0 or args.accel_fps <= 0:
        parser.error("--width, --height, --fps, --imgsz and --accel-fps must be positive")
    if args.warmup < 0 or args.est_refine_iter <= 0 or args.track_refine_iter <= 0:
        parser.error("warmup must be nonnegative and refinement iterations must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf and --iou must be in [0, 1]")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        parser.error("require 0 <= --min-depth < --max-depth")
    if args.pose_depth_band <= 0 or args.pose_depth_max_band < args.pose_depth_band:
        parser.error("require 0 < --pose-depth-band <= --pose-depth-max-band")
    if args.pose_depth_erode < 0 or (args.pose_depth_erode > 0 and args.pose_depth_erode % 2 == 0):
        parser.error("--pose-depth-erode must be zero or a positive odd integer")
    if not 0.0 <= args.min_pose_box_iou <= 1.0:
        parser.error("--min-pose-box-iou must be in [0, 1]")
    if args.max_pose_center_error < 0 or args.reregister_box_motion < 0:
        parser.error("pose center error and box motion thresholds must be nonnegative")
    if args.pose_lost_frames <= 0 or args.reregister_stable_frames <= 0:
        parser.error("pose lost/stable frame counts must be positive")
    tracker_positive = (
        args.tracker_accel_std,
        args.tracker_gate,
        args.tracker_stationary_speed,
        args.tracker_measurement_xy_std,
        args.tracker_measurement_z_std,
        args.max_pose_3d_error,
        args.max_upright_tilt,
        args.gravity_smoothing,
        args.pose_motion_speed,
        args.pose_motion_distance,
        args.max_pose_angular_speed,
    )
    if any(not np.isfinite(value) or value <= 0 for value in tracker_positive):
        parser.error("3D tracker noise, gate, speed, and pose error values must be positive")
    if args.tracker_max_missed < 0 or args.tracker_stationary_frames <= 0:
        parser.error("tracker missed frames must be nonnegative and stationary frames must be positive")
    if not 0.0 <= args.tracker_min_valid_depth_ratio <= 1.0:
        parser.error("--tracker-min-valid-depth-ratio must be in [0, 1]")
    if args.tracker_center_depth_offset is not None and args.tracker_center_depth_offset < 0:
        parser.error("--tracker-center-depth-offset must be nonnegative")
    if args.gravity_smoothing > 1.0:
        parser.error("--gravity-smoothing must be in (0, 1]")
    if not np.isfinite(args.imu_timeout) or args.imu_timeout <= 0:
        parser.error("--imu-timeout must be positive and finite")
    if args.imu_frame is not None and not args.imu_frame.strip().lstrip("/"):
        parser.error("--imu-frame must name a nonempty TF frame")
    if not np.isfinite(args.record_fps) or args.record_fps <= 0:
        parser.error("--record-fps must be positive and finite")
    if args.quantize is not None and str(args.device).lower() == "cpu":
        parser.error("--quantize requires CUDA")
    if args.pose_tracking_mode in ("upright-follow", "upright-hold") and not args.tracker_3d:
        parser.error("upright-follow and upright-hold require the 3D tracker")


def load_mesh(path: Path, trimesh_module):
    loaded = trimesh_module.load(str(path), process=False)
    if isinstance(loaded, trimesh_module.Scene):
        geometries = tuple(geometry for geometry in loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Mesh scene contains no geometry: {path}")
        loaded = trimesh_module.util.concatenate(geometries)
    if not isinstance(loaded, trimesh_module.Trimesh) or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Expected a nonempty triangle mesh: {path}")
    return loaded


def camera_matrix(intrinsics) -> np.ndarray:
    return np.array(
        [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def accelerometer_vector_in_color_frame(frameset, accel_to_color) -> np.ndarray | None:
    """Read one D435i acceleration sample and rotate it into color-camera axes."""
    if accel_to_color is None:
        return None
    try:
        accel_frame = frameset.first_or_default(rs.stream.accel)
        if not accel_frame:
            return None
        motion = accel_frame.as_motion_frame().get_motion_data()
        vector = [float(motion.x), float(motion.y), float(motion.z)]
        origin = np.asarray(
            rs.rs2_transform_point_to_point(accel_to_color, [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        tip = np.asarray(
            rs.rs2_transform_point_to_point(accel_to_color, vector),
            dtype=np.float64,
        )
        return tip - origin
    except RuntimeError:
        return None


def rotate_vector_by_quaternion_xyzw(
    vector: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Rotate a 3D vector by a ROS-order (x, y, z, w) quaternion."""
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("vector must contain three finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm < 1e-12:
        raise ValueError("quaternion must be nonzero")
    x, y, z, w = quaternion / quaternion_norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ value


def realsense_motion_xyz_format():
    """Return the motion-vector format name used by the installed SDK."""
    # librealsense renamed this enum in its Python bindings. 2.58 exposes
    # motion_xyz32f, while some older wheels expose motion_xyz.
    for name in ("motion_xyz32f", "motion_xyz"):
        value = getattr(rs.format, name, None)
        if value is not None:
            return value
    raise RuntimeError(
        "this pyrealsense2 build has no floating-point motion stream format "
        "(expected motion_xyz32f or motion_xyz)"
    )


def rotation_matrix_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return roll/pitch/yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    horizontal = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if horizontal > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def projected_bbox_xyxy(K: np.ndarray, center_pose: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Project the eight corners of a centered 3D bounding box."""
    corners = np.array(
        [
            [x, y, z]
            for x in (bbox[0, 0], bbox[1, 0])
            for y in (bbox[0, 1], bbox[1, 1])
            for z in (bbox[0, 2], bbox[1, 2])
        ],
        dtype=np.float64,
    )
    points_camera = (center_pose[:3, :3] @ corners.T).T + center_pose[:3, 3]
    if not np.all(np.isfinite(points_camera)) or np.any(points_camera[:, 2] <= 0.01):
        return None
    projected = (K @ points_camera.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(uv)):
        return None
    return np.concatenate((uv.min(axis=0), uv.max(axis=0)))


def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    intersection_min = np.maximum(first[:2], second[:2])
    intersection_max = np.minimum(first[2:], second[2:])
    intersection_size = np.maximum(0.0, intersection_max - intersection_min)
    intersection = float(np.prod(intersection_size))
    first_area = float(np.prod(np.maximum(0.0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0.0, second[2:] - second[:2])))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def assess_pose_projection(
    K: np.ndarray,
    center_pose: np.ndarray,
    bbox: np.ndarray,
    observed_box: np.ndarray,
    *,
    minimum_iou: float,
    maximum_center_error_ratio: float,
    reference_position_m: np.ndarray | None = None,
    maximum_position_error_m: float | None = None,
    upright_tilt_degrees: float | None = None,
    maximum_upright_tilt_degrees: float | None = None,
    angular_speed_degrees_s: float | None = None,
    maximum_angular_speed_degrees_s: float | None = None,
) -> PoseProjectionQuality:
    projected_box = projected_bbox_xyxy(K, center_pose, bbox)
    if projected_box is None:
        return PoseProjectionQuality(
            False,
            None,
            0.0,
            float("inf"),
            float("inf"),
            None,
            upright_tilt_degrees,
            angular_speed_degrees_s,
            "invalid projection",
        )

    observed_box = np.asarray(observed_box, dtype=np.float64)
    observed_size = np.maximum(0.0, observed_box[2:] - observed_box[:2])
    projected_size = np.maximum(0.0, projected_box[2:] - projected_box[:2])
    observed_area = float(np.prod(observed_size))
    projected_area = float(np.prod(projected_size))
    observed_center = (observed_box[:2] + observed_box[2:]) * 0.5
    projected_center = (projected_box[:2] + projected_box[2:]) * 0.5
    center_error_ratio = float(
        np.linalg.norm(projected_center - observed_center) / max(np.linalg.norm(observed_size), 1.0)
    )
    iou = box_iou_xyxy(projected_box, observed_box)
    area_ratio = projected_area / max(observed_area, 1.0)

    failures = []
    if iou < minimum_iou:
        failures.append(f"IoU {iou:.3f} < {minimum_iou:.3f}")
    if center_error_ratio > maximum_center_error_ratio:
        failures.append(f"center error {center_error_ratio:.3f} > {maximum_center_error_ratio:.3f}")
    position_error_m = None
    if reference_position_m is not None:
        reference_position_m = np.asarray(reference_position_m, dtype=np.float64).reshape(3)
        position_error_m = float(np.linalg.norm(center_pose[:3, 3] - reference_position_m))
        if maximum_position_error_m is not None and position_error_m > maximum_position_error_m:
            failures.append(f"3D error {position_error_m:.3f}m > {maximum_position_error_m:.3f}m")
    if (
        upright_tilt_degrees is not None
        and maximum_upright_tilt_degrees is not None
        and upright_tilt_degrees > maximum_upright_tilt_degrees
    ):
        failures.append(
            f"upright tilt {upright_tilt_degrees:.1f}deg > {maximum_upright_tilt_degrees:.1f}deg"
        )
    if (
        angular_speed_degrees_s is not None
        and maximum_angular_speed_degrees_s is not None
        and angular_speed_degrees_s > maximum_angular_speed_degrees_s
    ):
        failures.append(
            f"angular speed {angular_speed_degrees_s:.1f}deg/s > {maximum_angular_speed_degrees_s:.1f}deg/s"
        )
    return PoseProjectionQuality(
        valid=not failures,
        projected_box=projected_box,
        box_iou=iou,
        center_error_ratio=center_error_ratio,
        projected_to_observed_area_ratio=area_ratio,
        position_error_m=position_error_m,
        upright_tilt_degrees=upright_tilt_degrees,
        angular_speed_degrees_s=angular_speed_degrees_s,
        reason="ok" if not failures else "; ".join(failures),
    )


def detection_box_motion_ratio(current_box: np.ndarray, previous_box: np.ndarray | None) -> float:
    """Measure target translation/scale change relative to the previous box."""
    if previous_box is None:
        return float("inf")
    current_box = np.asarray(current_box, dtype=np.float64)
    previous_box = np.asarray(previous_box, dtype=np.float64)
    current_size = np.maximum(1.0, current_box[2:] - current_box[:2])
    previous_size = np.maximum(1.0, previous_box[2:] - previous_box[:2])
    current_center = (current_box[:2] + current_box[2:]) * 0.5
    previous_center = (previous_box[:2] + previous_box[2:]) * 0.5
    center_motion = float(
        np.linalg.norm(current_center - previous_center) / max(np.linalg.norm(previous_size), 1.0)
    )
    scale_motion = float(np.max(np.abs(current_size - previous_size) / previous_size))
    return max(center_motion, scale_motion)


def install_foundationpose_opencv_compat(*visualization_modules) -> None:
    """Allow FoundationPose debug text rendering on OpenCV 5.

    FoundationPose creates some debug mosaics as float arrays. OpenCV 5 only
    accepts uint8 images in ``putText`` and otherwise aborts pose registration
    after scoring has already finished.
    """
    for module in visualization_modules:
        original = module.cv_draw_text
        if getattr(original, "_foundationpose_uint8_compat", False):
            continue

        def cv_draw_text_uint8(image, *args, _original=original, **kwargs):
            if image.dtype != np.uint8:
                image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
            return _original(image, *args, **kwargs)

        cv_draw_text_uint8._foundationpose_uint8_compat = True
        module.cv_draw_text = cv_draw_text_uint8


def resize_segmentation_mask(mask_data, target_shape: tuple[int, int]) -> np.ndarray:
    mask = mask_data.detach().float().cpu().numpy()
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def filter_pose_depth(
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    enabled: bool,
    min_depth_m: float,
    max_depth_m: float,
    minimum_band_m: float,
    maximum_band_m: float,
    erode_kernel_size: int,
) -> PoseDepthFilterResult:
    """Reject isolated foreground/background depth values inside an object mask.

    The RGB segmentation mask is intentionally left unchanged. Only valid depth
    samples far away from the bottle's robust median surface are set to zero.
    Pixels outside the object mask remain untouched because FoundationPose uses
    the surrounding scene while refining and tracking.
    """
    if depth_m.ndim != 2 or object_mask.shape != depth_m.shape:
        raise ValueError("depth and object mask must have the same HxW shape")

    mask = object_mask.astype(bool)
    filtered = depth_m.astype(np.float32, copy=True)
    globally_valid = np.isfinite(filtered) & (filtered >= min_depth_m) & (filtered <= max_depth_m)
    filtered[~globally_valid] = 0.0
    valid_before_mask = mask & globally_valid
    mask_pixels = int(np.count_nonzero(mask))
    valid_before = int(np.count_nonzero(valid_before_mask))

    if not enabled or valid_before == 0:
        return PoseDepthFilterResult(
            depth_m=filtered,
            enabled=enabled,
            mask_pixels=mask_pixels,
            valid_before=valid_before,
            valid_after=valid_before,
            removed_pixels=0,
            reference_depth_m=None,
            median_absolute_deviation_m=None,
            effective_band_m=None,
        )

    reference_mask = mask
    if erode_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_kernel_size, erode_kernel_size),
        )
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        if np.count_nonzero(eroded & globally_valid) >= 20:
            reference_mask = eroded

    reference_values = filtered[reference_mask & globally_valid]
    reference_depth = float(np.median(reference_values))
    mad = float(np.median(np.abs(reference_values - reference_depth)))
    robust_sigma = 1.4826 * mad
    effective_band = min(maximum_band_m, max(minimum_band_m, 4.0 * robust_sigma))

    rejected = valid_before_mask & (np.abs(filtered - reference_depth) > effective_band)
    filtered[rejected] = 0.0
    removed_pixels = int(np.count_nonzero(rejected))
    valid_after = valid_before - removed_pixels
    return PoseDepthFilterResult(
        depth_m=filtered,
        enabled=True,
        mask_pixels=mask_pixels,
        valid_before=valid_before,
        valid_after=valid_after,
        removed_pixels=removed_pixels,
        reference_depth_m=reference_depth,
        median_absolute_deviation_m=mad,
        effective_band_m=effective_band,
    )


def make_rgbd_position_measurement(
    K: np.ndarray,
    observation: TargetObservation,
    depth_filter: PoseDepthFilterResult,
    *,
    center_depth_offset_m: float,
    minimum_valid_depth_ratio: float,
    tracker_config: TrackerConfig,
) -> RGBDPositionMeasurement | None:
    """Convert one YOLO mask and aligned depth map into a noisy 3D center."""
    valid = observation.mask & (depth_filter.depth_m > 0.0) & np.isfinite(depth_filter.depth_m)
    valid_pixels = int(np.count_nonzero(valid))
    valid_ratio = valid_pixels / max(1, depth_filter.mask_pixels)
    if valid_pixels < 20 or valid_ratio < minimum_valid_depth_ratio:
        return None

    depths = depth_filter.depth_m[valid].astype(np.float64)
    surface_depth = float(np.median(depths))
    mad = float(np.median(np.abs(depths - surface_depth)))
    center_depth = surface_depth + center_depth_offset_m
    x1, y1, x2, y2 = (float(value) for value in observation.box)
    pixel_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    ray = np.linalg.solve(K, np.array([pixel_center[0], pixel_center[1], 1.0], dtype=np.float64))
    position = ray * center_depth

    quality_scale = np.sqrt(1.0 + 3.0 * max(0.0, 1.0 - valid_ratio))
    robust_depth_std = 2.5 * 1.4826 * mad
    sigma_z = min(0.080, max(tracker_config.measurement_z_std_m, robust_depth_std) * quality_scale)
    pixel_to_meter = center_depth / max(float(K[0, 0]), float(K[1, 1]), 1.0)
    sigma_xy = max(tracker_config.measurement_xy_std_m, 1.5 * pixel_to_meter) * quality_scale
    covariance = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])
    return RGBDPositionMeasurement(
        position_m=position,
        covariance=covariance,
        pixel_center=pixel_center,
        surface_depth_m=surface_depth,
        center_depth_m=center_depth,
        center_depth_offset_m=center_depth_offset_m,
        valid_depth_ratio=valid_ratio,
        depth_mad_m=mad,
    )


def find_targets(
    model,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    args: argparse.Namespace,
) -> tuple[list[TargetObservation], object]:
    """Return every valid instance of the requested class in one frame."""
    result = model.predict(
        color_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        quantize=args.quantize,
        max_det=getattr(args, "max_det", 100),
        retina_masks=True,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], result

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    candidates = np.flatnonzero(class_ids == args.target_class)
    if candidates.size == 0:
        return [], result

    observations: list[TargetObservation] = []
    for selected_value in candidates:
        selected = int(selected_value)
        box = boxes[selected]
        mask = None
        mask_source = ""
        if (
            args.mask_source in ("auto", "seg")
            and result.masks is not None
            and selected < len(result.masks)
        ):
            mask = resize_segmentation_mask(result.masks.data[selected], depth_raw.shape)
            # A valid segmentation may legitimately have sparse/absent depth
            # on transparent bottles. Depth quality is assessed separately by
            # the RGB-D measurement stage, so do not discard the 2D instance.
            if np.count_nonzero(mask) >= args.min_mask_pixels:
                mask_source = "yolo-seg"
            else:
                mask = None
        if args.mask_source == "seg" and mask is None:
            continue

        depth_result = None
        if mask is None:
            depth_result = depth_box_mask(
                depth_raw,
                box,
                depth_scale=depth_scale,
                color_bgr=color_bgr,
                seed_ratio=args.seed_ratio,
                depth_tolerance_m=args.depth_tolerance,
                max_tolerance_m=args.max_depth_tolerance,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
                min_mask_pixels=args.min_mask_pixels,
                grabcut_iterations=args.grabcut_iterations,
            )
            if not depth_result.valid:
                continue
            mask = depth_result.mask
            mask_source = "aligned-depth+GrabCut"

        observations.append(
            TargetObservation(
                class_id=int(class_ids[selected]),
                class_name=str(model.names[int(class_ids[selected])]),
                confidence=float(confidences[selected]),
                box=box,
                mask=mask,
                mask_source=mask_source,
                depth_result=depth_result,
                detection_index=selected,
            )
        )
    return observations, result


def find_target(
    model,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    args: argparse.Namespace,
) -> tuple[TargetObservation | None, object]:
    """Backward-compatible single-instance selection by confidence."""
    observations, result = find_targets(model, color_bgr, depth_raw, depth_scale, args)
    if not observations:
        return None, result
    return max(observations, key=lambda item: item.confidence), result


def annotate_target(image: np.ndarray, observation: TargetObservation | None) -> np.ndarray:
    output = image.copy()
    if observation is None:
        return output
    # Magenta is reserved for YOLO so it remains distinguishable from the
    # green CAD projection drawn by FoundationPose.
    color = (255, 0, 255)
    output = overlay_mask(output, observation.mask, color)
    x1, y1, x2, y2 = (int(value) for value in observation.box)
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    text = f"{observation.class_name} {observation.confidence:.2f} | {observation.mask_source}"
    cv2.putText(output, text, (max(0, x1), max(22, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    return output


def annotate_3d_tracker(
    image: np.ndarray,
    K: np.ndarray,
    tracker_snapshot: TrackerSnapshot | None,
) -> np.ndarray:
    """Draw the filtered 3D center as a yellow image-space marker."""
    if tracker_snapshot is None or tracker_snapshot.position_m is None or tracker_snapshot.status == "lost":
        return image
    position = tracker_snapshot.position_m
    if not np.all(np.isfinite(position)) or position[2] <= 0.01:
        return image
    projected = K @ position
    uv = np.rint(projected[:2] / projected[2]).astype(int)
    height, width = image.shape[:2]
    if 0 <= uv[0] < width and 0 <= uv[1] < height:
        color = (0, 255, 255)
        cv2.circle(image, tuple(uv), 8, color, 2, cv2.LINE_AA)
        cv2.drawMarker(image, tuple(uv), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return image


def frame_record_metadata(
    observation: TargetObservation | None,
    pose: np.ndarray | None,
    pose_quality: PoseProjectionQuality | None,
    tracker_snapshot: TrackerSnapshot | None,
    rgbd_measurement: RGBDPositionMeasurement | None,
    gravity_snapshot: GravitySnapshot | None,
    pose_tracking_mode: str,
    motion_since_registration: bool,
) -> dict[str, object]:
    return {
        "observation": (
            None
            if observation is None
            else {
                "class_id": observation.class_id,
                "class_name": observation.class_name,
                "confidence": observation.confidence,
                "box_xyxy": [float(value) for value in observation.box],
                "mask_pixels": int(np.count_nonzero(observation.mask)),
            }
        ),
        "pose_ob_in_cam": None if pose is None else pose.tolist(),
        "pose_rpy_degrees": (
            None
            if pose is None
            else [float(value) for value in rotation_matrix_to_rpy_degrees(pose[:3, :3])]
        ),
        "pose_quality": None if pose_quality is None else pose_quality.metadata(),
        "rgbd_measurement": None if rgbd_measurement is None else rgbd_measurement.metadata(),
        "tracker": None if tracker_snapshot is None else tracker_snapshot.metadata(),
        "gravity": None if gravity_snapshot is None else gravity_snapshot.metadata(),
        "pose_tracking_mode": pose_tracking_mode,
        "motion_since_registration": motion_since_registration,
    }


def save_snapshot(
    output_dir: Path,
    color_bgr: np.ndarray,
    preview_bgr: np.ndarray,
    depth_raw: np.ndarray,
    observation: TargetObservation | None,
    pose: np.ndarray | None,
    K: np.ndarray,
    tracker_snapshot: TrackerSnapshot | None,
    rgbd_measurement: RGBDPositionMeasurement | None,
    gravity_snapshot: GravitySnapshot | None,
    pose_quality: PoseProjectionQuality | None,
    pose_tracking_mode: str,
    upright_source: str,
    motion_since_registration: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if not cv2.imwrite(str(output_dir / f"{timestamp}_rgb.png"), color_bgr):
        raise OSError("failed to save RGB snapshot")
    if not cv2.imwrite(str(output_dir / f"{timestamp}_depth.png"), depth_raw):
        raise OSError("failed to save depth snapshot")
    if not cv2.imwrite(str(output_dir / f"{timestamp}_preview.png"), preview_bgr):
        raise OSError("failed to save 2D/3D preview snapshot")
    if observation is not None:
        if not cv2.imwrite(str(output_dir / f"{timestamp}_mask.png"), observation.mask.astype(np.uint8) * 255):
            raise OSError("failed to save mask snapshot")
    np.savetxt(output_dir / f"{timestamp}_K.txt", K, fmt="%.9f")
    if pose is not None:
        np.savetxt(output_dir / f"{timestamp}_ob_in_cam.txt", pose, fmt="%.9f")
    tracking_metadata = {
        "rgbd_measurement": None if rgbd_measurement is None else rgbd_measurement.metadata(),
        "tracker": None if tracker_snapshot is None else tracker_snapshot.metadata(),
        "gravity": None if gravity_snapshot is None else gravity_snapshot.metadata(),
        "pose_quality": None if pose_quality is None else pose_quality.metadata(),
        "pose_tracking_mode": pose_tracking_mode,
        "upright_source": upright_source,
        "motion_since_registration": motion_since_registration,
    }
    with (output_dir / f"{timestamp}_tracker.json").open("w", encoding="utf-8") as file:
        json.dump(tracking_metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved snapshot: {output_dir / timestamp}")


def save_registration_bundle(
    output_dir: Path,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    observation: TargetObservation,
    K: np.ndarray,
    pose: np.ndarray,
    center_pose: np.ndarray,
    depth_filter: PoseDepthFilterResult,
    pose_quality: PoseProjectionQuality,
    tracker_snapshot: TrackerSnapshot | None,
    rgbd_measurement: RGBDPositionMeasurement | None,
    gravity_snapshot: GravitySnapshot | None,
    pose_tracking_mode: str,
    upright_source: str,
    motion_since_registration: bool,
) -> Path:
    """Persist every input needed to reproduce and inspect one registration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    bundle_dir = output_dir / "registrations" / timestamp
    bundle_dir.mkdir(parents=True, exist_ok=False)

    filtered_raw = np.rint(depth_filter.depth_m / depth_scale)
    filtered_raw = np.clip(filtered_raw, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    images = {
        "rgb.png": color_bgr,
        "depth_raw.png": depth_raw,
        "depth_filtered.png": filtered_raw,
        "mask.png": observation.mask.astype(np.uint8) * 255,
        "overlay_2d.png": annotate_target(color_bgr, observation),
    }
    for filename, image in images.items():
        if not cv2.imwrite(str(bundle_dir / filename), image):
            raise OSError(f"failed to save registration diagnostic: {bundle_dir / filename}")

    np.savetxt(bundle_dir / "K.txt", K, fmt="%.9f")
    np.savetxt(bundle_dir / "ob_in_cam.txt", pose, fmt="%.9f")
    np.savetxt(bundle_dir / "bbox_center_in_cam.txt", center_pose, fmt="%.9f")
    metadata = {
        "class_id": observation.class_id,
        "class_name": observation.class_name,
        "confidence": observation.confidence,
        "box_xyxy": [float(value) for value in observation.box],
        "mask_source": observation.mask_source,
        "depth_scale_m_per_unit": depth_scale,
        "pose_convention": "ob_in_cam maps the original CAD coordinate system into the color-camera frame",
        "camera_axes": "+X right, +Y down, +Z forward",
        "translation_m": [float(value) for value in pose[:3, 3]],
        "rotation_rpy_degrees": [float(value) for value in rotation_matrix_to_rpy_degrees(pose[:3, :3])],
        "rotation_convention": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll); pose maps CAD coordinates to camera coordinates",
        "pose_projection_quality": pose_quality.metadata(),
        "rgbd_measurement": None if rgbd_measurement is None else rgbd_measurement.metadata(),
        "tracker": None if tracker_snapshot is None else tracker_snapshot.metadata(),
        "gravity": None if gravity_snapshot is None else gravity_snapshot.metadata(),
        "pose_tracking_mode": pose_tracking_mode,
        "upright_source": upright_source,
        "motion_since_registration": motion_since_registration,
        "depth_filter": depth_filter.metadata(),
    }
    with (bundle_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved registration diagnostics: {bundle_dir}")
    return bundle_dir


def run(args: argparse.Namespace) -> int:
    yolo_path = args.yolo_model.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    foundationpose_root = args.foundationpose_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    record_dir = args.record_dir.expanduser().resolve()
    record_bag = None if args.record_bag is None else args.record_bag.expanduser().resolve()
    for path, label in ((yolo_path, "YOLO checkpoint"), (mesh_path, "mesh")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not (foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose repository not found: {foundationpose_root}")
    if record_bag is not None:
        if record_bag.suffix.lower() != ".bag":
            raise ValueError("--record-bag output must use the .bag extension")
        if record_bag.exists():
            raise FileExistsError(f"refusing to overwrite existing RealSense recording: {record_bag}")
        record_bag.parent.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(REPO_ROOT))
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
    install_foundationpose_opencv_compat(predict_pose_refine, predict_score)
    model = YOLO(str(yolo_path))
    if args.target_class not in model.names:
        raise ValueError(f"target class {args.target_class} is absent; model classes={model.names}")

    mesh = load_mesh(mesh_path, trimesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inverse_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack((-extents / 2.0, extents / 2.0), axis=0).reshape(2, 3)
    tracker_config = TrackerConfig(
        acceleration_std_mps2=args.tracker_accel_std,
        measurement_xy_std_m=args.tracker_measurement_xy_std,
        measurement_z_std_m=args.tracker_measurement_z_std,
        mahalanobis_gate=args.tracker_gate,
        max_missed_frames=args.tracker_max_missed,
        stationary_speed_mps=args.tracker_stationary_speed,
        stationary_frames=args.tracker_stationary_frames,
    )
    position_tracker = ConstantVelocity3DTracker(tracker_config) if args.tracker_3d else None
    gravity_filter = (
        GravityAxisFilter(smoothing=args.gravity_smoothing)
        if args.upright_source == "imu"
        else None
    )
    tracker_center_depth_offset = (
        float(args.tracker_center_depth_offset)
        if args.tracker_center_depth_offset is not None
        else float(np.mean(mesh.extents[:2]) * 0.5)
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
    # grasp_controller = RobotGraspController(
    #     enable_ros=args.enable_grasp,
    #     side=args.grasp_side,
    #     urdf_dir=str(REPO_ROOT / "bxi_teleop_v2" / "data"),
    # )

    # If --ros2 is given, subscribe to camera topics instead of using pyrealsense2.
    use_ros2 = getattr(args, "ros2", False)
    pipeline = None
    config = None
    ros_node = None
    ros_tf_buffer = None
    ros_helpers = []
    if not use_ros2:
        if rs is None:
            raise RuntimeError("pyrealsense2 not available; use --ros2 to subscribe to remote camera topics")
        context = rs.context()
        if not list(context.query_devices()):
            raise RuntimeError("No RealSense camera detected; check USB 3.0, cable, and device permissions")
        pipeline = rs.pipeline()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        if args.upright_source == "imu":
            config.enable_stream(rs.stream.accel, realsense_motion_xyz_format(), args.accel_fps)
        if record_bag is not None:
            config.enable_record_to_file(str(record_bag))
    else:
        # Lazy import rclpy and ROS2 message types when using ROS2 topics
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
        rclpy.init()
        ros_node = rclpy.create_node("d435i_foundationpose_bridge")
        if gravity_filter is not None:
            ros_tf_buffer = Buffer(node=ros_node)
            # The list keeps the listener (and its TF subscriptions) alive with the node.
            ros_helpers.append(TransformListener(ros_tf_buffer, ros_node, spin_thread=False))
        # storage for latest messages
        latest = {
            "rgb": None,
            "depth": None,
            "info": None,
            "depth_scale": None,
            "imu": None,
            "imu_sequence": 0,
        }

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
                    cv_img = image_msg_to_cv2(msg, desired_encoding=encoding)
                    latest[key] = cv_img
                except Exception as error:
                    print(f"ROS2 callback error ({key}): {error}", flush=True)

            return cb

        def camera_info_cb(msg):
            latest["info"] = msg

        def depth_scale_cb(msg):
            try:
                latest["depth_scale"] = float(msg.data)
            except Exception:
                latest["depth_scale"] = None

        def imu_cb(msg):
            latest["imu"] = msg
            latest["imu_sequence"] += 1

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
                else "imu=disabled by --upright-source"
            ),
        )

    started = False
    pose: np.ndarray | None = None
    observation: TargetObservation | None = None
    register_pending = bool(args.auto_register)
    recovery_pending = False
    pose_quality: PoseProjectionQuality | None = None
    invalid_pose_frames = 0
    previous_observation_box: np.ndarray | None = None
    stable_observation_frames = 0
    last_pose_status = ""
    tracker_snapshot: TrackerSnapshot | None = None
    rgbd_measurement: RGBDPositionMeasurement | None = None
    gravity_snapshot: GravitySnapshot | None = None
    accel_to_color = None
    reference_upright_axis: np.ndarray | None = None
    trusted_pose: np.ndarray | None = None
    registration_tracker_position: np.ndarray | None = None
    motion_since_registration = False
    last_pose_update_time: float | None = None
    preview_recorder = PreviewVideoRecorder(record_dir, args.record_fps)
    preview_auto_start_pending = bool(args.record_preview)
    grasp_executed_for_current_pose = False
    ros_wait_status_time = 0.0
    ros_wait_status_interval = 1.0
    last_ros_imu_sequence = 0
    last_ros_imu_accepted_time: float | None = None
    ros_imu_status = (
        "waiting for first message" if use_ros2 and gravity_filter is not None else "disabled"
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
        if not use_ros2 or gravity_filter is None:
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

        frame_description = source_frame
        if not target_frame:
            set_ros_imu_status(
                "waiting-camera-frame",
                f"received {source_frame}; waiting for CameraInfo.header.frame_id",
            )
            return
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
                acceleration = rotate_vector_by_quaternion_xyzw(
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
        if not use_ros2:
            try:
                profile = pipeline.start(config)
            except RuntimeError as error:
                if args.upright_source == "imu":
                    detail = str(error)
                    if "scan_element" in detail and "Permission denied" in detail:
                        hint = (
                            "D435i RGB/depth are accessible, but Linux denied access to its IIO "
                            "accelerometer. Install/reload the official librealsense udev rules and "
                            "reconnect the camera. Until then, add '--upright-source reference'."
                        )
                    else:
                        hint = (
                            "D435i RGB/depth/IMU streams could not start together. Check that "
                            f"--accel-fps {args.accel_fps} is supported (this D435i reports "
                            "100/200/400), or temporarily add '--upright-source reference'."
                        )
                    raise RuntimeError(f"{hint} Original RealSense error: {detail}") from error
                raise
            started = True
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            align_to_color = rs.align(rs.stream.color)
            if args.upright_source == "imu":
                try:
                    accel_profile = profile.get_stream(rs.stream.accel)
                    color_profile = profile.get_stream(rs.stream.color)
                    accel_to_color = accel_profile.get_extrinsics_to(color_profile)
                except RuntimeError as error:
                    raise RuntimeError(f"D435i accelerometer/color extrinsics unavailable: {error}") from error
            for _ in range(args.warmup):
                warmup_frames = pipeline.wait_for_frames()
                if gravity_filter is not None:
                    gravity_snapshot = gravity_filter.update(
                        accelerometer_vector_in_color_frame(warmup_frames, accel_to_color)
                    )
        else:
            depth_scale = None
            align_to_color = None
            for _ in range(max(1, args.warmup // 5)):
                rclpy.spin_once(ros_node, timeout_sec=0.05)
                for _ in range(8):
                    rclpy.spin_once(ros_node, timeout_sec=0.0)
                update_gravity_from_latest_ros_imu(latest.get("info"))

        print(f"YOLO: {yolo_path}")
        print(f"Mesh: {mesh_path}; extents(m): {extents}")
        depth_scale_text = "unknown" if depth_scale is None else f"{depth_scale:.9f}"
        print(
            f"Target: {args.target_class}={model.names[args.target_class]}; "
            f"depth scale: {depth_scale_text} m/unit"
        )
        if position_tracker is not None:
            print(
                f"3D tracker enabled: center-depth offset={tracker_center_depth_offset:.4f}m, "
                f"stationary={tracker_config.stationary_speed_mps:.3f}m/s for "
                f"{tracker_config.stationary_frames} frames"
            )
        if args.upright_source == "imu":
            imu_source = f"ROS2 {args.imu_topic}" if use_ros2 else "local D435i accelerometer"
            print(f"Upright constraint: {imu_source}, max tilt={args.max_upright_tilt:.1f}deg")
        elif args.upright_source == "reference":
            print(
                "Upright constraint: first accepted CAD Z axis, "
                f"max drift={args.max_upright_tilt:.1f}deg"
            )
        else:
            print("Upright constraint disabled")
        print(f"Pose tracking mode: {args.pose_tracking_mode}")
        if record_bag is not None:
            print(f"Raw RealSense RGB-D recording: {record_bag}")
        if args.enable_grasp:
            print("Grasp enabled: pose -> IK -> robot joint command (G triggers a manual grasp)")
        print(
            "R: register/re-register | C: clear pose | T: reset 3D tracker | "
            "V: preview recording | S/Space: save | G: grasp | Q/Esc: quit"
        )

        previous_time = time.perf_counter()
        smoothed_fps = 0.0
        while True:
            if use_ros2:
                rclpy.spin_once(ros_node, timeout_sec=0.02)
                for _ in range(8):
                    rclpy.spin_once(ros_node, timeout_sec=0.0)
                update_gravity_from_latest_ros_imu(latest.get("info"))
                rgb_msg = latest.get("rgb")
                depth_msg = latest.get("depth")
                info_msg = latest.get("info")
                depth_scale_val = latest.get("depth_scale")
                if rgb_msg is None or depth_msg is None or info_msg is None:
                    now = time.perf_counter()
                    if now - ros_wait_status_time >= ros_wait_status_interval:
                        ros_wait_status_time = now
                        topic_status = [
                            f"rgb={'ok' if rgb_msg is not None else 'missing'}",
                            f"depth={'ok' if depth_msg is not None else 'missing'}",
                            f"info={'ok' if info_msg is not None else 'missing'}",
                            f"depth_scale={'ok' if depth_scale_val is not None else 'missing'}",
                        ]
                        if gravity_filter is not None:
                            topic_status.append(f"imu={ros_imu_status}")
                        print("Waiting for ROS2 camera topics:", ", ".join(topic_status))
                    continue
                color_bgr = rgb_msg.copy()
                # depth may be 16UC1 or 32FC1 depending on publisher; accept as-is
                depth_raw = depth_msg.copy()
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                depth_scale = float(depth_scale_val) if depth_scale_val is not None else 0.001
                depth_m = depth_raw.astype(np.float32) * depth_scale
                depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
                # build intrinsics from CameraInfo
                try:
                    class DummyIntrinsics:
                        pass

                    intr = DummyIntrinsics()
                    intr.fx = info_msg.k[0]
                    intr.ppx = info_msg.k[2]
                    intr.fy = info_msg.k[4]
                    intr.ppy = info_msg.k[5]
                    K = camera_matrix(intr)
                except Exception:
                    continue
            else:
                raw_frames = pipeline.wait_for_frames()
                if gravity_filter is not None:
                    gravity_snapshot = gravity_filter.update(
                        accelerometer_vector_in_color_frame(raw_frames, accel_to_color)
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
                K = camera_matrix(intrinsics)

            start = time.perf_counter()
            if pose is None:
                grasp_executed_for_current_pose = False
            observation, _ = find_target(model, color_bgr, depth_raw, depth_scale, args)
            if observation is None:
                previous_observation_box = None
                stable_observation_frames = 0
            else:
                box_motion = detection_box_motion_ratio(observation.box, previous_observation_box)
                if box_motion <= args.reregister_box_motion:
                    stable_observation_frames += 1
                else:
                    stable_observation_frames = 1
                previous_observation_box = observation.box.copy()

            current_depth_filter: PoseDepthFilterResult | None = None
            rgbd_measurement = None
            if observation is not None:
                current_depth_filter = filter_pose_depth(
                    depth_m,
                    observation.mask,
                    enabled=args.pose_depth_filter,
                    min_depth_m=args.min_depth,
                    max_depth_m=args.max_depth,
                    minimum_band_m=args.pose_depth_band,
                    maximum_band_m=args.pose_depth_max_band,
                    erode_kernel_size=args.pose_depth_erode,
                )
                if position_tracker is not None:
                    rgbd_measurement = make_rgbd_position_measurement(
                        K,
                        observation,
                        current_depth_filter,
                        center_depth_offset_m=tracker_center_depth_offset,
                        minimum_valid_depth_ratio=args.tracker_min_valid_depth_ratio,
                        tracker_config=tracker_config,
                    )

            if position_tracker is not None:
                tracker_snapshot = position_tracker.step(
                    None if rgbd_measurement is None else rgbd_measurement.position_m,
                    start,
                    None if rgbd_measurement is None else rgbd_measurement.covariance,
                )

            if args.upright_source == "imu":
                ros_imu_is_fresh = (
                    not use_ros2
                    or (
                        last_ros_imu_accepted_time is not None
                        and start - last_ros_imu_accepted_time <= args.imu_timeout
                    )
                )
                expected_upright_axis = (
                    None
                    if gravity_snapshot is None or not ros_imu_is_fresh
                    else gravity_snapshot.axis_camera
                )
            elif args.upright_source == "reference":
                expected_upright_axis = reference_upright_axis
            else:
                expected_upright_axis = None

            tracker_position_for_pose = (
                None
                if tracker_snapshot is None
                or tracker_snapshot.position_m is None
                or tracker_snapshot.status == "lost"
                else tracker_snapshot.position_m
            )
            tracker_speed = (
                None
                if tracker_snapshot is None or tracker_snapshot.velocity_mps is None
                else float(np.linalg.norm(tracker_snapshot.velocity_mps))
            )
            if pose is not None and tracker_position_for_pose is not None:
                displacement = (
                    0.0
                    if registration_tracker_position is None
                    else float(np.linalg.norm(tracker_position_for_pose - registration_tracker_position))
                )
                if (
                    (tracker_speed is not None and tracker_speed >= args.pose_motion_speed)
                    or displacement >= args.pose_motion_distance
                ):
                    motion_since_registration = True

            if (
                pose is not None
                and motion_since_registration
                and args.reregister_after_motion
                and observation is not None
                and stable_observation_frames >= args.reregister_stable_frames
                and tracker_snapshot is not None
                and tracker_snapshot.is_stationary
            ):
                print("Moved target is stationary again; refreshing FoundationPose registration")
                estimator.pose_last = None
                pose = None
                trusted_pose = None
                register_pending = True
                recovery_pending = False
                pose_quality = None
                invalid_pose_frames = 0

            if pose is None:
                tracker_ready_for_registration = (
                    position_tracker is None
                    or (tracker_snapshot is not None and tracker_snapshot.is_stationary)
                )
                observation_ready_for_registration = (
                    observation is not None
                    and stable_observation_frames >= args.reregister_stable_frames
                    and tracker_ready_for_registration
                )
                if (
                    recovery_pending
                    and observation_ready_for_registration
                ):
                    print(
                        f"Target stable for {stable_observation_frames} frames; "
                        "starting automatic re-registration"
                    )
                    register_pending = True
                    recovery_pending = False
                imu_ready = args.upright_source != "imu" or expected_upright_axis is not None
                if register_pending and observation is not None and not imu_ready:
                    last_pose_status = (
                        f"waiting for ROS2 IMU gravity axis: {ros_imu_status}"
                        if use_ros2
                        else "waiting for D435i gravity axis"
                    )
                elif register_pending and observation is not None and not observation_ready_for_registration:
                    tracker_stable_frames = (
                        0 if tracker_snapshot is None else tracker_snapshot.stationary_frames
                    )
                    last_pose_status = (
                        "waiting for stable RGB-D target: "
                        f"2D {stable_observation_frames}/{args.reregister_stable_frames}, "
                        f"KF {tracker_stable_frames}/{tracker_config.stationary_frames}"
                    )
                if register_pending and observation_ready_for_registration and imu_ready:
                    assert current_depth_filter is not None
                    pose_depth = current_depth_filter
                    depth_metadata = pose_depth.metadata()
                    print(
                        f"Registering {observation.class_name}: conf={observation.confidence:.3f}, "
                        f"mask={observation.mask_source}, pixels={np.count_nonzero(observation.mask)}, "
                        f"valid-depth={depth_metadata['valid_ratio_before']:.1%}, "
                        f"removed={pose_depth.removed_pixels}, "
                        f"reference={pose_depth.reference_depth_m}, band={pose_depth.effective_band_m}"
                    )
                    pose = estimator.register(
                        K=K,
                        rgb=color_rgb,
                        depth=pose_depth.depth_m,
                        ob_mask=observation.mask,
                        iteration=args.est_refine_iter,
                    )
                    if args.snap_upright_registration and expected_upright_axis is not None:
                        raw_upright_tilt = axis_tilt_degrees(
                            pose[:3, 2], expected_upright_axis
                        )
                        pose = pose_with_upright_axis(
                            pose,
                            expected_upright_axis,
                            inverse_to_origin,
                        )
                        estimator.pose_last = pose.copy()
                        print(
                            "Applied known-upright constraint to registration: "
                            f"raw tilt={raw_upright_tilt:.1f}deg -> 0.0deg"
                        )
                    rpy_degrees = rotation_matrix_to_rpy_degrees(pose[:3, :3])
                    print("6D pose T_camera_object (meters):")
                    print(np.array2string(pose, precision=6, suppress_small=True))
                    print(
                        "translation xyz (m): "
                        f"{pose[0, 3]:+.6f}, {pose[1, 3]:+.6f}, {pose[2, 3]:+.6f}; "
                        "rotation rpy (deg): "
                        f"{rpy_degrees[0]:+.2f}, {rpy_degrees[1]:+.2f}, {rpy_degrees[2]:+.2f}"
                    )
                    center_pose = pose @ inverse_to_origin
                    upright_tilt = (
                        None
                        if expected_upright_axis is None
                        else axis_tilt_degrees(pose[:3, 2], expected_upright_axis)
                    )
                    tracker_reference = (
                        None
                        if tracker_snapshot is None or tracker_snapshot.position_m is None or tracker_snapshot.status == "lost"
                        else tracker_snapshot.position_m
                    )
                    pose_quality = assess_pose_projection(
                        K,
                        center_pose,
                        bbox,
                        observation.box,
                        minimum_iou=args.min_pose_box_iou,
                        maximum_center_error_ratio=args.max_pose_center_error,
                        reference_position_m=tracker_reference,
                        maximum_position_error_m=args.max_pose_3d_error,
                        upright_tilt_degrees=upright_tilt,
                        maximum_upright_tilt_degrees=args.max_upright_tilt,
                    )
                    position_error_text = (
                        "N/A" if pose_quality.position_error_m is None else f"{pose_quality.position_error_m:.3f}m"
                    )
                    print(
                        f"Pose projection quality: IoU={pose_quality.box_iou:.3f}, "
                        f"center-error={pose_quality.center_error_ratio:.3f}, "
                        f"area-ratio={pose_quality.projected_to_observed_area_ratio:.3f}, "
                        f"3D-error={position_error_text}, "
                        f"upright-tilt={'N/A' if upright_tilt is None else f'{upright_tilt:.1f}deg'}, "
                        f"result={pose_quality.reason}"
                    )
                    save_registration_bundle(
                        output_dir,
                        color_bgr,
                        depth_raw,
                        depth_scale,
                        observation,
                        K,
                        pose,
                        center_pose,
                        pose_depth,
                        pose_quality,
                        tracker_snapshot,
                        rgbd_measurement,
                        gravity_snapshot,
                        args.pose_tracking_mode,
                        args.upright_source,
                        motion_since_registration,
                    )
                    register_pending = False
                    if pose_quality.valid:
                        invalid_pose_frames = 0
                        last_pose_status = ""
                        trusted_pose = pose.copy()
                        registration_tracker_position = (
                            None if tracker_position_for_pose is None else tracker_position_for_pose.copy()
                        )
                        motion_since_registration = False
                        last_pose_update_time = start
                        if args.upright_source == "reference" and reference_upright_axis is None:
                            reference_upright_axis = pose[:3, 2].copy()
                    else:
                        print(
                            "Rejected implausible 6D pose; hold the target still and press R to try again. "
                            f"Reason: {pose_quality.reason}"
                        )
                        estimator.pose_last = None
                        pose = None
                        trusted_pose = None
                        last_pose_status = f"pose rejected: {pose_quality.reason}"
            else:
                tracking_depth = depth_m
                if current_depth_filter is not None:
                    tracking_depth = current_depth_filter.depth_m
                angular_speed_degrees_s = 0.0
                if args.pose_tracking_mode in ("upright-follow", "upright-hold"):
                    rotation_source_pose = trusted_pose if trusted_pose is not None else pose
                    if (
                        args.pose_tracking_mode == "upright-follow"
                        and expected_upright_axis is not None
                    ):
                        rotation_source_pose = pose_with_upright_axis(
                            rotation_source_pose,
                            expected_upright_axis,
                            inverse_to_origin,
                        )
                    if tracker_position_for_pose is not None:
                        pose = pose_with_center_position(
                            rotation_source_pose,
                            tracker_position_for_pose,
                            inverse_to_origin,
                        )
                    else:
                        pose = rotation_source_pose.copy()
                else:
                    previous_rotation = pose[:3, :3].copy()
                    pose = estimator.track_one(
                        rgb=color_rgb,
                        depth=tracking_depth,
                        K=K,
                        iteration=args.track_refine_iter,
                    )
                    pose_dt = max(1e-3, start - (last_pose_update_time or start))
                    angular_speed_degrees_s = (
                        relative_rotation_degrees(previous_rotation, pose[:3, :3]) / pose_dt
                    )
                last_pose_update_time = start
                center_pose = pose @ inverse_to_origin
                upright_tilt = (
                    None
                    if expected_upright_axis is None
                    else axis_tilt_degrees(pose[:3, 2], expected_upright_axis)
                )
                if observation is None:
                    pose_quality = None
                    invalid_pose_frames += 1
                else:
                    tracker_reference = (
                        None
                        if tracker_snapshot is None or tracker_snapshot.position_m is None or tracker_snapshot.status == "lost"
                        else tracker_snapshot.position_m
                    )
                    pose_quality = assess_pose_projection(
                        K,
                        center_pose,
                        bbox,
                        observation.box,
                        minimum_iou=args.min_pose_box_iou,
                        maximum_center_error_ratio=args.max_pose_center_error,
                        reference_position_m=tracker_reference,
                        maximum_position_error_m=args.max_pose_3d_error,
                        upright_tilt_degrees=upright_tilt,
                        maximum_upright_tilt_degrees=args.max_upright_tilt,
                        angular_speed_degrees_s=angular_speed_degrees_s,
                        maximum_angular_speed_degrees_s=args.max_pose_angular_speed,
                    )
                    invalid_pose_frames = 0 if pose_quality.valid else invalid_pose_frames + 1

                if invalid_pose_frames >= args.pose_lost_frames:
                    reason = "YOLO target missing" if pose_quality is None else pose_quality.reason
                    print(
                        f"Pose tracking lost for {invalid_pose_frames} frames ({reason}); "
                        + ("waiting for a stable target to re-register" if args.auto_reregister else "press R to re-register")
                    )
                    estimator.pose_last = None
                    pose = None
                    trusted_pose = None
                    registration_tracker_position = None
                    motion_since_registration = False
                    pose_quality = None
                    invalid_pose_frames = 0
                    register_pending = False
                    recovery_pending = bool(args.auto_reregister)
                    previous_observation_box = None
                    stable_observation_frames = 0
                    last_pose_status = "pose lost"
            processing_ms = (time.perf_counter() - start) * 1000.0

            if pose is None:
                display = annotate_target(color_bgr, observation)
                if recovery_pending:
                    tracker_stability = (
                        ""
                        if tracker_snapshot is None
                        else f" | KF stable {tracker_snapshot.stationary_frames}/{tracker_config.stationary_frames}"
                    )
                    status = (
                        f"pose lost | 2D stable {stable_observation_frames}/{args.reregister_stable_frames}"
                        f"{tracker_stability}"
                    )
                elif last_pose_status:
                    status = last_pose_status
                else:
                    status = "press R to register" if observation is not None else "waiting for target"
            else:
                center_pose = pose @ inverse_to_origin
                base_bgr = annotate_target(color_bgr, observation) if args.show_2d_overlay else color_bgr
                base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
                display_rgb = draw_posed_3d_box(K, img=base_rgb, ob_in_cam=center_pose, bbox=bbox)
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
                tx, ty, tz = pose[:3, 3]
                roll, pitch, yaw = rotation_matrix_to_rpy_degrees(pose[:3, :3])
                quality_text = ""
                if pose_quality is not None:
                    quality_text = f" | IoU={pose_quality.box_iou:.2f}"
                    if pose_quality.upright_tilt_degrees is not None:
                        quality_text += f" tilt={pose_quality.upright_tilt_degrees:.1f}deg"
                    if not pose_quality.valid:
                        quality_text += f" suspect {invalid_pose_frames}/{args.pose_lost_frames}"
                if args.pose_tracking_mode == "upright-follow":
                    mode_text = "UP-FOLLOW"
                elif args.pose_tracking_mode == "upright-hold":
                    mode_text = "ROT-HOLD"
                else:
                    mode_text = "FP-TRACK"
                status = f"{mode_text} | t=({tx:+.3f}, {ty:+.3f}, {tz:+.3f})m{quality_text}"
                cv2.putText(
                    display,
                    f"rpy=({roll:+.1f}, {pitch:+.1f}, {yaw:+.1f})deg",
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if args.show_2d_overlay:
                    cv2.putText(
                        display,
                        "magenta=YOLO 2D | green=CAD 3D",
                        (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if args.upright_source == "imu":
                upright_source_text = "IMU"
            elif args.upright_source == "reference":
                upright_source_text = "REF"
            else:
                upright_source_text = "OFF"
            current_tilt = None if pose_quality is None else pose_quality.upright_tilt_degrees
            cv2.putText(
                display,
                f"UP={upright_source_text} tilt={'N/A' if current_tilt is None else f'{current_tilt:.1f}deg'} "
                f"moved={'yes' if motion_since_registration else 'no'}",
                (12, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

            display = annotate_3d_tracker(display, K, tracker_snapshot)
            if tracker_snapshot is not None and tracker_snapshot.position_m is not None:
                tracker_position = tracker_snapshot.position_m
                tracker_speed = float(np.linalg.norm(tracker_snapshot.velocity_mps))
                if tracker_snapshot.measurement_accepted is True:
                    measurement_state = "meas=ok"
                elif tracker_snapshot.measurement_accepted is False:
                    measurement_state = "meas=reject"
                else:
                    measurement_state = "predict-only"
                cv2.putText(
                    display,
                    f"KF3D {tracker_snapshot.status} p=({tracker_position[0]:+.3f},"
                    f"{tracker_position[1]:+.3f},{tracker_position[2]:+.3f})m "
                    f"v={tracker_speed:.3f}m/s {measurement_state}",
                    (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            cv2.putText(
                display,
                f"{status} | {smoothed_fps:.1f} FPS | {processing_ms:.1f} ms",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if (
                args.enable_grasp
                and pose is not None
                and pose_quality is not None
                and pose_quality.valid
                and not grasp_executed_for_current_pose
            ):
                print("Executing grasp from current pose")
                # grasp_controller.execute_grasp_from_pose(
                #     pose,
                #     side=args.grasp_side,
                #     approach_offset_m=args.grasp_approach_offset,
                #     grasp_offset_m=args.grasp_offset,
                # )
                grasp_executed_for_current_pose = True

            if preview_auto_start_pending:
                preview_recorder.start(display.shape)
                preview_auto_start_pending = False
            if preview_recorder.active:
                cv2.putText(
                    display,
                    "REC",
                    (display.shape[1] - 72, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(display, (display.shape[1] - 88, 23), 6, (0, 0, 255), cv2.FILLED)
            preview_recorder.write(
                display,
                frame_record_metadata(
                    observation,
                    pose,
                    pose_quality,
                    tracker_snapshot,
                    rgbd_measurement,
                    gravity_snapshot,
                    args.pose_tracking_mode,
                    motion_since_registration,
                ),
            )
            cv2.imshow("D435i YOLO + FoundationPose", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                estimator.pose_last = None
                pose = None
                register_pending = True
                recovery_pending = False
                pose_quality = None
                invalid_pose_frames = 0
                trusted_pose = None
                registration_tracker_position = None
                motion_since_registration = False
                last_pose_update_time = None
                last_pose_status = ""
            elif key in (ord("c"), ord("C")):
                estimator.pose_last = None
                pose = None
                register_pending = False
                recovery_pending = False
                pose_quality = None
                invalid_pose_frames = 0
                trusted_pose = None
                registration_tracker_position = None
                motion_since_registration = False
                last_pose_update_time = None
                last_pose_status = "pose cleared; press R to register"
            elif key in (ord("t"), ord("T")) and position_tracker is not None:
                position_tracker.reset()
                tracker_snapshot = position_tracker.snapshot()
                rgbd_measurement = None
                registration_tracker_position = None
                print("3D position tracker reset")
            elif key in (ord("v"), ord("V")):
                if preview_recorder.active:
                    preview_recorder.stop()
                else:
                    preview_recorder.start(display.shape)
            elif key in (ord("g"), ord("G")) and args.enable_grasp and pose is not None:
                print("Manual grasp trigger")
                # grasp_controller.execute_grasp_from_pose(
                #     pose,
                #     side=args.grasp_side,
                #     approach_offset_m=args.grasp_approach_offset,
                #     grasp_offset_m=args.grasp_offset,
                # )
                grasp_executed_for_current_pose = True
            elif key in (ord("s"), ord("S"), ord(" ")):
                save_snapshot(
                    output_dir / "snapshots",
                    color_bgr,
                    display,
                    depth_raw,
                    observation,
                    pose,
                    K,
                    tracker_snapshot,
                    rgbd_measurement,
                    gravity_snapshot,
                    pose_quality,
                    args.pose_tracking_mode,
                    args.upright_source,
                    motion_since_registration,
                )
        return 0
    finally:
        preview_recorder.stop()
        if started:
            pipeline.stop()
        if use_ros2:
            try:
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
