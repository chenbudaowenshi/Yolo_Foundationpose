#!/usr/bin/env python3
"""YOLO-seg + FoundationPose pose estimation for the ELF3 MuJoCo head camera.

Run ``my_sim/head_control_mujoco.py`` in another terminal first.  This program
waits until the head command and the measured head joints have stopped moving,
confirms the same target in several YOLO segmentation frames, aligns MuJoCo's
depth stream to the color stream, and then estimates the object's 6D pose.

The primary result is ``T_camera_object`` (object/CAD coordinates to the color
camera optical frame).  When synchronized odometry is available, the script
also reports ``T_world_object`` using the kinematics of the current ELF3 MJCF.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_PYTHON = Path("/home/robot/anaconda3/envs/pose/bin/python")
DEFAULT_YOLO_MODEL = REPO_ROOT / "ultralytics/ultralytics/yolov8n-seg.pt"
DEFAULT_OUTPUT = REPO_ROOT / "runs/foundationpose_mujoco/pose.json"

RGB_TOPIC = "/simulation/head_depth_camera/color/image_raw"
COLOR_INFO_TOPIC = "/simulation/head_depth_camera/color/camera_info"
DEPTH_TOPIC = "/simulation/head_depth_camera/depth/image_rect_raw"
DEPTH_INFO_TOPIC = "/simulation/head_depth_camera/depth/camera_info"
ACTUATOR_STATE_TOPIC = "/simulation/actuator_states"
HEAD_COMMAND_TOPIC = "/simulation/actuators_cmds_override"
ODOM_TOPIC = "/simulation/odom"
WINDOW_NAME = "MuJoCo YOLO-seg + FoundationPose"

ROS_SETUP_FILES = (
    Path("/opt/ros/humble/setup.bash"),
    Path("/opt/bxi/bxi_ros2_pkg/setup.bash"),
    REPO_ROOT / "bxi_rl_controller_ros2_example/install/setup.bash",
)


def ensure_pose_python() -> None:
    """Re-execute with the Python 3.10 environment shared by ROS and FoundationPose."""
    if __name__ != "__main__" or os.environ.get("MUJOCO_POSE_REEXEC") == "1":
        return
    if not POSE_PYTHON.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        expected = POSE_PYTHON.resolve()
    except OSError:
        return
    if current == expected:
        return

    environment = os.environ.copy()
    environment["MUJOCO_POSE_REEXEC"] = "1"
    environment.pop("PYTHONHOME", None)
    print(f"切换 Python 环境: {current} -> {expected}", file=sys.stderr, flush=True)
    os.execve(
        str(expected),
        [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


ensure_pose_python()


def ensure_ros_environment() -> None:
    """Load the ROS 2 and BXI overlays even when the terminal was not sourced.

    ``PYTHONPATH`` alone is not enough for generated ROS messages: their type
    support also needs shared libraries from the matching install prefixes.
    Re-executing through Bash applies the complete colcon environment before
    Python imports ``rclpy`` or ``communication.msg``.
    """
    if __name__ != "__main__" or os.environ.get("MUJOCO_POSE_ROS_BOOTSTRAPPED") == "1":
        return
    if any(argument in {"-h", "--help", "--self-test"} for argument in sys.argv[1:]):
        return

    missing = [str(path) for path in ROS_SETUP_FILES if not path.is_file()]
    if missing:
        raise SystemExit(
            "无法自动加载 ROS 2/BXI，缺少环境脚本：\n  " + "\n  ".join(missing)
        )

    environment = os.environ.copy()
    environment["MUJOCO_POSE_ROS_BOOTSTRAPPED"] = "1"
    environment.pop("PYTHONHOME", None)
    environment.pop("BASH_ENV", None)
    script_path = str(Path(__file__).resolve())
    shell_script = """
set -e
source "$1"
source "$2"
source "$3"
shift 3
exec "$@"
"""
    print("自动加载 ROS 2、BXI communication 和本地工作空间环境...", file=sys.stderr)
    os.execve(
        "/bin/bash",
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            shell_script,
            "get_pose_mujoco_ros_bootstrap",
            *(str(path) for path in ROS_SETUP_FILES),
            sys.executable,
            script_path,
            *sys.argv[1:],
        ],
        environment,
    )


ensure_ros_environment()

try:
    import cv2
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "缺少 numpy/OpenCV。请使用：\n"
        f"  {POSE_PYTHON} {Path(__file__).resolve()}\n"
        f"原始错误: {error}"
    ) from error


@dataclass(frozen=True)
class CameraInfoData:
    stamp_ns: int
    frame_id: str
    width: int
    height: int
    K: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True)
class CameraPacket:
    stamp_ns: int
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    color_info: CameraInfoData
    depth_info: CameraInfoData


@dataclass(frozen=True)
class HeadSample:
    stamp_ns: int
    received_at: float
    yaw_rad: float
    pitch_rad: float
    yaw_velocity: float
    pitch_velocity: float


@dataclass(frozen=True)
class CommandSample:
    stamp_ns: int
    received_at: float
    yaw_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class OdomSample:
    stamp_ns: int
    received_at: float
    frame_id: str
    child_frame_id: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class TargetObservation:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: np.ndarray
    mask: np.ndarray


@dataclass
class FoundationRuntime:
    yolo: Any
    target_class_id: int
    estimator: Any
    mesh: Any
    mesh_source: str
    symmetry_steps: int
    yaw_observable: bool
    bbox: np.ndarray
    inverse_to_origin: np.ndarray
    draw_box: Any
    draw_axis: Any


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "等待 MuJoCo 头部停止，使用 YOLO-seg 分割目标，并用 FoundationPose "
            "估计物体在头部彩色相机坐标系中的 6D 位姿。"
        )
    )
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument(
        "--target-class",
        default="bottle",
        help="YOLO 类别名称或编号；默认 bottle（COCO 编号 39）",
    )
    parser.add_argument(
        "--mesh",
        default="auto",
        help=(
            "CAD 文件，或 auto：按当前 bottle.xml 的圆柱/瓶肩/瓶颈/瓶盖生成米制网格"
        ),
    )
    parser.add_argument(
        "--symmetry-steps",
        type=int,
        help="绕物体 Z 轴的离散对称数；auto 网格默认 36，1 表示禁用",
    )
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics/CUDA 设备，例如 0")
    parser.add_argument("--min-mask-pixels", type=int, default=200)
    parser.add_argument(
        "--scan-fallback-timeout",
        type=float,
        default=0.0,
        help=(
            "头控若锁到其他类别并保持不动，经过此秒数仍看不到目标时由本节点继续扫描；"
            "0 表示禁用（默认；避免与外部 head_control 同时发布命令）"
        ),
    )
    parser.add_argument(
        "--allow-command-conflict",
        action="store_true",
        help=(
            "明确允许回退扫描与另一个 head_control 同时向 override 发布；"
            "通常不要启用，应先停止外部 head_control"
        ),
    )
    parser.add_argument("--scan-yaw-amplitude", type=float, default=0.45)
    parser.add_argument("--scan-period", type=float, default=4.0)
    parser.add_argument("--head-kp", type=float, default=16.747)
    parser.add_argument("--head-kd", type=float, default=1.066)
    parser.add_argument("--required-detections", type=int, default=3)
    parser.add_argument("--same-instance-iou", type=float, default=0.25)

    parser.add_argument("--head-stable-duration", type=float, default=0.60)
    parser.add_argument("--head-position-tolerance", type=float, default=0.015)
    parser.add_argument("--head-velocity-tolerance", type=float, default=0.06)
    parser.add_argument("--command-position-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--no-require-held-command",
        action="store_false",
        dest="require_held_command",
        help="不检查 head_control 是否持续发布相同角度，只检查真实关节是否稳定",
    )
    parser.set_defaults(require_held_command=True)
    parser.add_argument(
        "--ignore-head-stability",
        action="store_true",
        help="检测到目标后立即估计（头部仍在运动时不推荐）",
    )
    parser.add_argument("--state-sync-tolerance", type=float, default=0.50)

    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--depth-band", type=float, default=0.05)
    parser.add_argument("--depth-max-band", type=float, default=0.20)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=100)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument(
        "--min-pose-box-iou",
        type=float,
        default=0.30,
        help="投影 CAD 框与 YOLO 框的最小 IoU，用于拒绝错误位姿",
    )
    parser.add_argument(
        "--max-pose-center-error",
        type=float,
        default=0.30,
        help="投影框中心误差除以 YOLO 框对角线的最大值",
    )
    parser.add_argument("--debug", type=int, default=0)

    parser.add_argument("--continuous", action="store_true", help="注册后持续跟踪")
    parser.add_argument("--pose-rate", type=float, default=5.0)
    parser.add_argument("--lost-frames", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=0.0, help="等待超时秒数；0 表示无限")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示实时 OpenCV 识别窗口（默认开启；无桌面环境可用 --no-show）",
    )
    parser.add_argument(
        "--display-hold-sec", type=float, default=2.0,
        help="非连续模式下结果窗口保持时间（默认 2 秒；0 表示立即继续）",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存掩码、对齐深度和位姿可视化图",
    )
    parser.add_argument(
        "--world-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="有同步 /simulation/odom 时计算 ELF3 仿真世界坐标位姿",
    )

    parser.add_argument("--rgb-topic", default=RGB_TOPIC)
    parser.add_argument("--color-info-topic", default=COLOR_INFO_TOPIC)
    parser.add_argument("--depth-topic", default=DEPTH_TOPIC)
    parser.add_argument("--depth-info-topic", default=DEPTH_INFO_TOPIC)
    parser.add_argument("--actuator-state-topic", default=ACTUATOR_STATE_TOPIC)
    parser.add_argument("--head-command-topic", default=HEAD_COMMAND_TOPIC)
    parser.add_argument("--odom-topic", default=ODOM_TOPIC)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只运行深度对齐和坐标变换自检，不启动 ROS/GPU",
    )
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf 和 --iou 必须位于 [0, 1]")
    if args.imgsz <= 0 or args.min_mask_pixels <= 0 or args.required_detections <= 0:
        parser.error("图像尺寸、掩码像素数和连续检测帧数必须为正数")
    if not 0.0 <= args.same_instance_iou <= 1.0:
        parser.error("--same-instance-iou 必须位于 [0, 1]")
    positive = (
        args.head_stable_duration,
        args.head_position_tolerance,
        args.head_velocity_tolerance,
        args.command_position_tolerance,
        args.scan_yaw_amplitude,
        args.scan_period,
        args.head_kp,
        args.head_kd,
        args.state_sync_tolerance,
        args.depth_scale,
        args.max_depth,
        args.depth_band,
        args.depth_max_band,
        args.pose_rate,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        parser.error("稳定阈值、深度参数和频率必须为有限正数")
    if args.min_depth < 0.0 or args.max_depth <= args.min_depth:
        parser.error("要求 0 <= --min-depth < --max-depth")
    if args.depth_max_band < args.depth_band:
        parser.error("--depth-max-band 不能小于 --depth-band")
    if args.est_refine_iter <= 0 or args.track_refine_iter <= 0:
        parser.error("FoundationPose 迭代次数必须为正数")
    if not 0.0 <= args.min_pose_box_iou <= 1.0:
        parser.error("--min-pose-box-iou 必须位于 [0, 1]")
    if not math.isfinite(args.max_pose_center_error) or args.max_pose_center_error <= 0.0:
        parser.error("--max-pose-center-error 必须为有限正数")
    if args.lost_frames <= 0 or args.min_valid_depth_pixels <= 0:
        parser.error("丢失帧数和有效深度像素数必须为正数")
    if args.timeout < 0.0:
        parser.error("--timeout 不能为负数")
    if not math.isfinite(args.display_hold_sec) or args.display_hold_sec < 0.0:
        parser.error("--display-hold-sec 必须为有限非负数")
    if args.scan_fallback_timeout < 0.0:
        parser.error("--scan-fallback-timeout 不能为负数")
    if args.scan_fallback_timeout > 0.0 and not args.allow_command_conflict:
        parser.error(
            "启用 --scan-fallback-timeout 前应停止外部 head_control；"
            "若确认需要并发发布，请同时显式添加 --allow-command-conflict"
        )
    if args.symmetry_steps is not None and args.symmetry_steps <= 0:
        parser.error("--symmetry-steps 必须为正整数")


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_offset_seconds(sample_stamp_ns: int, image_stamp_ns: int) -> float | None:
    """Return a meaningful offset, or None when one publisher uses a zero clock."""
    if sample_stamp_ns == 0 or image_stamp_ns == 0:
        return None
    return (sample_stamp_ns - image_stamp_ns) / 1e9


def camera_info_from_message(message: Any) -> CameraInfoData:
    K = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError(f"无效相机内参 K={K.tolist()}")
    return CameraInfoData(
        stamp_ns=stamp_to_ns(message.header.stamp),
        frame_id=str(message.header.frame_id),
        width=int(message.width),
        height=int(message.height),
        K=K,
        distortion=np.asarray(message.d, dtype=np.float64),
    )


def image_message_to_array(message: Any, *, color: bool) -> np.ndarray:
    """Decode rgb8/bgr8 and 16UC1/mono16/32FC1 without cv_bridge."""
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"无效图像尺寸 {width}x{height}")

    if color:
        if encoding not in {"rgb8", "bgr8"}:
            raise ValueError(f"彩色图编码必须是 rgb8/bgr8，实际为 {message.encoding!r}")
        row_bytes = width * 3
        if step < row_bytes:
            raise ValueError(f"彩色图 step={step}，至少应为 {row_bytes}")
        needed = step * height
        if len(message.data) < needed:
            raise ValueError(f"彩色图数据只有 {len(message.data)} 字节，需要 {needed}")
        raw = np.frombuffer(message.data, dtype=np.uint8, count=needed)
        image = raw.reshape(height, step)[:, :row_bytes].reshape(height, width, 3)
        if encoding == "rgb8":
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image).copy()

    dtype: np.dtype[Any]
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    else:
        raise ValueError(f"深度图编码必须是 16UC1/mono16/32FC1，实际为 {message.encoding!r}")
    row_bytes = width * dtype.itemsize
    if step < row_bytes:
        raise ValueError(f"深度图 step={step}，至少应为 {row_bytes}")
    needed = step * height
    if len(message.data) < needed:
        raise ValueError(f"深度图数据只有 {len(message.data)} 字节，需要 {needed}")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=needed).reshape(height, step)
    compact = np.ascontiguousarray(rows[:, :row_bytes])
    image = compact.view(dtype).reshape(height, width)
    native_dtype = np.uint16 if dtype.kind == "u" else np.float32
    return image.astype(native_dtype, copy=False).copy()


def depth_to_meters(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth_raw.dtype == np.float32 or depth_raw.dtype == np.float64:
        return depth_raw.astype(np.float32, copy=True)
    return depth_raw.astype(np.float32) * float(depth_scale)


def depth_alignment_maps(
    depth_K: np.ndarray,
    color_K: np.ndarray,
    color_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map color pixels to the co-located MuJoCo depth renderer's pixels."""
    height, width = color_shape
    fx_c, fy_c = float(color_K[0, 0]), float(color_K[1, 1])
    cx_c, cy_c = float(color_K[0, 2]), float(color_K[1, 2])
    fx_d, fy_d = float(depth_K[0, 0]), float(depth_K[1, 1])
    cx_d, cy_d = float(depth_K[0, 2]), float(depth_K[1, 2])
    if min(fx_c, fy_c, fx_d, fy_d) <= 0.0:
        raise ValueError("相机焦距必须为正数")
    u_color, v_color = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = fx_d * (u_color - cx_c) / fx_c + cx_d
    map_y = fy_d * (v_color - cy_c) / fy_c + cy_d
    return map_x.astype(np.float32), map_y.astype(np.float32)


def align_depth_to_color(
    depth_m: np.ndarray,
    depth_K: np.ndarray,
    color_K: np.ndarray,
    color_shape: tuple[int, int],
    maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Align depth to color without resizing their different fields of view.

    MuJoCo renders both streams from the exact same camera pose, so inverse
    pinhole mapping is dense and does not require a depth/color extrinsic.
    """
    if depth_m.ndim != 2:
        raise ValueError(f"深度图必须是 HxW，实际为 {depth_m.shape}")
    if maps is None:
        maps = depth_alignment_maps(depth_K, color_K, color_shape)
    map_x, map_y = maps
    aligned = cv2.remap(
        depth_m.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.ascontiguousarray(aligned, dtype=np.float32)


def box_iou(first: np.ndarray, second: np.ndarray | None) -> float:
    if second is None:
        return 0.0
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    size = np.maximum(0.0, upper - lower)
    intersection = float(size[0] * size[1])
    first_area = float(np.prod(np.maximum(0.0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0.0, second[2:] - second[:2])))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def project_mesh_bbox(
    K: np.ndarray,
    pose: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray | None:
    points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    if not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 0.01):
        return None
    projected = (K @ points.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(pixels)):
        return None
    return np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))


def assess_pose_quality(
    K: np.ndarray,
    pose: np.ndarray,
    mesh: Any,
    observation: TargetObservation,
    minimum_iou: float,
    maximum_center_error: float,
) -> dict[str, Any]:
    projected = project_mesh_bbox(K, pose, np.asarray(mesh.vertices, dtype=np.float64))
    if projected is None:
        return {
            "valid": False,
            "reason": "CAD projection is invalid or behind the camera",
            "projected_box_xyxy": None,
            "box_iou": 0.0,
            "center_error_ratio": float("inf"),
        }
    overlap = box_iou(projected, observation.box_xyxy)
    observed_size = np.maximum(1.0, observation.box_xyxy[2:] - observation.box_xyxy[:2])
    observed_center = (observation.box_xyxy[:2] + observation.box_xyxy[2:]) * 0.5
    projected_center = (projected[:2] + projected[2:]) * 0.5
    center_error = float(
        np.linalg.norm(projected_center - observed_center)
        / max(np.linalg.norm(observed_size), 1.0)
    )
    valid = overlap >= minimum_iou and center_error <= maximum_center_error
    reason = (
        "ok"
        if valid
        else (
            f"projection mismatch: IoU={overlap:.3f} (min {minimum_iou:.3f}), "
            f"center={center_error:.3f} (max {maximum_center_error:.3f})"
        )
    )
    return {
        "valid": valid,
        "reason": reason,
        "projected_box_xyxy": projected.tolist(),
        "observed_box_xyxy": observation.box_xyxy.tolist(),
        "box_iou": overlap,
        "center_error_ratio": center_error,
    }


def resize_mask(mask_tensor: Any, shape: tuple[int, int]) -> np.ndarray:
    mask = mask_tensor.detach().float().cpu().numpy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"YOLO 掩码维度错误: {mask.shape}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(mask > 0.5)


def resolve_target_class(names: Any, requested: str) -> int:
    if isinstance(names, dict):
        normalized = {int(key): str(value) for key, value in names.items()}
    else:
        normalized = {index: str(value) for index, value in enumerate(names)}
    try:
        class_id = int(requested)
    except ValueError:
        matches = [
            class_id
            for class_id, name in normalized.items()
            if name.casefold() == str(requested).casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"YOLO 中找不到唯一类别 {requested!r}；可用类别为 {normalized}"
            )
        class_id = matches[0]
    if class_id not in normalized:
        raise ValueError(f"YOLO 中没有类别编号 {class_id}；可用类别为 {normalized}")
    return class_id


def detect_target(
    model: Any,
    color_bgr: np.ndarray,
    target_class_id: int,
    args: argparse.Namespace,
) -> TargetObservation | None:
    result = model.predict(
        color_bgr,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.masks is None:
        raise RuntimeError(
            "YOLO 返回了检测框但没有实例掩码；请使用 *-seg.pt 分割模型，而不是检测模型"
        )

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = result.boxes.conf.detach().cpu().numpy()
    candidates = np.flatnonzero(class_ids == target_class_id)
    if candidates.size == 0:
        return None
    selected = int(candidates[np.argmax(confidences[candidates])])
    mask = resize_mask(result.masks.data[selected], color_bgr.shape[:2])
    if int(np.count_nonzero(mask)) < args.min_mask_pixels:
        return None

    names = model.names
    class_name = str(names[target_class_id])
    return TargetObservation(
        class_id=target_class_id,
        class_name=class_name,
        confidence=float(confidences[selected]),
        box_xyxy=result.boxes.xyxy[selected].detach().cpu().numpy().astype(np.float64),
        mask=mask,
    )


def filter_depth_for_object(
    depth_m: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if depth_m.shape != mask.shape:
        raise ValueError(f"深度和掩码尺寸不同: {depth_m.shape} 与 {mask.shape}")
    filtered = depth_m.astype(np.float32, copy=True)
    globally_valid = (
        np.isfinite(filtered)
        & (filtered >= args.min_depth)
        & (filtered <= args.max_depth)
    )
    filtered[~globally_valid] = 0.0
    object_mask = mask.astype(bool)
    valid_in_mask = object_mask & globally_valid
    valid_before = int(np.count_nonzero(valid_in_mask))
    if valid_before < args.min_valid_depth_pixels:
        raise RuntimeError(
            f"目标掩码内只有 {valid_before} 个有效深度像素，"
            f"至少需要 {args.min_valid_depth_pixels} 个"
        )

    eroded = cv2.erode(
        object_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    reference_mask = eroded & globally_valid
    if int(np.count_nonzero(reference_mask)) < 20:
        reference_mask = valid_in_mask
    values = filtered[reference_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    band = min(args.depth_max_band, max(args.depth_band, 4.0 * 1.4826 * mad))
    rejected = valid_in_mask & (np.abs(filtered - median) > band)
    filtered[rejected] = 0.0
    valid_after = valid_before - int(np.count_nonzero(rejected))
    if valid_after < args.min_valid_depth_pixels:
        raise RuntimeError(
            f"深度去离群后只剩 {valid_after} 个目标深度像素，无法可靠注册"
        )
    metadata = {
        "mask_pixels": int(np.count_nonzero(object_mask)),
        "valid_before": valid_before,
        "valid_after": valid_after,
        "median_depth_m": median,
        "median_absolute_deviation_m": mad,
        "effective_band_m": float(band),
        "removed_pixels": valid_before - valid_after,
    }
    return filtered, metadata


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("四元数长度为零")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def rotation_matrix_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return XYZ roll/pitch/yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    horizontal = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if horizontal > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], horizontal)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], horizontal)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def elf3_world_camera_pose(head: HeadSample, odom: OdomSample) -> np.ndarray:
    """Analytic FK for the head camera in the current ELF3 MuJoCo XML."""
    world_R_base = quaternion_xyzw_to_matrix(odom.quaternion_xyzw)
    world_R_head = world_R_base @ rotation_z(head.yaw_rad) @ rotation_y(head.pitch_rad)

    torso_to_head = np.array([0.0, 0.0, 0.2995], dtype=np.float64)
    head_to_camera = np.array([0.0628, 0.0175, -0.048], dtype=np.float64)
    world_p_camera = (
        odom.position
        + world_R_base @ torso_to_head
        + world_R_head @ head_to_camera
    )

    # MuJoCo camera local: +X right, +Y up, -Z forward.
    # OpenCV optical: +X right, +Y down, +Z forward.
    head_R_optical = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = world_R_head @ head_R_optical
    transform[:3, 3] = world_p_camera
    return transform


def make_colored_mesh_part(mesh: Any, rgba: tuple[int, int, int, int]) -> Any:
    mesh.visual.vertex_colors = np.tile(
        np.asarray(rgba, dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    return mesh


def build_mujoco_bottle_mesh(trimesh_module: Any) -> Any:
    """Build the five visible primitives used by universal_test/bottle.xml."""
    translation = trimesh_module.transformations.translation_matrix
    blue = (20, 77, 173, 255)
    white = (235, 235, 224, 255)

    body = trimesh_module.creation.cylinder(radius=0.040, height=0.170, sections=64)
    body.apply_transform(translation([0.0, 0.0, 0.085]))
    shoulder = trimesh_module.creation.icosphere(subdivisions=3, radius=1.0)
    shoulder.apply_scale([0.040, 0.040, 0.025])
    shoulder.apply_transform(translation([0.0, 0.0, 0.177]))
    neck = trimesh_module.creation.cylinder(radius=0.018, height=0.050, sections=64)
    neck.apply_transform(translation([0.0, 0.0, 0.210]))
    cap = trimesh_module.creation.cylinder(radius=0.021, height=0.020, sections=64)
    cap.apply_transform(translation([0.0, 0.0, 0.243]))
    label = trimesh_module.creation.cylinder(radius=0.041, height=0.054, sections=64)
    label.apply_transform(translation([0.0, 0.0, 0.095]))

    parts = [
        make_colored_mesh_part(body, blue),
        make_colored_mesh_part(shoulder, blue),
        make_colored_mesh_part(neck, blue),
        make_colored_mesh_part(cap, white),
        make_colored_mesh_part(label, white),
    ]
    mesh = trimesh_module.util.concatenate(parts)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("程序化水瓶网格为空")
    return mesh


def load_mesh(mesh_argument: str, trimesh_module: Any) -> tuple[Any, str, bool]:
    if str(mesh_argument).casefold() == "auto":
        return build_mujoco_bottle_mesh(trimesh_module), "procedural:bottle.xml", True
    path = Path(mesh_argument).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CAD 网格不存在: {path}")
    loaded = trimesh_module.load(str(path), process=False)
    if isinstance(loaded, trimesh_module.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"CAD Scene 中没有几何体: {path}")
        loaded = trimesh_module.util.concatenate(geometries)
    if not isinstance(loaded, trimesh_module.Trimesh):
        raise TypeError(f"CAD 不是 Trimesh: {path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"CAD 网格为空: {path}")
    extents = np.asarray(loaded.extents, dtype=np.float64)
    if not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        raise ValueError(f"CAD 尺寸无效: {extents.tolist()}")
    if float(np.max(extents)) > 5.0 or float(np.min(extents)) < 1e-4:
        raise ValueError(
            f"CAD 尺寸 {extents.tolist()} 看起来不是米制；"
            "FoundationPose 和 MuJoCo 均要求网格顶点单位为米"
        )
    return loaded, str(path), False


def axial_symmetry_transforms(steps: int) -> np.ndarray | None:
    if steps <= 1:
        return None
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], steps, axis=0)
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)):
        transforms[index, :3, :3] = rotation_z(float(angle)).astype(np.float32)
    return transforms


def install_foundationpose_opencv_compat(*modules: Any) -> None:
    """Allow FoundationPose debug text helpers to accept float images on OpenCV 5."""
    for module in modules:
        original = module.cv_draw_text
        if getattr(original, "_foundationpose_uint8_compat", False):
            continue

        def cv_draw_text_uint8(image: np.ndarray, *values: Any, _original=original, **kwargs: Any):
            if image.dtype != np.uint8:
                image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
            return _original(image, *values, **kwargs)

        cv_draw_text_uint8._foundationpose_uint8_compat = True  # type: ignore[attr-defined]
        module.cv_draw_text = cv_draw_text_uint8


def load_foundation_runtime(args: argparse.Namespace) -> FoundationRuntime:
    yolo_path = args.yolo_model.expanduser().resolve()
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO-seg 权重不存在: {yolo_path}")
    if not (REPO_ROOT / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose 根目录不正确: {REPO_ROOT}")

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for path in (REPO_ROOT, REPO_ROOT / "mycpp/build"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import nvdiffrast.torch as dr
        import torch
        import trimesh
        from ultralytics import YOLO
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        from learning.training import predict_pose_refine, predict_score
        from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
    except Exception as error:
        raise RuntimeError(f"无法加载 YOLO/FoundationPose 依赖: {error}") from error

    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose 必须使用 NVIDIA CUDA，但当前 Python 看不到 GPU")
    device_text = str(args.device).strip().lower()
    try:
        if device_text.isdigit():
            torch.cuda.set_device(int(device_text))
        elif device_text.startswith("cuda:"):
            torch.cuda.set_device(int(device_text.split(":", 1)[1]))
    except (ValueError, RuntimeError) as error:
        raise RuntimeError(f"无效 CUDA 设备 {args.device!r}: {error}") from error

    set_logging_format()
    set_seed(0)
    install_foundationpose_opencv_compat(predict_pose_refine, predict_score)

    print(f"加载 YOLO-seg: {yolo_path}", flush=True)
    yolo = YOLO(str(yolo_path))
    if str(getattr(yolo, "task", "")).casefold() != "segment":
        raise ValueError(
            f"权重任务是 {getattr(yolo, 'task', None)!r}，必须提供 YOLO segmentation 权重"
        )
    target_class_id = resolve_target_class(yolo.names, str(args.target_class))
    print(f"YOLO 类别: {yolo.names}; 目标={target_class_id}:{yolo.names[target_class_id]}")

    mesh, mesh_source, procedural = load_mesh(args.mesh, trimesh)
    extents_actual = np.asarray(mesh.extents, dtype=np.float64)
    print(
        f"CAD: {mesh_source}; 尺寸={np.array2string(extents_actual, precision=4)} m; "
        f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}",
        flush=True,
    )
    symmetry_steps = args.symmetry_steps
    if symmetry_steps is None:
        symmetry_steps = 36 if procedural else 1
    symmetry_tfs = axial_symmetry_transforms(symmetry_steps)
    if symmetry_steps > 1:
        print(
            f"物体按绕 Z 轴 {symmetry_steps} 重对称处理；瓶身轴向 yaw 不可唯一观测",
            flush=True,
        )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack((-extents / 2.0, extents / 2.0), axis=0).reshape(2, 3)
    inverse_to_origin = np.linalg.inv(to_origin)

    output_dir = args.output.expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if procedural:
        mesh.export(str(output_dir / "mujoco_bottle_mesh.obj"))

    print("加载 FoundationPose scorer/refiner...", flush=True)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        symmetry_tfs=symmetry_tfs,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=args.debug,
        debug_dir=str(output_dir / "foundationpose_debug"),
        glctx=glctx,
    )
    print("FoundationPose 初始化完成", flush=True)
    return FoundationRuntime(
        yolo=yolo,
        target_class_id=target_class_id,
        estimator=estimator,
        mesh=mesh,
        mesh_source=mesh_source,
        symmetry_steps=symmetry_steps,
        yaw_observable=symmetry_steps <= 1,
        bbox=bbox,
        inverse_to_origin=inverse_to_origin,
        draw_box=draw_posed_3d_box,
        draw_axis=draw_xyz_axis,
    )


def validate_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(pose)):
        raise RuntimeError("FoundationPose 返回了 NaN/Inf")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise RuntimeError(f"FoundationPose 返回矩阵最后一行异常: {pose[3].tolist()}")
    determinant = float(np.linalg.det(pose[:3, :3]))
    if not 0.8 <= determinant <= 1.2:
        raise RuntimeError(f"FoundationPose 旋转矩阵 determinant={determinant:.5f}")
    return pose


def pose_fields(transform: np.ndarray) -> dict[str, Any]:
    return {
        "matrix": transform.tolist(),
        "translation_m": transform[:3, 3].tolist(),
        "quaternion_xyzw": rotation_matrix_to_quaternion_xyzw(transform[:3, :3]).tolist(),
        "rpy_degrees": rotation_matrix_to_rpy_degrees(transform[:3, :3]).tolist(),
    }


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    os.replace(temporary, path)


def make_pipeline_visualization(
    packet: CameraPacket,
    observation: TargetObservation | None,
    stage: str,
    *,
    head: HeadSample | None = None,
    pose: np.ndarray | None = None,
    runtime: FoundationRuntime | None = None,
    pose_quality: dict[str, Any] | None = None,
    detail: str | None = None,
) -> np.ndarray:
    """Build one BGR frame for every YOLO/FoundationPose pipeline stage."""
    visual = packet.color_bgr.copy()
    height, width = visual.shape[:2]

    if observation is not None:
        if observation.mask.shape != (height, width):
            raise ValueError(
                f"YOLO mask shape {observation.mask.shape} does not match "
                f"the color image {(height, width)}"
            )
        overlay = visual.copy()
        overlay[observation.mask] = np.array([40, 210, 40], dtype=np.uint8)
        visual = cv2.addWeighted(visual, 0.72, overlay, 0.28, 0.0)
        contours, _ = cv2.findContours(
            observation.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(visual, contours, -1, (40, 255, 40), 2)
        x1, y1, x2, y2 = np.rint(observation.box_xyxy).astype(int)
        x1, x2 = np.clip((x1, x2), 0, max(width - 1, 0))
        y1, y2 = np.clip((y1, y2), 0, max(height - 1, 0))
        cv2.rectangle(visual, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        label = f"{observation.class_name} {observation.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_y = int(y1) - 8
        if label_y - label_size[1] < 0:
            label_y = min(height - 4, int(y1) + label_size[1] + 8)
        label_left = int(x1)
        label_right = min(width - 1, label_left + label_size[0] + 10)
        cv2.rectangle(
            visual,
            (label_left, max(0, label_y - label_size[1] - 7)),
            (label_right, min(height - 1, label_y + 4)),
            (0, 190, 190),
            -1,
        )
        cv2.putText(
            visual,
            label,
            (label_left + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

    if pose is not None and runtime is not None:
        color_rgb = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
        center_pose = pose @ runtime.inverse_to_origin
        try:
            color_rgb = runtime.draw_box(
                packet.color_info.K,
                img=color_rgb,
                ob_in_cam=center_pose,
                bbox=runtime.bbox,
            )
            color_rgb = runtime.draw_axis(
                color_rgb,
                # The OBB uses its centered frame; XYZ uses the original CAD
                # frame (the procedural bottle's bottom-center).
                ob_in_cam=pose,
                scale=max(0.05, float(np.max(runtime.mesh.extents)) * 0.6),
                K=packet.color_info.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
        except Exception as error:
            print(f"警告：位姿可视化失败，但数值结果已保留: {error}", file=sys.stderr)
        visual = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    lines = [f"Status: {stage}"]
    if observation is None:
        lines.append("YOLO-seg: searching for target")
    else:
        lines.append(
            f"YOLO-seg: {observation.class_name}  conf={observation.confidence:.3f}  "
            f"mask={int(np.count_nonzero(observation.mask))} px"
        )
    if head is not None:
        lines.append(
            f"Head: yaw={math.degrees(head.yaw_rad):+.2f} deg  "
            f"pitch={math.degrees(head.pitch_rad):+.2f} deg"
        )
    if pose is not None:
        xyz = pose[:3, 3]
        rpy = rotation_matrix_to_rpy_degrees(pose[:3, :3])
        lines.append(
            f"T_camera_object xyz [m]: {xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}"
        )
        quality_text = ""
        if pose_quality is not None and "box_iou" in pose_quality:
            quality_text = f"  IoU={float(pose_quality['box_iou']):.3f}"
        lines.append(
            f"Object RPY [deg]: {rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}{quality_text}"
        )
        if runtime is not None and not runtime.yaw_observable:
            lines.append("Symmetry: rotation about object Z is ambiguous")
    if detail:
        lines.append(detail)

    font_scale = max(0.48, min(0.72, width / 1800.0))
    line_height = max(24, int(round(32 * font_scale / 0.72)))
    panel_height = min(height, 18 + line_height * len(lines))
    panel = visual[:panel_height].copy()
    panel[:] = 0
    visual[:panel_height] = cv2.addWeighted(
        visual[:panel_height], 0.30, panel, 0.70, 0.0
    )
    status_color = (
        (80, 255, 80)
        if "POSE OK" in stage
        else ((80, 80, 255) if "FAILED" in stage else (0, 220, 255))
    )
    for index, line in enumerate(lines):
        color = status_color if index == 0 else (245, 245, 245)
        cv2.putText(
            visual,
            line,
            (16, 10 + line_height * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        visual,
        "Q / Esc: exit",
        (max(16, width - 155), max(20, height - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(visual, dtype=np.uint8)


def make_visualization(
    packet: CameraPacket,
    observation: TargetObservation,
    pose: np.ndarray,
    runtime: FoundationRuntime,
    head: HeadSample,
    pose_quality: dict[str, Any] | None = None,
) -> np.ndarray:
    return make_pipeline_visualization(
        packet,
        observation,
        "FOUNDATIONPOSE POSE OK",
        head=head,
        pose=pose,
        runtime=runtime,
        pose_quality=pose_quality,
    )


def open_visualization_window() -> bool:
    """Create HighGUI safely; Qt may abort before raising if no display exists."""
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "警告：当前没有 DISPLAY/WAYLAND_DISPLAY，已禁用 OpenCV 弹窗；"
            "结果图片仍会保存。",
            file=sys.stderr,
            flush=True,
        )
        return False
    gui_line = next(
        (line.strip() for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:")),
        "",
    )
    if gui_line.upper().endswith("NONE"):
        print("警告：当前 OpenCV 没有 GUI 后端，已禁用弹窗。", file=sys.stderr, flush=True)
        return False

    # A stale DISPLAY or an unavailable Qt platform plugin makes Qt call
    # abort(), which cannot be caught as cv2.error. Probe HighGUI in a child
    # process so a broken desktop connection cannot terminate pose tracking.
    probe_code = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import cv2
import numpy as np
name = '__mujoco_pose_highgui_probe__'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.imshow(name, np.zeros((32, 32, 3), dtype=np.uint8))
cv2.waitKey(20)
cv2.destroyAllWindows()
"""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            f"警告：无法验证 OpenCV 图形界面 ({error})，已禁用弹窗；"
            "结果图片仍会保存。",
            file=sys.stderr,
            flush=True,
        )
        return False
    if probe.returncode != 0:
        print(
            "警告：DISPLAY/WAYLAND_DISPLAY 存在，但 OpenCV 无法连接图形界面；"
            "已禁用弹窗，结果图片仍会保存。请在本地图形终端运行，或检查 "
            "DISPLAY/XAUTHORITY。",
            file=sys.stderr,
            flush=True,
        )
        return False
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            waiting,
            "YOLO-seg + FoundationPose",
            (55, 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            waiting,
            "Waiting for synchronized MuJoCo RGB-D...",
            (55, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, waiting)
        cv2.waitKey(1)
    except cv2.error as error:
        raise RuntimeError(f"无法创建 OpenCV 窗口: {error}") from error
    print("OpenCV 实时窗口已开启；按 Q 或 Esc 退出。", flush=True)
    return True


def poll_visualization_window(delay_ms: int = 1) -> bool:
    """Pump HighGUI events and report whether the user kept the window open."""
    key = cv2.waitKey(max(1, int(delay_ms))) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        return False
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1.0
    except cv2.error:
        return False


def show_visualization_frame(visual: np.ndarray, *, block: bool = False) -> bool:
    """Show a frame and return False when Q/Esc or the window close button is used."""
    try:
        cv2.imshow(WINDOW_NAME, visual)
    except cv2.error:
        return False
    if not block:
        return poll_visualization_window(1)
    while poll_visualization_window(50):
        pass
    return False


def save_pose_result(
    args: argparse.Namespace,
    runtime: FoundationRuntime,
    packet: CameraPacket,
    observation: TargetObservation,
    head: HeadSample,
    head_sync_offset_s: float | None,
    depth_metadata: dict[str, Any],
    pose_quality: dict[str, Any],
    aligned_depth_m: np.ndarray,
    camera_pose: np.ndarray,
    odom: OdomSample | None,
    odom_sync_offset_s: float | None,
    sequence: int,
) -> tuple[dict[str, Any], np.ndarray]:
    seconds, nanoseconds = divmod(packet.stamp_ns, 1_000_000_000)
    record: dict[str, Any] = {
        "sequence": sequence,
        "timestamp": {"sec": seconds, "nanosec": nanoseconds},
        "target": {
            "class_id": observation.class_id,
            "class_name": observation.class_name,
            "confidence": observation.confidence,
            "box_xyxy": observation.box_xyxy.tolist(),
            "mask_pixels": int(np.count_nonzero(observation.mask)),
        },
        "head": {
            "yaw_joint": "head_z_joint",
            "pitch_joint": "head_y_joint",
            "yaw_rad": head.yaw_rad,
            "pitch_rad": head.pitch_rad,
            "yaw_deg": math.degrees(head.yaw_rad),
            "pitch_deg": math.degrees(head.pitch_rad),
            "yaw_velocity_rad_s": head.yaw_velocity,
            "pitch_velocity_rad_s": head.pitch_velocity,
            "image_state_time_offset_s": head_sync_offset_s,
        },
        "camera": {
            "frame_id": packet.color_info.frame_id,
            "K": packet.color_info.K.tolist(),
            "color_size": [packet.color_bgr.shape[1], packet.color_bgr.shape[0]],
            "native_depth_size": [packet.depth_raw.shape[1], packet.depth_raw.shape[0]],
        },
        "depth_filter": depth_metadata,
        "pose_quality": pose_quality,
        "mesh": {
            "source": runtime.mesh_source,
            "extents_m": np.asarray(runtime.mesh.extents, dtype=float).tolist(),
            "object_origin": "CAD origin; procedural mesh uses bottle bottom-center",
        },
        "T_camera_object": pose_fields(camera_pose),
        "pose_convention": (
            "T_camera_object maps original CAD/object coordinates into the OpenCV color "
            "optical frame (+X right, +Y down, +Z forward); translation is metres"
        ),
        "symmetry": {
            "axis": "object +Z",
            "steps": runtime.symmetry_steps,
            "yaw_observable": runtime.yaw_observable,
            "note": (
                "Rotation about object +Z is not unique for this declared axial symmetry."
                if not runtime.yaw_observable
                else "No axial symmetry was declared; yaw observability depends on the supplied CAD and RGB-D evidence."
            ),
        },
        "pipeline_config": {
            "yolo_model": str(args.yolo_model.expanduser().resolve()),
            "target_class": str(args.target_class),
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "image_size": args.imgsz,
            "est_refine_iterations": args.est_refine_iter,
            "track_refine_iterations": args.track_refine_iter,
            "minimum_pose_box_iou": args.min_pose_box_iou,
            "maximum_pose_center_error_ratio": args.max_pose_center_error,
        },
    }

    if args.world_pose and odom is not None:
        world_camera = elf3_world_camera_pose(head, odom)
        world_object = world_camera @ camera_pose
        record["T_world_camera"] = pose_fields(world_camera)
        record["T_world_object"] = pose_fields(world_object)
        record["world_pose_source"] = {
            "frame_id": odom.frame_id,
            "base_child_frame_id": odom.child_frame_id,
            "image_odom_time_offset_s": odom_sync_offset_s,
            "kinematics": "ELF3 MuJoCo analytic FK from odom + head_z_joint + head_y_joint",
        }

    output_path = args.output.expanduser().resolve()
    write_json_atomic(output_path, record)
    visual = make_visualization(
        packet,
        observation,
        camera_pose,
        runtime,
        head,
        pose_quality,
    )
    if args.save_images:
        prefix = output_path.with_suffix("")
        mask_path = Path(str(prefix) + "_mask.png")
        depth_path = Path(str(prefix) + "_aligned_depth_mm.png")
        preview_path = Path(str(prefix) + "_preview.png")
        cv2.imwrite(str(mask_path), observation.mask.astype(np.uint8) * 255)
        depth_mm = np.clip(aligned_depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)
        cv2.imwrite(str(preview_path), visual)

    print("\n" + "=" * 78)
    print(
        f"目标: {observation.class_name}  confidence={observation.confidence:.3f}  "
        f"stamp={seconds}.{nanoseconds:09d}"
    )
    print(
        f"真实头部角度: yaw(head_z)={head.yaw_rad:+.5f} rad "
        f"({math.degrees(head.yaw_rad):+.2f} deg), "
        f"pitch(head_y)={head.pitch_rad:+.5f} rad "
        f"({math.degrees(head.pitch_rad):+.2f} deg)"
    )
    print("T_camera_object [m]:")
    print(np.array2string(camera_pose, precision=6, suppress_small=True))
    print(f"camera translation xyz [m]: {camera_pose[:3, 3].tolist()}")
    print(
        "camera RPY [deg]: "
        f"{rotation_matrix_to_rpy_degrees(camera_pose[:3, :3]).tolist()} "
        + (
            "（配置了轴对称，瓶身自身 Z 轴旋转不唯一）"
            if not runtime.yaw_observable
            else ""
        )
    )
    if "T_world_object" in record:
        world_matrix = np.asarray(record["T_world_object"]["matrix"])
        print("T_world_object [m]:")
        print(np.array2string(world_matrix, precision=6, suppress_small=True))
    else:
        print("未获得同步 odom，因此本次只输出相机坐标系位姿。")
    print(f"结果已写入: {output_path}", flush=True)
    return record, visual


def run_self_test() -> int:
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    depth_K = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])
    color_K = np.array([[8.0, 0.0, 3.5], [0.0, 8.0, 2.5], [0.0, 0.0, 1.0]])
    aligned = align_depth_to_color(depth, depth_K, color_K, (6, 8))
    if aligned.shape != (6, 8) or np.count_nonzero(aligned == 2.0) < 30:
        raise AssertionError("depth alignment self-test failed")
    identity = quaternion_xyzw_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
    if not np.allclose(identity, np.eye(3)):
        raise AssertionError("quaternion self-test failed")
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation_z(0.4))
    if not np.allclose(quaternion_xyzw_to_matrix(quaternion), rotation_z(0.4), atol=1e-7):
        raise AssertionError("rotation round-trip self-test failed")
    synthetic_mesh = type(
        "SyntheticMesh",
        (),
        {"vertices": np.array([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]])},
    )()
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 2.0
    test_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    projected = project_mesh_bbox(test_K, pose, synthetic_mesh.vertices)
    if projected is None:
        raise AssertionError("pose projection self-test failed")
    observation = TargetObservation(
        class_id=0,
        class_name="test",
        confidence=1.0,
        box_xyxy=projected,
        mask=np.ones((2, 2), dtype=bool),
    )
    quality = assess_pose_quality(test_K, pose, synthetic_mesh, observation, 0.3, 0.3)
    if not quality["valid"]:
        raise AssertionError("pose quality self-test failed")
    print("自检通过：RGB-D 内参对齐、四元数和旋转变换正常。")
    return 0


def run(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile,
            ReliabilityPolicy,
            DurabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import CameraInfo, Image
        from nav_msgs.msg import Odometry
        from communication.msg import ActuatorCmds, ActuatorStates
    except Exception as error:
        raise RuntimeError(
            "无法导入 ROS 2 或 communication.msg。请在当前终端依次执行：\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /opt/bxi/bxi_ros2_pkg/setup.bash\n"
            "  source /home/robot/FoundationPose/"
            "bxi_rl_controller_ros2_example/install/setup.bash\n"
            f"然后重新运行本程序。原始错误: {error}"
        ) from error

    class MujocoPoseNode(Node):
        def __init__(self) -> None:
            super().__init__("mujoco_foundationpose")
            self._parts: dict[int, dict[str, Any]] = {}
            self._latest_color_info: CameraInfoData | None = None
            self._latest_depth_info: CameraInfoData | None = None
            self._packets: deque[CameraPacket] = deque(maxlen=2)
            self._completed_stamps: deque[int] = deque(maxlen=30)
            self.head_samples: deque[HeadSample] = deque(maxlen=600)
            self.command_samples: deque[CommandSample] = deque(maxlen=600)
            self.odom_samples: deque[OdomSample] = deque(maxlen=300)
            self.received = {
                "rgb": 0,
                "depth": 0,
                "color_info": 0,
                "depth_info": 0,
                "actuator": 0,
                "command": 0,
                "odom": 0,
            }
            self._last_decode_error_at = float("-inf")
            self._data_lock = threading.RLock()
            self.head_command_publisher = self.create_publisher(
                ActuatorCmds,
                args.head_command_topic,
                qos_profile_sensor_data,
            )
            self._subscriptions = [
                self.create_subscription(Image, args.rgb_topic, self._rgb_callback, qos_profile_sensor_data),
                self.create_subscription(Image, args.depth_topic, self._depth_callback, qos_profile_sensor_data),
                self.create_subscription(
                    CameraInfo,
                    args.color_info_topic,
                    self._color_info_callback,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    CameraInfo,
                    args.depth_info_topic,
                    self._depth_info_callback,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    ActuatorStates,
                    args.actuator_state_topic,
                    self._actuator_callback,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    ActuatorCmds,
                    args.head_command_topic,
                    self._command_callback,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    Odometry,
                    args.odom_topic,
                    self._odom_callback,
                    qos_profile_sensor_data,
                ),
            ]
            # Camera QoS changed between MuJoCo ROS releases.  Subscribe with
            # a reliable fallback as well as the standard sensor-data profile;
            # DDS will activate whichever is compatible with the publisher.
            reliable_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._subscriptions.extend(
                (
                    self.create_subscription(Image, args.rgb_topic, self._rgb_callback, reliable_qos),
                    self.create_subscription(Image, args.depth_topic, self._depth_callback, reliable_qos),
                    self.create_subscription(CameraInfo, args.color_info_topic, self._color_info_callback, reliable_qos),
                    self.create_subscription(CameraInfo, args.depth_info_topic, self._depth_info_callback, reliable_qos),
                )
            )

        def publish_scan_command(self, yaw_rad: float, pitch_rad: float = 0.0) -> None:
            command = ActuatorCmds()
            command.header.stamp = self.get_clock().now().to_msg()
            command.actuators_name = ["head_z_joint", "head_y_joint"]
            command.pos = [float(yaw_rad), float(pitch_rad)]
            command.kp = [float(args.head_kp), float(args.head_kp)]
            command.kd = [float(args.head_kd), float(args.head_kd)]
            self.head_command_publisher.publish(command)

        def _warn_decode(self, label: str, error: Exception) -> None:
            now = time.monotonic()
            if now - self._last_decode_error_at >= 3.0:
                self.get_logger().warning(f"{label} 消息解析失败: {error}")
                self._last_decode_error_at = now

        def _rgb_callback(self, message: Any) -> None:
            try:
                stamp = stamp_to_ns(message.header.stamp)
                image = image_message_to_array(message, color=True)
                with self._data_lock:
                    self._parts.setdefault(stamp, {})["rgb"] = image
                    self.received["rgb"] += 1
                    self._try_make_packet(stamp)
                    self._prune_parts()
            except Exception as error:
                self._warn_decode("RGB", error)

        def _depth_callback(self, message: Any) -> None:
            try:
                stamp = stamp_to_ns(message.header.stamp)
                image = image_message_to_array(message, color=False)
                with self._data_lock:
                    self._parts.setdefault(stamp, {})["depth"] = image
                    self.received["depth"] += 1
                    self._try_make_packet(stamp)
                    self._prune_parts()
            except Exception as error:
                self._warn_decode("Depth", error)

        def _color_info_callback(self, message: Any) -> None:
            try:
                info = camera_info_from_message(message)
                with self._data_lock:
                    self._latest_color_info = info
                    self._parts.setdefault(info.stamp_ns, {})["color_info"] = info
                    self.received["color_info"] += 1
                    self._try_make_packet(info.stamp_ns)
                    self._try_pending_packets()
            except Exception as error:
                self._warn_decode("Color CameraInfo", error)

        def _depth_info_callback(self, message: Any) -> None:
            try:
                info = camera_info_from_message(message)
                with self._data_lock:
                    self._latest_depth_info = info
                    self._parts.setdefault(info.stamp_ns, {})["depth_info"] = info
                    self.received["depth_info"] += 1
                    self._try_make_packet(info.stamp_ns)
                    self._try_pending_packets()
            except Exception as error:
                self._warn_decode("Depth CameraInfo", error)

        def _try_pending_packets(self) -> None:
            for stamp in tuple(self._parts):
                self._try_make_packet(stamp)

        def _prune_parts(self) -> None:
            while len(self._parts) > 30:
                self._parts.pop(next(iter(self._parts)))

        def _try_make_packet(self, stamp: int) -> None:
            if stamp in self._completed_stamps:
                return
            parts = self._parts.get(stamp)
            if not parts or "rgb" not in parts or "depth" not in parts:
                return
            color_info = parts.get("color_info")
            depth_info = parts.get("depth_info")
            if color_info is None and self._latest_color_info is not None:
                if stamp == 0 or self._latest_color_info.stamp_ns in (0, stamp):
                    color_info = self._latest_color_info
            if depth_info is None and self._latest_depth_info is not None:
                if stamp == 0 or self._latest_depth_info.stamp_ns in (0, stamp):
                    depth_info = self._latest_depth_info
            if color_info is None or depth_info is None:
                return
            color_image = parts["rgb"]
            depth_image = parts["depth"]
            if color_image.shape[:2] != (color_info.height, color_info.width):
                raise ValueError(
                    f"RGB {color_image.shape[:2]} 与 CameraInfo "
                    f"{(color_info.height, color_info.width)} 不一致"
                )
            if depth_image.shape != (depth_info.height, depth_info.width):
                raise ValueError(
                    f"Depth {depth_image.shape} 与 CameraInfo "
                    f"{(depth_info.height, depth_info.width)} 不一致"
                )
            packet = CameraPacket(
                stamp_ns=stamp,
                color_bgr=color_image,
                depth_raw=depth_image,
                color_info=color_info,
                depth_info=depth_info,
            )
            self._packets.append(packet)
            self._completed_stamps.append(stamp)
            self._parts.pop(stamp, None)

        def _actuator_callback(self, message: Any) -> None:
            try:
                indices = {str(name): index for index, name in enumerate(message.name)}
                yaw_index = indices["head_z_joint"]
                pitch_index = indices["head_y_joint"]
                positions = np.array(
                    [message.position[yaw_index], message.position[pitch_index]],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(positions)):
                    raise ValueError(f"头部位置不是有限数: {positions.tolist()}")
                yaw_velocity = (
                    float(message.velocity[yaw_index])
                    if yaw_index < len(message.velocity)
                    else float("nan")
                )
                pitch_velocity = (
                    float(message.velocity[pitch_index])
                    if pitch_index < len(message.velocity)
                    else float("nan")
                )
                with self._data_lock:
                    self.head_samples.append(
                        HeadSample(
                            stamp_ns=stamp_to_ns(message.header.stamp),
                            received_at=time.monotonic(),
                            yaw_rad=float(positions[0]),
                            pitch_rad=float(positions[1]),
                            yaw_velocity=yaw_velocity,
                            pitch_velocity=pitch_velocity,
                        )
                    )
                    self.received["actuator"] += 1
            except (KeyError, IndexError, ValueError) as error:
                self._warn_decode("ActuatorStates", error)

        def _command_callback(self, message: Any) -> None:
            try:
                indices = {
                    str(name): index for index, name in enumerate(message.actuators_name)
                }
                yaw_index = indices["head_z_joint"]
                pitch_index = indices["head_y_joint"]
                positions = np.array(
                    [message.pos[yaw_index], message.pos[pitch_index]], dtype=np.float64
                )
                if not np.all(np.isfinite(positions)):
                    raise ValueError(f"头部命令不是有限数: {positions.tolist()}")
                with self._data_lock:
                    self.command_samples.append(
                        CommandSample(
                            stamp_ns=stamp_to_ns(message.header.stamp),
                            received_at=time.monotonic(),
                            yaw_rad=float(positions[0]),
                            pitch_rad=float(positions[1]),
                        )
                    )
                    self.received["command"] += 1
            except (KeyError, IndexError, ValueError):
                # Other override publishers may command unrelated joints.
                return

        def _odom_callback(self, message: Any) -> None:
            pose = message.pose.pose
            sample = OdomSample(
                stamp_ns=stamp_to_ns(message.header.stamp),
                received_at=time.monotonic(),
                frame_id=str(message.header.frame_id),
                child_frame_id=str(message.child_frame_id),
                position=np.array(
                    [pose.position.x, pose.position.y, pose.position.z],
                    dtype=np.float64,
                ),
                quaternion_xyzw=np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                    dtype=np.float64,
                ),
            )
            with self._data_lock:
                self.odom_samples.append(sample)
                self.received["odom"] += 1

        def pop_latest_packet(self) -> CameraPacket | None:
            with self._data_lock:
                if not self._packets:
                    return None
                packet = self._packets[-1]
                self._packets.clear()
                return packet

        def head_stability(self) -> tuple[bool, str]:
            if args.ignore_head_stability:
                return True, "ignored by --ignore-head-stability"
            now = time.monotonic()
            with self._data_lock:
                head_samples = tuple(self.head_samples)
                command_samples = tuple(self.command_samples)
            if not head_samples:
                return False, f"等待 {args.actuator_state_topic}"
            if now - head_samples[-1].received_at > 0.5:
                return False, "头部关节反馈已超时"
            start = now - args.head_stable_duration
            head_window = [sample for sample in head_samples if sample.received_at >= start]
            if len(head_window) < 3 or head_window[-1].received_at - head_window[0].received_at < 0.8 * args.head_stable_duration:
                return False, "正在积累真实关节稳定窗口"
            yaws = np.asarray([sample.yaw_rad for sample in head_window])
            pitches = np.asarray([sample.pitch_rad for sample in head_window])
            if float(np.ptp(yaws)) > args.head_position_tolerance:
                return False, "head_z_joint 仍在转动"
            if float(np.ptp(pitches)) > args.head_position_tolerance:
                return False, "head_y_joint 仍在转动"
            velocities = np.asarray(
                [
                    value
                    for sample in head_window
                    for value in (sample.yaw_velocity, sample.pitch_velocity)
                    if math.isfinite(value)
                ],
                dtype=np.float64,
            )
            if velocities.size and float(np.max(np.abs(velocities))) > args.head_velocity_tolerance:
                return False, "真实头部速度尚未降到阈值"

            if args.require_held_command:
                if not command_samples:
                    return False, f"等待 head_control 命令 {args.head_command_topic}"
                if now - command_samples[-1].received_at > 0.5:
                    return False, "head_control 命令已超时"
                command_window = [
                    sample for sample in command_samples if sample.received_at >= start
                ]
                if (
                    len(command_window) < 3
                    or command_window[-1].received_at - command_window[0].received_at
                    < 0.8 * args.head_stable_duration
                ):
                    return False, "正在确认 head_control 已锁定角度"
                command_yaws = np.asarray([sample.yaw_rad for sample in command_window])
                command_pitches = np.asarray([sample.pitch_rad for sample in command_window])
                if max(float(np.ptp(command_yaws)), float(np.ptp(command_pitches))) > args.command_position_tolerance:
                    return False, "head_control 仍在发布扫描轨迹"
            return True, "head command held and measured joints stable"

        def nearest_head(self, stamp_ns: int) -> tuple[HeadSample | None, float | None]:
            with self._data_lock:
                head_samples = tuple(self.head_samples)
            if not head_samples:
                return None, None
            stamped = [sample for sample in head_samples if sample.stamp_ns != 0]
            if stamp_ns != 0 and stamped:
                sample = min(stamped, key=lambda value: abs(value.stamp_ns - stamp_ns))
                offset = stamp_offset_seconds(sample.stamp_ns, stamp_ns)
                if offset is not None and abs(offset) <= args.state_sync_tolerance:
                    return sample, offset
                # MuJoCo and ROS callbacks can be delayed by GPU inference.
                # A recent odometry sample is still valid for this static
                # scene; reject only when it is clearly stale.
                if time.monotonic() - sample.received_at <= 1.0:
                    return sample, offset
                return None, offset
            sample = head_samples[-1]
            if time.monotonic() - sample.received_at <= 0.5:
                return sample, None
            return None, None

        def command_matches_packet(self, stamp_ns: int) -> bool:
            if args.ignore_head_stability or not args.require_held_command:
                return True
            with self._data_lock:
                command_samples = tuple(self.command_samples)
            stamped = [sample for sample in command_samples if sample.stamp_ns != 0]
            if stamp_ns == 0 or not stamped:
                return bool(command_samples) and (
                    time.monotonic() - command_samples[-1].received_at <= 0.5
                )
            sample = min(stamped, key=lambda value: abs(value.stamp_ns - stamp_ns))
            offset = stamp_offset_seconds(sample.stamp_ns, stamp_ns)
            if offset is not None and abs(offset) <= args.state_sync_tolerance:
                return True
            return False

        def nearest_odom(self, stamp_ns: int) -> tuple[OdomSample | None, float | None]:
            with self._data_lock:
                odom_samples = tuple(self.odom_samples)
            if not odom_samples:
                return None, None
            stamped = [sample for sample in odom_samples if sample.stamp_ns != 0]
            if stamp_ns != 0 and stamped:
                sample = min(stamped, key=lambda value: abs(value.stamp_ns - stamp_ns))
                offset = stamp_offset_seconds(sample.stamp_ns, stamp_ns)
                if offset is not None and abs(offset) <= args.state_sync_tolerance:
                    return sample, offset
                return None, offset
            sample = odom_samples[-1]
            if time.monotonic() - sample.received_at <= 0.5:
                return sample, None
            return None, None

        def status_text(self) -> str:
            with self._data_lock:
                received = dict(self.received)
            missing = [name for name, count in received.items() if count == 0]
            return "received=" + str(received) + (f", missing={missing}" if missing else "")

    # Do not let rclpy parse this application's non-ROS argparse flags.
    rclpy.init(args=[])
    node = MujocoPoseNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="mujoco_pose_ros_executor",
        daemon=True,
    )
    executor_thread.start()
    print(
        "ROS 订阅已启动：\n"
        f"  RGB:        {args.rgb_topic}\n"
        f"  Depth:      {args.depth_topic}\n"
        f"  Color K:    {args.color_info_topic}\n"
        f"  Depth K:    {args.depth_info_topic}\n"
        f"  Head state: {args.actuator_state_topic}\n"
        f"  Head lock:  {args.head_command_topic}\n"
        "等待同一时间戳 RGB-D、目标分割和头部锁定...",
        flush=True,
    )

    try:
        runtime = load_foundation_runtime(args)
    except Exception:
        # Cleanup can race with a callback that already removed the node from
        # the executor.  Treat that state as already-cleaned instead of
        # masking the actual FoundationPose loading error.
        try:
            executor.shutdown(timeout_sec=2.0)
        except (ValueError, RuntimeError):
            pass
        executor_thread.join(timeout=2.0)
        try:
            node.destroy_node()
        except (ValueError, RuntimeError):
            pass
        rclpy.try_shutdown()
        raise

    gui_active = False

    started_at = time.monotonic()
    last_status_at = float("-inf")
    last_yolo_at = float("-inf")
    stable_since = float("inf")
    was_stable = False
    target_missing_since: float | None = None
    fallback_scanning = False
    fallback_scan_started_at = 0.0
    previous_box: np.ndarray | None = None
    consecutive = 0
    registered = False
    lost_frames = 0
    last_pose_at = float("-inf")
    last_camera_pose: np.ndarray | None = None
    last_pose_quality: dict[str, Any] | None = None
    output_sequence = 0
    alignment_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    try:
        gui_active = bool(args.show and open_visualization_window())
        while rclpy.ok():
            now = time.monotonic()
            if args.timeout > 0.0 and now - started_at > args.timeout:
                raise TimeoutError(f"等待目标超过 {args.timeout:.1f} 秒；{node.status_text()}")

            packet = node.pop_latest_packet()
            if packet is None:
                if now - last_status_at >= 3.0:
                    print(f"等待相机数据：{node.status_text()}", flush=True)
                    last_status_at = now
                if gui_active and not poll_visualization_window(1):
                    break
                time.sleep(0.01)
                continue

            if now - last_yolo_at < 1.0 / max(args.pose_rate * 2.0, 1.0):
                if gui_active and not poll_visualization_window(1):
                    break
                continue
            last_yolo_at = now
            stable, stable_reason = node.head_stability()
            if fallback_scanning:
                stable = False
                stable_reason = "目标类别回退扫描中"
            head, head_offset = node.nearest_head(packet.stamp_ns)
            if head is None:
                stable = False
                stable_reason = (
                    "没有与图像同步的头部反馈"
                    if head_offset is None
                    else f"头部反馈与图像相差 {head_offset:+.3f}s"
                )
            elif not node.command_matches_packet(packet.stamp_ns):
                stable = False
                stable_reason = "图像时间附近没有头部锁定命令"

            if stable and not was_stable:
                stable_since = now
            elif not stable:
                stable_since = float("inf")
            was_stable = stable

            # Only use a fresh camera packet received after the lock/stability
            # transition; this discards frames queued while the head was moving.
            if stable and now - stable_since < 0.05:
                consecutive = 0
                if gui_active:
                    visual = make_pipeline_visualization(
                        packet,
                        None,
                        "WAITING FOR FRESH STABLE RGB-D",
                        head=head,
                    )
                    if not show_visualization_frame(visual):
                        break
                continue

            observation = detect_target(
                runtime.yolo,
                packet.color_bgr,
                runtime.target_class_id,
                args,
            )
            if fallback_scanning:
                phase = 2.0 * math.pi * (now - fallback_scan_started_at) / args.scan_period
                node.publish_scan_command(args.scan_yaw_amplitude * math.sin(phase))
            if observation is None:
                consecutive = 0
                previous_box = None
                held_without_target = stable or fallback_scanning
                if held_without_target and target_missing_since is None:
                    target_missing_since = now
                should_start_fallback = (
                    args.scan_fallback_timeout > 0.0
                    and stable
                    and target_missing_since is not None
                    and now - target_missing_since >= args.scan_fallback_timeout
                )
                if fallback_scanning or should_start_fallback:
                    if not fallback_scanning:
                        fallback_scanning = True
                        fallback_scan_started_at = now
                        print(
                            "头控保持不动但未找到目标；本节点接管 override，继续按目标类别扫描。",
                            flush=True,
                        )
                        node.publish_scan_command(0.0)
                if registered:
                    lost_frames += 1
                    if lost_frames >= args.lost_frames:
                        print("目标连续丢失，清除 FoundationPose 跟踪状态并等待重新注册。")
                        runtime.estimator.pose_last = None
                        registered = False
                        last_camera_pose = None
                        last_pose_quality = None
                        lost_frames = 0
                if now - last_status_at >= 3.0:
                    print(
                        f"YOLO-seg 尚未检测到 {runtime.yolo.names[runtime.target_class_id]!r}; "
                        f"头部状态: {stable_reason}",
                        flush=True,
                    )
                    last_status_at = now
                if gui_active:
                    stage = "YOLO SEARCHING - HEAD SCANNING" if fallback_scanning else "YOLO SEARCHING"
                    visual = make_pipeline_visualization(
                        packet,
                        None,
                        stage,
                        head=head,
                        detail="Waiting for target segmentation",
                    )
                    if not show_visualization_frame(visual):
                        break
                continue

            target_missing_since = None
            if fallback_scanning:
                assert head is not None
                # Hold the measured angle immediately. Subsequent frames must
                # pass the normal command/joint stability windows before pose.
                node.publish_scan_command(head.yaw_rad, head.pitch_rad)
                fallback_scanning = False
                consecutive = 0
                stable_since = float("inf")
                was_stable = False
                print(
                    f"目标 {observation.class_name} 已进入视野；停止回退扫描并保持真实头部角度。",
                    flush=True,
                )
                if gui_active:
                    visual = make_pipeline_visualization(
                        packet,
                        observation,
                        "YOLO TARGET FOUND - LOCKING HEAD",
                        head=head,
                    )
                    if not show_visualization_frame(visual):
                        break
                continue

            lost_frames = 0
            same_instance = box_iou(observation.box_xyxy, previous_box) >= args.same_instance_iou
            previous_box = observation.box_xyxy.copy()
            if not stable:
                consecutive = 0
                if now - last_status_at >= 2.0:
                    print(
                        f"已分割到 {observation.class_name} ({observation.confidence:.3f})，"
                        f"但仍在等待头部停止：{stable_reason}",
                        flush=True,
                    )
                    last_status_at = now
                if gui_active:
                    visual = make_pipeline_visualization(
                        packet,
                        observation,
                        "YOLO DETECTED - WAITING FOR HEAD",
                        head=head,
                        detail="Waiting for synchronized, stationary head joints",
                    )
                    if not show_visualization_frame(visual):
                        break
                continue
            consecutive = consecutive + 1 if same_instance else 1

            should_register = not registered and consecutive >= args.required_detections
            should_track = (
                registered
                and args.continuous
                and now - last_pose_at >= 1.0 / args.pose_rate
            )
            if not should_register and not should_track:
                if not registered:
                    print(
                        f"目标与头部锁定确认 {consecutive}/{args.required_detections}: "
                        f"{observation.class_name} conf={observation.confidence:.3f}",
                        flush=True,
                    )
                if gui_active:
                    stage = (
                        f"YOLO CONFIRMING {consecutive}/{args.required_detections}"
                        if not registered
                        else "FOUNDATIONPOSE TRACK READY"
                    )
                    visual = make_pipeline_visualization(
                        packet,
                        observation,
                        stage,
                        head=head,
                        pose=last_camera_pose if registered else None,
                        runtime=runtime if registered else None,
                        pose_quality=last_pose_quality if registered else None,
                    )
                    if not show_visualization_frame(visual):
                        break
                continue
            assert head is not None

            color_shape = packet.color_bgr.shape[:2]
            cache_key = (
                packet.depth_raw.shape,
                color_shape,
                tuple(np.round(packet.depth_info.K.reshape(-1), 8)),
                tuple(np.round(packet.color_info.K.reshape(-1), 8)),
            )
            maps = alignment_cache.get(cache_key)
            if maps is None:
                maps = depth_alignment_maps(
                    packet.depth_info.K,
                    packet.color_info.K,
                    color_shape,
                )
                alignment_cache[cache_key] = maps
                distortion = np.concatenate(
                    (packet.color_info.distortion, packet.depth_info.distortion)
                )
                if distortion.size and np.max(np.abs(distortion)) > 1e-8:
                    node.get_logger().warning(
                        "CameraInfo 含非零畸变；当前 MuJoCo 对齐按无畸变针孔模型计算"
                    )
                print(
                    f"深度对齐: native={packet.depth_raw.shape[::-1]} -> "
                    f"color={color_shape[::-1]}（使用两组 CameraInfo，不做简单 resize）",
                    flush=True,
                )
            try:
                native_depth_m = depth_to_meters(packet.depth_raw, args.depth_scale)
                aligned_depth_m = align_depth_to_color(
                    native_depth_m,
                    packet.depth_info.K,
                    packet.color_info.K,
                    color_shape,
                    maps,
                )
                filtered_depth_m, depth_metadata = filter_depth_for_object(
                    aligned_depth_m,
                    observation.mask,
                    args,
                )
                color_rgb = cv2.cvtColor(packet.color_bgr, cv2.COLOR_BGR2RGB)

                if gui_active:
                    stage = (
                        "FOUNDATIONPOSE REGISTERING..."
                        if should_register
                        else "FOUNDATIONPOSE TRACKING..."
                    )
                    visual = make_pipeline_visualization(
                        packet,
                        observation,
                        stage,
                        head=head,
                        detail=f"Valid object depth: {depth_metadata['valid_after']} px",
                    )
                    if not show_visualization_frame(visual):
                        break

                if should_register:
                    print(
                        f"头部已停止且 YOLO-seg 连续确认目标，开始 FoundationPose register "
                        f"(valid depth={depth_metadata['valid_after']})...",
                        flush=True,
                    )
                    camera_pose = runtime.estimator.register(
                        K=packet.color_info.K,
                        rgb=color_rgb,
                        depth=filtered_depth_m,
                        ob_mask=observation.mask,
                        iteration=args.est_refine_iter,
                    )
                else:
                    camera_pose = runtime.estimator.track_one(
                        rgb=color_rgb,
                        depth=filtered_depth_m,
                        K=packet.color_info.K,
                        iteration=args.track_refine_iter,
                    )
                camera_pose = validate_pose(camera_pose)
                pose_quality = assess_pose_quality(
                    packet.color_info.K,
                    camera_pose,
                    runtime.mesh,
                    observation,
                    args.min_pose_box_iou,
                    args.max_pose_center_error,
                )
                pose_depth_error = abs(
                    float(camera_pose[2, 3]) - float(depth_metadata["median_depth_m"])
                )
                pose_quality["translation_z_vs_mask_median_error_m"] = pose_depth_error
                pose_quality["maximum_translation_depth_error_m"] = args.depth_max_band
                if pose_depth_error > args.depth_max_band:
                    pose_quality["valid"] = False
                    pose_quality["reason"] = (
                        f"pose z and mask median depth differ by {pose_depth_error:.3f} m "
                        f"(max {args.depth_max_band:.3f} m)"
                    )
                if not pose_quality["valid"]:
                    raise RuntimeError(f"位姿质量检查未通过: {pose_quality['reason']}")
                if should_register:
                    registered = True
            except (RuntimeError, ValueError) as error:
                print(f"本帧位姿估计失败，将等待新 RGB-D 重新注册: {error}", file=sys.stderr)
                runtime.estimator.pose_last = None
                registered = False
                last_camera_pose = None
                last_pose_quality = None
                consecutive = 0
                if gui_active:
                    visual = make_pipeline_visualization(
                        packet,
                        observation,
                        "FOUNDATIONPOSE FAILED - RETRYING",
                        head=head,
                        detail="Pose rejected; waiting for a new synchronized RGB-D frame",
                    )
                    if not show_visualization_frame(visual):
                        break
                continue
            last_pose_at = time.monotonic()
            last_camera_pose = camera_pose.copy()
            last_pose_quality = dict(pose_quality)
            output_sequence += 1
            odom, odom_offset = node.nearest_odom(packet.stamp_ns)
            _, visual = save_pose_result(
                args,
                runtime,
                packet,
                observation,
                head,
                head_offset,
                depth_metadata,
                pose_quality,
                aligned_depth_m,
                camera_pose,
                odom,
                odom_offset,
                output_sequence,
            )

            if gui_active:
                if not show_visualization_frame(visual, block=False):
                    break
                if not args.continuous and args.display_hold_sec > 0.0:
                    hold_until = time.monotonic() + args.display_hold_sec
                    while time.monotonic() < hold_until:
                        if not poll_visualization_window(30):
                            return 0
            if not args.continuous:
                return 0
    except KeyboardInterrupt:
        print("\n用户终止。")
        return 130
    finally:
        if gui_active:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            executor.shutdown(timeout_sec=2.0)
        except (ValueError, RuntimeError):
            # rclpy may have removed the node during shutdown already.
            pass
        executor_thread.join(timeout=2.0)
        try:
            node.destroy_node()
        except (ValueError, RuntimeError):
            pass
        rclpy.try_shutdown()
    return 0


def main() -> int:
    try:
        parser = build_argument_parser()
        args = parser.parse_args()
        validate_arguments(args, parser)
        if args.self_test:
            return run_self_test()
        return run(args)
    except Exception as error:
        print(f"错误: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
