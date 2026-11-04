#!/usr/bin/env python3
"""Publish autonomous FoundationPose output as ROS2 pose/status topics and TF.

The autonomous vision process atomically writes ``latest_state.json``. This
node turns its ``pose_camera_object`` matrix (T_camera_CAD_original) into a
``geometry_msgs/PoseStamped`` without coupling ROS arm control to the vision
process. The pose is the original CAD/model origin, not the detected object's
geometric center. Consumers must gate motion on ``object_pose_valid`` and the
age of ``PoseStamped.header.stamp``; separate ROS topics are not atomic.

For a robot-frame IK target, provide a calibrated, time-correct TF from the
robot base to the camera and a grasp transform defined in the CAD frame. A
typical wrist target is:

    T_base_wrist = T_base_camera @ T_camera_CAD @ T_CAD_TCP @ inv(T_wrist_TCP)

The source timestamp is the state file's Unix/system-time mtime, not the image
acquisition timestamp. This approximation is suitable for a fixed camera. An
eye-in-hand camera requires the vision process to export the source image's ROS
timestamp before the pose is used with a dynamic TF.

Dynamic TF output is disabled by default because an invalid/stale TF cannot be
retracted. If enabled, consumers must still check the validity topic and pose
timestamp instead of looking up the latest TF alone.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = REPO_ROOT / "runs/autonomous_vision/latest_state.json"


@dataclass(frozen=True)
class PoseState:
    valid: bool
    reason: str
    pose_camera_object: np.ndarray | None
    camera_frame: str
    source_camera_frame: str
    source_status: str
    source_wall_time: str | None
    source_stamp_s: float | None
    state_age_s: float | None
    track_id: int | None
    class_id: int | None
    class_name: str | None
    pose_age_s: float | None
    yaw_observable: bool | None
    imu_enabled: bool | None
    imu_fresh: bool | None
    ros2_imu_fresh: bool | None
    ros2_imu_age_s: float | None

    def status_metadata(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "camera_frame": self.camera_frame or None,
            "source_camera_frame": self.source_camera_frame or None,
            "camera_frame_override_applied": bool(
                self.camera_frame and self.camera_frame != self.source_camera_frame
            ),
            "source_status": self.source_status or None,
            "source_wall_time": self.source_wall_time,
            "source_stamp_s": self.source_stamp_s,
            "state_age_s": self.state_age_s,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "pose_semantics": "T_camera_CAD_original",
            "translation_unit": "m",
            "quaternion_order": "xyzw",
            "timestamp_semantics": "state_file_mtime_unix_system_time",
            "pose_age_s": self.pose_age_s,
            "yaw_observable": self.yaw_observable,
            "source_imu_enabled": self.imu_enabled,
            "imu_fresh": self.imu_fresh,
            "ros2_imu_fresh": self.ros2_imu_fresh,
            "ros2_imu_age_s": self.ros2_imu_age_s,
        }


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a finite proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        raise ValueError("rotation is not orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=1e-3):
        raise ValueError(f"rotation determinant must be +1, got {determinant:.6f}")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("failed to construct a valid quaternion")
    quaternion /= norm
    # q and -q encode the same rotation. A stable sign avoids discontinuous
    # values in logs and downstream filters.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def epoch_seconds_to_time_message(epoch_seconds: float) -> TimeMessage:
    """Convert a positive Unix timestamp to a ROS builtin Time message."""
    value = float(epoch_seconds)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("source timestamp must be nonnegative and finite")
    seconds = math.floor(value)
    nanoseconds = int(round((value - seconds) * 1_000_000_000.0))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return TimeMessage(sec=int(seconds), nanosec=nanoseconds)


def _invalid_pose_state(
    reason: str,
    *,
    state_age_s: float | None = None,
    source_stamp_s: float | None = None,
    source_status: str = "",
) -> PoseState:
    return PoseState(
        valid=False,
        reason=reason,
        pose_camera_object=None,
        camera_frame="",
        source_camera_frame="",
        source_status=source_status,
        source_wall_time=None,
        source_stamp_s=source_stamp_s,
        state_age_s=state_age_s,
        track_id=None,
        class_id=None,
        class_name=None,
        pose_age_s=None,
        yaw_observable=None,
        imu_enabled=None,
        imu_fresh=None,
        ros2_imu_fresh=None,
        ros2_imu_age_s=None,
    )


def _json_integer(value: object) -> int | None:
    """Return a JSON integer while rejecting booleans (``bool`` subclasses ``int``)."""
    return value if type(value) is int else None


def _optional_nonnegative_float(value: object) -> float | None:
    """Parse optional finite, nonnegative JSON numeric metadata."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) and parsed >= 0.0 else None


def load_pose_state(
    state_file: Path,
    *,
    now_epoch_s: float | None = None,
    maximum_state_age_s: float = 1.0,
    maximum_future_skew_s: float = 0.0,
    require_observed_this_frame: bool = True,
    require_pose_quality: bool = True,
    require_imu_fresh: bool = False,
    require_yaw_observable: bool = False,
    require_optical_frame: bool = True,
    camera_frame_override: str = "",
) -> PoseState:
    """Read and strictly validate one atomically written autonomous state file."""
    if now_epoch_s is not None:
        now_epoch_s = float(now_epoch_s)
    if now_epoch_s is not None and not np.isfinite(now_epoch_s):
        raise ValueError("now_epoch_s must be finite")
    if not np.isfinite(maximum_state_age_s) or maximum_state_age_s <= 0.0:
        raise ValueError("maximum_state_age_s must be positive and finite")
    if not np.isfinite(maximum_future_skew_s) or maximum_future_skew_s < 0.0:
        raise ValueError("maximum_future_skew_s must be nonnegative and finite")

    state_age_s: float | None = None
    source_stamp_s: float | None = None
    try:
        # fstat the opened descriptor so an atomic path replacement cannot pair
        # one JSON generation with another generation's timestamp.
        with state_file.open("r", encoding="utf-8") as file:
            stat = os.fstat(file.fileno())
            source_stamp_s = float(stat.st_mtime)
            # Read the clock after opening/fstat when no test time was supplied;
            # this avoids rejecting a file atomically replaced just after entry.
            current_epoch_s = time.time() if now_epoch_s is None else now_epoch_s
            state_age_s = current_epoch_s - source_stamp_s
            if state_age_s < -maximum_future_skew_s:
                return _invalid_pose_state(
                    f"state file timestamp is {abs(state_age_s):.3f}s in the future "
                    f"(limit {maximum_future_skew_s:.3f}s)",
                    state_age_s=state_age_s,
                    source_stamp_s=source_stamp_s,
                )
            if state_age_s > maximum_state_age_s:
                return _invalid_pose_state(
                    f"state file is stale ({state_age_s:.3f}s > "
                    f"{maximum_state_age_s:.3f}s)",
                    state_age_s=state_age_s,
                    source_stamp_s=source_stamp_s,
                )
            state = json.load(file)
    except FileNotFoundError:
        return _invalid_pose_state(f"state file not found: {state_file}")
    except (OSError, json.JSONDecodeError) as error:
        return _invalid_pose_state(
            f"cannot read state file: {error}",
            state_age_s=state_age_s,
            source_stamp_s=source_stamp_s,
        )
    if not isinstance(state, dict):
        return _invalid_pose_state(
            "state root must be a JSON object",
            state_age_s=state_age_s,
            source_stamp_s=source_stamp_s,
        )

    source_status = str(state.get("status", ""))
    source_camera_frame = str(state.get("camera_frame", "")).strip().lstrip("/")
    camera_frame = camera_frame_override.strip().lstrip("/") or str(
        state.get("camera_frame", "")
    ).strip().lstrip("/")
    source_wall_time_value = state.get("wall_time")
    source_wall_time = (
        str(source_wall_time_value) if source_wall_time_value is not None else None
    )
    selected_target = state.get("selected_target")
    track_id = _json_integer(state.get("selected_track_id"))
    class_id = _json_integer(state.get("requested_class_id"))
    class_name_value = state.get("requested_class_name")
    class_name = str(class_name_value) if class_name_value is not None else None
    pose_age_s = _optional_nonnegative_float(state.get("pose_age_s"))
    yaw_value = state.get("yaw_observable")
    yaw_observable = yaw_value if isinstance(yaw_value, bool) else None
    imu_enabled_value = state.get("imu_enabled")
    imu_enabled = imu_enabled_value if isinstance(imu_enabled_value, bool) else None
    ros2_imu_value = state.get("ros2_imu_fresh")
    ros2_imu_fresh = ros2_imu_value if isinstance(ros2_imu_value, bool) else None
    # New state files expose a source-independent freshness field. Fall back to
    # the original ROS2-only field so existing state files remain compatible.
    imu_value = state.get("imu_fresh") if "imu_fresh" in state else ros2_imu_value
    imu_fresh = imu_value if isinstance(imu_value, bool) else None
    ros2_imu_age_s = _optional_nonnegative_float(state.get("ros2_imu_age_s"))

    common = {
        "camera_frame": camera_frame,
        "source_camera_frame": source_camera_frame,
        "source_status": source_status,
        "source_wall_time": source_wall_time,
        "source_stamp_s": source_stamp_s,
        "state_age_s": state_age_s,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "pose_age_s": pose_age_s,
        "yaw_observable": yaw_observable,
        "imu_enabled": imu_enabled,
        "imu_fresh": imu_fresh,
        "ros2_imu_fresh": ros2_imu_fresh,
        "ros2_imu_age_s": ros2_imu_age_s,
    }
    invalid_reason = ""
    if _json_integer(state.get("schema_version")) != 1:
        invalid_reason = "unsupported or missing schema_version (expected 1)"
    elif not source_camera_frame:
        invalid_reason = "source camera_frame is empty"
    elif not camera_frame:
        invalid_reason = "camera_frame is empty"
    elif require_optical_frame and "optical" not in camera_frame.lower():
        invalid_reason = (
            f"camera_frame name '{camera_frame}' does not contain 'optical'; "
            "set camera_frame_override to a real color optical TF frame, or disable "
            "the name check only after verifying optical-axis semantics"
        )
    elif not isinstance(selected_target, dict):
        invalid_reason = "no selected target"
    elif track_id is None or _json_integer(selected_target.get("track_id")) != track_id:
        invalid_reason = "selected target track_id does not match selected_track_id"
    elif class_id is None or _json_integer(selected_target.get("class_id")) != class_id:
        invalid_reason = "selected target class_id does not match requested_class_id"
    elif class_name is None or str(selected_target.get("class_name", "")) != class_name:
        invalid_reason = "selected target class_name does not match requested_class_name"
    elif require_observed_this_frame and selected_target.get("observed_this_frame") is not True:
        invalid_reason = "selected target was not observed this frame"
    elif require_pose_quality:
        pose_quality = state.get("pose_quality")
        if not isinstance(pose_quality, dict) or pose_quality.get("valid") is not True:
            invalid_reason = "pose quality is not valid"
    if not invalid_reason and require_imu_fresh and imu_value is not True:
        invalid_reason = "IMU data is missing or stale"
    if not invalid_reason and require_yaw_observable and yaw_value is not True:
        invalid_reason = "object yaw is not observable"

    raw_pose = state.get("pose_camera_object")
    pose = None
    if not invalid_reason:
        try:
            pose = np.asarray(raw_pose, dtype=np.float64)
        except (TypeError, ValueError):
            invalid_reason = "pose_camera_object is not numeric"
        else:
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                invalid_reason = "pose_camera_object must be a finite 4x4 matrix"
            elif not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
                invalid_reason = "pose_camera_object has an invalid homogeneous row"
            else:
                try:
                    rotation_matrix_to_quaternion_xyzw(pose[:3, :3])
                except ValueError as error:
                    invalid_reason = str(error)

    return PoseState(
        valid=not invalid_reason,
        reason="ok" if not invalid_reason else invalid_reason,
        pose_camera_object=pose if not invalid_reason else None,
        **common,
    )


class FoundationPoseStateNode(Node):
    """ROS2 bridge from the autonomous JSON state to pose, validity and TF outputs."""

    def __init__(self) -> None:
        super().__init__("foundationpose_pose_state_node")
        self._declare_parameters()
        self._load_parameters()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pose_publisher = self.create_publisher(PoseStamped, self.pose_topic, qos)
        self.valid_publisher = self.create_publisher(Bool, self.valid_topic, qos)
        self.status_publisher = self.create_publisher(String, self.status_topic, qos)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self._last_log_state: tuple[bool, str] | None = None
        self._last_tf_key: tuple[float, str, str, int | None] | None = None
        self.timer = self.create_timer(1.0 / self.poll_rate_hz, self._publish_latest_state)

        self.get_logger().info(f"Reading FoundationPose state: {self.state_file}")
        self.get_logger().info(
            f"Publishing pose={self.pose_topic}, valid={self.valid_topic}, "
            f"status={self.status_topic}, child_frame={self.child_frame_id}"
        )
        if self.camera_frame_override:
            self.get_logger().warn(
                "camera_frame_override only relabels the optical-coordinate pose; "
                "it does not apply a TF rotation or translation"
            )
        if not self.require_optical_frame:
            self.get_logger().warn(
                "Optical-frame name checking is disabled; confirm that CameraInfo.frame_id "
                "uses +X right, +Y down, +Z forward before commanding a robot"
            )
        if self.publish_tf:
            self.get_logger().warn(
                "TF output is enabled. A stale target TF cannot be retracted; IK must also "
                "gate on the validity topic and PoseStamped source timestamp"
            )
        self.get_logger().info(
            f"IMU freshness gate: {'enabled' if self.use_imu else 'disabled'}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("state_file", str(DEFAULT_STATE_FILE))
        self.declare_parameter("poll_rate_hz", 20.0)
        self.declare_parameter("maximum_state_age_s", 0.5)
        self.declare_parameter("maximum_future_skew_s", 0.0)
        self.declare_parameter("pose_topic", "/foundationpose/object_pose")
        self.declare_parameter("valid_topic", "/foundationpose/object_pose_valid")
        self.declare_parameter("status_topic", "/foundationpose/object_pose_status")
        self.declare_parameter("child_frame_id", "foundationpose_object_cad")
        self.declare_parameter("camera_frame_override", "")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("require_observed_this_frame", True)
        self.declare_parameter("require_pose_quality", True)
        self.declare_parameter("use_imu", False)
        self.declare_parameter("require_yaw_observable", False)
        self.declare_parameter("require_optical_frame", True)

    def _load_parameters(self) -> None:
        self.state_file = Path(str(self.get_parameter("state_file").value)).expanduser().resolve()
        self.poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self.maximum_state_age_s = float(self.get_parameter("maximum_state_age_s").value)
        self.maximum_future_skew_s = float(
            self.get_parameter("maximum_future_skew_s").value
        )
        self.pose_topic = str(self.get_parameter("pose_topic").value).strip()
        self.valid_topic = str(self.get_parameter("valid_topic").value).strip()
        self.status_topic = str(self.get_parameter("status_topic").value).strip()
        self.child_frame_id = str(self.get_parameter("child_frame_id").value).strip().lstrip("/")
        self.camera_frame_override = str(
            self.get_parameter("camera_frame_override").value
        ).strip().lstrip("/")
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.require_observed_this_frame = bool(
            self.get_parameter("require_observed_this_frame").value
        )
        self.require_pose_quality = bool(self.get_parameter("require_pose_quality").value)
        self.use_imu = bool(self.get_parameter("use_imu").value)
        self.require_yaw_observable = bool(
            self.get_parameter("require_yaw_observable").value
        )
        self.require_optical_frame = bool(self.get_parameter("require_optical_frame").value)

        if not np.isfinite(self.poll_rate_hz) or self.poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive and finite")
        if not np.isfinite(self.maximum_state_age_s) or self.maximum_state_age_s <= 0.0:
            raise ValueError("maximum_state_age_s must be positive and finite")
        if not np.isfinite(self.maximum_future_skew_s) or self.maximum_future_skew_s < 0.0:
            raise ValueError("maximum_future_skew_s must be nonnegative and finite")
        if self.has_parameter("use_sim_time") and bool(
            self.get_parameter("use_sim_time").value
        ):
            raise ValueError(
                "use_sim_time is unsupported because the JSON source uses Unix/system time"
            )
        for name, value in (
            ("pose_topic", self.pose_topic),
            ("valid_topic", self.valid_topic),
            ("status_topic", self.status_topic),
            ("child_frame_id", self.child_frame_id),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")

    def _publish_latest_state(self) -> None:
        state = load_pose_state(
            self.state_file,
            maximum_state_age_s=self.maximum_state_age_s,
            maximum_future_skew_s=self.maximum_future_skew_s,
            require_observed_this_frame=self.require_observed_this_frame,
            require_pose_quality=self.require_pose_quality,
            require_imu_fresh=self.use_imu,
            require_yaw_observable=self.require_yaw_observable,
            require_optical_frame=self.require_optical_frame,
            camera_frame_override=self.camera_frame_override,
        )

        if state.valid and state.source_stamp_s is None:
            state = replace(state, valid=False, reason="valid pose has no source timestamp")
        if state.valid and state.pose_camera_object is None:
            state = replace(state, valid=False, reason="valid pose matrix is missing")
        if (
            state.valid
            and self.tf_broadcaster is not None
            and state.camera_frame == self.child_frame_id
        ):
            state = replace(
                state,
                valid=False,
                reason=(
                    "TF parent and child frame must differ "
                    f"('{state.camera_frame}')"
                ),
            )

        valid_message = Bool()
        valid_message.data = state.valid

        status_message = String()
        status_metadata = state.status_metadata()
        status_metadata["imu_freshness_gate_enabled"] = self.use_imu
        status_message.data = json.dumps(status_metadata, ensure_ascii=False)

        log_reason = state.reason.partition(" (")[0]
        log_state = (state.valid, log_reason)
        if log_state != self._last_log_state:
            if state.valid:
                self.get_logger().info(
                    f"Valid pose: track={state.track_id}, class={state.class_name}, "
                    f"frame={state.camera_frame}"
                )
            else:
                self.get_logger().warn(f"6D pose invalid: {state.reason}")
            self._last_log_state = log_state

        if not state.valid or state.pose_camera_object is None:
            # Fail closed immediately. No pose/TF is emitted while invalid.
            self.valid_publisher.publish(valid_message)
            self.status_publisher.publish(status_message)
            return
        pose = state.pose_camera_object
        quaternion = rotation_matrix_to_quaternion_xyzw(pose[:3, :3])
        pose_message = PoseStamped()
        pose_message.header.stamp = epoch_seconds_to_time_message(state.source_stamp_s)
        pose_message.header.frame_id = state.camera_frame
        pose_message.pose.position.x = float(pose[0, 3])
        pose_message.pose.position.y = float(pose[1, 3])
        pose_message.pose.position.z = float(pose[2, 3])
        pose_message.pose.orientation.x = float(quaternion[0])
        pose_message.pose.orientation.y = float(quaternion[1])
        pose_message.pose.orientation.z = float(quaternion[2])
        pose_message.pose.orientation.w = float(quaternion[3])
        # Publish the matching pose before raising validity. Late/jittery IK
        # consumers must additionally match/check the source timestamp.
        self.pose_publisher.publish(pose_message)

        if self.tf_broadcaster is not None:
            tf_key = (
                state.source_stamp_s,
                state.camera_frame,
                self.child_frame_id,
                state.track_id,
            )
            if tf_key != self._last_tf_key:
                transform = TransformStamped()
                transform.header = pose_message.header
                transform.child_frame_id = self.child_frame_id
                transform.transform.translation.x = pose_message.pose.position.x
                transform.transform.translation.y = pose_message.pose.position.y
                transform.transform.translation.z = pose_message.pose.position.z
                transform.transform.rotation = pose_message.pose.orientation
                self.tf_broadcaster.sendTransform(transform)
                self._last_tf_key = tf_key

        self.status_publisher.publish(status_message)
        self.valid_publisher.publish(valid_message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FoundationPoseStateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
