#!/usr/bin/env python3
"""Replay an arm-IK joint trajectory into an already running ROS2 MuJoCo sim.

This helper deliberately uses a *separate* simulation override topic.  The
normal ``simulation/actuators_cmds`` topic is owned by the gait/controller
node, and publishing a second stream there would race that controller.  The
controller must be started with its explicit simulation arm-override option
and subscribe to ``simulation/arm_ik_cmds_override`` before this program is
run.

The process is intentionally ROS-only (it does not import MuJoCo), so it must
run with a Python 3.10 ROS2 environment, for example::

    source /opt/ros/humble/setup.bash
    source /opt/bxi/bxi_ros2_pkg/setup.bash
    python my_sim/arm_ik_ros_bridge.py trajectory.json

The trajectory JSON format is small and stable::

    {"joint_names": ["l_shoulder_y_joint", ...],
     "frames": [{"time_s": 0.0, "position": [...],
                 "kp": [...], "kd": [...]}, ...]}

Frames are linearly interpolated and published at 50 Hz.  A fresh command is
published continuously so the controller's watchdog cannot release the arm
while a long waypoint is held.  On every exit an empty command is sent to
release the override.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


DEFAULT_TOPIC = "/simulation/arm_ik_cmds_override"
DEFAULT_STATE_TOPIC = "/simulation/actuator_states"
DEFAULT_ODOM_TOPIC = "/simulation/odom"
DEFAULT_MOTION_TOPIC = "/motion_commands"
DEFAULT_RATE_HZ = 50.0
DEFAULT_TIMEOUT_SEC = 2.0
DEFAULT_NAV_RATE_HZ = 20.0
# MuJoCo odometry has a few millimetres of gait jitter near standstill.  Keep
# the acceptance band below the old 8 cm dead-zone so the bridge does not
# accept a pose that is visibly short of the IK reach stance.
DEFAULT_NAV_POSITION_TOLERANCE = 0.07
DEFAULT_NAV_HEADING_TOLERANCE = 0.12
DEFAULT_NAV_TIMEOUT_SEC = 45.0
DEFAULT_STAGE_MIN_DURATION_SEC = 0.75
DEFAULT_STAGE_MAX_JOINT_SPEED = 0.40
DEFAULT_FINAL_HOLD_SEC = 3.0
# Conservative limits for the learned gait.  Large velocity/yaw steps can
# excite torso roll when the controller is under CPU load.
NAV_MAX_VX = 0.22
NAV_MAX_VY = 0.12
NAV_MAX_YAW_RATE = 0.40
# The learned normal gait has a translational command dead-zone around
# 0.08-0.10 m/s.  A lower floor lets the bridge approach a target forever
# without taking another step (typically stalling 0.12-0.14 m away).
NAV_MIN_TRANSLATION_SPEED = 0.14
NAV_FINAL_APPROACH_SPEED = 0.21
NAV_SETTLE_SEC = 1.5

# Gains from the ELF3 MuJoCo actuator calibration.  A blanket ``kp=45,
# kd=1`` makes the shoulder under-damped; the resulting torque impulse is one
# of the main causes of a forward tip when the first IK waypoint is sent.
ARM_PD_GAINS = {
    "l_shoulder_y_joint": (54.224, 3.452),
    "l_shoulder_x_joint": (54.224, 3.452),
    "l_shoulder_z_joint": (16.747, 1.066),
    "l_elbow_y_joint": (54.224, 3.452),
    "l_wrist_x_joint": (16.747, 1.066),
    "l_wrist_y_joint": (16.747, 1.066),
    "l_wrist_z_joint": (16.747, 1.066),
    "r_shoulder_y_joint": (54.224, 3.452),
    "r_shoulder_x_joint": (54.224, 3.452),
    "r_shoulder_z_joint": (16.747, 1.066),
    "r_elbow_y_joint": (54.224, 3.452),
    "r_wrist_x_joint": (16.747, 1.066),
    "r_wrist_y_joint": (16.747, 1.066),
    "r_wrist_z_joint": (16.747, 1.066),
}


def _arm_gains(names: Sequence[str]) -> tuple[list[float], list[float]]:
    return (
        [ARM_PD_GAINS.get(name, (45.0, 1.0))[0] for name in names],
        [ARM_PD_GAINS.get(name, (45.0, 1.0))[1] for name in names],
    )

# The interface is deliberately limited to the fourteen arm hinges.  This is
# a second line of defence in addition to the controller-side validation.
ARM_JOINTS = frozenset(
    (
        "l_shoulder_y_joint",
        "l_shoulder_x_joint",
        "l_shoulder_z_joint",
        "l_elbow_y_joint",
        "l_wrist_x_joint",
        "l_wrist_y_joint",
        "l_wrist_z_joint",
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
    )
)


def _as_finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _finite_sequence(value: Any, field: str, expected: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(
            f"trajectory {field} must contain exactly {expected} values"
        )
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"trajectory {field} must contain only numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"trajectory {field} must contain only finite values")
    return result


def load_trajectory(path: Path) -> dict[str, Any]:
    """Validate and normalize a trajectory JSON file."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"trajectory JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read trajectory JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("trajectory JSON root must be an object")

    raw_names = payload.get("joint_names")
    if not isinstance(raw_names, list) or not raw_names:
        raise ValueError("trajectory joint_names must be a non-empty list")
    names = tuple(str(name) for name in raw_names)
    if any(not name for name in names):
        raise ValueError("trajectory joint_names must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("trajectory joint_names contains duplicates")
    unexpected = tuple(name for name in names if name not in ARM_JOINTS)
    if unexpected:
        raise ValueError(
            "arm IK bridge rejects non-arm joints: " + ", ".join(unexpected)
        )

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("trajectory frames must be a non-empty list")
    frames: list[dict[str, Any]] = []
    previous_time = -math.inf
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            raise ValueError(f"trajectory frame {index} must be an object")
        try:
            time_s = float(raw_frame["time_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"trajectory frame {index} has invalid time_s") from error
        if not math.isfinite(time_s) or time_s < 0.0 or time_s < previous_time:
            raise ValueError(
                f"trajectory frame {index} time_s must be finite and non-decreasing"
            )
        previous_time = time_s
        positions = _finite_sequence(raw_frame.get("position"), "position", len(names))
        # Conservative defaults match the BXI controller's normal arm gains;
        # callers can provide per-joint gains when the model requires them.
        kp = _finite_sequence(
            raw_frame.get("kp", [45.0] * len(names)), "kp", len(names)
        )
        kd = _finite_sequence(
            raw_frame.get("kd", [1.0] * len(names)), "kd", len(names)
        )
        if any(value < 0.0 for value in kp + kd):
            raise ValueError(f"trajectory frame {index} gains must be non-negative")
        frames.append(
            {
                "time_s": time_s,
                "position": positions,
                "kp": kp,
                "kd": kd,
            }
        )

    return {"joint_names": names, "frames": tuple(frames), "source": str(path)}


def _load_json_object(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"report JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read report JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("report JSON root must be an object")
    return payload


def _report_stance(report: dict[str, Any]) -> tuple[float, float]:
    """Extract the torso XY used by IK, with backward-compatible fallbacks.

    ``arm_ik.py`` may correct the gait's residual position before pinning its
    kinematic reach model.  Replaying the solved arm vectors at the raw
    walked pose would translate the TCP by that correction, so prefer the
    recorded reach stance.  Older reports lack that section; their planned
    stance is a closer match to the IK model than the raw gait capture.
    """
    reach = report.get("reach_stance")
    if isinstance(reach, dict):
        position = reach.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return (
                _as_finite_float(position[0], "reach_stance.position[0]"),
                _as_finite_float(position[1], "reach_stance.position[1]"),
            )
    plan = report.get("plan")
    if isinstance(plan, dict):
        position = plan.get("stance_xy")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return (
                _as_finite_float(position[0], "plan.stance_xy[0]"),
                _as_finite_float(position[1], "plan.stance_xy[1]"),
            )
    walked = report.get("walked_stance")
    if isinstance(walked, dict):
        position = walked.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return (
                _as_finite_float(position[0], "walked_stance.position[0]"),
                _as_finite_float(position[1], "walked_stance.position[1]"),
            )
    raise ValueError(
        "report has no reach_stance.position, plan.stance_xy, or "
        "walked_stance.position; rerun arm_ik.py with --report-json"
    )


def _report_grasp_point(report: dict[str, Any]) -> Optional[tuple[float, float]]:
    plan = report.get("plan")
    if not isinstance(plan, dict):
        return None
    point = plan.get("grasp_point")
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return (
        _as_finite_float(point[0], "plan.grasp_point[0]"),
        _as_finite_float(point[1], "plan.grasp_point[1]"),
    )


def _report_arm_waypoints(report: dict[str, Any]) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Read solved IK stage joint vectors from an arm_ik reach report."""
    ik = report.get("ik")
    if not isinstance(ik, dict):
        raise ValueError("report has no ik section")
    raw_stages = ik.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("report ik.stages must be a list")

    names: Optional[tuple[str, ...]] = None
    waypoints: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict) or not raw_stage.get("solved"):
            continue
        raw_names = raw_stage.get("joint_names")
        raw_positions = raw_stage.get("joint_positions")
        if not isinstance(raw_names, list) or not isinstance(raw_positions, list):
            continue
        current_names = tuple(str(name) for name in raw_names)
        if not current_names or len(current_names) != len(raw_positions):
            raise ValueError(f"report ik stage {index} has mismatched joint vectors")
        if any(name not in ARM_JOINTS for name in current_names):
            raise ValueError(f"report ik stage {index} contains a non-arm joint")
        if names is None:
            names = current_names
        elif current_names != names:
            # A report generated by one run should preserve ordering.  Reject
            # ambiguity instead of silently commanding the wrong wrist joint.
            raise ValueError("report IK stages use inconsistent joint ordering")
        positions = _finite_sequence(raw_positions, "joint_positions", len(names))
        waypoints.append(
            {
                "stage": str(raw_stage.get("stage", f"stage_{index}")),
                "position": positions,
            }
        )

    if names is None or not waypoints:
        raise ValueError("report contains no solved IK stages")
    return names, tuple(waypoints)


def load_report(path: Path) -> dict[str, Any]:
    """Normalize an ``arm_ik.py --report-json`` result for live replay."""
    report = _load_json_object(path)
    names, waypoints = _report_arm_waypoints(report)
    stance = _report_stance(report)
    return {
        "source": str(path.expanduser().resolve()),
        "stance_xy": stance,
        "grasp_xy": _report_grasp_point(report),
        "joint_names": names,
        "waypoints": waypoints,
    }


def trajectory_from_report(
    report: dict[str, Any],
    initial_positions: Sequence[float],
    *,
    max_joint_speed: float = DEFAULT_STAGE_MAX_JOINT_SPEED,
    minimum_stage_duration: float = DEFAULT_STAGE_MIN_DURATION_SEC,
) -> dict[str, Any]:
    """Build a smooth trajectory from current state through every IK stage."""
    names = tuple(report["joint_names"])
    if len(initial_positions) != len(names):
        raise ValueError("initial arm state length does not match report IK joints")
    max_joint_speed = _as_finite_float(max_joint_speed, "max_joint_speed")
    minimum_stage_duration = _as_finite_float(
        minimum_stage_duration, "minimum_stage_duration"
    )
    if max_joint_speed <= 0.0 or minimum_stage_duration <= 0.0:
        raise ValueError("trajectory timing parameters must be positive")

    current = [
        _as_finite_float(value, "initial arm position") for value in initial_positions
    ]
    arm_kp, arm_kd = _arm_gains(names)
    frames: list[dict[str, Any]] = [
        {
            "time_s": 0.0,
            "position": current,
            "kp": arm_kp,
            "kd": arm_kd,
        }
    ]
    elapsed = 0.0
    for waypoint in report["waypoints"]:
        target = list(waypoint["position"])
        delta = max(abs(float(a) - float(b)) for a, b in zip(current, target))
        duration = max(minimum_stage_duration, delta / max_joint_speed)
        elapsed += duration
        frames.append(
            {
                "time_s": elapsed,
                "position": target,
                "kp": arm_kp,
                "kd": arm_kd,
                "stage": waypoint["stage"],
            }
        )
        current = target
    return {"joint_names": names, "frames": tuple(frames), "source": report["source"]}


def interpolate_frames(
    frames: Sequence[dict[str, Any]], time_s: float
) -> tuple[list[float], list[float], list[float]]:
    """Linearly interpolate one normalized trajectory at ``time_s``."""
    if not frames:
        raise ValueError("cannot interpolate an empty trajectory")
    if time_s <= frames[0]["time_s"]:
        frame = frames[0]
        return frame["position"], frame["kp"], frame["kd"]
    if time_s >= frames[-1]["time_s"]:
        frame = frames[-1]
        return frame["position"], frame["kp"], frame["kd"]

    for left, right in zip(frames, frames[1:]):
        left_time = float(left["time_s"])
        right_time = float(right["time_s"])
        if time_s > right_time:
            continue
        span = right_time - left_time
        alpha = 1.0 if span <= 0.0 else (time_s - left_time) / span
        position = [
            float(a + alpha * (b - a))
            for a, b in zip(left["position"], right["position"])
        ]
        # Gains are interpolated too, which avoids a discontinuous torque jump
        # when a trajectory was authored with stage-specific gains.
        kp = [float(a + alpha * (b - a)) for a, b in zip(left["kp"], right["kp"])]
        kd = [float(a + alpha * (b - a)) for a, b in zip(left["kd"], right["kd"])]
        return position, kp, kd
    frame = frames[-1]
    return frame["position"], frame["kp"], frame["kd"]


def _import_ros() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import ROS lazily so JSON validation can run without ROS installed."""
    try:
        import rclpy
        from communication.msg import ActuatorCmds, ActuatorStates, MotionCommands
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
    except ImportError as error:
        raise RuntimeError(
            "ROS2 communication messages are unavailable. Source these first:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /opt/bxi/bxi_ros2_pkg/setup.bash"
        ) from error
    return rclpy, Node, ActuatorCmds, ActuatorStates, MotionCommands, Odometry


def _topic_name(topic: str) -> str:
    topic = str(topic).strip()
    if not topic:
        raise ValueError("ROS topic must not be empty")
    return topic if topic.startswith("/") else "/" + topic


def _yaw_from_xyzw(quaternion: Sequence[float]) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _endpoint_name(endpoint: Any) -> tuple[str, str]:
    return (str(getattr(endpoint, "node_name", "")), str(getattr(endpoint, "node_namespace", "")))


def run_bridge(
    trajectory: Optional[dict[str, Any]] = None,
    *,
    report: Optional[dict[str, Any]] = None,
    navigate: bool = False,
    topic: str = DEFAULT_TOPIC,
    state_topic: str = DEFAULT_STATE_TOPIC,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
    motion_topic: str = DEFAULT_MOTION_TOPIC,
    rate_hz: float = DEFAULT_RATE_HZ,
    wait_timeout: float = DEFAULT_TIMEOUT_SEC,
    nav_rate_hz: float = DEFAULT_NAV_RATE_HZ,
    nav_timeout: float = DEFAULT_NAV_TIMEOUT_SEC,
    nav_position_tolerance: float = DEFAULT_NAV_POSITION_TOLERANCE,
    nav_heading_tolerance: float = DEFAULT_NAV_HEADING_TOLERANCE,
    stage_max_joint_speed: float = DEFAULT_STAGE_MAX_JOINT_SPEED,
    stage_min_duration: float = DEFAULT_STAGE_MIN_DURATION_SEC,
    hold_final: bool = False,
    final_hold_sec: float = DEFAULT_FINAL_HOLD_SEC,
    require_subscriber: bool = True,
    dry_run: bool = False,
) -> int:
    """Navigate and/or publish an arm trajectory into the live simulation."""
    if trajectory is None and report is None:
        raise ValueError("either trajectory or report is required")
    if trajectory is not None and report is not None:
        raise ValueError("trajectory and report are mutually exclusive")
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("--rate must be finite and positive")
    if not math.isfinite(wait_timeout) or wait_timeout < 0.0:
        raise ValueError("--wait-timeout must be finite and non-negative")
    if not math.isfinite(nav_rate_hz) or nav_rate_hz <= 0.0:
        raise ValueError("--nav-rate must be finite and positive")
    if not math.isfinite(nav_timeout) or nav_timeout <= 0.0:
        raise ValueError("--nav-timeout must be finite and positive")
    if not math.isfinite(nav_position_tolerance) or nav_position_tolerance <= 0.0:
        raise ValueError("--nav-position-tolerance must be finite and positive")
    if not math.isfinite(nav_heading_tolerance) or nav_heading_tolerance <= 0.0:
        raise ValueError("--nav-heading-tolerance must be finite and positive")
    if not math.isfinite(final_hold_sec) or final_hold_sec < 0.0:
        raise ValueError("--final-hold-sec must be finite and non-negative")
    topic = _topic_name(topic)
    state_topic = _topic_name(state_topic)
    odom_topic = _topic_name(odom_topic)
    motion_topic = _topic_name(motion_topic)

    if report is not None:
        names = list(report["joint_names"])
        frames = None
        duration = None
    else:
        assert trajectory is not None
        names = list(trajectory["joint_names"])
        frames = trajectory["frames"]
        duration = float(frames[-1]["time_s"])

    if dry_run:
        if report is not None:
            detail = "report={}, stance=({}, {})".format(
                report["source"], report["stance_xy"][0], report["stance_xy"][1]
            )
            print(
                "report valid: {} arm joints, {}, navigate={}".format(
                    len(names), detail, navigate
                )
            )
            return 0
        assert duration is not None
        print(
            "trajectory valid: {} joints, {:.3f}s, topic={}".format(
                len(names), duration, topic
            )
        )
        return 0

    rclpy, Node, ActuatorCmds, ActuatorStates, MotionCommands, Odometry = _import_ros()

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("arm_ik_ros_bridge")
            # Sensor-data QoS matches the simulation controller's actuator
            # streams and avoids a reliable/best-effort incompatibility.
            from rclpy.qos import qos_profile_sensor_data

            self.publisher = self.create_publisher(
                ActuatorCmds, topic, qos_profile_sensor_data
            )
            self.subscription = self.create_subscription(
                ActuatorStates,
                state_topic,
                self._state_callback,
                qos_profile_sensor_data,
            )
            self.odom_subscription = None
            self.motion_publisher = None
            if navigate:
                self.motion_publisher = self.create_publisher(
                    MotionCommands, motion_topic, qos_profile_sensor_data
                )
                self.odom_subscription = self.create_subscription(
                    Odometry,
                    odom_topic,
                    self._odom_callback,
                    qos_profile_sensor_data,
                )
            self.state_names: tuple[str, ...] = ()
            self.state_positions: dict[str, float] = {}
            self.last_state_monotonic = 0.0
            self.odom_xy: Optional[tuple[float, float]] = None
            self.odom_yaw: Optional[float] = None
            self.last_odom_monotonic = 0.0

        def _state_callback(self, message: Any) -> None:
            observed = tuple(str(name) for name in message.name)
            if observed:
                self.state_names = observed
                self.state_positions = {
                    name: float(position)
                    for name, position in zip(observed, message.position)
                    if math.isfinite(float(position))
                }
                self.last_state_monotonic = time.monotonic()

        def _odom_callback(self, message: Any) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            xy = (float(position.x), float(position.y))
            yaw = _yaw_from_xyzw((orientation.x, orientation.y, orientation.z, orientation.w))
            if all(math.isfinite(value) for value in (*xy, yaw)):
                self.odom_xy = xy
                self.odom_yaw = yaw
                self.last_odom_monotonic = time.monotonic()

        def publish_frame(
            self,
            position: Iterable[float],
            kp: Iterable[float],
            kd: Iterable[float],
        ) -> None:
            message = ActuatorCmds()
            message.header.frame_id = "elf3"
            message.header.stamp = self.get_clock().now().to_msg()
            message.actuators_name = list(names)
            message.pos = list(position)
            message.kp = list(kp)
            message.kd = list(kd)
            message.vel = [0.0] * len(names)
            message.torque = [0.0] * len(names)
            self.publisher.publish(message)

        def release(self) -> None:
            # An empty named command is the controller's documented release
            # operation; publish repeatedly for DDS delivery reliability.
            message = ActuatorCmds()
            for _ in range(3):
                self.publisher.publish(message)
                time.sleep(0.01)

        def publish_motion(self, vx: float, vy: float, yaw_rate: float) -> None:
            if self.motion_publisher is None:
                return
            message = MotionCommands()
            message.vel_des.x = float(vx)
            message.vel_des.y = float(vy)
            message.yawdot_des = float(yaw_rate)
            self.motion_publisher.publish(message)

        def stop_motion(self) -> None:
            for _ in range(3):
                self.publish_motion(0.0, 0.0, 0.0)
                time.sleep(0.01)

    rclpy.init()
    node: Optional[BridgeNode] = None
    try:
        node = BridgeNode()
        warned_state = False
        warned_subscriber = False
        state_deadline = time.monotonic() + wait_timeout
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=min(0.05, 1.0 / rate_hz))
            now = time.monotonic()
            if node.state_names:
                missing = tuple(name for name in names if name not in node.state_names)
                if missing:
                    raise RuntimeError(
                        "simulation actuator state is missing arm joints: "
                        + ", ".join(missing)
                    )
            elif now >= state_deadline:
                if not warned_state:
                    print(
                        f"arm_ik bridge: no actuator state received on {state_topic}",
                        file=sys.stderr,
                    )
                    warned_state = True
                raise RuntimeError("timed out waiting for simulation actuator state")

            subscribers = node.get_subscriptions_info_by_topic(topic)
            # The bridge itself is a publisher, so this list must contain the
            # dedicated controller subscriber.  Refuse to publish into a topic
            # nobody owns; this catches a forgotten controller option early.
            if require_subscriber and not subscribers:
                if now >= state_deadline:
                    if not warned_subscriber:
                        print(
                            f"arm_ik bridge: no controller subscribes to {topic}; "
                            "enable the simulation arm override first",
                            file=sys.stderr,
                        )
                        warned_subscriber = True
                    raise RuntimeError("no subscriber for arm IK override topic")
                continue
            break

        if report is not None:
            if navigate:
                target_xy = tuple(float(value) for value in report["stance_xy"])
                grasp_xy = report.get("grasp_xy")
                motion_subscribers = node.get_subscriptions_info_by_topic(motion_topic)
                if not motion_subscribers:
                    raise RuntimeError(
                        f"no controller subscribes to {motion_topic}; "
                        "the ELF3 controller must be in the normal gait mode"
                    )
                print(
                    "navigation target=({:.3f}, {:.3f}), motion_topic={}, "
                    "subscribers={}".format(
                        target_xy[0], target_xy[1], motion_topic, len(motion_subscribers)
                    ),
                    flush=True,
                )
                nav_deadline = time.monotonic() + nav_timeout
                nav_period = 1.0 / nav_rate_hz
                # Keep the command stream bounded.  The controller consumes
                # the newest velocity command, so publishing faster than the
                # requested navigation rate only adds DDS traffic and can
                # starve actuator-state callbacks on a loaded machine.
                next_nav_tick = 0.0
                settled_since: Optional[float] = None
                last_nav_report = float("-inf")
                while rclpy.ok():
                    now = time.monotonic()
                    rclpy.spin_once(node, timeout_sec=min(0.05, nav_period))
                    if node.odom_xy is None or node.odom_yaw is None:
                        if now >= nav_deadline:
                            raise RuntimeError(
                                f"timed out waiting for odometry on {odom_topic}"
                            )
                        continue
                    if now >= nav_deadline:
                        node.stop_motion()
                        raise RuntimeError("navigation timed out before the report stance")

                    dx = target_xy[0] - node.odom_xy[0]
                    dy = target_xy[1] - node.odom_xy[1]
                    distance = math.hypot(dx, dy)
                    target_yaw = (
                        math.atan2(grasp_xy[1] - target_xy[1], grasp_xy[0] - target_xy[0])
                        if grasp_xy is not None
                        else math.atan2(dy, dx)
                    )
                    heading_error = _wrap_angle(target_yaw - node.odom_yaw)
                    if now - last_nav_report >= 2.0:
                        print(
                            "navigation odom=({:.3f}, {:.3f}) yaw={:.1f}deg "
                            "distance={:.3f} heading_error={:.1f}deg".format(
                                node.odom_xy[0],
                                node.odom_xy[1],
                                math.degrees(node.odom_yaw),
                                distance,
                                math.degrees(heading_error),
                            ),
                            flush=True,
                        )
                        last_nav_report = now
                    if distance <= nav_position_tolerance and abs(heading_error) <= nav_heading_tolerance:
                        node.stop_motion()
                        if settled_since is None:
                            settled_since = time.monotonic()
                        if time.monotonic() - settled_since >= 0.5:
                            break
                    elif now >= next_nav_tick:
                        settled_since = None
                        # Convert world-frame error into body-frame gait command.
                        c = math.cos(node.odom_yaw)
                        s = math.sin(node.odom_yaw)
                        body_x = c * dx + s * dy
                        body_y = -s * dx + c * dy
                        if distance < 0.20:
                            # The normal gait ignores very small commands.  A
                            # short, bounded push keeps the base moving through
                            # the final 20 cm; the position check stops it before
                            # the hand-off target.
                            speed = NAV_FINAL_APPROACH_SPEED
                        else:
                            speed = min(
                                NAV_MAX_VX,
                                max(NAV_MIN_TRANSLATION_SPEED, 0.65 * distance),
                            )
                        norm = max(distance, 1.0e-9)
                        vx = max(-NAV_MAX_VX, min(NAV_MAX_VX, speed * body_x / norm))
                        vy = max(-NAV_MAX_VY, min(NAV_MAX_VY, speed * body_y / norm))
                        yaw_rate = max(-NAV_MAX_YAW_RATE, min(NAV_MAX_YAW_RATE, 1.0 * heading_error))
                        node.publish_motion(vx, vy, yaw_rate)
                        next_nav_tick = now + nav_period
                node.stop_motion()
                # Let the gait controller complete its braking cycle and let
                # torso oscillations decay before arm commands change the
                # upper-body load.
                settle_until = time.monotonic() + NAV_SETTLE_SEC
                while rclpy.ok() and time.monotonic() < settle_until:
                    node.stop_motion()
                    rclpy.spin_once(node, timeout_sec=0.05)

            # The initial frame is taken from the live actuator state, which
            # makes the first arm command continuous with the gait pose.
            missing = tuple(name for name in names if name not in node.state_positions)
            if missing:
                raise RuntimeError(
                    "simulation actuator state is missing arm joints: "
                    + ", ".join(missing)
                )
            initial = [node.state_positions[name] for name in names]
            trajectory = trajectory_from_report(
                report,
                initial,
                max_joint_speed=stage_max_joint_speed,
                minimum_stage_duration=stage_min_duration,
            )
            frames = trajectory["frames"]
            duration = float(frames[-1]["time_s"])

        assert frames is not None and duration is not None
        started = time.monotonic()
        period = 1.0 / rate_hz
        next_tick = started
        while rclpy.ok():
            now = time.monotonic()
            elapsed = now - started
            # Keep the gait command latched at zero while the arm trajectory
            # is active.  A single stop frame can be lost over DDS, and the
            # controller otherwise continues the last walking command while
            # the arm shifts the centre of mass.
            if navigate:
                node.publish_motion(0.0, 0.0, 0.0)
            position, kp, kd = interpolate_frames(frames, elapsed)
            node.publish_frame(position, kp, kd)
            if elapsed >= duration:
                # Keep the solved pose visible when requested; otherwise hold
                # long enough to settle before releasing the mailbox.
                hold_until = (
                    math.inf
                    if hold_final
                    else now + max(0.25, final_hold_sec)
                )
                while rclpy.ok() and time.monotonic() < hold_until:
                    if navigate:
                        node.publish_motion(0.0, 0.0, 0.0)
                    node.publish_frame(frames[-1]["position"], frames[-1]["kp"], frames[-1]["kd"])
                    rclpy.spin_once(node, timeout_sec=period)
                break
            rclpy.spin_once(node, timeout_sec=0.0)
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if node is not None:
            # Do not let cleanup errors hide the navigation/IK failure (or a
            # Ctrl-C).  Once another callback has shut down the context DDS
            # publication is no longer valid, so skip the best-effort release
            # in that case.
            if rclpy.ok():
                try:
                    node.release()
                except Exception as error:
                    print(f"arm_ik bridge: release failed: {error}", file=sys.stderr)
                if navigate:
                    try:
                        node.stop_motion()
                    except Exception as error:
                        print(
                            f"arm_ik bridge: motion stop failed: {error}",
                            file=sys.stderr,
                        )
            try:
                node.destroy_node()
            except Exception as error:
                print(f"arm_ik bridge: node cleanup failed: {error}", file=sys.stderr)
        if rclpy.ok():
            rclpy.try_shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        help="trajectory JSON (use --report for an arm_ik reach report)",
    )
    parser.add_argument(
        "--report",
        "--report-json",
        dest="report",
        type=Path,
        help=(
            "arm_ik.py --report-json output; derive waypoints from its IK "
            "stages (aliases: --report, --report-json)"
        ),
    )
    parser.add_argument(
        "--navigate",
        action="store_true",
        help="drive the already-running robot to report.plan.stance_xy first",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="dedicated arm override topic")
    parser.add_argument("--state-topic", default=DEFAULT_STATE_TOPIC, help="actuator state topic")
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC, help="world odometry topic")
    parser.add_argument("--motion-topic", default=DEFAULT_MOTION_TOPIC, help="MotionCommands topic")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="publish rate in Hz")
    parser.add_argument("--nav-rate", type=float, default=DEFAULT_NAV_RATE_HZ, help="navigation command rate in Hz")
    parser.add_argument("--nav-timeout", type=float, default=DEFAULT_NAV_TIMEOUT_SEC, help="navigation timeout in seconds")
    parser.add_argument("--nav-position-tolerance", type=float, default=DEFAULT_NAV_POSITION_TOLERANCE, help="navigation XY tolerance in metres")
    parser.add_argument("--nav-heading-tolerance", type=float, default=DEFAULT_NAV_HEADING_TOLERANCE, help="navigation yaw tolerance in radians")
    parser.add_argument("--stage-max-joint-speed", type=float, default=DEFAULT_STAGE_MAX_JOINT_SPEED, help="maximum interpolated arm joint speed in rad/s")
    parser.add_argument("--stage-min-duration", type=float, default=DEFAULT_STAGE_MIN_DURATION_SEC, help="minimum duration per IK waypoint in seconds")
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="seconds to wait for state/subscriber (default: 2)",
    )
    parser.add_argument(
        "--allow-no-subscriber",
        action="store_true",
        help="validation-only/testing escape hatch; not recommended for a live run",
    )
    parser.add_argument(
        "--hold-final",
        action="store_true",
        help="keep publishing the final grasp pose until Ctrl-C",
    )
    parser.add_argument(
        "--final-hold-sec",
        type=float,
        default=DEFAULT_FINAL_HOLD_SEC,
        help="seconds to hold the final pose before releasing (default: 3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate JSON without ROS")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.report is not None and arguments.trajectory is not None:
            parser.error("provide either TRAJECTORY or --report, not both")
        if arguments.report is None and arguments.trajectory is None:
            parser.error("provide TRAJECTORY or --report")
        if arguments.navigate and arguments.report is None:
            parser.error("--navigate requires --report")
        report = load_report(arguments.report) if arguments.report is not None else None
        trajectory = (
            None
            if report is not None
            else load_trajectory(arguments.trajectory)
        )
        return run_bridge(
            trajectory,
            report=report,
            navigate=arguments.navigate,
            topic=arguments.topic,
            state_topic=arguments.state_topic,
            odom_topic=arguments.odom_topic,
            motion_topic=arguments.motion_topic,
            rate_hz=arguments.rate,
            wait_timeout=arguments.wait_timeout,
            nav_rate_hz=arguments.nav_rate,
            nav_timeout=arguments.nav_timeout,
            nav_position_tolerance=arguments.nav_position_tolerance,
            nav_heading_tolerance=arguments.nav_heading_tolerance,
            stage_max_joint_speed=arguments.stage_max_joint_speed,
            stage_min_duration=arguments.stage_min_duration,
            hold_final=arguments.hold_final,
            final_hold_sec=arguments.final_hold_sec,
            require_subscriber=not arguments.allow_no_subscriber,
            dry_run=arguments.dry_run,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"arm_ik_ros_bridge: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
