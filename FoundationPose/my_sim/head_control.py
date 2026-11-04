#!/usr/bin/env python3
"""Minimal ELF3 real-hardware head controller.

The controller accepts only the two named head joints and republishes their
position targets often enough to stay inside the hardware override timeout.
It does not change the robot state machine; put the robot in the appropriate
zero-position/PD state before starting this node.

Before running on hardware, source the ROS and BXI workspaces and verify that
the hardware node is ``hardware_elf3_head`` (31-DoF configuration).  The
default targets are deliberately small and conservative::

    python3 my_sim/head_control.py

Use ``--duration`` for a bounded test and ``Ctrl-C`` to release the override::

    python3 my_sim/head_control.py --yaw 0.20 --pitch -0.10 --duration 10
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import rclpy
from communication.msg import ActuatorCmds
from rclpy.node import Node


TOPIC_NAME = "/hardware/actuators_cmds_override"
JOINT_NAMES = ["head_z_joint", "head_y_joint"]
DEFAULT_KP = 16.747
DEFAULT_KD = 1.066
DEFAULT_RATE_HZ = 20.0

# ELF3 mechanical limits.  Keep the command-line defaults well inside them.
YAW_LIMIT_RAD = 0
PITCH_LIMIT_RAD = 0.785


def _bounded(value: float, limit: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or abs(value) > limit:
        raise ValueError(
            f"{name} must be finite and within +/-{limit:.3f} rad; got {value!r}"
        )
    return value


class HeadController(Node):
    """Publish a named-joint position override for the real ELF3 hardware."""

    def __init__(
        self,
        *,
        yaw: float = 0.20,
        pitch: float = -0.10,
        rate_hz: float = DEFAULT_RATE_HZ,
    ) -> None:
        super().__init__("head_controller")
        self.publisher = self.create_publisher(ActuatorCmds, TOPIC_NAME, 10)
        self.yaw = _bounded(yaw, YAW_LIMIT_RAD, "yaw")
        self.pitch = _bounded(pitch, PITCH_LIMIT_RAD, "pitch")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        self.rate_hz = float(rate_hz)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_command)
        self.get_logger().info(
            f"publishing head override on {TOPIC_NAME} at {self.rate_hz:g} Hz: "
            f"yaw={self.yaw:+.3f} rad, pitch={self.pitch:+.3f} rad"
        )

    def publish_command(self) -> None:
        """Publish one complete command; array order follows JOINT_NAMES."""
        message = ActuatorCmds()
        message.actuators_name = list(JOINT_NAMES)
        message.pos = [self.yaw, self.pitch]
        message.kp = [DEFAULT_KP, DEFAULT_KP]
        message.kd = [DEFAULT_KD, DEFAULT_KD]
        # vel and torque are intentionally omitted and therefore remain zero.
        self.publisher.publish(message)

    def release_override(self) -> None:
        """Clear the named override immediately instead of waiting 0.2 s."""
        self.publisher.publish(ActuatorCmds())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw", type=float, default=0.20, help="head_z_joint target (rad)")
    parser.add_argument("--pitch", type=float, default=-0.10, help="head_y_joint target (rad)")
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE_HZ, help="publish rate (Hz)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds (default: run until Ctrl-C)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0.0
    ):
        raise SystemExit("--duration must be finite and positive")

    rclpy.init()
    node: Optional[HeadController] = None
    try:
        node = HeadController(yaw=args.yaw, pitch=args.pitch, rate_hz=args.rate)
        if args.duration is None:
            rclpy.spin(node)
        else:
            end_time = node.get_clock().now().nanoseconds + int(args.duration * 1e9)
            while rclpy.ok() and node.get_clock().now().nanoseconds < end_time:
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.release_override()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
