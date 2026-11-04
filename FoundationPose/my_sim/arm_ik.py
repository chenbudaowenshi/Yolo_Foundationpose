#!/usr/bin/env python3
"""Walk the ELF3 humanoid to a perceived object and reach it with arm IK.

The pipeline has three stages that mirror the existing tooling in this
repository:

1. Read ``T_world_object`` from the JSON that ``my_demo/get_pose_mujoco.py``
   writes after FoundationPose estimates the target pose.
2. Drive the BXI AMP learned gait from ``my_sim/bxi_gait_controller.py`` with a
   closed-loop heading/distance controller until the torso stands within arm
   reach of the object.
3. Hand the stance over to a kinematic-base model and solve the arm with the
   damped least-squares Jacobian IK already provided by ``my_sim/get_sim.py``
   (``solve_hand_ik``) through a pre-grasp standoff and a final grasp pose.

Why two models?  Every arm hinge is also part of ``POLICY_JOINT_ROWS``, so the
AMP policy rewrites its own PD targets for those joints 50 times a second and
observes their deviation as balance feedback.  Overriding the arm targets while
the policy runs makes the robot walk away from its stance and eventually fall
(measured: the torso drifts several metres).  ``my_sim/get_sim.py`` already
encodes this constraint by refusing to combine ``--ik-target`` with
``--rl-walk``.  This pipeline follows the same rule: the gait policy owns the
approach, then the joint configuration and torso pose transfer to a
``mobile_base`` model whose kinematic torso holds still while the arm moves.

The ELF3 hand in ``elf3_hand.xml`` is a single collision sphere with no
actuated fingers, so "grasp" here means placing the hand TCP on the object's
grasp point.  Closing a gripper needs a hand model that has finger joints.

Run inside the Python 3.8 ``mj`` environment, which owns MuJoCo and
onnxruntime::

    /home/robot/anaconda3/envs/mj/bin/python my_sim/arm_ik.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MY_SIM_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MY_SIM_DIR.parent
MJ_PYTHON = "/home/robot/anaconda3/envs/mj/bin/python"
ROS_PYTHON = "/home/robot/anaconda3/envs/pose/bin/python"
ROS_DOMAIN_ID_MIN = 0
ROS_DOMAIN_ID_MAX = 232


def _restart_with_mj_python() -> None:
    """Re-exec under the mj environment, which owns MuJoCo and onnxruntime.

    Mirrors the same trick as ``my_sim/head_control_mujoco.py``, so the script
    works from whichever Conda environment the user happens to have active.
    """
    if __name__ != "__main__" or not os.path.isfile(MJ_PYTHON):
        return
    if os.path.realpath(sys.executable) == os.path.realpath(MJ_PYTHON):
        return
    try:
        import mujoco  # noqa: F401  (probe only)

        if getattr(sys.modules["mujoco"], "__file__", None) is not None:
            return
    except ImportError:
        pass

    print(
        "Switching from {} to {} for MuJoCo + onnxruntime...".format(
            sys.executable, MJ_PYTHON
        ),
        file=sys.stderr,
        flush=True,
    )
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    os.execve(
        MJ_PYTHON,
        [MJ_PYTHON, os.path.abspath(__file__), *sys.argv[1:]],
        environment,
    )


_restart_with_mj_python()
# Append rather than insert: my_sim/mujoco/ holds MJCF assets and has no
# __init__.py, so putting my_sim first would let it win the name "mujoco" as a
# PEP 420 namespace package.  The real extension module would then be shadowed
# and "import mujoco" would succeed with an empty module, surfacing later as a
# confusing AttributeError instead of ModuleNotFoundError.
if str(MY_SIM_DIR) not in sys.path:
    sys.path.append(str(MY_SIM_DIR))

import get_sim  # noqa: E402  (path bootstrap above keeps sibling imports working)

DEFAULT_POSE_JSON = REPOSITORY_ROOT / "runs/foundationpose_mujoco/pose.json"

# Measured by sampling the seven arm hinges: the TCP reaches at most 0.622 m
# from the shoulder body.  Stand closer than that so IK keeps a usable margin.
ARM_REACH_LIMIT = 0.622
DEFAULT_STANDOFF_DISTANCE = 0.45
DEFAULT_PREGRASP_OFFSET = 0.12

# The gait policy settles around this torso height once it stops walking.
STANDING_TORSO_HEIGHT = 1.036

# Gait tuning for the approach controller.  The AMP policy accepts body-frame
# vx, vy and yaw rate, clipped to COMMAND_MINIMUM/COMMAND_MAXIMUM.
POSITION_TOLERANCE = 0.09
HEADING_TOLERANCE = 0.20
TURN_IN_PLACE_THRESHOLD = 0.35
APPROACH_TIMEOUT_SEC = 40.0
SETTLE_DURATION_SEC = 1.5

# Once the stance position is reached, only resume walking if the torso drifts
# past this multiple of the tolerance.  Without hysteresis the controller
# flip-flops between seeking position and seeking heading, and the AMP policy's
# yaw deadband turns that into a limit cycle that never converges.
POSITION_HYSTERESIS = 1.6

# Cap fine heading alignment.  The reach stage re-pins the torso yaw toward the
# object regardless, so a few degrees of residual heading error costs nothing;
# spinning in place chasing it would just burn the approach budget.
HEADING_ALIGN_TIMEOUT_SEC = 4.0

# The bottle body origin sits at its base, so lift the grasp point into the
# cylindrical body instead of clamping the neck or the floor contact.
DEFAULT_GRASP_HEIGHT_OFFSET = 0.10


def verify_runtime(mujoco: Any) -> None:
    """Fail early and clearly when the interpreter lacks a usable MuJoCo.

    The ``pose`` and ``foundationpose`` environments have no ``mujoco`` package.
    Because ``my_sim/mujoco/`` is a directory of MJCF assets, an import there
    can silently resolve to a namespace package whose ``__file__`` is None, so
    the failure would otherwise appear as a missing attribute much later.
    """
    if getattr(mujoco, "__file__", None) is None:
        raise RuntimeError(
            "'import mujoco' resolved to the asset directory my_sim/mujoco "
            "instead of the MuJoCo package, which means this interpreter "
            "({}) has no MuJoCo installed.\nRun the pipeline with the mj "
            "environment instead:\n  {} my_sim/arm_ik.py".format(
                sys.executable, MJ_PYTHON
            )
        )
    if not hasattr(mujoco, "MjSpec"):
        raise RuntimeError(
            "MuJoCo {} has no MjSpec, which this pipeline needs to build the "
            "robot-only scene.\nUse the mj environment (MuJoCo 3.2.3):\n"
            "  {} my_sim/arm_ik.py".format(
                getattr(mujoco, "__version__", "?"), MJ_PYTHON
            )
        )


def load_object_pose(pose_path: Path) -> Dict[str, Any]:
    """Read ``T_world_object`` from the perception pipeline's JSON record.

    ``my_demo/get_pose_mujoco.py`` only writes ``T_world_object`` when it runs
    with ``--world-pose`` and receives odometry, so a missing key means the
    camera pose was never resolved into the world frame.
    """
    pose_path = pose_path.expanduser().resolve()
    if not pose_path.is_file():
        raise FileNotFoundError(
            "object pose JSON not found: {}\n"
            "Run my_demo/get_pose_mujoco.py --world-pose first.".format(pose_path)
        )
    with pose_path.open("r", encoding="utf-8") as pose_file:
        record = json.load(pose_file)

    if "T_world_object" not in record:
        raise RuntimeError(
            "{} has no 'T_world_object'; rerun the pose node with --world-pose "
            "so the camera pose is resolved through odometry".format(pose_path)
        )
    quality = record.get("pose_quality", {})
    if quality and not quality.get("valid", True):
        raise RuntimeError(
            "the recorded pose is flagged invalid ({}); "
            "rerun perception before reaching".format(quality.get("reason", "unknown"))
        )

    matrix = record["T_world_object"]["matrix"]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise RuntimeError("T_world_object.matrix must be a 4x4 nested list")

    return {
        "matrix": matrix,
        "translation": (
            float(matrix[0][3]),
            float(matrix[1][3]),
            float(matrix[2][3]),
        ),
        "class_name": str(record.get("target", {}).get("class_name", "object")),
        "extents": record.get("mesh", {}).get("extents_m"),
        "yaw_observable": bool(record.get("symmetry", {}).get("yaw_observable", True)),
        "source": str(pose_path),
    }


def grasp_point_from_pose(
    numpy: Any, pose: Dict[str, Any], height_offset: float
) -> Any:
    """Lift the reported object origin to a graspable point on the body.

    FoundationPose reports the CAD origin, which for the procedural bottle mesh
    sits at the bottom centre.  Raising it by ``height_offset`` targets the
    cylindrical body rather than the base or the narrow neck.
    """
    grasp = numpy.asarray(pose["translation"], dtype=numpy.float64).copy()
    grasp[2] += float(height_offset)
    return grasp


def approach_frame(
    numpy: Any, base_xy: Any, grasp_point: Any
) -> Tuple[float, Any]:
    """Return the yaw that faces the object and the horizontal approach axis."""
    delta = numpy.asarray(grasp_point[:2], dtype=numpy.float64) - numpy.asarray(
        base_xy, dtype=numpy.float64
    )
    planar_distance = float(numpy.linalg.norm(delta))
    if planar_distance <= 1.0e-6:
        raise RuntimeError(
            "the torso is directly under the object, so the approach "
            "direction is undefined"
        )
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    axis = numpy.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=numpy.float64)
    return yaw, axis


def standing_target(
    numpy: Any, grasp_point: Any, standoff_distance: float
) -> Any:
    """Pick the torso ground position that keeps the grasp point in reach.

    The robot stops on the line between its spawn point and the object, so the
    object stays in front of the torso instead of beside it.
    """
    grasp_xy = numpy.asarray(grasp_point[:2], dtype=numpy.float64)
    radius = float(numpy.linalg.norm(grasp_xy))
    if radius <= 1.0e-6:
        raise RuntimeError("the object is at the world origin; cannot pick a stance")
    if standoff_distance >= radius:
        # Already close enough that walking outward would be counterproductive.
        return numpy.zeros(2, dtype=numpy.float64)
    return grasp_xy * (1.0 - standoff_distance / radius)


def base_yaw(numpy: Any, quaternion_wxyz: Any) -> float:
    """Extract the world yaw from a MuJoCo ``w x y z`` base quaternion."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ObjectApproachController:
    """Steer the AMP gait toward a standing pose in front of the object.

    The learned policy only accepts body-frame velocities, so the object pose
    is turned into a heading error and a distance error.  Large heading errors
    turn in place first; otherwise forward speed and yaw rate blend so the walk
    curves onto the target.
    """

    def __init__(
        self,
        numpy: Any,
        gait_controller: Any,
        root_qpos_address: int,
        target_xy: Any,
        move_speed: float,
        turn_speed: float,
        position_tolerance: float = POSITION_TOLERANCE,
        heading_tolerance: float = HEADING_TOLERANCE,
    ) -> None:
        if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive and finite")
        if not math.isfinite(heading_tolerance) or heading_tolerance <= 0.0:
            raise ValueError("heading_tolerance must be positive and finite")
        self.numpy = numpy
        self.gait_controller = gait_controller
        self.root = int(root_qpos_address)
        self.target_xy = numpy.asarray(target_xy, dtype=numpy.float64).reshape(2)
        self.move_speed = float(move_speed)
        self.turn_speed = float(turn_speed)
        self.position_tolerance = float(position_tolerance)
        self.heading_tolerance = float(heading_tolerance)
        self.final_yaw: Optional[float] = None
        self.arrived = False
        # Latched once the stance is first reached, so small position jitter no
        # longer restarts the walk phase.
        self.in_position = False
        self.align_deadline: Optional[float] = None

    def base_state(self, data: Any) -> Tuple[Any, float]:
        """Return the planar torso position and its world yaw."""
        position = self.numpy.asarray(
            data.qpos[self.root : self.root + 2], dtype=self.numpy.float64
        ).copy()
        yaw = base_yaw(self.numpy, data.qpos[self.root + 3 : self.root + 7])
        return position, yaw

    def set_final_heading(self, grasp_point: Any) -> None:
        """Face the object once the stance position has been reached."""
        self.final_yaw = math.atan2(
            float(grasp_point[1]) - float(self.target_xy[1]),
            float(grasp_point[0]) - float(self.target_xy[0]),
        )

    def update(self, data: Any) -> Dict[str, float]:
        """Issue one velocity command and report the remaining errors."""
        numpy = self.numpy
        position, yaw = self.base_state(data)
        delta = self.target_xy - position
        distance = float(numpy.linalg.norm(delta))

        # Hysteresis: tight tolerance to declare arrival, looser one to give it
        # up again.  The AMP gait cannot hold a position to millimetres, so a
        # single threshold makes the walk/align modes oscillate forever.
        if not self.in_position:
            self.in_position = distance <= self.position_tolerance
        elif distance > self.position_tolerance * POSITION_HYSTERESIS:
            self.in_position = False
            self.align_deadline = None

        if not self.in_position:
            heading_error = wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - yaw)
            if abs(heading_error) > TURN_IN_PLACE_THRESHOLD:
                # Turning first avoids long curved detours and keeps the
                # object inside the head camera's field of view.
                self.gait_controller.set_command(
                    0.0,
                    0.0,
                    float(numpy.clip(1.5 * heading_error, -self.turn_speed, self.turn_speed)),
                )
            else:
                self.gait_controller.set_command(
                    float(numpy.clip(0.9 * distance, 0.10, self.move_speed)),
                    0.0,
                    float(
                        numpy.clip(
                            1.2 * heading_error,
                            -0.75 * self.turn_speed,
                            0.75 * self.turn_speed,
                        )
                    ),
                )
            return {"distance": distance, "heading_error": heading_error, "arrived": 0.0}

        # Within the stance tolerance: rotate to face the object, then hold.
        if self.final_yaw is None:
            self.gait_controller.set_command(0.0, 0.0, 0.0)
            self.arrived = True
            return {"distance": distance, "heading_error": 0.0, "arrived": 1.0}

        if self.align_deadline is None:
            self.align_deadline = float(data.time) + HEADING_ALIGN_TIMEOUT_SEC

        heading_error = wrap_angle(self.final_yaw - yaw)
        aligned = abs(heading_error) <= self.heading_tolerance
        if not aligned and float(data.time) < self.align_deadline:
            self.gait_controller.set_command(
                0.0,
                0.0,
                float(numpy.clip(1.5 * heading_error, -self.turn_speed, self.turn_speed)),
            )
            return {"distance": distance, "heading_error": heading_error, "arrived": 0.0}

        self.gait_controller.set_command(0.0, 0.0, 0.0)
        self.arrived = True
        return {"distance": distance, "heading_error": heading_error, "arrived": 1.0}


def choose_grasp_side(
    numpy: Any,
    mujoco: Any,
    model: Any,
    data: Any,
    grasp_point: Any,
    requested_side: str,
) -> str:
    """Pick the arm whose TCP already sits closer to the grasp point."""
    if requested_side != "auto":
        return requested_side
    best_side = None
    best_distance = float("inf")
    for side, (site_name, _, _) in get_sim.TCP_SITES.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        distance = float(numpy.linalg.norm(grasp_point - data.site_xpos[site_id]))
        if distance < best_distance:
            best_distance = distance
            best_side = side
    if best_side is None:
        raise RuntimeError("no hand TCP sites are present in the compiled model")
    return best_side


def grasp_orientation(mujoco: Any, numpy: Any, approach_axis: Any, side: str) -> Any:
    """Build a TCP quaternion whose +X axis points along the approach.

    The recorded pose declares the bottle axially symmetric
    (``symmetry.yaw_observable`` is false), so the object's own yaw carries no
    information.  The hand orientation therefore comes from the robot-to-object
    approach direction: TCP +X reaches forward and the remaining axes are built
    from world up, mirrored per side so each wrist curls inward.
    """
    forward = numpy.asarray(approach_axis, dtype=numpy.float64).copy()
    forward[2] = 0.0
    forward_norm = float(numpy.linalg.norm(forward))
    if forward_norm <= 1.0e-9:
        raise RuntimeError("the approach axis is vertical; cannot build a TCP frame")
    forward /= forward_norm

    world_up = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float64)
    lateral = numpy.cross(world_up, forward)
    lateral /= float(numpy.linalg.norm(lateral))
    if side == "right":
        # Mirror the wrist so both hands curl toward the object rather than
        # rolling the right wrist through its +/-0.7854 rad limit.
        lateral = -lateral

    rotation = numpy.column_stack((forward, lateral, numpy.cross(forward, lateral)))
    quaternion = numpy.zeros(4, dtype=numpy.float64)
    mujoco.mju_mat2Quat(quaternion, numpy.ascontiguousarray(rotation.reshape(9)))
    return quaternion


def reach_object(
    mujoco: Any,
    numpy: Any,
    model: Any,
    data: Any,
    grasp_point: Any,
    approach_axis: Any,
    side: str,
    pregrasp_offset: float,
    use_orientation: bool,
    ik_arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Walk the hand along the approach axis, keeping the last reachable pose.

    Every waypoint lies on the approach axis so the hand slides in horizontally
    instead of dropping onto the object from above.  Solves run in sequence
    without resetting ``qpos``, so each one warm-starts from the previous
    solution; that matters because ``solve_hand_ik`` caps its per-iteration
    joint step, and a cold start from the walking pose can stall short of a
    target that is reachable via intermediate poses.

    If a later waypoint fails, the last solved configuration is kept and
    reported rather than discarded, because that is still the closest pose the
    arm can actually hold.
    """
    quaternion = (
        grasp_orientation(mujoco, numpy, approach_axis, side)
        if use_orientation
        else None
    )
    axis = numpy.asarray(approach_axis, dtype=numpy.float64)
    target_point = numpy.asarray(grasp_point, dtype=numpy.float64)
    offset = float(pregrasp_offset)

    # An intermediate step between pre-grasp and grasp keeps each solve's
    # displacement small, which is what makes the warm start effective.
    waypoints = [("pregrasp", target_point - axis * offset)]
    if offset > 0.0:
        waypoints.append(("approach", target_point - axis * (offset * 0.5)))
    waypoints.append(("grasp", target_point))

    stages = []
    for label, target in waypoints:
        snapshot = data.qpos.copy()
        attempts = [(label, quaternion)]
        if quaternion is not None:
            # Orientation is the usual culprit at the edge of the workspace, so
            # fall back to position-only rather than losing the waypoint.
            attempts.append((label + "_position_only", None))

        for attempt_label, attempt_quaternion in attempts:
            try:
                result = get_sim.solve_hand_ik(
                    mujoco,
                    numpy,
                    model,
                    data,
                    side=side,
                    target_position=target,
                    target_quaternion=attempt_quaternion,
                    **ik_arguments,
                )
            except RuntimeError as error:
                data.qpos[:] = snapshot
                mujoco.mj_forward(model, data)
                stages.append(
                    {"stage": attempt_label, "solved": False, "detail": str(error)}
                )
                continue

            if attempt_quaternion is None:
                # Stop asking for orientation on later waypoints too, otherwise
                # the next solve fights the compromise this one just made.
                quaternion = None
            stages.append(
                {
                    "stage": attempt_label,
                    "solved": True,
                    "target": numpy.asarray(target).tolist(),
                    "iterations": int(result["iterations"]),
                    "position_error": float(result["position_error"]),
                    "rotation_error": float(result["rotation_error"]),
                    "achieved_position": result["achieved_position"].tolist(),
                    "joint_names": list(result["joint_names"]),
                    "joint_positions": result["joint_positions"].tolist(),
                }
            )
            break

    solved = [stage for stage in stages if stage.get("solved")]
    return {
        "side": side,
        "used_orientation": quaternion is not None,
        "stages": stages,
        "reached": bool(solved and solved[-1]["stage"].startswith("grasp")),
        "final_stage": solved[-1] if solved else None,
    }


def capture_stance(mujoco: Any, model: Any, data: Any, gait_controller: Any) -> Dict[str, Any]:
    """Record the torso pose and joint angles reached at the end of the walk."""
    root = gait_controller._root_qpos_address
    joint_angles = {}
    for joint_name in gait_controller.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_angles[joint_name] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
    return {
        "position": (float(data.qpos[root]), float(data.qpos[root + 1])),
        # MuJoCo stores a free-joint quaternion immediately after xyz in
        # wxyz order.  Keep the measured heading in the report so a live
        # bridge can place the simulator at the same stance without having to
        # infer it from the (usually approximate) planned target.
        "yaw_rad": float(base_yaw(None, data.qpos[root + 3 : root + 7])),
        "torso_height": float(data.qpos[root + 2]),
        "joint_angles": joint_angles,
        "simulated_time_s": float(data.time),
    }


def build_reach_model(mujoco: Any, numpy: Any, scene: str) -> Tuple[Any, Any]:
    """Compile the kinematic-base twin used for the reach stage.

    ``mobile_base=True`` replaces the free root joint with a mocap body, so the
    torso stays exactly where it is placed while the arm moves.  That removes
    the balance coupling that makes arm IK unusable under the gait policy.
    """
    model = get_sim.load_mjcf_model(
        mujoco,
        get_sim.SCENE_MODELS[scene],
        robot_only=True,
        mobile_base=True,
        add_ground=True,
        ground_priority=1,
    )
    get_sim.configure_elf3_joint_damping(mujoco, model)
    if model.nmocap < 1:
        raise RuntimeError("the reach model has no mocap body to pin the torso")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data


def stance_position_for_reach(
    numpy: Any,
    walked_xy: Any,
    grasp_point: Any,
    standoff_distance: float,
) -> Tuple[Any, float]:
    """Pull the pinned torso onto the intended standoff along its own heading.

    The AMP gait only parks within ``POSITION_TOLERANCE`` of the stance target,
    and that residual is enough to push a low or far grasp point out of the
    arm's workspace.  Because the reach stage pins the torso kinematically
    anyway, the leftover approach-direction error can be removed here: keep the
    heading the robot actually walked to and correct only the distance.
    """
    walked_xy = numpy.asarray(walked_xy, dtype=numpy.float64).reshape(2)
    grasp_xy = numpy.asarray(grasp_point[:2], dtype=numpy.float64)
    delta = grasp_xy - walked_xy
    distance = float(numpy.linalg.norm(delta))
    if distance <= 1.0e-6:
        raise RuntimeError(
            "the torso is directly under the object, so the approach "
            "direction is undefined"
        )
    direction = delta / distance
    corrected = grasp_xy - direction * float(standoff_distance)
    return corrected, float(numpy.linalg.norm(corrected - walked_xy))


def apply_stance(
    mujoco: Any,
    numpy: Any,
    model: Any,
    data: Any,
    stance: Dict[str, Any],
    grasp_point: Any,
    base_xy: Any,
) -> float:
    """Place the kinematic torso at ``base_xy``, facing the object."""
    for joint_name, angle in stance["joint_angles"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(
                "the reach model is missing joint '{}'".format(joint_name)
            )
        data.qpos[int(model.jnt_qposadr[joint_id])] = angle

    base_xy = numpy.asarray(base_xy, dtype=numpy.float64).reshape(2)
    yaw, _ = approach_frame(numpy, base_xy, grasp_point)
    data.mocap_pos[0] = (base_xy[0], base_xy[1], stance["torso_height"])
    data.mocap_quat[0] = (
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return yaw


def run_pipeline(arguments: argparse.Namespace) -> int:
    """Walk to the perceived object and reach it, then report what happened."""
    mujoco, numpy = get_sim.import_runtime()
    verify_runtime(mujoco)

    pose = load_object_pose(arguments.pose_json)
    grasp_point = grasp_point_from_pose(numpy, pose, arguments.grasp_height_offset)
    stance_xy = standing_target(numpy, grasp_point, arguments.standoff_distance)

    print("Target: {} from {}".format(pose["class_name"], pose["source"]))
    print(
        "  T_world_object translation = ({:.3f}, {:.3f}, {:.3f}) m".format(*pose["translation"])
    )
    print("  grasp point = ({:.3f}, {:.3f}, {:.3f}) m".format(*grasp_point))
    print(
        "  stance target = ({:.3f}, {:.3f}) m, standoff = {:.2f} m".format(
            stance_xy[0], stance_xy[1], arguments.standoff_distance
        )
    )
    if not pose["yaw_observable"]:
        print(
            "  object yaw is unobservable (axial symmetry); the hand approach "
            "direction comes from the robot-to-object vector"
        )

    ik_arguments = {
        "max_iterations": arguments.ik_max_iterations,
        "tolerance": arguments.ik_tolerance,
        "damping": arguments.ik_damping,
    }

    stance, approach_status = walk_to_stance(
        mujoco, numpy, arguments, grasp_point, stance_xy
    )
    ik_report, reach_model, reach_data, hold_status = reach_from_stance(
        mujoco, numpy, arguments, grasp_point, stance, ik_arguments
    )
    status = hold_status if approach_status == "arrived" else approach_status

    achieved = report_final_state(
        mujoco, numpy, reach_model, reach_data, grasp_point, stance, ik_report, status
    )

    if arguments.viewer and arguments.live_ros:
        raise RuntimeError("--live-ros cannot be combined with --viewer")
    if arguments.viewer:
        show_reach_pose(mujoco, numpy, reach_model, reach_data)

    report_path = arguments.report_json
    if arguments.live_ros and report_path is None:
        # Keep a deterministic hand-off artifact so the child ROS process can
        # consume exactly the measurements produced by this planner run.
        report_path = REPOSITORY_ROOT / "runs/arm_ik/reach_report.json"

    if report_path is not None:
        # Resolve before both the atomic write and the child ROS process.  The
        # child runs from the repository root, while users commonly pass a
        # path relative to ``my_sim`` (for example ``../runs/...``).
        report_path = report_path.expanduser().resolve()
        write_report(
            report_path, pose, grasp_point, stance_xy, stance, ik_report, achieved
        )
        print("Report written to {}".format(report_path))

    if arguments.live_ros:
        if report_path is None:  # defensive; live_ros always assigns one above
            raise RuntimeError("--live-ros requires a report path")
        return run_live_ros_bridge(report_path, arguments)

    return 0 if status == "reached" else 1


def walk_to_stance(
    mujoco: Any,
    numpy: Any,
    arguments: argparse.Namespace,
    grasp_point: Any,
    stance_xy: Any,
) -> Tuple[Dict[str, Any], str]:
    """Phase 1: walk the physical floating-base robot to its stance."""
    model = get_sim.load_mjcf_model(
        mujoco,
        get_sim.SCENE_MODELS[arguments.scene],
        robot_only=True,
        add_ground=True,
        ground_priority=0,
    )
    data = mujoco.MjData(model)
    gait_controller_class = get_sim.import_gait_controller_class()
    gait_controller = gait_controller_class(
        mujoco,
        numpy,
        model,
        data,
        model_path=arguments.gait_model,
        move_speed=arguments.move_speed,
        turn_speed=arguments.turn_speed,
    )

    approach = ObjectApproachController(
        numpy,
        gait_controller,
        gait_controller._root_qpos_address,
        stance_xy,
        move_speed=arguments.move_speed,
        turn_speed=arguments.turn_speed,
    )
    approach.set_final_heading(grasp_point)

    timestep = float(model.opt.timestep)
    approach_steps = int(math.ceil(arguments.approach_timeout / timestep))
    settle_steps = int(math.ceil(SETTLE_DURATION_SEC / timestep))

    status = "approach_timeout"
    arrival_step: Optional[int] = None
    for step in range(approach_steps + settle_steps):
        state = approach.update(data)
        if arrival_step is None and approach.arrived:
            arrival_step = step
            status = "arrived"
            position, yaw = approach.base_state(data)
            print(
                "Arrived after {:.2f}s at ({:.3f}, {:.3f}), yaw={:.1f}deg, "
                "residual distance={:.3f} m".format(
                    data.time,
                    position[0],
                    position[1],
                    math.degrees(yaw),
                    state["distance"],
                )
            )
        elif arrival_step is None and step >= approach_steps:
            gait_controller.set_command(0.0, 0.0, 0.0)
            arrival_step = step
            print(
                "Approach timed out after {:.1f}s with {:.3f} m remaining; "
                "reaching from wherever the robot stopped".format(
                    arguments.approach_timeout, state["distance"]
                )
            )
        elif arrival_step is not None and step - arrival_step >= settle_steps:
            # The torso needs a moment to stop swaying, otherwise the captured
            # stance describes a pose the robot has already left.
            break

        gait_controller.step(data)
        mujoco.mj_step(model, data)
        if not numpy.isfinite(data.qpos).all():
            raise RuntimeError("gait simulation diverged: qpos is not finite")

    stance = capture_stance(mujoco, model, data, gait_controller)
    if stance["torso_height"] < 0.85:
        status = "fell_while_walking"
    print(
        "Stance captured at ({:.3f}, {:.3f}), torso height {:.3f} m, "
        "after {:.2f}s of walking".format(
            stance["position"][0],
            stance["position"][1],
            stance["torso_height"],
            stance["simulated_time_s"],
        )
    )
    gait_controller.close()
    return stance, status


def reach_from_stance(
    mujoco: Any,
    numpy: Any,
    arguments: argparse.Namespace,
    grasp_point: Any,
    stance: Dict[str, Any],
    ik_arguments: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Any, Any, str]:
    """Phase 2+3: transfer the stance to the kinematic model and reach."""
    model, data = build_reach_model(mujoco, numpy, arguments.scene)
    reach_xy, correction = stance_position_for_reach(
        numpy, stance["position"], grasp_point, arguments.standoff_distance
    )
    # The gait can stop inside its position tolerance.  IK then pins the
    # kinematic torso at a corrected XY position so the arm sees the exact
    # standoff used by the solved joint vectors.  Preserve that position in
    # the hand-off report; a live replay must navigate to this pose rather
    # than the uncorrected walked stance.
    stance["reach_position"] = (
        float(reach_xy[0]),
        float(reach_xy[1]),
    )
    stance["reach_correction_m"] = float(correction)
    yaw = apply_stance(mujoco, numpy, model, data, stance, grasp_point, reach_xy)
    _, approach_axis = approach_frame(numpy, reach_xy, grasp_point)
    print(
        "Reach model: kinematic torso pinned at ({:.3f}, {:.3f}), yaw {:.1f}deg "
        "({:.3f} m stance correction; the gait policy is released so the arm "
        "can move)".format(reach_xy[0], reach_xy[1], math.degrees(yaw), correction)
    )

    side = choose_grasp_side(numpy, mujoco, model, data, grasp_point, arguments.ik_side)
    ik_report = reach_object(
        mujoco,
        numpy,
        model,
        data,
        grasp_point,
        approach_axis,
        side,
        arguments.pregrasp_offset,
        not arguments.ignore_orientation,
        ik_arguments,
    )

    print("IK arm: {} hand".format(ik_report["side"]))
    for stage in ik_report["stages"]:
        if stage.get("solved"):
            print(
                "  {:<26s} solved in {:3d} iterations, position error {:.5f} m".format(
                    stage["stage"], stage["iterations"], stage["position_error"]
                )
            )
        else:
            print("  {:<26s} failed: {}".format(stage["stage"], stage["detail"]))

    if ik_report["final_stage"] is None:
        print("No reachable arm configuration; the arm stays at its walking pose.")
        return ik_report, model, data, "ik_failed"

    # Hold the solved configuration with torque control so the reported TCP is
    # a physically settled pose rather than a raw kinematic assignment.
    hold_controller = get_sim.elf3_joint_hold_controller(mujoco, numpy, model, data)
    hold_steps = int(math.ceil(arguments.hold_duration / float(model.opt.timestep)))
    for _ in range(hold_steps):
        hold_controller(data)
        mujoco.mj_step(model, data)
        if not numpy.isfinite(data.qpos).all():
            raise RuntimeError("reach simulation diverged: qpos is not finite")

    return (
        ik_report,
        model,
        data,
        "reached" if ik_report["reached"] else "pregrasp_only",
    )

def show_reach_pose(mujoco: Any, numpy: Any, model: Any, data: Any) -> None:
    """Keep the solved pose on screen under the joint-hold controller."""
    hold_controller = get_sim.elf3_joint_hold_controller(mujoco, numpy, model, data)
    environment = get_sim.MujocoEnvironment(
        mujoco,
        numpy,
        model,
        data,
        source="arm_ik reach pose",
        controller=hold_controller,
        robot_only=True,
        base_mode="mobile",
        has_ground=True,
        base_driver=None,
        gait_controller=None,
        ik_result=None,
    )
    try:
        environment.run_viewer(duration=None, maximum_steps=None, realtime_factor=1.0)
    except RuntimeError as error:
        print("Viewer unavailable: {}".format(error))


def report_final_state(
    mujoco: Any,
    numpy: Any,
    model: Any,
    data: Any,
    grasp_point: Any,
    stance: Dict[str, Any],
    ik_report: Optional[Dict[str, Any]],
    status: str,
) -> Dict[str, Any]:
    """Print and return the measured outcome after the hold phase."""
    torso_height = float(stance["torso_height"])
    standing = torso_height > 0.85

    achieved: Dict[str, Any] = {
        "status": status,
        "walk_time_s": float(stance["simulated_time_s"]),
        "stance_position": [float(value) for value in stance["position"]],
        "torso_height_m": torso_height,
        "still_standing": standing,
    }

    if ik_report is not None and ik_report["final_stage"] is not None:
        side = ik_report["side"]
        site_name = get_sim.TCP_SITES[side][0]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        tcp = numpy.asarray(data.site_xpos[site_id], dtype=numpy.float64).copy()
        # Measured after the hold phase, so this includes any sag the joint-hold
        # torques allow rather than just the kinematic IK residual above.
        error = float(numpy.linalg.norm(grasp_point - tcp))
        achieved["tcp_position"] = tcp.tolist()
        achieved["tcp_to_grasp_error_m"] = error
        achieved["side"] = side
        print(
            "Executed {} TCP = ({:.3f}, {:.3f}, {:.3f}), "
            "distance to grasp point = {:.3f} m".format(side, tcp[0], tcp[1], tcp[2], error)
        )

    print(
        "Final status: {} | torso height {:.3f} m ({})".format(
            status, torso_height, "standing" if standing else "FELL"
        )
    )
    print(
        "Note: elf3_hand.xml has no actuated fingers, so the hand is placed on "
        "the grasp point but cannot close around it."
    )
    return achieved


def write_report(
    report_path: Path,
    pose: Dict[str, Any],
    grasp_point: Any,
    stance_xy: Any,
    stance: Dict[str, Any],
    ik_report: Optional[Dict[str, Any]],
    achieved: Dict[str, Any],
) -> None:
    """Persist the run for comparison against the perception record."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "object": {
            "class_name": pose["class_name"],
            "pose_source": pose["source"],
            "T_world_object_translation": list(pose["translation"]),
            "yaw_observable": pose["yaw_observable"],
        },
        "plan": {
            "grasp_point": [float(value) for value in grasp_point],
            "stance_xy": [float(value) for value in stance_xy],
        },
        "walked_stance": {
            "position": [float(value) for value in stance["position"]],
            "yaw_rad": float(stance.get("yaw_rad", 0.0)),
            "torso_height_m": float(stance["torso_height"]),
            "walk_time_s": float(stance["simulated_time_s"]),
            # Preserve the complete 29-DOF pose.  The arm IK stages only
            # contain the seven arm joints, so this map is required for a
            # process that replays the result in the ROS2 simulator.
            "joint_angles": {
                str(name): float(angle)
                for name, angle in stance.get("joint_angles", {}).items()
            },
        },
        "reach_stance": {
            "position": [
                float(value)
                for value in stance.get("reach_position", stance["position"])
            ],
            "correction_m": float(stance.get("reach_correction_m", 0.0)),
        },
        "ik": ik_report,
        "result": achieved,
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")
    temporary.replace(report_path)


def run_live_ros_bridge(
    report_path: Path,
    arguments: argparse.Namespace,
) -> int:
    """Replay a completed plan through the controller in an existing ROS sim.

    ``arm_ik.py`` intentionally remains a MuJoCo/onnxruntime (Python 3.8)
    planner.  ROS 2 and the generated ``communication`` messages live in the
    Python 3.10 environment, so the bridge is launched as a child process
    after sourcing the three ROS workspaces.  No MuJoCo viewer is created here;
    the already-running ROS2 MuJoCo process remains the only window.
    """
    # ``run_live_ros_bridge`` launches the child with ``cwd`` set to the
    # repository root.  Resolve the hand-off artifact here so a path supplied
    # from ``my_sim`` (for example ``../runs/arm_ik/reach_report.json``) is not
    # interpreted a second time relative to the child working directory.
    report_path = report_path.expanduser().resolve()

    bridge = MY_SIM_DIR / "arm_ik_ros_bridge.py"
    if not bridge.is_file():
        raise FileNotFoundError("ROS bridge not found: {}".format(bridge))
    if not os.path.isfile(ROS_PYTHON):
        raise RuntimeError(
            "ROS Python interpreter not found: {}".format(ROS_PYTHON)
        )

    setup_files = (
        Path("/opt/ros/humble/setup.bash"),
        Path("/opt/bxi/bxi_ros2_pkg/setup.bash"),
        REPOSITORY_ROOT / "bxi_rl_controller_ros2_example/install/setup.bash",
    )
    missing = [str(path) for path in setup_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            "cannot start the ROS bridge; missing setup files:\n  "
            + "\n  ".join(missing)
        )

    bridge_args = [
        ROS_PYTHON,
        str(bridge),
        "--report",
        str(report_path),
        "--wait-timeout",
        str(arguments.ros_wait_timeout),
        "--final-hold-sec",
        str(arguments.ros_final_hold_sec),
    ]
    if arguments.ros_navigate:
        bridge_args.append("--navigate")
    if arguments.ros_hold_final:
        bridge_args.append("--hold-final")

    # The standard bxi_teleop_v2 launcher uses domain 88 in its tmux panes.
    # An explicit CLI value wins over a stale ROS_DOMAIN_ID inherited from a
    # shell (for example a perception terminal that uses domain 91).  When no
    # CLI value is supplied, preserve an already-exported domain and fall back
    # to the documented simulation default.
    environment = os.environ.copy()
    configured_domain = arguments.ros_domain_id
    if configured_domain is None:
        configured_domain = int(environment.get("ROS_DOMAIN_ID", "88"))
    environment["ROS_DOMAIN_ID"] = str(configured_domain)
    environment.setdefault("ROS_LOCALHOST_ONLY", "1")
    environment.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    command = "source {} && source {} && source {} && exec {}".format(
        *(shlex.quote(str(path)) for path in setup_files),
        " ".join(shlex.quote(str(value)) for value in bridge_args),
    )
    print(
        "Live ROS replay: existing MuJoCo window via {} (domain={})".format(
            "/simulation/arm_ik_cmds_override", configured_domain
        ),
        flush=True,
    )
    completed = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pose-json",
        type=Path,
        default=DEFAULT_POSE_JSON,
        help="pose record written by my_demo/get_pose_mujoco.py --world-pose",
    )
    parser.add_argument(
        "--scene",
        choices=tuple(get_sim.SCENE_MODELS),
        default="elf3_hand",
        help="MJCF to load (default: elf3_hand, which has hand meshes)",
    )
    parser.add_argument(
        "--ik-side",
        choices=("auto", "left", "right"),
        default="auto",
        help="arm used for the reach; auto picks the closer hand (default: auto)",
    )
    parser.add_argument(
        "--standoff-distance",
        type=float,
        default=DEFAULT_STANDOFF_DISTANCE,
        help=(
            "horizontal torso-to-object distance to stand at, in metres "
            "(default: {:.2f}; the arm reaches {:.2f} m from the shoulder)".format(
                DEFAULT_STANDOFF_DISTANCE, ARM_REACH_LIMIT
            )
        ),
    )
    parser.add_argument(
        "--pregrasp-offset",
        type=float,
        default=DEFAULT_PREGRASP_OFFSET,
        help="pre-grasp retreat along the approach axis in metres (default: 0.12)",
    )
    parser.add_argument(
        "--grasp-height-offset",
        type=float,
        default=DEFAULT_GRASP_HEIGHT_OFFSET,
        help=(
            "metres to raise the grasp point above the reported object origin; "
            "the procedural bottle mesh has its origin at the base (default: 0.10)"
        ),
    )
    parser.add_argument(
        "--ignore-orientation",
        action="store_true",
        help="solve position-only IK and let the wrist settle wherever it lands",
    )
    parser.add_argument(
        "--approach-timeout",
        type=float,
        default=APPROACH_TIMEOUT_SEC,
        help="give up walking after this many simulated seconds (default: 25)",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=4.0,
        help="simulated seconds to hold the reach before reporting (default: 4)",
    )
    parser.add_argument(
        "--gait-model",
        type=Path,
        default=get_sim.BXI_GAIT_MODEL,
        help="AMP locomotion ONNX policy",
    )
    parser.add_argument(
        "--move-speed",
        type=float,
        default=0.45,
        help="approach walking speed in m/s (default: 0.45)",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=0.80,
        help="approach yaw rate in rad/s (default: 0.80)",
    )
    parser.add_argument(
        "--ik-max-iterations",
        type=int,
        default=300,
        help="maximum damped-least-squares iterations (default: 300)",
    )
    parser.add_argument(
        "--ik-tolerance",
        type=float,
        default=1.0e-4,
        help="IK convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1.0e-2,
        help="IK damping coefficient (default: 1e-2)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="write the plan and measured result to this JSON path",
    )
    parser.add_argument(
        "--live-ros",
        action="store_true",
        help=(
            "after planning, replay the walk/reach stages through the already "
            "running ROS2 MuJoCo window; does not open a viewer"
        ),
    )
    parser.add_argument(
        "--ros-navigate",
        action="store_true",
        help=(
            "with --live-ros, drive the existing ROS robot to the reported "
            "stance using /motion_commands before arm IK (default: off)"
        ),
    )
    parser.add_argument(
        "--ros-hold-final",
        action="store_true",
        help="with --live-ros, keep the final grasp command until Ctrl-C",
    )
    parser.add_argument(
        "--ros-final-hold-sec",
        type=float,
        default=3.0,
        help="live ROS final-pose hold before release (default: 3 s)",
    )
    parser.add_argument(
        "--ros-wait-timeout",
        type=float,
        default=5.0,
        help="live ROS startup/state timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help=(
            "ROS_DOMAIN_ID for --live-ros (overrides the environment); if omitted, "
            "use the exported value or 88"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="accepted for symmetry with get_sim.py; this pipeline never opens a viewer",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the MuJoCo viewer on the final reach pose so it can be inspected",
    )
    return parser


def validate_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if not math.isfinite(arguments.standoff_distance) or arguments.standoff_distance <= 0.0:
        parser.error("--standoff-distance must be positive and finite")
    if arguments.standoff_distance >= ARM_REACH_LIMIT:
        parser.error(
            "--standoff-distance {:.2f} m is at or beyond the measured arm reach "
            "of {:.2f} m, so IK cannot converge".format(
                arguments.standoff_distance, ARM_REACH_LIMIT
            )
        )
    if not math.isfinite(arguments.pregrasp_offset) or arguments.pregrasp_offset < 0.0:
        parser.error("--pregrasp-offset must be non-negative and finite")
    if not math.isfinite(arguments.grasp_height_offset):
        parser.error("--grasp-height-offset must be finite")
    if not math.isfinite(arguments.approach_timeout) or arguments.approach_timeout <= 0.0:
        parser.error("--approach-timeout must be positive and finite")
    if not math.isfinite(arguments.hold_duration) or arguments.hold_duration <= 0.0:
        parser.error("--hold-duration must be positive and finite")
    if not math.isfinite(arguments.move_speed) or arguments.move_speed <= 0.0:
        parser.error("--move-speed must be positive and finite")
    if not math.isfinite(arguments.turn_speed) or arguments.turn_speed <= 0.0:
        parser.error("--turn-speed must be positive and finite")
    if arguments.ik_max_iterations <= 0:
        parser.error("--ik-max-iterations must be positive")
    if not math.isfinite(arguments.ik_tolerance) or arguments.ik_tolerance <= 0.0:
        parser.error("--ik-tolerance must be positive and finite")
    if not math.isfinite(arguments.ik_damping) or arguments.ik_damping <= 0.0:
        parser.error("--ik-damping must be positive and finite")
    if not math.isfinite(arguments.ros_final_hold_sec) or arguments.ros_final_hold_sec < 0.0:
        parser.error("--ros-final-hold-sec must be finite and non-negative")
    if not math.isfinite(arguments.ros_wait_timeout) or arguments.ros_wait_timeout < 0.0:
        parser.error("--ros-wait-timeout must be finite and non-negative")
    if arguments.ros_domain_id is not None and (
        arguments.ros_domain_id < ROS_DOMAIN_ID_MIN
        or arguments.ros_domain_id > ROS_DOMAIN_ID_MAX
    ):
        parser.error("--ros-domain-id must be between 0 and 232")


def main(argv: Optional[Any] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(arguments, parser)
    try:
        return run_pipeline(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        # These are the expected setup failures (no pose file, invalid record,
        # missing policy).  A traceback would bury the actionable message.
        print("arm_ik: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
