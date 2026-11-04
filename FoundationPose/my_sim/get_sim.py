#!/usr/bin/env python3
"""Run Elf3 learned locomotion or hand IK on a small ground plane.

The default model is ``my_sim/mujoco/elf3.xml``.  Use ``--scene elf3_hand``
to load the local hand-mesh variant.  For the built-in models, the loader keeps
only the ``torso_link`` robot subtree and adds a checker ground.  With no IK
request, the default mode runs the BXI ``amp_terrain.onnx`` learned humanoid
gait against the physical floating-base robot.  IK requests use the stable
planar mocap base unless a base mode is selected explicitly.  In the viewer,
W/S move forward/backward, A/D rotate, and X/Space stops.  Commands remain
active until another motion key or the stop key is pressed.  The source XML
files are not rewritten, and no URDF conversion is performed here.

Run from the ``mj`` Conda environment::

    conda activate mj
    python my_sim/get_sim.py

Position IK example (world-frame XYZ in metres)::

    python my_sim/get_sim.py --scene elf3_hand \
        --ik-side left --ik-target 0.42 0.25 1.02
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple


MY_SIM_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = MY_SIM_DIR / "mujoco"
BXI_GAIT_MODEL = (
    MY_SIM_DIR.parent
    / "bxi_rl_controller_ros2_example"
    / "src"
    / "bxi_example_py_elf3"
    / "mods"
    / "com.bxi.basic_actions"
    / "assets"
    / "amp_terrain.onnx"
)
SCENE_MODELS = {
    "elf3": MUJOCO_DIR / "elf3.xml",
    "elf3_hand": MUJOCO_DIR / "elf3_hand.xml",
}
ARM_JOINT_NAMES = {
    "left": (
        "l_shoulder_y_joint",
        "l_shoulder_x_joint",
        "l_shoulder_z_joint",
        "l_elbow_y_joint",
        "l_wrist_x_joint",
        "l_wrist_y_joint",
        "l_wrist_z_joint",
    ),
    "right": (
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
    ),
}
TCP_SITES = {
    "left": ("l_hand_tcp", "l_wrist_z_link", (0.11, -0.01, 0.0)),
    "right": ("r_hand_tcp", "r_wrist_z_link", (0.11, 0.01, 0.0)),
}
STANDING_TORSO_HEIGHT = 1.0625
DEFAULT_MOVE_SPEED = 0.40
DEFAULT_TURN_SPEED = 0.80

Controller = Callable[[Any], None]


def import_runtime() -> Tuple[Any, Any]:
    """Import MuJoCo lazily so ``--help`` works outside the mj environment."""
    try:
        mujoco = importlib.import_module("mujoco")
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MuJoCo is not installed in the active Python environment.\n"
            "Run: conda activate mj\n"
            "Then: python my_sim/get_sim.py"
        ) from error
    return mujoco, numpy


def add_default_ground(
    mujoco: Any, spec: Any, ground_priority: int = 1
) -> None:
    """Add a compact checker plane and a fixed overhead light."""
    if ground_priority not in (0, 1):
        raise ValueError("ground_priority must be 0 or 1")
    texture = spec.add_texture()
    texture.name = "default_ground_checker"
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    texture.rgb1[:] = (0.22, 0.25, 0.28)
    texture.rgb2[:] = (0.42, 0.45, 0.48)
    texture.width = 512
    texture.height = 512
    texture.nchannel = 3
    texture.gridsize[:] = (1, 1)
    texture.random = 0.01

    material = spec.add_material()
    material.name = "default_ground_material"
    material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    material.texuniform = 1
    material.texrepeat[:] = (20.0, 20.0)
    material.reflectance = 0.05

    ground = spec.worldbody.add_geom()
    ground.name = "default_ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.pos[:] = (0.0, 0.0, 0.0)
    ground.size[:] = (10.0, 10.0, 0.1)
    ground.material = material.name
    ground.condim = 3
    # The learned gait needs priority=0 so the feet's priority=1 friction
    # matches the BXI training model.  The kinematic modes retain priority=1.
    ground.priority = ground_priority
    ground.friction[:] = (1.0, 0.01, 0.001)

    light = spec.worldbody.add_light()
    light.name = "default_overhead_light"
    light.pos[:] = (0.0, 0.0, 4.0)
    light.dir[:] = (0.0, 0.0, -1.0)
    light.ambient[:] = (0.25, 0.25, 0.25)
    light.diffuse[:] = (0.7, 0.7, 0.7)


def load_mjcf_model(
    mujoco: Any,
    model_path: Path,
    fixed_base: bool = False,
    robot_only: bool = False,
    mobile_base: bool = False,
    add_ground: bool = False,
    ground_priority: int = 1,
) -> Any:
    """Load MJCF and optionally build a compact Elf3 ground scene."""
    if fixed_base and mobile_base:
        raise ValueError("fixed_base and mobile_base are mutually exclusive")
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError("MuJoCo model not found: {}".format(model_path))
    if not fixed_base and not robot_only and not mobile_base and not add_ground:
        return mujoco.MjModel.from_xml_path(str(model_path))

    spec = mujoco.MjSpec()
    spec.from_file(str(model_path))

    if robot_only:
        world = spec.worldbody
        torso_found = False
        body = world.first_body()
        while body is not None:
            next_body = world.next_body(body)
            if body.name == "torso_link":
                torso_found = True
            else:
                spec.detach_body(body)
            body = next_body
        if not torso_found:
            raise RuntimeError("robot-only loading requires a 'torso_link' body")

        # Terrain XML can also add static world geoms, markers, and lights that
        # are not children of a body.  Remove those while keeping everything
        # mounted inside the torso_link robot subtree.
        for element_kind in ("geom", "site", "camera", "light", "frame"):
            first_element = getattr(world, "first_{}".format(element_kind))
            next_element = getattr(world, "next_{}".format(element_kind))
            element = first_element()
            while element is not None:
                following_element = next_element(element)
                element.delete()
                element = following_element

        # Drop terrain-only visual assets too, including its skybox, so the
        # compiled model contains robot resources rather than a hidden scene.
        for asset_kind in ("material", "texture"):
            for asset in list(getattr(spec, asset_kind)):
                if asset.name.startswith("ut_"):
                    asset.delete()

        # Add explicit hand reference points for Jacobian IK.  They are added
        # only to the in-memory model, leaving the user's source MJCF intact.
        for side, (site_name, body_name, site_position) in TCP_SITES.items():
            wrist = spec.find_body(body_name)
            if wrist is None:
                raise RuntimeError(
                    "{} arm is missing end-effector body '{}'".format(
                        side, body_name
                    )
                )
            site = wrist.add_site(name=site_name)
            site.pos[:] = site_position
            site.size[:] = (0.012, 0.0, 0.0)
            site.group = 4
            if side == "left":
                site.rgba[:] = (0.1, 0.7, 1.0, 1.0)
            else:
                site.rgba[:] = (1.0, 0.45, 0.1, 1.0)

    torso = spec.find_body("torso_link")
    if (fixed_base or mobile_base or add_ground) and torso is None:
        raise RuntimeError("Elf3 scene requires a 'torso_link' body")

    if add_ground:
        add_default_ground(mujoco, spec, ground_priority=ground_priority)
        torso.pos[2] = STANDING_TORSO_HEIGHT

    if fixed_base or mobile_base:
        root_joint = torso.first_joint()
        if root_joint is None or root_joint.type != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Elf3 MJCF does not contain a floating root joint")
        root_joint.delete()
        if mobile_base:
            torso.mocap = True

    return spec.compile()


def elf3_joint_hold_controller(
    mujoco: Any, numpy: Any, model: Any, data: Any
) -> Controller:
    """Hold the initial Elf3 joint angles without providing balance control."""
    # The terrain also contains freely moving rocks.  Only Elf3's named root
    # joint determines whether gravity compensation is valid for the robot.
    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint"
    )
    floating_base = (
        root_joint_id >= 0
        and model.jnt_type[root_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    )
    controlled_joints = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise RuntimeError(
                "Elf3 actuator {} is not a joint actuator".format(actuator_id)
            )
        if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
            raise RuntimeError(
                "Elf3 actuator {} is not a torque motor".format(actuator_id)
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        controlled_joints.append(
            (actuator_id, qpos_address, dof_address, float(data.qpos[qpos_address]))
        )
    if not controlled_joints:
        raise RuntimeError("Elf3 MJCF does not contain controllable hinge motors")

    def control(sim_data: Any) -> None:
        for actuator_id, qpos_address, dof_address, target in controlled_joints:
            torque = 45.0 * (target - sim_data.qpos[qpos_address])
            # qfrc_bias assumes zero base acceleration, so it is valid for the
            # fixed-base inspection mode but not for a freely falling robot.
            if not floating_base:
                torque += float(sim_data.qfrc_bias[dof_address])
            if model.actuator_ctrllimited[actuator_id]:
                lower, upper = model.actuator_ctrlrange[actuator_id]
                torque = float(numpy.clip(torque, lower, upper))
            sim_data.ctrl[actuator_id] = torque

    return control


def configure_elf3_joint_damping(mujoco: Any, model: Any) -> None:
    """Add stable implicit damping without an explicit high-gain D term."""
    minimum_damping = 0.5
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        dof_address = int(model.jnt_dofadr[joint_id])
        model.dof_damping[dof_address] = max(
            float(model.dof_damping[dof_address]), minimum_damping
        )


class KeyboardBaseController:
    """Drive an upright mocap torso over the XY plane from viewer keys."""

    _FORWARD_KEYS = (ord("W"), 265)  # W or Up
    _BACKWARD_KEYS = (ord("S"), 264)  # S or Down
    _LEFT_KEYS = (ord("A"), 263)  # A or Left
    _RIGHT_KEYS = (ord("D"), 262)  # D or Right
    _STOP_KEYS = (ord("X"), 32)  # X or Space

    def __init__(
        self,
        mujoco: Any,
        numpy: Any,
        model: Any,
        move_speed: float,
        turn_speed: float,
    ) -> None:
        torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        if torso_id < 0:
            raise RuntimeError("keyboard drive requires a 'torso_link' body")
        mocap_id = int(model.body_mocapid[torso_id])
        if mocap_id < 0:
            raise RuntimeError("keyboard drive requires a mocap torso_link")
        self.numpy = numpy
        self.mocap_id = mocap_id
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self._lock = threading.Lock()

    def set_command(self, linear_velocity: float, angular_velocity: float) -> None:
        """Set planar body-frame speed in m/s and yaw speed in rad/s."""
        with self._lock:
            self.linear_velocity = float(linear_velocity)
            self.angular_velocity = float(angular_velocity)

    def stop(self) -> None:
        self.set_command(0.0, 0.0)

    def handle_key(self, keycode: int) -> None:
        """Handle GLFW-style key codes supplied by launch_passive."""
        if keycode in self._FORWARD_KEYS:
            self.set_command(self.move_speed, 0.0)
            description = "forward"
        elif keycode in self._BACKWARD_KEYS:
            self.set_command(-self.move_speed, 0.0)
            description = "backward"
        elif keycode in self._LEFT_KEYS:
            self.set_command(0.0, self.turn_speed)
            description = "rotate left"
        elif keycode in self._RIGHT_KEYS:
            self.set_command(0.0, -self.turn_speed)
            description = "rotate right"
        elif keycode in self._STOP_KEYS:
            self.stop()
            description = "stop"
        else:
            return
        print("Keyboard command: {}".format(description), flush=True)

    def advance(self, data: Any, timestep: float) -> None:
        """Integrate the active command into mocap position and yaw."""
        with self._lock:
            linear_velocity = self.linear_velocity
            angular_velocity = self.angular_velocity

        quaternion = data.mocap_quat[self.mocap_id]
        w, x, y, z = quaternion
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        yaw += angular_velocity * timestep

        position = data.mocap_pos[self.mocap_id]
        position[0] += linear_velocity * math.cos(yaw) * timestep
        position[1] += linear_velocity * math.sin(yaw) * timestep
        position[0:2] = self.numpy.clip(position[0:2], -9.0, 9.0)
        position[2] = STANDING_TORSO_HEIGHT
        quaternion[:] = (
            math.cos(0.5 * yaw),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
        )


def solve_hand_ik(
    mujoco: Any,
    numpy: Any,
    model: Any,
    data: Any,
    side: str,
    target_position: Any,
    target_quaternion: Optional[Any] = None,
    max_iterations: int = 300,
    tolerance: float = 1.0e-4,
    damping: float = 1.0e-2,
) -> dict[str, Any]:
    """Solve one arm with damped least-squares MuJoCo Jacobian IK.

    Positions are expressed in the MuJoCo world frame.  If supplied, the
    target quaternion uses MuJoCo's ``w x y z`` order.  Only the selected
    arm's seven hinge joints are updated.
    """
    if side not in ARM_JOINT_NAMES:
        raise ValueError("IK side must be 'left' or 'right'")
    if max_iterations <= 0:
        raise ValueError("IK max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("IK tolerance must be positive and finite")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("IK damping must be positive and finite")

    target = numpy.asarray(target_position, dtype=numpy.float64)
    if target.shape != (3,) or not numpy.isfinite(target).all():
        raise ValueError("IK target position must contain three finite values")

    quaternion = None
    if target_quaternion is not None:
        quaternion = numpy.asarray(target_quaternion, dtype=numpy.float64)
        if quaternion.shape != (4,) or not numpy.isfinite(quaternion).all():
            raise ValueError("IK target quaternion must contain four finite values")
        quaternion_norm = float(numpy.linalg.norm(quaternion))
        if quaternion_norm <= 1.0e-12:
            raise ValueError("IK target quaternion must be non-zero")
        quaternion = numpy.ascontiguousarray(quaternion / quaternion_norm)

    site_name = TCP_SITES[side][0]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError("IK site '{}' is missing from the model".format(site_name))

    joint_ids = []
    qpos_addresses = []
    dof_addresses = []
    for joint_name in ARM_JOINT_NAMES[side]:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError("IK joint '{}' is missing".format(joint_name))
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError("IK joint '{}' is not a hinge".format(joint_name))
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    jacobian_position = numpy.zeros((3, model.nv), dtype=numpy.float64)
    jacobian_rotation = numpy.zeros((3, model.nv), dtype=numpy.float64)
    rotation_error = numpy.zeros(3, dtype=numpy.float64)
    current_quaternion = numpy.zeros(4, dtype=numpy.float64)
    orientation_weight = 0.25
    maximum_joint_step = 0.20
    starting_qpos = data.qpos.copy()
    starting_qvel = data.qvel.copy()

    position_error_norm = math.inf
    rotation_error_norm = 0.0
    for iteration in range(max_iterations + 1):
        mujoco.mj_forward(model, data)
        position_error = target - data.site_xpos[site_id]
        position_error_norm = float(numpy.linalg.norm(position_error))

        if quaternion is not None:
            mujoco.mju_mat2Quat(
                current_quaternion, data.site_xmat[site_id]
            )
            mujoco.mju_subQuat(
                rotation_error, quaternion, current_quaternion
            )
            # mju_subQuat expresses the difference in the current TCP frame,
            # while mj_jacSite returns angular velocity in the world frame.
            rotation_error[:] = (
                data.site_xmat[site_id].reshape(3, 3) @ rotation_error
            )
            rotation_error_norm = float(numpy.linalg.norm(rotation_error))
        else:
            rotation_error[:] = 0.0
            rotation_error_norm = 0.0

        if (
            position_error_norm <= tolerance
            and rotation_error_norm <= tolerance
        ):
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mju_mat2Quat(
                current_quaternion, data.site_xmat[site_id]
            )
            return {
                "side": side,
                "site": site_name,
                "iterations": iteration,
                "position_error": position_error_norm,
                "rotation_error": rotation_error_norm,
                "achieved_position": data.site_xpos[site_id].copy(),
                "achieved_quaternion": current_quaternion.copy(),
                "joint_names": ARM_JOINT_NAMES[side],
                "joint_positions": numpy.asarray(
                    [data.qpos[address] for address in qpos_addresses],
                    dtype=numpy.float64,
                ),
            }
        if iteration == max_iterations:
            break

        mujoco.mj_jacSite(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        arm_jacobian_position = jacobian_position[:, dof_addresses]
        if quaternion is None:
            task_jacobian = arm_jacobian_position
            task_error = position_error
        else:
            task_jacobian = numpy.vstack(
                (
                    arm_jacobian_position,
                    orientation_weight
                    * jacobian_rotation[:, dof_addresses],
                )
            )
            task_error = numpy.concatenate(
                (position_error, orientation_weight * rotation_error)
            )

        regularized = (
            task_jacobian @ task_jacobian.T
            + (damping * damping)
            * numpy.eye(task_jacobian.shape[0], dtype=numpy.float64)
        )
        try:
            task_step = numpy.linalg.solve(regularized, task_error)
        except numpy.linalg.LinAlgError as error:
            data.qpos[:] = starting_qpos
            data.qvel[:] = starting_qvel
            mujoco.mj_forward(model, data)
            raise RuntimeError("IK linear solve failed: {}".format(error)) from error
        joint_step = task_jacobian.T @ task_step
        joint_step_norm = float(numpy.linalg.norm(joint_step))
        if joint_step_norm > maximum_joint_step:
            joint_step *= maximum_joint_step / joint_step_norm

        for index, (joint_id, qpos_address) in enumerate(
            zip(joint_ids, qpos_addresses)
        ):
            joint_value = float(data.qpos[qpos_address] + joint_step[index])
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                joint_value = float(numpy.clip(joint_value, lower, upper))
            data.qpos[qpos_address] = joint_value

    data.qpos[:] = starting_qpos
    data.qvel[:] = starting_qvel
    mujoco.mj_forward(model, data)
    raise RuntimeError(
        "{} hand IK did not converge after {} iterations: "
        "position error={:.6f}m, rotation error={:.6f}rad".format(
            side,
            max_iterations,
            position_error_norm,
            rotation_error_norm,
        )
    )


class MujocoEnvironment:
    """Small wrapper around ``MjModel`` and ``MjData`` simulation loops."""

    def __init__(
        self,
        mujoco: Any,
        numpy: Any,
        model: Any,
        data: Any,
        source: str,
        controller: Optional[Controller],
        robot_only: bool,
        base_mode: str,
        has_ground: bool,
        base_driver: Optional[KeyboardBaseController],
        gait_controller: Optional[Any],
        ik_result: Optional[dict[str, Any]],
    ) -> None:
        self.mujoco = mujoco
        self.numpy = numpy
        self.model = model
        self.data = data
        self.source = source
        self.controller = controller
        self.robot_only = robot_only
        self.base_mode = base_mode
        self.fixed_base = base_mode == "fixed"
        self.mobile_base = base_mode == "mobile"
        self.gait_mode = base_mode == "gait"
        self.has_ground = has_ground
        self.base_driver = base_driver
        self.gait_controller = gait_controller
        self.ik_result = ik_result
        self.initial_qpos = data.qpos.copy()
        self.initial_mocap_pos = data.mocap_pos.copy()
        self.initial_mocap_quat = data.mocap_quat.copy()

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.mocap_pos[:] = self.initial_mocap_pos
        self.data.mocap_quat[:] = self.initial_mocap_quat
        if self.base_driver is not None:
            self.base_driver.stop()
        self.mujoco.mj_forward(self.model, self.data)
        if self.gait_controller is not None:
            self.gait_controller.reset(self.data, reset_pose=False)

    def step(self) -> None:
        if self.base_driver is not None:
            self.base_driver.advance(self.data, self.model.opt.timestep)
        if self.gait_controller is not None:
            self.gait_controller.step(self.data)
        elif self.controller is not None:
            self.controller(self.data)
        self.mujoco.mj_step(self.model, self.data)
        if not self.numpy.isfinite(self.data.qpos).all():
            raise RuntimeError("simulation diverged: qpos contains non-finite values")
        if not self.numpy.isfinite(self.data.qvel).all():
            raise RuntimeError("simulation diverged: qvel contains non-finite values")

    def _finished(
        self,
        start_simulation_time: float,
        completed_steps: int,
        duration: Optional[float],
        maximum_steps: Optional[int],
    ) -> bool:
        if duration is not None and self.data.time - start_simulation_time >= duration:
            return True
        return maximum_steps is not None and completed_steps >= maximum_steps

    def run_headless(
        self, duration: Optional[float], maximum_steps: Optional[int]
    ) -> int:
        if duration is None and maximum_steps is None:
            maximum_steps = 1000
        start_simulation_time = float(self.data.time)
        completed_steps = 0
        while not self._finished(
            start_simulation_time, completed_steps, duration, maximum_steps
        ):
            self.step()
            completed_steps += 1
        return completed_steps

    def run_viewer(
        self,
        duration: Optional[float],
        maximum_steps: Optional[int],
        realtime_factor: float,
    ) -> int:
        try:
            viewer_module = importlib.import_module("mujoco.viewer")
            if self.gait_controller is not None:
                key_callback = self.gait_controller.handle_key
            elif self.base_driver is not None:
                key_callback = self.base_driver.handle_key
            else:
                key_callback = None
            viewer_context = viewer_module.launch_passive(
                self.model,
                self.data,
                key_callback=key_callback,
            )
        except Exception as error:
            raise RuntimeError(
                "cannot start the MuJoCo viewer; check DISPLAY/GLFW, or use --headless"
            ) from error

        start_simulation_time = float(self.data.time)
        completed_steps = 0
        with viewer_context as viewer:
            with viewer.lock():
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -18.0
                torso_id = self.mujoco.mj_name2id(
                    self.model,
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    "torso_link",
                )
                if torso_id >= 0:
                    viewer.cam.lookat[:] = self.data.xpos[torso_id]
                    viewer.cam.lookat[2] -= 0.35
                    viewer.cam.distance = 2.6
                    viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = torso_id
                    viewer.cam.fixedcamid = -1
                else:
                    viewer.cam.lookat[:] = self.model.stat.center
                    viewer.cam.distance = max(
                        0.5, 1.35 * float(self.model.stat.extent)
                    )
            while viewer.is_running() and not self._finished(
                start_simulation_time, completed_steps, duration, maximum_steps
            ):
                wall_step_started = time.perf_counter()
                self.step()
                completed_steps += 1
                viewer.sync()
                target_period = self.model.opt.timestep / realtime_factor
                remaining = target_period - (time.perf_counter() - wall_step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
        return completed_steps

    def close(self) -> None:
        if self.gait_controller is not None:
            self.gait_controller.close()


def import_gait_controller_class() -> Any:
    """Import the Python-3.8-compatible BXI policy adapter lazily."""
    module_name = (
        "{}.bxi_gait_controller".format(__package__)
        if __package__
        else "bxi_gait_controller"
    )
    return importlib.import_module(module_name).BxiAmpGaitController


def build_environment(args: argparse.Namespace) -> MujocoEnvironment:
    mujoco, numpy = import_runtime()
    if args.model is not None:
        model_path = args.model.expanduser().resolve()
    else:
        model_path = SCENE_MODELS[args.scene]

    resolved_path = model_path.resolve()
    is_elf3 = resolved_path in {
        scene_path.resolve() for scene_path in SCENE_MODELS.values()
    }
    if is_elf3:
        default_base_mode = "mobile" if args.ik_target is not None else "gait"
    else:
        default_base_mode = "source"
    base_mode = args.base_mode or default_base_mode
    if base_mode != "source" and not is_elf3:
        raise RuntimeError(
            "base-mode options are only supported for elf3.xml and elf3_hand.xml"
        )
    if base_mode == "gait" and args.ik_target is not None:
        raise RuntimeError(
            "hand IK cannot be combined with --rl-walk; use --mobile-base "
            "or --fixed-base for IK"
        )
    if base_mode == "gait" and args.no_control:
        raise RuntimeError("--no-control cannot be combined with --rl-walk")

    model = load_mjcf_model(
        mujoco,
        resolved_path,
        fixed_base=base_mode == "fixed",
        robot_only=is_elf3,
        mobile_base=base_mode == "mobile",
        add_ground=is_elf3,
        ground_priority=0 if base_mode == "gait" else 1,
    )
    if is_elf3 and base_mode != "gait":
        configure_elf3_joint_damping(mujoco, model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    gait_controller = None
    if base_mode == "gait":
        gait_controller_class = import_gait_controller_class()
        gait_controller = gait_controller_class(
            mujoco,
            numpy,
            model,
            data,
            model_path=args.gait_model,
            move_speed=args.move_speed,
            turn_speed=args.turn_speed,
        )

    ik_result = None
    if args.ik_target is not None:
        if not is_elf3:
            raise RuntimeError("IK options require elf3.xml or elf3_hand.xml")
        if base_mode in ("floating", "gait"):
            raise RuntimeError(
                "IK visualization requires --mobile-base or --fixed-base"
            )
        ik_result = solve_hand_ik(
            mujoco,
            numpy,
            model,
            data,
            side=args.ik_side,
            target_position=args.ik_target,
            target_quaternion=args.ik_quat,
            max_iterations=args.ik_max_iterations,
            tolerance=args.ik_tolerance,
            damping=args.ik_damping,
        )
        # Save the solution in keyframe 0 without changing model.qpos0 (which
        # would redefine the hinge reference angles and corrupt kinematics).
        if model.nkey > 0:
            model.key_qpos[0, :] = data.qpos

    controller = None
    if not args.no_control and is_elf3 and base_mode != "gait":
        controller = elf3_joint_hold_controller(mujoco, numpy, model, data)

    base_driver = None
    if base_mode == "mobile":
        base_driver = KeyboardBaseController(
            mujoco,
            numpy,
            model,
            move_speed=args.move_speed,
            turn_speed=args.turn_speed,
        )

    return MujocoEnvironment(
        mujoco,
        numpy,
        model,
        data,
        str(resolved_path),
        controller,
        robot_only=is_elf3,
        base_mode=base_mode,
        has_ground=is_elf3,
        base_driver=base_driver,
        gait_controller=gait_controller,
        ik_result=ik_result,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=tuple(SCENE_MODELS),
        default="elf3",
        help=(
            "built-in MJCF to load: elf3.xml (default) or elf3_hand.xml"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="load an arbitrary MJCF/XML file; overrides --scene",
    )
    parser.add_argument(
        "--ik-side",
        choices=tuple(ARM_JOINT_NAMES),
        default="left",
        help="arm used by --ik-target (default: left)",
    )
    parser.add_argument(
        "--ik-target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="solve the selected hand TCP to world-frame XYZ metres",
    )
    parser.add_argument(
        "--ik-quat",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help="optional world-frame TCP orientation quaternion (w x y z)",
    )
    parser.add_argument(
        "--ik-max-iterations",
        type=int,
        default=300,
        help="maximum damped-least-squares IK iterations (default: 300)",
    )
    parser.add_argument(
        "--ik-tolerance",
        type=float,
        default=1.0e-4,
        help="IK position/orientation tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1.0e-2,
        help="IK damped-least-squares coefficient (default: 1e-2)",
    )
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument(
        "--rl-walk",
        "--gait",
        dest="base_mode",
        action="store_const",
        const="gait",
        help=(
            "run the physical floating-base BXI learned gait "
            "(default for Elf3 without IK)"
        ),
    )
    base_group.add_argument(
        "--mobile-base",
        dest="base_mode",
        action="store_const",
        const="mobile",
        help="use the stable kinematic planar base (default when IK is requested)",
    )
    base_group.add_argument(
        "--fixed-base",
        dest="base_mode",
        action="store_const",
        const="fixed",
        help="keep torso_link static on the ground",
    )
    base_group.add_argument(
        "--floating-base",
        dest="base_mode",
        action="store_const",
        const="floating",
        help="retain the physical free root (no balance policy, so it can fall)",
    )
    parser.set_defaults(base_mode=None)
    parser.add_argument(
        "--gait-model",
        type=Path,
        default=BXI_GAIT_MODEL,
        help="AMP locomotion ONNX model used by --rl-walk",
    )
    parser.add_argument(
        "--move-speed",
        type=float,
        default=DEFAULT_MOVE_SPEED,
        help="keyboard forward/backward speed in m/s (default: 0.4)",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=DEFAULT_TURN_SPEED,
        help="keyboard yaw speed in rad/s (default: 0.8)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run physics without opening the viewer",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help=(
            "stop after this many simulated seconds; GUI otherwise runs until "
            "closed (the first limit wins when combined with --steps)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="stop after this many steps (the first limit wins with --duration)",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="GUI simulation speed relative to real time (default: 1.0)",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="disable the non-gait Elf3 joint-hold controller",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0.0
    ):
        parser.error("--duration must be positive and finite")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if not math.isfinite(args.realtime_factor) or args.realtime_factor <= 0.0:
        parser.error("--realtime-factor must be positive and finite")
    if not math.isfinite(args.move_speed) or args.move_speed <= 0.0:
        parser.error("--move-speed must be positive and finite")
    if not math.isfinite(args.turn_speed) or args.turn_speed <= 0.0:
        parser.error("--turn-speed must be positive and finite")
    if args.ik_quat is not None and args.ik_target is None:
        parser.error("--ik-quat requires --ik-target")
    if args.ik_max_iterations <= 0:
        parser.error("--ik-max-iterations must be positive")
    if not math.isfinite(args.ik_tolerance) or args.ik_tolerance <= 0.0:
        parser.error("--ik-tolerance must be positive and finite")
    if not math.isfinite(args.ik_damping) or args.ik_damping <= 0.0:
        parser.error("--ik-damping must be positive and finite")


def run(args: argparse.Namespace) -> int:
    environment = build_environment(args)
    print(
        "Loaded MuJoCo environment: {}\n"
        "  bodies={}, joints={}, dofs={}, actuators={}, meshes={}, sensors={}, "
        "timestep={}s".format(
            environment.source,
            environment.model.nbody,
            environment.model.njnt,
            environment.model.nv,
            environment.model.nu,
            environment.model.nmesh,
            environment.model.nsensor,
            environment.model.opt.timestep,
        )
    )
    if environment.robot_only:
        print(
            "Mode: Elf3 with compact checker ground "
            "(the large terrain was excluded in memory)."
        )
        if environment.gait_mode:
            print(
                "Mode: physical floating-base BXI AMP humanoid gait "
                "(50Hz policy, 500Hz joint PD)."
            )
            print(
                "Gait policy: {}".format(environment.gait_controller.model_path)
            )
            print(
                "Keyboard: W/Up=walk forward, S/Down=walk backward, "
                "A/Left=turn left, D/Right=turn right, X/Space=stop."
            )
            print("Keyboard commands remain active until the next key press.")
        elif environment.mobile_base:
            print(
                "Keyboard: W/Up=forward, S/Down=backward, "
                "A/Left=turn left, D/Right=turn right, X/Space=stop."
            )
            print("Keyboard commands remain active until the next key press.")
            print(
                "Note: navigation uses a stable upright mocap base; "
                "it is not a learned biped gait."
            )
        elif environment.fixed_base:
            print("Note: Elf3 is fixed at torso_link on the ground.")
        elif environment.base_mode == "floating":
            print(
                "Note: Elf3 has a physical free base but no balance policy; "
                "falling is expected."
            )
    if environment.ik_result is not None:
        result = environment.ik_result
        print(
            "IK solved: side={}, site={}, iterations={}, position_error={:.6f}m, "
            "rotation_error={:.6f}rad, achieved_xyz={}".format(
                result["side"],
                result["site"],
                result["iterations"],
                result["position_error"],
                result["rotation_error"],
                environment.numpy.array2string(
                    result["achieved_position"], precision=6
                ),
            )
        )
        print(
            "IK joint solution (rad): {}".format(
                ", ".join(
                    "{}={:.6f}".format(name, value)
                    for name, value in zip(
                        result["joint_names"], result["joint_positions"]
                    )
                )
            )
        )

    started = time.perf_counter()
    try:
        if args.headless:
            completed_steps = environment.run_headless(args.duration, args.steps)
        else:
            print("MuJoCo viewer started. Close the window or press Esc to exit.")
            completed_steps = environment.run_viewer(
                args.duration, args.steps, args.realtime_factor
            )
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    print(
        "Simulation finished: steps={}, sim_time={:.3f}s, wall_time={:.3f}s".format(
            completed_steps, environment.data.time, elapsed
        )
    )
    return 0


def main(argv: Optional[Any] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
