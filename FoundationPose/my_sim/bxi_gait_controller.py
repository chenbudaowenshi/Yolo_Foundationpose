#!/usr/bin/env python3
"""Minimal ELF3 AMP gait controller for the local MuJoCo simulation.

This module mirrors the normal walking policy contract from
``bxi_rl_controller_ros2_example`` without importing its ROS 2 framework.  It
therefore remains usable from the Python 3.8 ``mj`` Conda environment.  The
controller expects the original ELF3 floating-base MJCF, torque motors, and
the ``amp_terrain.onnx`` policy.

The policy runs at 50 Hz while joint PD torques are refreshed at every MuJoCo
physics step.  All joints, actuators, and IMU sensors are resolved by name so
the implementation does not depend on qpos, qvel, or actuator array order.
"""

from __future__ import annotations

import importlib
import math
import threading
from pathlib import Path
from typing import Any, Optional, Tuple


POLICY_PERIOD = 0.02
POLICY_HISTORY_LENGTH = 10
POLICY_FRAME_SIZE = 96
POLICY_INPUT_SIZE = POLICY_HISTORY_LENGTH * POLICY_FRAME_SIZE
POLICY_ACTION_SIZE = 29
POLICY_OUTPUT_SIZE = 32
DEFAULT_BASE_HEIGHT = 1.0625

# Command limits used by the package's normal walking state: vx, vy, yaw rate.
COMMAND_MINIMUM = (-1.0, -0.5, -1.5)
COMMAND_MAXIMUM = (1.0, 0.5, 1.5)

# Every row is: joint name, default position, kp, kd, action scale.  This is the
# exact ELF3 Isaac joint contract used by HumanoidGaitPolicyLiteIsaaclab and
# amp_terrain.onnx in bxi_rl_controller_ros2_example.
POLICY_JOINT_ROWS = (
    ("l_shoulder_y_joint", 0.2, 54.224, 3.452, 0.231),
    ("r_shoulder_y_joint", 0.2, 54.224, 3.452, 0.231),
    ("waist_y_joint", 0.0, 108.448, 6.904, 0.231),
    ("l_shoulder_x_joint", 0.2, 54.224, 3.452, 0.231),
    ("r_shoulder_x_joint", -0.2, 54.224, 3.452, 0.231),
    ("waist_x_joint", 0.0, 162.672, 10.356, 0.154),
    ("l_shoulder_z_joint", 0.0, 16.747, 1.066, 0.373),
    ("r_shoulder_z_joint", 0.0, 16.747, 1.066, 0.373),
    ("waist_z_joint", 0.0, 176.421, 11.231, 0.213),
    ("l_elbow_y_joint", 0.6, 54.224, 3.452, 0.231),
    ("r_elbow_y_joint", 0.6, 54.224, 3.452, 0.231),
    ("l_hip_y_joint", -0.3, 176.421, 11.231, 0.213),
    ("r_hip_y_joint", -0.3, 176.421, 11.231, 0.213),
    ("l_wrist_x_joint", 0.0, 16.747, 1.066, 0.373),
    ("r_wrist_x_joint", 0.0, 16.747, 1.066, 0.373),
    ("l_hip_x_joint", 0.0, 176.421, 11.231, 0.213),
    ("r_hip_x_joint", 0.0, 176.421, 11.231, 0.213),
    ("l_wrist_y_joint", 0.0, 16.747, 1.066, 0.373),
    ("r_wrist_y_joint", 0.0, 16.747, 1.066, 0.373),
    ("l_hip_z_joint", 0.0, 54.224, 3.452, 0.231),
    ("r_hip_z_joint", 0.0, 54.224, 3.452, 0.231),
    ("l_wrist_z_joint", 0.0, 16.747, 1.066, 0.373),
    ("r_wrist_z_joint", 0.0, 16.747, 1.066, 0.373),
    ("l_knee_y_joint", 0.6, 176.421, 11.231, 0.213),
    ("r_knee_y_joint", 0.6, 176.421, 11.231, 0.213),
    ("l_ankle_y_joint", -0.3, 33.493, 2.132, 0.373),
    ("r_ankle_y_joint", -0.3, 33.493, 2.132, 0.373),
    ("l_ankle_x_joint", 0.0, 21.771, 1.386, 0.230),
    ("r_ankle_x_joint", 0.0, 21.771, 1.386, 0.230),
)

HEAD_JOINT_ROWS = (
    ("head_z_joint", 0.0, 16.747, 1.066),
    ("head_y_joint", 0.0, 16.747, 1.066),
)


class KeyboardVelocityCommand:
    """Persistent body-frame velocity command controlled by viewer keys."""

    _FORWARD_KEYS = (ord("W"), 265)  # W or Up
    _BACKWARD_KEYS = (ord("S"), 264)  # S or Down
    _LEFT_KEYS = (ord("A"), 263)  # A or Left
    _RIGHT_KEYS = (ord("D"), 262)  # D or Right
    _STOP_KEYS = (ord("X"), 32)  # X or Space

    def __init__(self, numpy: Any, move_speed: float, turn_speed: float) -> None:
        if not math.isfinite(move_speed) or move_speed <= 0.0:
            raise ValueError("move_speed must be positive and finite")
        if not math.isfinite(turn_speed) or turn_speed <= 0.0:
            raise ValueError("turn_speed must be positive and finite")
        self.numpy = numpy
        self.move_speed = float(move_speed)
        self.turn_speed = float(turn_speed)
        self._command = numpy.zeros(3, dtype=numpy.float32)
        self._lock = threading.Lock()

    def set_command(
        self, forward_velocity: float, lateral_velocity: float, yaw_rate: float
    ) -> None:
        values = self.numpy.asarray(
            (forward_velocity, lateral_velocity, yaw_rate),
            dtype=self.numpy.float32,
        )
        if values.shape != (3,) or not self.numpy.isfinite(values).all():
            raise ValueError("velocity command must contain three finite values")
        values = self.numpy.clip(values, COMMAND_MINIMUM, COMMAND_MAXIMUM)
        with self._lock:
            self._command[:] = values

    def stop(self) -> None:
        self.set_command(0.0, 0.0, 0.0)

    def copy_into(self, output: Any) -> None:
        with self._lock:
            output[:] = self._command

    def snapshot(self) -> Any:
        with self._lock:
            return self._command.copy()

    def handle_key(self, keycode: int) -> None:
        """Handle GLFW-style key codes supplied by ``launch_passive``."""
        if keycode in self._FORWARD_KEYS:
            self.set_command(self.move_speed, 0.0, 0.0)
            description = "walk forward"
        elif keycode in self._BACKWARD_KEYS:
            self.set_command(-self.move_speed, 0.0, 0.0)
            description = "walk backward"
        elif keycode in self._LEFT_KEYS:
            self.set_command(0.0, 0.0, self.turn_speed)
            description = "turn left"
        elif keycode in self._RIGHT_KEYS:
            self.set_command(0.0, 0.0, -self.turn_speed)
            description = "turn right"
        elif keycode in self._STOP_KEYS:
            self.stop()
            description = "stop"
        else:
            return
        print("Learned gait command: {}".format(description), flush=True)


class BxiAmpGaitController:
    """Run the BXI ELF3 AMP walking policy directly against ``MjData``."""

    def __init__(
        self,
        mujoco: Any,
        numpy: Any,
        model: Any,
        data: Any,
        model_path: Any,
        move_speed: float = 0.4,
        turn_speed: float = 0.8,
        base_height: float = DEFAULT_BASE_HEIGHT,
        policy_period: float = POLICY_PERIOD,
    ) -> None:
        self.mujoco = mujoco
        self.numpy = numpy
        self.model = model
        self.data = data
        self.base_height = float(base_height)
        self.policy_period = float(policy_period)
        if not math.isfinite(self.base_height) or self.base_height <= 0.0:
            raise ValueError("base_height must be positive and finite")
        if not math.isfinite(self.policy_period) or self.policy_period <= 0.0:
            raise ValueError("policy_period must be positive and finite")
        if self.model.opt.timestep <= 0.0:
            raise ValueError("MuJoCo timestep must be positive")

        self.command_source = KeyboardVelocityCommand(
            numpy, move_speed=move_speed, turn_speed=turn_speed
        )
        self.handle_key = self.command_source.handle_key

        self._resolve_model_contract()
        self._create_policy_arrays()
        self._open_session(Path(model_path).expanduser().resolve())
        self._initialize_pose(self.data)
        self._reset_policy_state(self.data)
        self._initial_qpos = self.data.qpos.copy()

    def _resolve_model_contract(self) -> None:
        root_joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "world_joint"
        )
        if root_joint_id < 0:
            raise RuntimeError(
                "AMP gait requires the ELF3 floating root joint 'world_joint'"
            )
        if (
            self.model.jnt_type[root_joint_id]
            != self.mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError("AMP gait requires world_joint to be a free joint")
        self._root_qpos_address = int(self.model.jnt_qposadr[root_joint_id])

        policy_joint_ids = []
        policy_qpos_addresses = []
        policy_dof_addresses = []
        policy_actuator_ids = []
        for row in POLICY_JOINT_ROWS:
            joint_name = row[0]
            joint_id, qpos_address, dof_address, actuator_id = (
                self._resolve_torque_joint(joint_name)
            )
            policy_joint_ids.append(joint_id)
            policy_qpos_addresses.append(qpos_address)
            policy_dof_addresses.append(dof_address)
            policy_actuator_ids.append(actuator_id)

        head_qpos_addresses = []
        head_dof_addresses = []
        head_actuator_ids = []
        for row in HEAD_JOINT_ROWS:
            _, qpos_address, dof_address, actuator_id = self._resolve_torque_joint(
                row[0]
            )
            head_qpos_addresses.append(qpos_address)
            head_dof_addresses.append(dof_address)
            head_actuator_ids.append(actuator_id)

        np = self.numpy
        self._policy_joint_ids = np.asarray(policy_joint_ids, dtype=np.int32)
        self._policy_qpos_addresses = np.asarray(
            policy_qpos_addresses, dtype=np.int32
        )
        self._policy_dof_addresses = np.asarray(
            policy_dof_addresses, dtype=np.int32
        )
        self._policy_actuator_ids = np.asarray(
            policy_actuator_ids, dtype=np.int32
        )
        self._head_qpos_addresses = np.asarray(
            head_qpos_addresses, dtype=np.int32
        )
        self._head_dof_addresses = np.asarray(head_dof_addresses, dtype=np.int32)
        self._head_actuator_ids = np.asarray(head_actuator_ids, dtype=np.int32)

        self._policy_ctrl_minimum = self.model.actuator_ctrlrange[
            self._policy_actuator_ids, 0
        ].copy()
        self._policy_ctrl_maximum = self.model.actuator_ctrlrange[
            self._policy_actuator_ids, 1
        ].copy()
        self._head_ctrl_minimum = self.model.actuator_ctrlrange[
            self._head_actuator_ids, 0
        ].copy()
        self._head_ctrl_maximum = self.model.actuator_ctrlrange[
            self._head_actuator_ids, 1
        ].copy()

        self._gyro_address = self._resolve_sensor("torso_ang_vel", 3)
        self._quaternion_address = self._resolve_sensor("torso_quat", 4)

    def _resolve_torque_joint(self, joint_name: str) -> Tuple[int, int, int, int]:
        joint_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError("ELF3 joint '{}' is missing".format(joint_name))
        if self.model.jnt_type[joint_id] != self.mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError("ELF3 joint '{}' is not a hinge".format(joint_name))

        actuator_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if actuator_id < 0:
            raise RuntimeError(
                "ELF3 actuator '{}' is missing".format(joint_name)
            )
        if (
            self.model.actuator_trntype[actuator_id]
            != self.mujoco.mjtTrn.mjTRN_JOINT
            or int(self.model.actuator_trnid[actuator_id, 0]) != joint_id
        ):
            raise RuntimeError(
                "ELF3 actuator '{}' does not drive its namesake joint".format(
                    joint_name
                )
            )
        if (
            self.model.actuator_biastype[actuator_id]
            != self.mujoco.mjtBias.mjBIAS_NONE
        ):
            raise RuntimeError(
                "ELF3 actuator '{}' must be a torque motor".format(joint_name)
            )
        if not self.model.actuator_ctrllimited[actuator_id]:
            raise RuntimeError(
                "ELF3 actuator '{}' must define ctrlrange".format(joint_name)
            )
        return (
            joint_id,
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
            actuator_id,
        )

    def _resolve_sensor(self, sensor_name: str, expected_dimension: int) -> int:
        sensor_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise RuntimeError("ELF3 sensor '{}' is missing".format(sensor_name))
        dimension = int(self.model.sensor_dim[sensor_id])
        if dimension != expected_dimension:
            raise RuntimeError(
                "ELF3 sensor '{}' has dimension {}, expected {}".format(
                    sensor_name, dimension, expected_dimension
                )
            )
        return int(self.model.sensor_adr[sensor_id])

    def _create_policy_arrays(self) -> None:
        np = self.numpy
        self.joint_names = tuple(row[0] for row in POLICY_JOINT_ROWS)
        self.default_position = np.asarray(
            [row[1] for row in POLICY_JOINT_ROWS], dtype=np.float64
        )
        self.kp = np.asarray(
            [row[2] for row in POLICY_JOINT_ROWS], dtype=np.float64
        )
        self.kd = np.asarray(
            [row[3] for row in POLICY_JOINT_ROWS], dtype=np.float64
        )
        self.action_scale = np.asarray(
            [row[4] for row in POLICY_JOINT_ROWS], dtype=np.float64
        )
        self._head_default = np.asarray(
            [row[1] for row in HEAD_JOINT_ROWS], dtype=np.float64
        )
        self._head_kp = np.asarray(
            [row[2] for row in HEAD_JOINT_ROWS], dtype=np.float64
        )
        self._head_kd = np.asarray(
            [row[3] for row in HEAD_JOINT_ROWS], dtype=np.float64
        )

        self._single_observation = np.zeros(
            POLICY_FRAME_SIZE, dtype=np.float32
        )
        self._history = np.zeros(
            (POLICY_HISTORY_LENGTH, POLICY_FRAME_SIZE), dtype=np.float32
        )
        self._policy_input = np.zeros((1, POLICY_INPUT_SIZE), dtype=np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._action = np.zeros(POLICY_ACTION_SIZE, dtype=np.float32)
        self._target = self.default_position.copy()
        self._estimated_velocity = np.zeros(3, dtype=np.float32)
        self._policy_torque = np.zeros(POLICY_ACTION_SIZE, dtype=np.float64)
        self._head_torque = np.zeros(len(HEAD_JOINT_ROWS), dtype=np.float64)

    def _open_session(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError("AMP ONNX model not found: {}".format(model_path))
        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "onnxruntime is required for learned gait. In the mj environment "
                "run: python -m pip install 'onnxruntime==1.19.2'"
            ) from error

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs":
            raise RuntimeError("AMP policy must have one input named 'obs'")
        if tuple(inputs[0].shape) != (1, POLICY_INPUT_SIZE):
            raise RuntimeError(
                "AMP policy input shape is {}, expected (1, {})".format(
                    inputs[0].shape, POLICY_INPUT_SIZE
                )
            )
        if not outputs or outputs[0].name != "actions":
            raise RuntimeError("AMP policy first output must be named 'actions'")
        if tuple(outputs[0].shape) != (1, POLICY_OUTPUT_SIZE):
            raise RuntimeError(
                "AMP policy output shape is {}, expected (1, {})".format(
                    outputs[0].shape, POLICY_OUTPUT_SIZE
                )
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.model_path = str(model_path)

    def _initialize_pose(self, data: Any) -> None:
        root = self._root_qpos_address
        data.qpos[root : root + 7] = (
            0.0,
            0.0,
            self.base_height,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        data.qpos[self._policy_qpos_addresses] = self.default_position
        data.qpos[self._head_qpos_addresses] = self._head_default
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, data)

    def _reset_policy_state(self, data: Any) -> None:
        self.command_source.stop()
        self._command.fill(0.0)
        self._action.fill(0.0)
        self._estimated_velocity.fill(0.0)
        self._target[:] = self.default_position
        self._build_single_observation(data)
        self._history[:] = self._single_observation
        self._policy_input[0, :] = self._history.reshape(-1)
        self._next_policy_time = float(data.time)

    def reset(self, data: Optional[Any] = None, reset_pose: bool = True) -> None:
        """Reset the gait state and optionally restore the nominal start pose."""
        sim_data = self.data if data is None else data
        if reset_pose:
            self._initialize_pose(sim_data)
        else:
            self.mujoco.mj_forward(self.model, sim_data)
        self._reset_policy_state(sim_data)

    def set_command(
        self, forward_velocity: float, lateral_velocity: float, yaw_rate: float
    ) -> None:
        self.command_source.set_command(
            forward_velocity, lateral_velocity, yaw_rate
        )

    @property
    def command(self) -> Any:
        return self.command_source.snapshot()

    @property
    def target_position(self) -> Any:
        return self._target.copy()

    @property
    def estimated_velocity(self) -> Any:
        return self._estimated_velocity.copy()

    def _build_single_observation(self, data: Any) -> None:
        np = self.numpy
        observation = self._single_observation
        observation.fill(0.0)

        observation[0:3] = data.sensordata[
            self._gyro_address : self._gyro_address + 3
        ]
        quaternion = data.sensordata[
            self._quaternion_address : self._quaternion_address + 4
        ]
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(quaternion_norm) or quaternion_norm <= 1.0e-12:
            raise RuntimeError("torso_quat sensor returned an invalid quaternion")
        w, x, y, z = quaternion / quaternion_norm
        observation[3:6] = (
            2.0 * (w * y - x * z),
            -2.0 * (w * x + y * z),
            2.0 * (x * x + y * y) - 1.0,
        )

        self.command_source.copy_into(self._command)
        observation[6:9] = self._command
        np.subtract(
            data.qpos[self._policy_qpos_addresses],
            self.default_position,
            out=observation[9:38],
        )
        observation[38:67] = data.qvel[self._policy_dof_addresses]
        observation[67:96] = self._action
        if not np.isfinite(observation).all():
            raise RuntimeError("AMP observation contains non-finite values")

    def update_policy(self, data: Optional[Any] = None, force: bool = False) -> bool:
        """Run one due 50 Hz policy update; return whether inference ran."""
        sim_data = self.data if data is None else data
        now = float(sim_data.time)
        tolerance = max(1.0e-12, 0.25 * float(self.model.opt.timestep))
        if not force and now + tolerance < self._next_policy_time:
            return False

        self._build_single_observation(sim_data)
        self._history[:-1] = self._history[1:]
        self._history[-1] = self._single_observation
        self._policy_input[0, :] = self._history.reshape(-1)
        raw_output = self._session.run(
            [self._output_name], {self._input_name: self._policy_input}
        )[0]
        output = self.numpy.asarray(raw_output, dtype=self.numpy.float32).reshape(-1)
        if output.shape != (POLICY_OUTPUT_SIZE,):
            raise RuntimeError(
                "AMP inference returned shape {}, expected ({},)".format(
                    output.shape, POLICY_OUTPUT_SIZE
                )
            )
        if not self.numpy.isfinite(output).all():
            raise RuntimeError("AMP inference returned non-finite actions")

        self._action[:] = output[:POLICY_ACTION_SIZE]
        self._estimated_velocity[:] = output[POLICY_ACTION_SIZE:POLICY_OUTPUT_SIZE]
        self._target[:] = self.default_position
        self._target += self._action * self.action_scale

        if force:
            self._next_policy_time = now + self.policy_period
        else:
            while self._next_policy_time <= now + tolerance:
                self._next_policy_time += self.policy_period
        return True

    def apply_control(self, data: Optional[Any] = None) -> None:
        """Apply policy and head-hold PD torques for one physics step."""
        sim_data = self.data if data is None else data
        np = self.numpy

        np.subtract(
            self._target,
            sim_data.qpos[self._policy_qpos_addresses],
            out=self._policy_torque,
        )
        self._policy_torque *= self.kp
        self._policy_torque -= self.kd * sim_data.qvel[
            self._policy_dof_addresses
        ]
        np.clip(
            self._policy_torque,
            self._policy_ctrl_minimum,
            self._policy_ctrl_maximum,
            out=self._policy_torque,
        )
        sim_data.ctrl[self._policy_actuator_ids] = self._policy_torque

        np.subtract(
            self._head_default,
            sim_data.qpos[self._head_qpos_addresses],
            out=self._head_torque,
        )
        self._head_torque *= self._head_kp
        self._head_torque -= self._head_kd * sim_data.qvel[
            self._head_dof_addresses
        ]
        np.clip(
            self._head_torque,
            self._head_ctrl_minimum,
            self._head_ctrl_maximum,
            out=self._head_torque,
        )
        sim_data.ctrl[self._head_actuator_ids] = self._head_torque

    def step(self, data: Optional[Any] = None) -> bool:
        """Run due inference and apply PD control before ``mj_step``."""
        sim_data = self.data if data is None else data
        updated = self.update_policy(sim_data)
        self.apply_control(sim_data)
        return updated

    def __call__(self, data: Any) -> None:
        self.step(data)

    def close(self) -> None:
        """Release the ONNX Runtime session and stop motion commands."""
        self.command_source.stop()
        self._session = None


__all__ = [
    "BxiAmpGaitController",
    "COMMAND_MAXIMUM",
    "COMMAND_MINIMUM",
    "DEFAULT_BASE_HEIGHT",
    "KeyboardVelocityCommand",
    "POLICY_JOINT_ROWS",
    "POLICY_PERIOD",
]
