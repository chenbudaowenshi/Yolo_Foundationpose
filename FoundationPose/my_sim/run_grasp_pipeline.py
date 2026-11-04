#!/usr/bin/env python3
"""Run the complete ROS2 MuJoCo perception-to-grasp workflow.

The workflow uses one ROS2 MuJoCo window and connects the existing tools in
their intended order:

1. ``example_launch_demo.py enable_arm_ik:=true`` starts MuJoCo and the ELF3
   controller.
2. ``head_control_mujoco.py`` sweeps the head and holds it after YOLO finds the
   bottle.
3. ``get_pose_mujoco.py`` registers the 6D pose and writes ``T_world_object``.
4. ``arm_ik.py --live-ros`` solves the arm and replays the result through the
   running ROS2 simulation (with optional base navigation).

The individual programs use different Python environments.  This launcher
sources the ROS/BXI overlays for every child and invokes the explicit pose or
MuJoCo interpreter, so the user does not need to keep four terminals in sync.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MY_SIM = REPOSITORY_ROOT / "my_sim"
ROS_PYTHON = Path("/home/robot/anaconda3/envs/pose/bin/python")
MJ_PYTHON = Path("/home/robot/anaconda3/envs/mj/bin/python")
ROS2_SETUP = Path("/opt/ros/humble/setup.bash")
BXI_SETUP = Path("/opt/bxi/bxi_ros2_pkg/setup.bash")
LOCAL_SETUP = REPOSITORY_ROOT / "bxi_rl_controller_ros2_example/install/setup.bash"

DEFAULT_DOMAIN = 91
DEFAULT_POSE = REPOSITORY_ROOT / "runs/foundationpose_mujoco/pose.json"
DEFAULT_REPORT = REPOSITORY_ROOT / "runs/arm_ik/reach_report.json"
SIM_TOPICS = (
    "/simulation/actuator_states",
    "/simulation/odom",
    "/simulation/head_depth_camera/color/image_raw",
    "/simulation/head_depth_camera/depth/image_rect_raw",
)
# These two topics are emitted by every supported MuJoCo launch.  Camera
# topic names changed between simulator package versions, so requiring all
# four entries to detect an existing graph can miss a perfectly valid window
# and start a second simulator.
SIM_CORE_TOPICS = frozenset(("/simulation/actuator_states", "/simulation/odom"))
ARM_OVERRIDE_TOPIC = "/simulation/arm_ik_cmds_override"


class PipelineError(RuntimeError):
    """An actionable workflow setup or stage failure."""


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll()


def _quote_command(arguments: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(argument)) for argument in arguments)


def _check_files() -> None:
    missing = [
        path
        for path in (ROS2_SETUP, BXI_SETUP, LOCAL_SETUP, ROS_PYTHON, MJ_PYTHON)
        if not path.is_file()
    ]
    if missing:
        details = "\n  ".join(str(path) for path in missing)
        raise PipelineError(
            "required ROS/Python files are missing:\n  "
            + details
            + "\nRebuild/source the ROS2 workspace before starting the pipeline."
        )

    launch_file = (
        REPOSITORY_ROOT
        / "bxi_rl_controller_ros2_example/install/share/bxi_example_py_elf3/launch/example_launch_demo.py"
    )
    if not launch_file.is_file():
        raise PipelineError(
            "installed bxi_example_py_elf3 launch file is missing: "
            f"{launch_file}. Build and source the workspace before running."
        )
    try:
        launch_text = launch_file.read_text(encoding="utf-8")
    except OSError as error:
        raise PipelineError(f"cannot read installed launch file: {error}") from error
    if "enable_arm_ik" not in launch_text or "arguments=[xml_file]" not in launch_text:
        raise PipelineError(
            "the installed bxi_example_py_elf3 launch file is stale (it must "
            "contain enable_arm_ik and pass the MJCF path as arguments=[xml_file]). "
            "Rebuild and source the workspace:\n"
            "  cd /home/robot/FoundationPose/bxi_rl_controller_ros2_example\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /opt/bxi/bxi_ros2_pkg/setup.bash\n"
            "  colcon build --merge-install --packages-select bxi_example_py_elf3\n"
            "  source install/setup.bash"
        )


def _child_environment(domain_id: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    # Do not force an RMW implementation or FastDDS transport profile here.
    # The MuJoCo launch may have been started with a site-specific RMW/XML
    # configuration; overriding it in child nodes makes actuator topics visible
    # while large Image publishers disappear (publisher count=0).  Inherit the
    # launch terminal's settings and only pin the requested domain.
    environment["PYTHONUNBUFFERED"] = "1"
    # Keep a deterministic ROS log location.  Some installations make the
    # user's default ~/.ros directory read-only; rclpy then fails before DDS
    # discovery and the symptom looks like a camera with zero publishers.
    ros_log_dir = Path("/tmp/ros_log")
    try:
        ros_log_dir.mkdir(parents=True, exist_ok=True)
        environment.setdefault("ROS_LOG_DIR", str(ros_log_dir))
    except OSError:
        # Keep the inherited setting if /tmp is unexpectedly unavailable.
        pass
    return environment


def _ros_environment_summary(environment: dict[str, str]) -> str:
    keys = (
        "ROS_DOMAIN_ID",
        "RMW_IMPLEMENTATION",
        "ROS_LOCALHOST_ONLY",
        "FASTDDS_BUILTIN_TRANSPORTS",
        "FASTDDS_DEFAULT_PROFILES_FILE",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "CYCLONEDDS_URI",
    )
    return ", ".join(f"{key}={environment.get(key, '<unset>')}" for key in keys)


def _shell_command(arguments: Sequence[object]) -> list[str]:
    setup = " && ".join(
        f"source {shlex.quote(str(path))}"
        for path in (ROS2_SETUP, BXI_SETUP, LOCAL_SETUP)
    )
    command = f"{setup} && exec {_quote_command(arguments)}"
    return ["/bin/bash", "--noprofile", "--norc", "-c", command]


def _start_process(
    name: str,
    arguments: Sequence[object],
    environment: dict[str, str],
) -> ManagedProcess:
    print(f"[pipeline] starting {name}: {_quote_command(arguments)}", flush=True)
    try:
        process = subprocess.Popen(
            _shell_command(arguments),
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        raise PipelineError(f"could not start {name}: {error}") from error
    return ManagedProcess(name, process)


def _signal_process(process: ManagedProcess, sig: int) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.process.pid, sig)
    except ProcessLookupError:
        pass


def _stop_process(process: Optional[ManagedProcess], *, force: bool = False) -> None:
    if process is None or process.returncode is not None:
        return
    _signal_process(process, signal.SIGTERM if force else signal.SIGINT)
    try:
        process.process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process(process, signal.SIGTERM)
    try:
        process.process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process(process, signal.SIGKILL)
    try:
        process.process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _topic_names(environment: dict[str, str]) -> set[str]:
    command = ["ros2", "topic", "list", "--no-daemon", "--spin-time", "1.0"]
    try:
        result = subprocess.run(
            _shell_command(command),
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _detect_existing_simulation(
    environment: dict[str, str], *, probe_sec: float = 3.0
) -> bool:
    """Probe briefly before spawning MuJoCo to avoid a startup race.

    ``ros2 launch`` publishes the core topics asynchronously.  Without this
    grace period, invoking the pipeline immediately after launch can observe
    an empty graph and create a duplicate simulator a moment later.
    """
    deadline = time.monotonic() + max(0.0, float(probe_sec))
    while True:
        if SIM_CORE_TOPICS.issubset(_topic_names(environment)):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _discover_camera_topics(environment: dict[str, str]) -> tuple[str, str] | None:
    """Find the head RGB/depth topics advertised by the running simulator."""
    # Camera publishers carry large Image endpoints and can take several
    # seconds to appear after a ROS daemon restart.  Core actuator/odom topics
    # are discovered faster, so allow a longer independent camera probe.
    deadline = time.monotonic() + 12.0
    while True:
        topics = _topic_names(environment)
        color = sorted(
            topic for topic in topics
            if "head_depth_camera" in topic and "/color/" in topic
            and topic.endswith(("/image_raw", "/image"))
        )
        depth = sorted(
            topic for topic in topics
            if "head_depth_camera" in topic and "/depth/" in topic
            and topic.endswith(("/image_rect_raw", "/image_raw", "/image"))
        )
        if color and depth:
            return color[0], depth[0]
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    return None


def _topic_has_subscriber(topic: str, environment: dict[str, str]) -> bool:
    command = ["ros2", "topic", "info", topic, "--no-daemon", "--spin-time", "0.5"]
    try:
        result = subprocess.run(
            _shell_command(command),
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    match = re.search(r"Subscription count:\s*(\d+)", result.stdout)
    return bool(match and int(match.group(1)) > 0)


def _topic_has_publisher(topic: str, environment: dict[str, str]) -> bool:
    """Return true only after a live DDS publisher endpoint is discovered."""
    command = ["ros2", "topic", "info", topic, "--no-daemon", "--spin-time", "0.8"]
    try:
        result = subprocess.run(
            _shell_command(command),
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    match = re.search(r"Publisher count:\s*(\d+)", result.stdout)
    return bool(match and int(match.group(1)) > 0)


def _wait_for_camera_publisher(
    rgb_topic: str, environment: dict[str, str], timeout: float = 30.0
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    last_report = float("-inf")
    while time.monotonic() < deadline:
        if _topic_has_publisher(rgb_topic, environment):
            print(f"[pipeline] camera publisher is ready on {rgb_topic}", flush=True)
            return True
        now = time.monotonic()
        if now - last_report >= 5.0:
            print(
                f"[pipeline] waiting for camera publisher on {rgb_topic}; "
                "DDS discovery is still in progress",
                flush=True,
            )
            last_report = now
        time.sleep(1.0)
    print(
        f"[pipeline] camera publisher was not confirmed after {timeout:.0f}s; "
        "starting head control anyway so its diagnostics can identify the issue",
        flush=True,
    )
    return False


def _wait_for_simulation(
    simulation: Optional[ManagedProcess],
    environment: dict[str, str],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_report = float("-inf")
    # Core state topics are sufficient to identify a live simulator.  Camera
    # publishers may appear later (or be omitted by a reduced MJCF), and the
    # head/pose nodes already have their own image timeout diagnostics.  Do not
    # block the whole pipeline on camera discovery here.
    required = set(SIM_CORE_TOPICS)
    while time.monotonic() < deadline:
        if simulation is not None and simulation.returncode is not None:
            raise PipelineError(
                f"MuJoCo launch exited before the ROS graph was ready "
                f"(code {simulation.returncode})"
            )
        topics = _topic_names(environment)
        missing = required - topics
        if not missing:
            camera_topics = sorted(
                topic for topic in topics if topic.startswith("/simulation/")
                and "/image_" in topic
            )
            suffix = " (camera topics discovered)" if camera_topics else ""
            print(
                "[pipeline] MuJoCo odometry and actuator topics are ready" + suffix,
                flush=True,
            )
            return
        now = time.monotonic()
        if now - last_report >= 5.0:
            print(
                "[pipeline] waiting for ROS simulation topics; missing: "
                + ", ".join(sorted(missing)),
                flush=True,
            )
            last_report = now
        time.sleep(1.0)
    raise PipelineError(
        f"timed out after {timeout:.1f}s waiting for ROS simulation topics. "
        "Check ROS_DOMAIN_ID and the MuJoCo/controller logs."
    )


def _wait_for_arm_override(
    simulation: Optional[ManagedProcess],
    environment: dict[str, str],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if simulation is not None and simulation.returncode is not None:
            raise PipelineError(
                "MuJoCo/controller exited before the arm IK override subscriber "
                f"appeared (code {simulation.returncode})"
            )
        if _topic_has_subscriber(ARM_OVERRIDE_TOPIC, environment):
            print(f"[pipeline] arm IK override subscriber is ready on {ARM_OVERRIDE_TOPIC}", flush=True)
            return
        time.sleep(1.0)
    raise PipelineError(
        f"no controller subscribes to {ARM_OVERRIDE_TOPIC}. Start the controller "
        "with enable_arm_ik:=true and rebuild/source the workspace."
    )


def _wait_for_controller_normal(
    simulation: Optional[ManagedProcess],
    environment: dict[str, str],
    timeout: float,
) -> None:
    """Wait until the ELF3 state machine has released reset/zero-torque.

    Head overrides are intentionally ignored while the controller is in
    ``zero_torque``.  Starting the scanner as soon as the topic exists can
    therefore leave the head apparently frozen.  Poll the published state
    snapshot before launching the scanner.
    """
    deadline = time.monotonic() + timeout
    topic = "/simulation/state_machine_info"
    while time.monotonic() < deadline:
        if simulation is not None and simulation.returncode is not None:
            raise PipelineError(
                "MuJoCo/controller exited before reaching normal state "
                f"(code {simulation.returncode})"
            )
        command = ["ros2", "topic", "echo", topic, "--once", "--no-daemon", "--spin-time", "0.5"]
        try:
            result = subprocess.run(
                _shell_command(command), cwd=str(REPOSITORY_ROOT), env=environment,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            text = result.stdout.lower()
            # ``state_machine_info`` is a JSON string message, rendered by
            # ``ros2 topic echo`` as ``data: '{..."current":{"name":...}}'``.
            # Restrict the match to the current-state object so graph entries
            # mentioning normal/zero_torque do not produce a false positive.
            current_match = re.search(r'"current"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text)
            if current_match is None:
                current_match = re.search(r"current.*?name:\s*([^\\n,}]+)", text, re.DOTALL)
            if current_match is not None and current_match.group(1).strip().endswith("/normal"):
                print("[pipeline] controller state is normal; head overrides are active", flush=True)
                return
            # Older controller builds may publish a flattened snapshot or
            # quote the JSON payload differently.  Once reset has completed,
            # the presence of the normal state is sufficient; zero_torque is
            # only the initial state and is not reported as current here.
            if "com.bxi.basic_actions/normal" in text and "current" in text:
                print("[pipeline] controller state is normal; head overrides are active", flush=True)
                return
        time.sleep(0.5)
    raise PipelineError(
        "controller did not reach normal state; head commands would be ignored. "
        "Check state_machine_info and reset/controller logs."
    )


def _wait_for_process(
    process: ManagedProcess,
    timeout: float,
    *,
    watch: Optional[ManagedProcess] = None,
    watches: Sequence[Optional[ManagedProcess]] = (),
) -> int:
    deadline = None if timeout <= 0.0 else time.monotonic() + timeout
    watched = tuple(item for item in (watch, *watches) if item is not None)
    while True:
        returncode = process.returncode
        if returncode is not None:
            return int(returncode)
        for item in watched:
            if item.returncode is not None:
                raise PipelineError(
                    f"{item.name} exited unexpectedly (code {item.returncode}) "
                    f"while waiting for {process.name}"
                )
        if deadline is not None and time.monotonic() >= deadline:
            raise PipelineError(f"{process.name} timed out after {timeout:.1f}s")
        time.sleep(0.2)


def _load_world_pose(
    path: Path,
    started_at_ns: int,
    previous_mtime_ns: Optional[int],
) -> dict[str, object]:
    if not path.is_file():
        raise PipelineError(f"pose stage finished without writing {path}")
    try:
        stat = path.stat()
        # ext4 timestamps are commonly rounded to the nearest second in
        # containerized environments.  Allow that rounding, while still
        # rejecting an untouched file from an earlier run.
        if stat.st_mtime_ns + 1_000_000_000 < started_at_ns or (
            previous_mtime_ns is not None and stat.st_mtime_ns <= previous_mtime_ns
        ):
            raise PipelineError(
                f"{path} is older than this pose stage; refusing to use a stale result"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read pose result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PipelineError(
            "pose result must be a JSON object containing T_world_object"
        )
    transform = payload.get("T_world_object")
    if not isinstance(transform, dict):
        raise PipelineError(
            "pose result has no T_world_object object. Ensure get_pose_mujoco.py "
            "ran with --world-pose and wrote a complete result."
        )
    matrix = transform.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 4 or any(
        not isinstance(row, list) or len(row) != 4 for row in matrix
    ):
        raise PipelineError(
            "pose result has no valid T_world_object. Ensure /simulation/odom is "
            "available in the same ROS domain and rerun perception."
        )
    try:
        values = [float(value) for row in matrix for value in row]
    except (TypeError, ValueError) as error:
        raise PipelineError("T_world_object contains non-numeric values") from error
    if not all(math.isfinite(value) for value in values):
        raise PipelineError("T_world_object contains non-finite values")
    quality = payload.get("pose_quality")
    if isinstance(quality, dict) and quality.get("valid") is False:
        raise PipelineError(f"pose quality rejected the result: {quality.get('reason', 'unknown')}")
    print(
        "[pipeline] valid T_world_object received: "
        f"({float(matrix[0][3]):.3f}, {float(matrix[1][3]):.3f}, {float(matrix[2][3]):.3f}) m",
        flush=True,
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start MuJoCo, scan the head, estimate 6D pose, and execute arm IK."
    )
    parser.add_argument("--ros-domain-id", type=int, default=DEFAULT_DOMAIN)
    parser.add_argument("--pose-json", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sim-start-timeout", type=float, default=60.0)
    parser.add_argument("--pose-timeout", type=float, default=300.0)
    parser.add_argument(
        "--arm-timeout",
        type=float,
        default=0.0,
        help="IK timeout in seconds; 0 means no timeout (default)",
    )
    parser.add_argument("--ros-wait-timeout", type=float, default=20.0)
    parser.add_argument("--ros-final-hold-sec", type=float, default=5.0)
    _add_boolean_option(
        parser,
        "ros_navigate",
        default=True,
        help="navigate the live base to the computed stance (default: enabled)",
    )
    _add_boolean_option(
        parser,
        "ros_hold_final",
        default=False,
        help="keep the final arm pose until Ctrl-C (default: disabled)",
    )
    _add_boolean_option(
        parser,
        "show",
        default=True,
        help="show the FoundationPose OpenCV preview (default: enabled)",
    )
    _add_boolean_option(
        parser,
        "keep_sim",
        default=True,
        help="leave the ROS2 MuJoCo window running after success (default: enabled)",
    )
    parser.add_argument(
        "--reuse-sim",
        action="store_true",
        help=(
            "attach to an already running ROS2 MuJoCo graph instead of launching one "
            "(auto-detected when /simulation/actuator_states and /simulation/odom exist)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the child commands without starting processes",
    )
    return parser


def _add_boolean_option(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    """Add ``--foo``/``--no-foo`` flags on Python 3.8 and newer."""
    positive = "--" + name.replace("_", "-")
    negative = "--no-" + name.replace("_", "-")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(positive, dest=name, action="store_true", help=help)
    group.add_argument(
        negative,
        dest=name,
        action="store_false",
        help="disable this option",
    )
    parser.set_defaults(**{name: bool(default)})


def _validate_arguments(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 <= arguments.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be between 0 and 232")
    for name in ("sim_start_timeout", "pose_timeout", "arm_timeout", "ros_wait_timeout", "ros_final_hold_sec"):
        value = float(getattr(arguments, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if arguments.sim_start_timeout == 0.0:
        parser.error("--sim-start-timeout must be greater than zero")


def run(arguments: argparse.Namespace) -> int:
    _check_files()
    pose_path = arguments.pose_json.expanduser().resolve()
    report_path = arguments.report_json.expanduser().resolve()
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous_pose_mtime_ns: Optional[int] = None
    try:
        previous_pose_mtime_ns = pose_path.stat().st_mtime_ns
    except FileNotFoundError:
        pass
    environment = _child_environment(arguments.ros_domain_id)
    print(f"[pipeline] ROS child environment: {_ros_environment_summary(environment)}", flush=True)
    detection_marker = MY_SIM / ".head_detection.ready"
    detection_marker.unlink(missing_ok=True)

    simulation_command = [
        "ros2",
        "launch",
        "bxi_example_py_elf3",
        "example_launch_demo.py",
        "enable_arm_ik:=true",
    ]
    head_command = [str(ROS_PYTHON), str(MY_SIM / "head_control_mujoco.py")]
    head_command.append("--show" if arguments.show else "--no-show")
    head_command.extend(["--detection-file", str(detection_marker)])
    pose_command = [
        str(ROS_PYTHON),
        str(MY_SIM / "get_pose_mujoco.py"),
        "--world-pose",
        "--output",
        str(pose_path),
        "--save-images",
    ]
    pose_command.append("--show" if arguments.show else "--no-show")
    pose_command.extend(["--timeout", str(arguments.pose_timeout)])
    arm_command = [
        str(MJ_PYTHON),
        str(MY_SIM / "arm_ik.py"),
        "--pose-json",
        str(pose_path),
        "--report-json",
        str(report_path),
        "--live-ros",
        "--ros-domain-id",
        str(arguments.ros_domain_id),
        "--ros-wait-timeout",
        str(arguments.ros_wait_timeout),
        "--ros-final-hold-sec",
        str(arguments.ros_final_hold_sec),
    ]
    if arguments.ros_navigate:
        arm_command.append("--ros-navigate")
    if arguments.ros_hold_final:
        arm_command.append("--ros-hold-final")

    if arguments.dry_run:
        print(f"ROS_DOMAIN_ID={arguments.ros_domain_id}")
        print("simulation:", _quote_command(simulation_command))
        print("head:", _quote_command(head_command))
        print("pose:", _quote_command(pose_command))
        print("arm:", _quote_command(arm_command))
        return 0

    # Reuse an already running MuJoCo graph automatically.  Older versions
    # required ``--reuse-sim`` and checked camera topics too strictly; when a
    # camera name differed, a second simulator was launched in the same ROS
    # domain and navigation commands were split between two controllers.
    # ``--reuse-sim`` remains accepted as an explicit override.
    if not arguments.reuse_sim and _detect_existing_simulation(environment):
        arguments.reuse_sim = True
        print(
            "[pipeline] detected an existing ROS2 MuJoCo graph; "
            "reusing its window (no second simulator will be started)",
            flush=True,
        )

    simulation: Optional[ManagedProcess] = None
    head: Optional[ManagedProcess] = None
    pose: Optional[ManagedProcess] = None
    arm: Optional[ManagedProcess] = None
    success = False
    interrupted = False
    try:
        if arguments.reuse_sim:
            print("[pipeline] reusing the existing ROS2 MuJoCo graph", flush=True)
        else:
            simulation = _start_process("MuJoCo launch", simulation_command, environment)
        _wait_for_simulation(simulation, environment, arguments.sim_start_timeout)
        discovered_camera = _discover_camera_topics(environment)
        if discovered_camera is None:
            # The standard ELF3 MJCF uses these names.  Topic discovery can
            # transiently return an incomplete set while DDS is rebuilding
            # large Image endpoints; do not abort a valid launch in that
            # window.  The head/pose nodes will report a real transport error
            # if the publishers are genuinely absent.
            rgb_topic = "/simulation/head_depth_camera/color/image_raw"
            depth_topic = "/simulation/head_depth_camera/depth/image_rect_raw"
            print(
                "[pipeline] camera topic discovery was incomplete; using the "
                f"standard ELF3 topics rgb={rgb_topic}, depth={depth_topic}",
                flush=True,
            )
        else:
            rgb_topic, depth_topic = discovered_camera
            print(
                f"[pipeline] using MuJoCo camera topics: rgb={rgb_topic}, depth={depth_topic}",
                flush=True,
            )
        _wait_for_camera_publisher(rgb_topic, environment)
        head_command.extend(["--camera-topic", rgb_topic])
        pose_command.extend(["--rgb-topic", rgb_topic, "--depth-topic", depth_topic])
        _wait_for_arm_override(simulation, environment, arguments.sim_start_timeout)
        # The controller logs the reset transition to ``normal`` but some
        # builds do not publish state_machine_info.  The override subscriber
        # appears before reset completes, so use a short settling delay rather
        # than blocking forever on that optional status topic.
        time.sleep(2.0)
        if simulation is not None and simulation.returncode is not None:
            raise PipelineError(
                "MuJoCo/controller exited during controller startup "
                f"(code {simulation.returncode})"
            )

        head = _start_process("head control", head_command, environment)
        # Let the head node initialize its model and publish its first command;
        # get_pose itself still waits for synchronized camera/state samples.
        init_deadline = time.monotonic() + 1.0
        while time.monotonic() < init_deadline:
            if head.returncode is not None:
                raise PipelineError(
                    f"head control exited immediately (code {head.returncode})"
                )
            if simulation is not None and simulation.returncode is not None:
                raise PipelineError(
                    "MuJoCo launch exited while head control was starting "
                    f"(code {simulation.returncode})"
                )
            time.sleep(min(0.1, max(0.0, init_deadline - time.monotonic())))
        if head.returncode is not None:
            raise PipelineError(f"head control exited immediately (code {head.returncode})")

        # The perception order is deliberate: wait for the scanner to latch a
        # bottle before starting FoundationPose.  The head node exits shortly
        # after detection, leaving its final joint command in the controller's
        # mailbox until the watchdog releases it.
        detection_deadline = time.monotonic() + arguments.pose_timeout
        while not detection_marker.is_file():
            if head.returncode is not None:
                raise PipelineError(f"head scan/detection failed (code {head.returncode})")
            if simulation is not None and simulation.returncode is not None:
                raise PipelineError(
                    "MuJoCo/controller exited during head scan "
                    f"(code {simulation.returncode})"
                )
            if time.monotonic() >= detection_deadline:
                raise PipelineError(f"head scan timed out after {arguments.pose_timeout:.1f}s")
            time.sleep(0.2)
        print("[pipeline] bottle detected; head command remains active for pose estimation", flush=True)

        pose_started_at_ns = time.time_ns()
        pose = _start_process("6D pose estimation", pose_command, environment)
        # Both publishers are required for synchronized RGB-D/world-pose
        # samples.  Fail promptly if either the head node or MuJoCo dies,
        # instead of waiting for the full perception timeout.
        pose_code = _wait_for_process(
            pose,
            arguments.pose_timeout,
            watch=head,
            watches=(simulation,),
        )
        if pose_code != 0:
            raise PipelineError(f"6D pose estimation failed (code {pose_code})")
        _load_world_pose(pose_path, pose_started_at_ns, previous_pose_mtime_ns)

        arm = _start_process("arm IK replay", arm_command, environment)
        arm_code = _wait_for_process(arm, arguments.arm_timeout, watch=simulation)
        if arm_code != 0:
            raise PipelineError(f"arm IK replay failed (code {arm_code})")
        if not report_path.is_file():
            raise PipelineError(f"arm IK completed without writing {report_path}")
        print("[pipeline] complete: the hand trajectory was replayed in the ROS2 MuJoCo window", flush=True)
        success = True
        return 0
    except KeyboardInterrupt:
        interrupted = True
        print("\n[pipeline] interrupted by user", flush=True)
        return 130
    finally:
        # The head publisher must stay alive through pose and arm replay, but
        # it is safe to stop it once the arm stage is complete.  Its controller
        # mailbox watchdog then releases only the head joints.
        _stop_process(arm)
        _stop_process(pose)
        _stop_process(head)
        if simulation is not None and (not success or not arguments.keep_sim or interrupted):
            _stop_process(simulation)
        elif simulation is not None and success:
            print("[pipeline] leaving the MuJoCo window running (use --no-keep-sim to stop it)", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _validate_arguments(arguments, parser)
    try:
        return run(arguments)
    except PipelineError as error:
        print(f"run_grasp_pipeline: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
