"""Exercise the public shell entry points without starting ROS nodes or CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "my_demo/scripts/run_sb_vision.sh"
ENV_SCRIPT = ROOT / "my_demo/ros2_handoff_env.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sb launch test ", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.probe = Path(self.directory.name) / "python probe"
        self.probe.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
            "    'logs': os.environ.get('ROS_LOG_DIR'),\n"
            "    'extensions': os.environ.get('TORCH_EXTENSIONS_DIR'),\n"
            "    'pycache': os.environ.get('PYTHONPYCACHEPREFIX')}))\n"
        )
        self.probe.chmod(0o755)
        self.env = os.environ.copy()
        for name in ("ROS_LOG_DIR", "TORCH_EXTENSIONS_DIR", "SB_DDS_INTERFACE", "SB_DDS_PEERS"):
            self.env.pop(name, None)
        self.env["SB_VISION_PYTHON"] = str(self.probe)

    def launch(self, *arguments):
        result = subprocess.run(
            ["bash", str(LAUNCHER), *arguments], cwd="/tmp", env=self.env,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_launch_from_tmp_resolves_entry_and_default_paths(self):
        state = self.launch("--metrics-jsonl", "reports/live metrics.jsonl")
        self.assertEqual(state["argv"], [str(ROOT / "my_demo/sb_vision_node.py"),
                                         "--metrics-jsonl", "reports/live metrics.jsonl"])
        self.assertEqual(state["cwd"], str(ROOT))
        self.assertEqual(state["logs"], str(ROOT / "runs/sb_vision/ros_logs"))
        self.assertEqual(state["extensions"], str(ROOT / ".cache/torch_extensions"))
        self.assertEqual(state["pycache"], str(ROOT / ".cache/python"))

    def test_custom_log_and_extension_paths_are_preserved(self):
        self.env["ROS_LOG_DIR"] = str(Path(self.directory.name) / "custom logs")
        self.env["TORCH_EXTENSIONS_DIR"] = str(Path(self.directory.name) / "custom extensions")
        state = self.launch("--help")
        self.assertEqual(state["logs"], self.env["ROS_LOG_DIR"])
        self.assertEqual(state["extensions"], self.env["TORCH_EXTENSIONS_DIR"])

    def test_sourcing_environment_preserves_caller_variables(self):
        self.assert_source_preserves_variables(expected_status=0)

    def test_invalid_interface_propagates_failure_without_clobbering_caller(self):
        self.env["SB_DDS_INTERFACE"] = "invalid interface"
        self.assert_source_preserves_variables(expected_status=1)

    def assert_source_preserves_variables(self, expected_status):
        script = '''
_sb_root=caller_root
_sb_local_ros=caller_dependencies
_sb_peers=caller_peers
_sb_peer=caller_peer
source "$1"
sb_test_status=$?
printf '%s\n' "$sb_test_status" "$_sb_root" "$_sb_local_ros" "$_sb_peers" "$_sb_peer"
'''
        result = subprocess.run(
            ["bash", "-c", script, "sb-launch-test", str(ENV_SCRIPT)],
            cwd="/tmp", env=self.env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-5:],
                         [str(expected_status), "caller_root", "caller_dependencies",
                          "caller_peers", "caller_peer"])


if __name__ == "__main__":
    unittest.main()
