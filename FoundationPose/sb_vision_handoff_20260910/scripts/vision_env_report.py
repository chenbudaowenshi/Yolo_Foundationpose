#!/usr/bin/env python3
"""Report what an environment IS.  Run it in yours, run it in mine, compare.

    python3 vision_env_report.py                # 用当前解释器
    python3 vision_env_report.py --json         # 便于两边 diff

This is a QUESTION, not a gate.  The vision engineer owns their environment; whether FoundationPose runs on
3.10 or 3.11 is their call and their expertise.  What this script settles is a decision on OUR side: which
integration shape to use.

The checks below are the preconditions for putting the vision code in the SAME PROCESS as rclpy, which on
Humble is a `cpython-310` binary extension:
  - Python 3.10          : rclpy's .so is cp310; 3.11 cannot load it, no workaround
  - rclpy importable     : the ROS environment is actually sourced
  - ObjectPose importable: our custom message package is on the path
  - numpy 1.x            : ROS Humble's message conversion is built against numpy 1
  - torch sees the GPU and actually executes a CUDA matrix multiplication;
    the architecture list is diagnostic, not an exact-string compatibility gate

All green -> the simple shape works (they hand over a plain Python class, we wrap the ROS node around it).
Any red -> the decoupled shape (separate process + IPC), which is equally workable and costs THEM nothing;
the extra bridge is on our side.  Neither outcome is a failure on their part, and the script says so.

Exit code is 0 / non-zero只为脚本化方便，不是"合格 / 不合格"。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys

#: Reported for comparison but never fatal - a missing one just means "that half isn't installed here".
SOFT_MODULES = ("ultralytics", "cv2", "pyrealsense2", "scipy", "nvdiffrast", "pytorch3d",
                "warp", "trimesh", "open3d", "pyrender", "kaolin")


def probe(name: str) -> str | None:
    """Version string of an importable module, or None.  Never raises: a broken module is a missing one."""
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a module that explodes on import is unusable either way
        return None
    return str(getattr(module, "__version__", "installed"))


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return {}
    line = out.stdout.strip().splitlines()
    if not line:
        return {}
    parts = [p.strip() for p in line[0].split(",")]
    return {"name": parts[0], "driver": parts[1] if len(parts) > 1 else "?",
            "memory": parts[2] if len(parts) > 2 else "?"}


def torch_report() -> dict:
    """Everything about torch that can make it useless here, in one dict."""
    info: dict = {"version": None, "cuda_build": None, "available": False,
                  "device": None, "capability": None, "arch_list": [], "arch_ok": None}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["version"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    try:
        info["available"] = bool(torch.cuda.is_available())
        info["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    if not info["available"]:
        return info
    major, minor = torch.cuda.get_device_capability(0)
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = f"{major}.{minor}"
    # Discovery alone does not establish that CUDA kernels can execute.
    want = f"sm_{major}{minor}"
    info["native_arch_listed"] = want in info["arch_list"]
    # Absence of an exact sm_XX entry does not imply incompatibility. Verify a
    # kernel on this device; keep the build's architecture list for diagnostics.
    try:
        probe_tensor = torch.ones((16, 16), device="cuda")
        info["arch_ok"] = float((probe_tensor @ probe_tensor)[0, 0].item()) == 16.0
        torch.cuda.synchronize()
        info["kernel_probe"] = "matrix_multiply_passed"
    except Exception as exc:
        info["arch_ok"] = False
        info["kernel_probe"] = f"{type(exc).__name__}: {exc}"
    info["want_arch"] = want
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="机器可读输出（方便两台机器 diff）")
    args = parser.parse_args()

    py = platform.python_version()
    py_ok = sys.version_info[:2] == (3, 10)
    rclpy_v = probe("rclpy")
    msg_ok = False
    try:
        from sb_communication.msg import ObjectPose  # noqa: F401
        msg_ok = True
    except Exception:  # noqa: BLE001
        pass
    numpy_v = probe("numpy")
    numpy_ok = bool(numpy_v) and numpy_v.split(".")[0] == "1"
    torch_i = torch_report()
    gpu = gpu_info()
    soft = {name: probe(name) for name in SOFT_MODULES}

    checks = [
        ("Python 是 3.10（rclpy 的 .so 是 cp310，3.11 加载不了）", py_ok, py),
        ("rclpy 可导入（ROS 环境已 source）", rclpy_v is not None, rclpy_v or "缺失"),
        ("sb_communication/ObjectPose 可导入", msg_ok, "OK" if msg_ok else "缺失，需 source 我们的 ws"),
        ("numpy 是 1.x（Humble 的消息转换按 numpy 1 编译）", numpy_ok, numpy_v or "缺失"),
        ("torch 可用且能看到 GPU", bool(torch_i["available"]), torch_i.get("version") or "缺失"),
        (f"torch CUDA 内核实测（设备 {torch_i.get('want_arch', 'sm_??')}）",
         bool(torch_i.get("arch_ok")), f"cu{torch_i.get('cuda_build')} arch={','.join(torch_i['arch_list']) or '-'}"),
    ]

    # A stable short hash of the versions that matter, so two people can compare environments by one token
    # instead of eyeballing twenty lines.
    material = json.dumps({"py": py, "numpy": numpy_v, "torch": torch_i.get("version"),
                           "cuda": torch_i.get("cuda_build"), **soft}, sort_keys=True)
    fingerprint = hashlib.sha256(material.encode()).hexdigest()[:12]

    if args.json:
        print(json.dumps({"python": py, "executable": sys.executable,
                          "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                          "ros_distro": os.environ.get("ROS_DISTRO"), "rclpy": rclpy_v,
                          "object_pose": msg_ok, "numpy": numpy_v, "torch": torch_i, "gpu": gpu,
                          "modules": soft, "fingerprint": fingerprint,
                          "hard_pass": all(ok for _, ok, _ in checks)}, ensure_ascii=False, indent=2))
        return 0 if all(ok for _, ok, _ in checks) else 1

    print("=" * 74)
    print("视觉节点环境体检")
    print("=" * 74)
    print(f"  解释器      {sys.executable}")
    print(f"  Python      {py}")
    print(f"  conda 环境  {os.environ.get('CONDA_DEFAULT_ENV') or '（不在 conda 里）'}")
    print(f"  ROS_DISTRO  {os.environ.get('ROS_DISTRO') or '（未 source）'}")
    if gpu:
        print(f"  GPU         {gpu['name']} | 驱动 {gpu['driver']} | 显存 {gpu['memory']}")
    if torch_i.get("capability"):
        print(f"  算力         {torch_i['capability']}  (需要 {torch_i['want_arch']} 的核)")

    print("\n同进程前提（决定我们用哪种对接形态，不是对你的要求）")
    for name, ok, detail in checks:
        print(f"  {'是' if ok else '否'}  {name}")
        print(f"      └─ {detail}")

    print("\n依赖版本（只做比对，不判定）")
    for name, version in soft.items():
        print(f"      {name:14} {version or '—'}")

    failed = [name for name, ok, _ in checks if not ok]
    print("\n" + "-" * 74)
    print(f"  环境指纹  {fingerprint}      ← 两边各跑一次，指纹一样就是同一套环境")
    if not failed:
        print("  结论      这个环境可以和 ROS 同进程 → 走「形态 B」：你交一个纯 Python 类，")
        print("            ROS 那层我来包，你完全不用碰 ROS。这是最省事的一种。")
    else:
        print(f"  结论      有 {len(failed)} 项不满足同进程的前提 → 走「形态 C」：你的进程独立跑，")
        print("            中间用 IPC，我这边多写一个桥接节点。")
        print("            **这不是你的问题，也不需要你改环境**——桥接的活在我这边。")
        if not py_ok:
            print(f"            （关键那项是 Python：这里是 {py}，rclpy 需要 3.10）")
        if rclpy_v is None or not msg_ok:
            print("            （如果你本来就打算走同进程：source /opt/ros/humble/setup.bash")
            print("              和 ws/install/setup.bash 之后再跑一次）")
    print("\n  把这份输出发回给我，连同 docs/vision_handoff.md 第 8.3 节那九个问题的回答。")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
