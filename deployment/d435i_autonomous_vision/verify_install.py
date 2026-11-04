#!/usr/bin/env python3
"""Verify files, Python modules, CUDA, FoundationPose, and optional D435i access."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--foundationpose-root",
        type=Path,
        default=Path(os.environ.get("FOUNDATIONPOSE_ROOT", BUNDLE_ROOT / "FoundationPose")),
    )
    value.add_argument("--require-camera", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    foundationpose_root = args.foundationpose_root.expanduser().resolve()
    required_files = [
        BUNDLE_ROOT / "examples/d435i_autonomous_vision.py",
        BUNDLE_ROOT / "examples/d435i_foundationpose.py",
        BUNDLE_ROOT / "examples/async_pose_worker.py",
        BUNDLE_ROOT / "examples/multi_object_rgbd_tracker.py",
        BUNDLE_ROOT / "examples/rgbd_kalman_tracker.py",
        BUNDLE_ROOT / "examples/upright_pose_utils.py",
        BUNDLE_ROOT / "examples/depth_mask_utils.py",
        BUNDLE_ROOT / "ultralytics/__init__.py",
        BUNDLE_ROOT / "runs/segment/drink_seg/weights/best.pt",
        BUNDLE_ROOT / "mesh/yinliao_edit_fixed.obj",
        foundationpose_root / "estimater.py",
        foundationpose_root / "weights/2023-10-28-18-33-37/model_best.pth",
        foundationpose_root / "weights/2024-01-11-20-02-45/model_best.pth",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        print("FAIL: missing required files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    module_names = (
        "cv2",
        "numpy",
        "scipy",
        "pyrealsense2",
        "torch",
        "torchvision",
        "trimesh",
        "nvdiffrast.torch",
        "pytorch3d",
        "open3d",
        "omegaconf",
    )
    failures: list[str] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as error:  # verification should report every missing module
            failures.append(f"{name}: {error}")
    if failures:
        print("FAIL: Python dependency checks:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    import pyrealsense2 as rs
    import torch

    if not torch.cuda.is_available():
        print("FAIL: PyTorch cannot access a CUDA GPU", file=sys.stderr)
        return 1

    mycpp_build = foundationpose_root / "mycpp/build"
    sys.path.insert(0, str(foundationpose_root))
    sys.path.insert(0, str(mycpp_build))
    try:
        importlib.import_module("mycpp")
        importlib.import_module("estimater")
    except Exception as error:
        print(f"FAIL: FoundationPose import: {error}", file=sys.stderr)
        print("Rebuild FoundationPose/mycpp on this machine.", file=sys.stderr)
        return 1

    devices = list(rs.context().query_devices())
    if args.require_camera and not devices:
        print("FAIL: no RealSense camera detected", file=sys.stderr)
        return 1

    print("PASS: application files and Python dependencies")
    print(f"PASS: CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"PASS: FoundationPose: {foundationpose_root}")
    if devices:
        print(f"PASS: RealSense devices: {len(devices)}")
    else:
        print("INFO: no RealSense detected (use --require-camera to make this an error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

