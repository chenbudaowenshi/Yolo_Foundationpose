#!/usr/bin/env python3
"""Offline smoke validation for multi-instance tracking on the existing RGB-D dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import d435i_foundationpose as base
from d435i_autonomous_vision import InstancePayload
from multi_object_rgbd_tracker import Detection3D, MultiObjectRGBDTracker, NearestTargetSelector
from rgbd_kalman_tracker import TrackerConfig


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", type=Path, default=ROOT / "runs/segment/drink_seg/weights/best.pt")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "configs/datasets/image")
    parser.add_argument("--depth-dir", type=Path, default=ROOT / "configs/datasets/depth")
    parser.add_argument("--label-dir", type=Path, default=ROOT / "configs/datasets/labels")
    parser.add_argument(
        "--K-file",
        type=Path,
        default=ROOT / "runs/foundationpose_realtime/snapshots/20260724_112413_027_K.txt",
    )
    parser.add_argument("--target-class", type=int, default=2)
    parser.add_argument("--start", type=int, default=240)
    parser.add_argument("--end", type=int, default=280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/autonomous_vision/offline_validation.json")
    parser.add_argument("--trust-checkpoints", action="store_true")
    return parser


def ground_truth_count(path: Path, target_class: int) -> int:
    if not path.is_file():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and int(float(line.split()[0])) == target_class
    )


def main() -> int:
    args = build_parser().parse_args()
    if not args.trust_checkpoints:
        raise SystemExit("add --trust-checkpoints after confirming the local checkpoint is trusted")
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    model = YOLO(str(args.yolo_model.expanduser().resolve()))
    K = np.loadtxt(args.K_file.expanduser().resolve())
    tracker_config = TrackerConfig(max_missed_frames=30)
    manager = MultiObjectRGBDTracker(
        tracker_config,
        confirmation_hits=3,
        maximum_detection_misses=30,
        maximum_3d_distance_m=0.20,
    )
    selector = NearestTargetSelector(lost_grace_frames=15)
    base_parser = base.build_parser()
    inference_args = base_parser.parse_args(
        [
            "--target-class",
            str(args.target_class),
            "--mask-source",
            "seg",
            "--conf",
            str(args.conf),
            "--trust-checkpoints",
        ]
    )
    inference_args.max_det = 100

    frames: list[dict[str, object]] = []
    total_ground_truth = 0
    total_predictions = 0
    exact_count_frames = 0
    depth_usable_predictions = 0
    selected_ids: list[int] = []
    for frame_number in range(args.start, args.end + 1):
        stem = f"frame_{frame_number:06d}"
        image_path = args.image_dir / f"{stem}.png"
        depth_path = args.depth_dir / f"{stem}.png"
        if not image_path.is_file() or not depth_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            continue
        observations, _ = base.find_targets(model, image, depth, args.depth_scale, inference_args)
        depth_m = depth.astype(np.float32) * args.depth_scale
        detections: list[Detection3D] = []
        for observation in observations:
            depth_filter = base.filter_pose_depth(
                depth_m,
                observation.mask,
                enabled=True,
                min_depth_m=0.1,
                max_depth_m=5.0,
                minimum_band_m=0.055,
                maximum_band_m=0.080,
                erode_kernel_size=7,
            )
            measurement = base.make_rgbd_position_measurement(
                K,
                observation,
                depth_filter,
                center_depth_offset_m=0.03255,
                minimum_valid_depth_ratio=0.20,
                tracker_config=tracker_config,
            )
            if measurement is not None:
                depth_usable_predictions += 1
            detections.append(
                Detection3D(
                    box=observation.box,
                    confidence=observation.confidence,
                    position_m=None if measurement is None else measurement.position_m,
                    covariance=None if measurement is None else measurement.covariance,
                    payload=InstancePayload(observation, depth_filter, measurement),
                )
            )
        views = manager.update(detections, frame_number / 30.0)
        selected = selector.select(views)
        if selected is not None:
            selected_ids.append(selected.track_id)
        truth = ground_truth_count(args.label_dir / f"{stem}.txt", args.target_class)
        predicted = len(observations)
        total_ground_truth += truth
        total_predictions += predicted
        exact_count_frames += int(truth == predicted)
        frames.append(
            {
                "stem": stem,
                "ground_truth_instances": truth,
                "predicted_instances": predicted,
                "active_track_ids": [view.track_id for view in views],
                "selected_track_id": None if selected is None else selected.track_id,
                "selected_distance_m": None if selected is None else selected.distance_m,
            }
        )

    report = {
        "target_class_id": args.target_class,
        "target_class_name": str(model.names[args.target_class]),
        "frames": len(frames),
        "ground_truth_instances": total_ground_truth,
        "predicted_instances": total_predictions,
        "exact_instance_count_rate": exact_count_frames / max(1, len(frames)),
        "depth_usable_prediction_rate": depth_usable_predictions / max(1, total_predictions),
        "unique_selected_track_ids": sorted(set(selected_ids)),
        "frame_results": frames,
        "warning": "This is a temporal/count smoke test on the existing random-split dataset, not an independent accuracy benchmark.",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frame_results"}, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
