#!/usr/bin/env python3
"""Check, train, and validate a custom YOLOv8 instance-segmentation dataset.

Examples:
    python examples/custom_segmentation.py check
    python examples/custom_segmentation.py train --trust-checkpoints
    python examples/custom_segmentation.py val \
        --weights runs/segment/drink_seg/weights/best.pt --trust-checkpoints
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "datasets/custom_segmentation_sam2/data.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics/models/v8/seg/yolov8n-seg.yaml"
DEFAULT_DETECT_WEIGHTS = REPO_ROOT / "runs/custom/baseline/weights/best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_dataset_config(data_file: Path) -> dict[str, Any]:
    data_file = data_file.expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_file}")
    config = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("path", "train", "val", "names") if key not in config]
    if missing:
        raise ValueError(f"Missing YAML keys: {', '.join(missing)}")

    root = Path(config["path"]).expanduser()
    if not root.is_absolute():
        raise ValueError("Use an absolute path in data.yaml 'path:'")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    names = config["names"]
    if isinstance(names, list):
        class_ids = list(range(len(names)))
    elif isinstance(names, dict):
        class_ids = sorted(int(class_id) for class_id in names)
    else:
        raise TypeError("'names:' must be a list or an ID-to-name mapping")
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"Class IDs must be contiguous and start at 0; got {class_ids}")

    config["_file"] = data_file
    config["_root"] = root
    config["_class_count"] = len(class_ids)
    return config


def labels_dir_for(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Image directory must contain an 'images' component: {images_dir}") from exc
    return Path(*parts[:index], "labels", *parts[index + 1 :])


def polygon_area(points: list[float]) -> float:
    xs = points[0::2]
    ys = points[1::2]
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - ys[i] * xs[(i + 1) % len(xs)] for i in range(len(xs)))) / 2.0


def check_split(config: dict[str, Any], split: str) -> tuple[int, int, int]:
    images_dir = config["_root"] / config[split]
    labels_dir = labels_dir_for(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{split} images not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"{split} labels not found: {labels_dir}")

    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    label_files = 0
    instance_count = 0
    errors: list[str] = []
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        if not label.is_file():
            continue
        label_files += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
                errors.append(
                    f"{label}:{line_number}: expected class plus >=3 xy points; got {len(fields)} fields"
                )
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                errors.append(f"{label}:{line_number}: non-numeric value")
                continue
            class_value = values[0]
            class_id = int(class_value)
            coordinates = values[1:]
            if class_value != class_id or not 0 <= class_id < config["_class_count"]:
                errors.append(f"{label}:{line_number}: invalid class ID {class_value}")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
                errors.append(f"{label}:{line_number}: polygon coordinates must be finite and normalized to [0, 1]")
            elif polygon_area(coordinates) < 1e-5:
                errors.append(f"{label}:{line_number}: polygon area is zero or too small")
            instance_count += 1

    if errors:
        preview = "\n".join(errors[:20])
        extra = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Invalid segmentation labels:\n{preview}{extra}")
    return len(images), label_files, instance_count


def check_dataset(data_file: Path) -> dict[str, Any]:
    config = load_dataset_config(data_file)
    print(f"Dataset root: {config['_root']}")
    print(f"Classes: {config['_class_count']}")
    for split in ("train", "val"):
        images, labels, instances = check_split(config, split)
        print(f"{split}: {images} images, {labels} label files, {instances} instances")
    print("Segmentation dataset check passed.")
    return config


def enable_trusted_legacy_checkpoints(confirmed: bool) -> None:
    if not confirmed:
        raise SystemExit(
            "This old YOLOv8 release fully deserializes .pt files. "
            "Confirm the checkpoints are trusted, then add --trust-checkpoints."
        )
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print("Legacy checkpoint loading enabled for this process only.")


def train(args: argparse.Namespace) -> None:
    check_dataset(args.data)
    model_path = args.model.expanduser().resolve()
    init_detect = None if args.no_detect_init else args.init_detect_weights.expanduser().resolve()
    needs_checkpoint = model_path.suffix.lower() == ".pt" or init_detect is not None
    if needs_checkpoint:
        enable_trusted_legacy_checkpoints(args.trust_checkpoints)
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation model/config not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError(f"--model must be a segmentation model; inferred task={model.task}")

    if init_detect is not None:
        if model_path.suffix.lower() != ".yaml":
            raise ValueError("--init-detect-weights is only used when --model is a segmentation YAML")
        if not init_detect.is_file():
            raise FileNotFoundError(f"Detection checkpoint not found: {init_detect}")
        detector = YOLO(str(init_detect))
        if detector.task != "detect":
            raise ValueError(f"Expected detection initialization weights, got task={detector.task}")
        model.model.load(detector.model)
        # YOLO.train() only forwards an in-memory model to the trainer when a
        # checkpoint marker is truthy. This preserves the transferred backbone
        # while the segmentation prototype/head remains newly initialized.
        model.ckpt = {"transferred_from": str(init_detect)}
        print(f"Initialized matching segmentation backbone/neck layers from: {init_detect}")

    model.train(
        data=str(args.data.expanduser().resolve()),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        val=True,
        plots=True,
        close_mosaic=args.close_mosaic,
    )
    best = Path(model.trainer.best).resolve()
    print(f"Best checkpoint: {best}")
    print(
        "Independent validation:\n"
        f"python examples/custom_segmentation.py val --weights {best} --trust-checkpoints"
    )


def validate(args: argparse.Namespace) -> None:
    check_dataset(args.data)
    enable_trusted_legacy_checkpoints(args.trust_checkpoints)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError(f"Expected segmentation checkpoint, got task={model.task}")
    model.val(
        data=str(args.data.expanduser().resolve()),
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        plots=True,
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trust-checkpoints", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    check_parser.set_defaults(func=lambda args: check_dataset(args.data))

    train_parser = subparsers.add_parser("train")
    add_runtime_arguments(train_parser)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--init-detect-weights", type=Path, default=DEFAULT_DETECT_WEIGHTS)
    train_parser.add_argument("--no-detect-init", action="store_true", help="train segmentation model from scratch")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=30)
    train_parser.add_argument("--close-mosaic", type=int, default=10)
    train_parser.add_argument("--cache", action="store_true")
    train_parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs/segment")
    train_parser.add_argument("--name", default="drink_seg")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--exist-ok", action="store_true")
    train_parser.set_defaults(func=train)

    val_parser = subparsers.add_parser("val")
    add_runtime_arguments(val_parser)
    val_parser.add_argument("--weights", type=Path, required=True)
    val_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
