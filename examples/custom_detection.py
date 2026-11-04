#!/usr/bin/env python3
"""Train and validate a custom YOLOv8 object-detection dataset.

Examples:
    python examples/custom_detection.py check
    python examples/custom_detection.py train --trust-checkpoints
    python examples/custom_detection.py val --weights runs/custom/exp/weights/best.pt --trust-checkpoints

This repository is Ultralytics 8.0.29, while the configured environment uses
PyTorch 2.7 for RTX 50-series GPU support. PyTorch 2.7 restricts legacy pickle
checkpoints by default. The ``--trust-checkpoints`` flag enables legacy loading
for this process only. Use it only for checkpoints you created yourself or got
from a trusted source.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "configs/custom_detection.yaml"
DEFAULT_MODEL = REPO_ROOT / "ultralytics/models/v8/seg/yolov8n-seg.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_dataset_config(data_file: Path) -> dict[str, Any]:
    """Load the dataset YAML and perform inexpensive checks before training."""
    data_file = data_file.expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_file}")

    with data_file.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    missing_keys = [key for key in ("path", "train", "val", "names") if key not in config]
    if missing_keys:
        raise ValueError(f"Missing YAML keys: {', '.join(missing_keys)}")

    root = Path(config["path"]).expanduser()
    if not root.is_absolute():
        raise ValueError("Use an absolute dataset path in 'path:' to avoid ambiguous path resolution.")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    names = config["names"]
    if isinstance(names, list):
        class_ids = list(range(len(names)))
    elif isinstance(names, dict):
        class_ids = sorted(int(class_id) for class_id in names)
    else:
        raise TypeError("'names:' must be a list or a mapping such as {0: person, 1: car}.")
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"Class IDs must be contiguous and start at 0; got {class_ids}.")

    config["_file"] = data_file
    config["_root"] = root
    config["_class_count"] = len(class_ids)
    return config


def labels_dir_for(images_dir: Path) -> Path:
    """Map .../images/<split> to .../labels/<split>."""
    parts = list(images_dir.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Image directory must contain an 'images' component: {images_dir}") from exc
    return Path(*parts[:index], "labels", *parts[index + 1 :])


def check_split(config: dict[str, Any], split: str) -> tuple[int, int, int]:
    """Validate one split and return image, label, and missing-label counts."""
    images_dir = config["_root"] / config[split]
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{split} image directory not found: {images_dir}")

    labels_dir = labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"{split} label directory not found: {labels_dir}")

    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found in: {images_dir}")

    label_count = 0
    missing_label_count = 0
    errors: list[str] = []
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        if not label.exists():
            # A missing label is accepted by YOLO as a background-only image.
            missing_label_count += 1
            continue

        label_count += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"{label}:{line_number}: expected 5 fields, got {len(fields)}")
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                errors.append(f"{label}:{line_number}: contains a non-numeric value")
                continue
            class_value, *box = values
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < config["_class_count"]:
                errors.append(f"{label}:{line_number}: invalid class ID {class_value}")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in box):
                errors.append(f"{label}:{line_number}: box values must be normalized to [0, 1]")

    if errors:
        preview = "\n".join(errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Invalid labels:\n{preview}{suffix}")
    return len(images), label_count, missing_label_count


def check_dataset(data_file: Path) -> dict[str, Any]:
    """Check train/val directories, class IDs, and YOLO label rows."""
    config = load_dataset_config(data_file)
    print(f"Dataset root: {config['_root']}")
    print(f"Classes: {config['_class_count']}")
    for split in ("train", "val"):
        images, labels, missing = check_split(config, split)
        print(f"{split}: {images} images, {labels} label files, {missing} background/missing labels")
    print("Dataset check passed.")
    return config


def enable_trusted_legacy_checkpoints(confirmed: bool) -> None:
    """Enable old YOLO checkpoint loading only after an explicit trust decision."""
    if not confirmed:
        raise SystemExit(
            "This old YOLOv8 release must fully deserialize .pt checkpoints. "
            "Review the checkpoint source, then rerun with --trust-checkpoints."
        )
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print("Legacy checkpoint loading enabled for this process only; use trusted .pt files only.")


def train(args: argparse.Namespace) -> None:
    check_dataset(args.data)
    enable_trusted_legacy_checkpoints(args.trust_checkpoints)

    # Import after the process-scoped checkpoint policy has been decided.
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        val=True,
        plots=True,
    )

    best = Path(model.trainer.best).resolve()
    last = Path(model.trainer.last).resolve()
    print(f"Best checkpoint: {best}")
    print(f"Last checkpoint: {last}")
    print(
        "Independent validation:\n"
        f"python examples/custom_detection.py val --weights {best} --trust-checkpoints"
    )


def validate(args: argparse.Namespace) -> None:
    check_dataset(args.data)
    enable_trusted_legacy_checkpoints(args.trust_checkpoints)
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.val(
        data=str(args.data.resolve()),
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        plots=True,
    )
    print(f"Validation completed with: {weights}")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640, help="square input image size")
    parser.add_argument("--batch", type=int, default=16, help="batch size; reduce it if CUDA runs out of memory")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument("--workers", type=int, default=8, help="data-loader worker processes")
    parser.add_argument(
        "--trust-checkpoints",
        action="store_true",
        help="allow this process to fully load trusted legacy .pt checkpoints",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="check dataset paths and YOLO labels")
    check_parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset YAML path")
    check_parser.set_defaults(func=lambda args: check_dataset(args.data))

    train_parser = subparsers.add_parser("train", help="train and validate after every epoch")
    add_runtime_arguments(train_parser)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="model .yaml or trusted .pt path")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=30, help="early-stopping patience")
    train_parser.add_argument("--cache", action="store_true", help="cache images in RAM")
    train_parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs/custom")
    train_parser.add_argument("--name", default="exp")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--exist-ok", action="store_true", help="reuse an existing run directory")
    train_parser.set_defaults(func=train)

    val_parser = subparsers.add_parser("val", help="independently validate a trained checkpoint")
    add_runtime_arguments(val_parser)
    val_parser.add_argument("--weights", type=Path, required=True, help="trusted best.pt or last.pt path")
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
