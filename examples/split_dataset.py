#!/usr/bin/env python3
"""Randomly split RGB images, labels, and optional depth maps into train/val sets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "configs/datasets"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/custom_detection"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = IMAGE_SUFFIXES | {".npy"}


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index files by stem and reject ambiguous duplicate stems."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in indexed:
            raise ValueError(f"Duplicate stem '{path.stem}': {indexed[path.stem]} and {path}")
        indexed[path.stem] = path
    return indexed


def load_classes(classes_file: Path) -> list[str]:
    if not classes_file.is_file():
        raise FileNotFoundError(f"Class list not found: {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in: {classes_file}")
    return names


def count_instances(stems: list[str], labels: dict[str, Path], class_count: int) -> Counter[int]:
    """Count labeled objects per class and catch invalid class IDs."""
    counts: Counter[int] = Counter()
    for stem in stems:
        label = labels.get(stem)
        if label is None:
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number}: expected 5 fields, got {len(fields)}")
            try:
                class_value = float(fields[0])
                class_id = int(class_value)
            except ValueError as exc:
                raise ValueError(f"{label}:{line_number}: invalid class ID '{fields[0]}'") from exc
            if class_value != class_id or not 0 <= class_id < class_count:
                raise ValueError(f"{label}:{line_number}: class ID {class_value} is outside [0, {class_count - 1}]")
            counts[class_id] += 1
    return counts


def copy_split(
    split: str,
    stems: list[str],
    images: dict[str, Path],
    labels: dict[str, Path],
    depths: dict[str, Path] | None,
    output_root: Path,
) -> None:
    image_output = output_root / "images" / split
    label_output = output_root / "labels" / split
    depth_output = output_root / "depth" / split if depths is not None else None

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    if depth_output is not None:
        depth_output.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image = images[stem]
        shutil.copy2(image, image_output / image.name)
        if stem in labels:
            label = labels[stem]
            shutil.copy2(label, label_output / label.name)
        if depths is not None and depth_output is not None:
            depth = depths[stem]
            shutil.copy2(depth, depth_output / depth.name)


def write_metadata(
    output_root: Path,
    source_root: Path,
    names: list[str],
    train_stems: list[str],
    val_stems: list[str],
    val_ratio: float,
    seed: int,
    has_depth: bool,
) -> None:
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": None,
        "names": dict(enumerate(names)),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    manifest = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "seed": seed,
        "val_ratio": val_ratio,
        "has_depth": has_depth,
        "train": train_stems,
        "val": val_stems,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_distribution(split: str, stems: list[str], counts: Counter[int], names: list[str]) -> None:
    distribution = ", ".join(f"{class_id}:{names[class_id]}={counts[class_id]}" for class_id in range(len(names)))
    print(f"{split}: {len(stems)} images; instances: {distribution}")


def split_dataset(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    image_dir = source_root / args.image_dir
    label_dir = source_root / args.label_dir
    depth_dir = source_root / args.depth_dir

    images = index_files(image_dir, IMAGE_SUFFIXES)
    labels = index_files(label_dir, {".txt"})
    labels.pop("classes", None)
    names = load_classes(label_dir / "classes.txt")
    if len(images) < 2:
        raise ValueError(f"At least 2 images are required; found {len(images)}")

    missing_labels = sorted(set(images) - set(labels))
    extra_labels = sorted(set(labels) - set(images))
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no label file and will be treated as background images.")
    if extra_labels:
        print(f"WARNING: {len(extra_labels)} label files have no matching image and will be ignored.")

    depths: dict[str, Path] | None = None
    if depth_dir.exists():
        depths = index_files(depth_dir, DEPTH_SUFFIXES)
        missing_depths = sorted(set(images) - set(depths))
        extra_depths = sorted(set(depths) - set(images))
        if missing_depths or extra_depths:
            raise ValueError(
                "Depth maps must have a one-to-one stem match with RGB images; "
                f"missing={len(missing_depths)}, extra={len(extra_depths)}"
            )
        print(f"Depth maps detected: {len(depths)}")
    else:
        print(f"No depth directory found at {depth_dir}; splitting RGB images and labels only.")

    stems = sorted(images)
    random.Random(args.seed).shuffle(stems)
    val_count = max(1, min(len(stems) - 1, round(len(stems) * args.val_ratio)))
    val_stems = sorted(stems[:val_count])
    train_stems = sorted(stems[val_count:])

    train_counts = count_instances(train_stems, labels, len(names))
    val_counts = count_instances(val_stems, labels, len(names))
    print_distribution("train", train_stems, train_counts, names)
    print_distribution("val", val_stems, val_counts, names)
    print(f"Random seed: {args.seed}; requested val ratio: {args.val_ratio:.2%}")

    if args.dry_run:
        print("Dry run only: no files were copied.")
        return

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}\n"
            "Choose another --output-root or remove the old split after checking it."
        )

    copy_split("train", train_stems, images, labels, depths, output_root)
    copy_split("val", val_stems, images, labels, depths, output_root)
    write_metadata(
        output_root,
        source_root,
        names,
        train_stems,
        val_stems,
        args.val_ratio,
        args.seed,
        has_depth=depths is not None,
    )
    print(f"Split completed: {output_root}")
    print(f"Dataset config: {output_root / 'data.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", default="image", help="RGB image directory name under source-root")
    parser.add_argument("--label-dir", default="labels", help="label directory name under source-root")
    parser.add_argument("--depth-dir", default="depth", help="optional depth directory name under source-root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation ratio in the open interval (0, 1)")
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for a reproducible split")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the split without copying")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")
    try:
        split_dataset(args)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
