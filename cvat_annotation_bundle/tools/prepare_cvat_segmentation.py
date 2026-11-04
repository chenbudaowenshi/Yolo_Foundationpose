#!/usr/bin/env python3
"""Package existing YOLO segmentation labels for two CVAT tasks.

Images are embedded by default because the CVAT 2.70 Ultralytics importer
reads image dimensions from the annotation ZIP even when task media already
comes from a connected file share. Upload the generated ZIP with annotation
format "Ultralytics YOLO Segmentation 1.0".
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets/custom_segmentation_sam2"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/cvat_import"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def index_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in indexed:
            raise ValueError(f"Duplicate stem '{path.stem}' in {directory}")
        indexed[path.stem] = path
    return indexed


def load_names(dataset_root: Path) -> dict[int, str]:
    data_file = dataset_root / "data.yaml"
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_file}")
    config = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, list):
        return dict(enumerate(str(name) for name in names))
    if isinstance(names, dict):
        normalized = {int(class_id): str(name) for class_id, name in names.items()}
        if sorted(normalized) == list(range(len(normalized))):
            return normalized
    raise ValueError(f"Invalid contiguous class mapping in {data_file}")


def validate_label(path: Path, class_count: int) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"{path}:{line_number}: not a YOLO segmentation polygon")
        values = [float(value) for value in fields]
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class ID {values[0]}")
        if not all(0.0 <= coordinate <= 1.0 for coordinate in values[1:]):
            raise ValueError(f"{path}:{line_number}: coordinates must be normalized to [0, 1]")


def make_package(dataset_root: Path, output_dir: Path, split: str, include_images: bool) -> Path:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = index_by_stem(image_dir, IMAGE_SUFFIXES)
    labels = index_by_stem(label_dir, {".txt"})
    if set(images) != set(labels):
        missing = len(set(images) - set(labels))
        extra = len(set(labels) - set(images))
        raise ValueError(f"{split}: image/label stem mismatch; missing={missing}, extra={extra}")
    names = load_names(dataset_root)
    for label in labels.values():
        validate_label(label, len(names))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"drink_seg_{split}_annotations.zip"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {output}")

    # CVAT task frames are matched by filename. Each independent CVAT task can
    # therefore use the conventional 'train' subset inside its annotation ZIP,
    # while the output filename retains the original train/val identity.
    data = {"path": "./", "train": "train.txt", "names": names}
    data_stream = io.StringIO()
    yaml.safe_dump(data, data_stream, sort_keys=False, allow_unicode=True)
    manifest = "".join(f"images/train/{images[stem].name}\n" for stem in sorted(images))

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("data.yaml", data_stream.getvalue())
        archive.writestr("train.txt", manifest)
        for stem in sorted(images):
            archive.write(labels[stem], f"labels/train/{stem}.txt")
            if include_images:
                archive.write(images[stem], f"images/train/{images[stem].name}")
    print(f"{split}: {len(images)} images/labels -> {output} ({output.stat().st_size} bytes)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--include-images", dest="include_images", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--labels-only",
        dest="include_images",
        action="store_false",
        help="omit images (not compatible with the CVAT 2.70 import used here)",
    )
    parser.set_defaults(include_images=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    splits = ("train", "val") if args.split == "both" else (args.split,)
    try:
        for split in splits:
            make_package(dataset_root, output_dir, split, args.include_images)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
