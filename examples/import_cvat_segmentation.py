#!/usr/bin/env python3
"""Validate and safely merge CVAT Ultralytics segmentation exports.

CVAT can reorder class IDs independently in each task. This importer maps IDs
by class name back to the canonical IDs in the destination data.yaml, requires
a one-to-one label/image stem match, validates every polygon, and creates a
backup before replacing any generated SAM2 labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets/custom_segmentation_sam2"
DEFAULT_EXPORTS = REPO_ROOT / "datasets/cvat_exports"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_names(value: Any, source: str) -> dict[int, str]:
    if isinstance(value, list):
        names = dict(enumerate(str(name) for name in value))
    elif isinstance(value, dict):
        names = {int(class_id): str(name) for class_id, name in value.items()}
    else:
        raise ValueError(f"{source}: 'names' must be a list or mapping")
    if sorted(names) != list(range(len(names))):
        raise ValueError(f"{source}: class IDs must be contiguous from 0")
    if len(set(names.values())) != len(names):
        raise ValueError(f"{source}: class names must be unique")
    return names


def canonical_names(dataset_root: Path) -> dict[int, str]:
    data_file = dataset_root / "data.yaml"
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_file}")
    config = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    return normalize_names(config.get("names"), str(data_file))


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"ZIP symlink is not allowed: {info.filename}")
        members.append(info)
    return members


def read_archive_names(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], path: Path) -> dict[int, str]:
    yaml_members = [info for info in members if PurePosixPath(info.filename).name == "data.yaml"]
    if len(yaml_members) != 1:
        raise ValueError(f"{path}: expected exactly one data.yaml, found {len(yaml_members)}")
    config = yaml.safe_load(archive.read(yaml_members[0]).decode("utf-8")) or {}
    return normalize_names(config.get("names"), f"{path}:data.yaml")


def polygon_area(coordinates: list[float]) -> float:
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return abs(
        sum(xs[index] * ys[(index + 1) % len(xs)] - ys[index] * xs[(index + 1) % len(xs)] for index in range(len(xs)))
    ) / 2.0


def remap_label(
    content: str,
    source: str,
    archive_names: dict[int, str],
    canonical_id_for_name: dict[str, int],
) -> tuple[str, Counter[int]]:
    output: list[str] = []
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"{source}:{line_number}: invalid YOLO segmentation field count {len(fields)}")
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: non-numeric field") from error
        archive_id = int(class_value)
        if class_value != archive_id or archive_id not in archive_names:
            raise ValueError(f"{source}:{line_number}: invalid archive class ID {class_value}")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{source}:{line_number}: coordinates must be finite and normalized to [0, 1]")
        if polygon_area(coordinates) < 1e-7:
            raise ValueError(f"{source}:{line_number}: polygon area is zero or too small")
        class_name = archive_names[archive_id]
        if class_name not in canonical_id_for_name:
            raise ValueError(f"{source}:{line_number}: unknown class name '{class_name}'")
        canonical_id = canonical_id_for_name[class_name]
        output.append(str(canonical_id) + " " + " ".join(fields[1:]))
        counts[canonical_id] += 1
    return "\n".join(output) + ("\n" if output else ""), counts


def expected_stems(dataset_root: Path, split: str) -> set[str]:
    directory = dataset_root / "images" / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Image split not found: {directory}")
    return {path.stem for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def load_export(
    path: Path,
    split: str,
    dataset_root: Path,
    canonical: dict[int, str],
) -> tuple[dict[str, str], Counter[int], dict[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CVAT export not found: {path}")
    canonical_id_for_name = {name: class_id for class_id, name in canonical.items()}
    with zipfile.ZipFile(path) as archive:
        members = safe_members(archive)
        archive_names = read_archive_names(archive, members, path)
        if set(archive_names.values()) != set(canonical.values()):
            raise ValueError(
                f"{path}: class names differ; archive={archive_names}, canonical={canonical}"
            )
        label_members = [
            info
            for info in members
            if not info.is_dir()
            and PurePosixPath(info.filename).suffix.lower() == ".txt"
            and "labels" in PurePosixPath(info.filename).parts
        ]
        labels: dict[str, str] = {}
        counts: Counter[int] = Counter()
        for info in label_members:
            stem = PurePosixPath(info.filename).stem
            if stem in labels:
                raise ValueError(f"{path}: duplicate label stem '{stem}'")
            content = archive.read(info).decode("utf-8")
            remapped, file_counts = remap_label(
                content, f"{path}:{info.filename}", archive_names, canonical_id_for_name
            )
            labels[stem] = remapped
            counts.update(file_counts)

    expected = expected_stems(dataset_root, split)
    missing = sorted(expected - set(labels))
    extra = sorted(set(labels) - expected)
    if missing or extra:
        raise ValueError(
            f"{split}: CVAT labels do not match images; missing={len(missing)}, extra={len(extra)}"
        )
    return labels, counts, archive_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_count(dataset_root: Path, split: str, labels: dict[str, str]) -> int:
    changed = 0
    for stem, content in labels.items():
        current = dataset_root / "labels" / split / f"{stem}.txt"
        if not current.is_file() or current.read_text(encoding="utf-8") != content:
            changed += 1
    return changed


def create_backup(dataset_root: Path, output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = output_dir / f"labels_before_cvat_{timestamp}.zip"
    with zipfile.ZipFile(backup, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(dataset_root / "data.yaml", "data.yaml")
        for split in ("train", "val"):
            for label in sorted((dataset_root / "labels" / split).glob("*.txt")):
                archive.write(label, f"labels/{split}/{label.name}")
    return backup


def apply_labels(dataset_root: Path, loaded: dict[str, dict[str, str]]) -> None:
    with tempfile.TemporaryDirectory(prefix=".cvat-label-stage-", dir=dataset_root) as temp:
        stage = Path(temp)
        for split, labels in loaded.items():
            split_stage = stage / split
            split_stage.mkdir(parents=True)
            for stem, content in labels.items():
                (split_stage / f"{stem}.txt").write_text(content, encoding="utf-8")
        for split, labels in loaded.items():
            destination = dataset_root / "labels" / split
            for stem in labels:
                os.replace(stage / split / f"{stem}.txt", destination / f"{stem}.txt")
    for cache in (dataset_root / "labels").glob("*.cache"):
        cache.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--val-zip", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--apply", action="store_true", help="backup and replace labels after validation")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    exports_dir = args.exports_dir.expanduser().resolve()
    paths = {
        "train": args.train_zip.expanduser().resolve(),
        "val": args.val_zip.expanduser().resolve(),
    }
    try:
        canonical = canonical_names(dataset_root)
        loaded: dict[str, dict[str, str]] = {}
        counts: dict[str, Counter[int]] = {}
        archive_names: dict[str, dict[int, str]] = {}
        changes: dict[str, int] = {}
        for split in ("train", "val"):
            loaded[split], counts[split], archive_names[split] = load_export(
                paths[split], split, dataset_root, canonical
            )
            changes[split] = changed_count(dataset_root, split, loaded[split])
            distribution = ", ".join(
                f"{class_id}:{canonical[class_id]}={counts[split][class_id]}" for class_id in canonical
            )
            print(
                f"{split}: {len(loaded[split])} label files; changed={changes[split]}; "
                f"instances: {distribution}; archive names={archive_names[split]}"
            )
        print("All CVAT labels passed structural, class-name, coordinate, and stem checks.")
        if not args.apply:
            print("Dry run only. Add --apply to create a backup and merge these labels.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = create_backup(dataset_root, exports_dir, timestamp)
        copied_exports: dict[str, str] = {}
        for split, source in paths.items():
            destination = exports_dir / f"{split}_cvat_export_{timestamp}.zip"
            shutil.copy2(source, destination)
            copied_exports[split] = str(destination)
        apply_labels(dataset_root, loaded)
        summary = {
            "timestamp": timestamp,
            "dataset_root": str(dataset_root),
            "canonical_names": canonical,
            "source_exports": {split: str(path) for split, path in paths.items()},
            "source_sha256": {split: sha256(path) for split, path in paths.items()},
            "archived_exports": copied_exports,
            "backup": str(backup),
            "changed_label_files": changes,
            "instances": {split: dict(counts[split]) for split in counts},
        }
        summary_path = exports_dir / f"import_summary_{timestamp}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Backup: {backup}")
        print(f"Import summary: {summary_path}")
        print("CVAT labels merged successfully; stale YOLO cache files were removed.")
    except (FileExistsError, FileNotFoundError, OSError, TypeError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
