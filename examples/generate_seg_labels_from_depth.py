#!/usr/bin/env python3
"""Generate reviewable YOLO-seg labels from boxes and aligned D435i depth.

The generated polygons are bootstrapping labels, not unquestionable ground
truth. Review the images under ``previews/`` before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

from depth_mask_utils import depth_box_mask, mask_to_yolo_polygon, overlay_mask, yolo_box_to_xyxy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "configs/datasets"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/custom_segmentation_auto"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "datasets/custom_detection/split_manifest.json"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = IMAGE_SUFFIXES | {".npy"}
COLORS = ((0, 165, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (255, 255, 0))


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate stem '{path.stem}': {files[path.stem]} and {path}")
        files[path.stem] = path
    return files


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class list not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Class list is empty: {path}")
    return names


def load_detection_labels(path: Path, class_count: int) -> list[tuple[int, list[float]]]:
    objects: list[tuple[int, list[float]]] = []
    if not path.is_file():
        return objects
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected detection label with 5 fields, got {len(fields)}")
        values = [float(value) for value in fields]
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class ID {values[0]}")
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"{path}:{line_number}: normalized box values must be in [0, 1]")
        objects.append((class_id, values[1:]))
    return objects


def read_depth(path: Path):
    if path.suffix.lower() == ".npy":
        import numpy as np

        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"Could not read a single-channel depth map: {path}")
    return depth


def choose_splits(stems: list[str], manifest_path: Path | None, val_ratio: float, seed: int) -> dict[str, str]:
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assignment = {stem: "train" for stem in manifest.get("train", [])}
        assignment.update({stem: "val" for stem in manifest.get("val", [])})
        if all(stem in assignment for stem in stems):
            print(f"Reusing detection split: {manifest_path}")
            return assignment
        print("WARNING: split manifest does not cover every source image; creating a new deterministic split.")

    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_ratio)))
    val_stems = set(shuffled[:val_count])
    return {stem: ("val" if stem in val_stems else "train") for stem in stems}


def write_dataset_yaml(output_root: Path, names: list[str]) -> None:
    config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": None,
        "names": dict(enumerate(names)),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def format_polygon(class_id: int, polygon: list[float]) -> str:
    return str(class_id) + " " + " ".join(f"{value:.6f}" for value in polygon)


def generate(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}\n"
            "Use another --output-root so unreviewed labels are never mixed with old data."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    image_dir = source_root / args.image_dir
    depth_dir = source_root / args.depth_dir
    label_dir = source_root / args.label_dir
    images = index_files(image_dir, IMAGE_SUFFIXES)
    depths = index_files(depth_dir, DEPTH_SUFFIXES)
    labels = index_files(label_dir, {".txt"})
    labels.pop("classes", None)
    names = load_classes(label_dir / "classes.txt")

    if not images:
        raise ValueError(f"No RGB images found in {image_dir}")
    missing_depth = sorted(set(images) - set(depths))
    extra_depth = sorted(set(depths) - set(images))
    if missing_depth or extra_depth:
        raise ValueError(f"RGB/depth stems must match exactly; missing={len(missing_depth)}, extra={len(extra_depth)}")

    stems = sorted(images)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        stems = stems[: args.limit]
        print(f"Pilot mode: processing the first {len(stems)} images")
    manifest_path = None if args.ignore_split_manifest else args.split_manifest.expanduser().resolve()
    split_for = choose_splits(stems, manifest_path, args.val_ratio, args.seed)
    write_dataset_yaml(output_root, names)

    report_rows: list[dict[str, object]] = []
    accepted_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    rejected_images = 0

    for index, stem in enumerate(stems, start=1):
        image_path = images[stem]
        depth_path = depths[stem]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        depth = read_depth(depth_path)
        if image is None:
            raise ValueError(f"Could not read RGB image: {image_path}")
        if image.shape[:2] != depth.shape:
            raise ValueError(f"RGB/depth shape mismatch for {stem}: {image.shape[:2]} vs {depth.shape}")

        objects = load_detection_labels(labels.get(stem, Path("/__missing__")), len(names))
        segmentation_lines: list[str] = []
        instance_masks: list[tuple[int, object]] = []
        image_failed = False
        for object_index, (class_id, box_xywhn) in enumerate(objects):
            box_xyxy = yolo_box_to_xyxy(box_xywhn, image.shape[1], image.shape[0])
            result = depth_box_mask(
                depth,
                box_xyxy,
                depth_scale=args.depth_scale,
                color_bgr=None if args.no_grabcut else image,
                seed_ratio=args.seed_ratio,
                depth_tolerance_m=args.depth_tolerance,
                max_tolerance_m=args.max_depth_tolerance,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
                min_mask_pixels=args.min_mask_pixels,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                grabcut_iterations=0 if args.no_grabcut else args.grabcut_iterations,
            )
            polygon = mask_to_yolo_polygon(result.mask, epsilon_ratio=args.polygon_epsilon) if result.valid else None
            status = "ok" if polygon is not None else "failed"
            reason = result.reason if result.reason is not None else (None if polygon is not None else "contour has <3 points")
            report_rows.append(
                {
                    "stem": stem,
                    "split": split_for[stem],
                    "object_index": object_index,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "status": status,
                    "reason": reason or "",
                    "reference_depth_m": "" if result.reference_depth_m is None else f"{result.reference_depth_m:.5f}",
                    "tolerance_m": "" if result.tolerance_m is None else f"{result.tolerance_m:.5f}",
                    "mask_pixels": result.area_pixels,
                    "mask_box_ratio": f"{result.area_ratio:.5f}",
                }
            )
            if polygon is None:
                image_failed = True
                continue
            segmentation_lines.append(format_polygon(class_id, polygon))
            instance_masks.append((class_id, result))

        # An image with one missed object would teach the model that the missed
        # object is background, so reject the entire image until it is reviewed.
        if image_failed:
            rejected_images += 1
            continue

        split = split_for[stem]
        for kind in ("images", "labels", "depth", "previews"):
            (output_root / kind / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_root / "images" / split / image_path.name)
        shutil.copy2(depth_path, output_root / "depth" / split / depth_path.name)
        (output_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(segmentation_lines) + ("\n" if segmentation_lines else ""),
            encoding="utf-8",
        )

        preview = image.copy()
        for class_id, result in instance_masks:
            preview = overlay_mask(preview, result.mask, COLORS[class_id % len(COLORS)])
            x1, y1, x2, y2 = result.box
            cv2.rectangle(preview, (x1, y1), (x2, y2), COLORS[class_id % len(COLORS)], 1)
            cv2.putText(
                preview,
                f"{names[class_id]} z={result.reference_depth_m:.2f}m",
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[class_id % len(COLORS)],
                2,
                cv2.LINE_AA,
            )
            class_counts[class_id] += 1
        preview_path = output_root / "previews" / split / f"{stem}.jpg"
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Failed to write preview: {preview_path}")
        accepted_counts[split] += 1

        if index % 50 == 0 or index == len(stems):
            print(f"Processed {index}/{len(stems)} images")

    report_path = output_root / "review.csv"
    fieldnames = [
        "stem",
        "split",
        "object_index",
        "class_id",
        "class_name",
        "status",
        "reason",
        "reference_depth_m",
        "tolerance_m",
        "mask_pixels",
        "mask_box_ratio",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_images": len(stems),
        "accepted_train_images": accepted_counts["train"],
        "accepted_val_images": accepted_counts["val"],
        "rejected_images": rejected_images,
        "instances_per_class": {names[class_id]: class_counts[class_id] for class_id in range(len(names))},
        "parameters": vars(args) | {"source_root": str(source_root), "output_root": str(output_root)},
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Accepted train images: {accepted_counts['train']}")
    print(f"Accepted val images: {accepted_counts['val']}")
    print(f"Rejected images requiring manual annotation: {rejected_images}")
    print(f"Review previews carefully: {output_root / 'previews'}")
    print(f"Failure/details report: {report_path}")
    print(f"Dataset YAML: {output_root / 'data.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--label-dir", default="labels")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="meters per uint16 depth unit")
    parser.add_argument("--depth-tolerance", type=float, default=0.045, help="minimum foreground depth band, meters")
    parser.add_argument("--max-depth-tolerance", type=float, default=0.12, help="maximum adaptive depth band, meters")
    parser.add_argument("--seed-ratio", type=float, default=0.35, help="central box ratio used as object depth seed")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--min-mask-pixels", type=int, default=150)
    parser.add_argument("--min-area-ratio", type=float, default=0.04)
    parser.add_argument("--max-area-ratio", type=float, default=0.97)
    parser.add_argument("--grabcut-iterations", type=int, default=3)
    parser.add_argument("--no-grabcut", action="store_true", help="use only aligned depth and connectivity")
    parser.add_argument("--polygon-epsilon", type=float, default=0.005, help="contour simplification ratio")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--ignore-split-manifest", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="pilot run on the first N images")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be in (0, 1)")
    if not 0.0 < args.min_area_ratio < args.max_area_ratio <= 1.0:
        parser.error("require 0 < --min-area-ratio < --max-area-ratio <= 1")
    try:
        generate(args)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
