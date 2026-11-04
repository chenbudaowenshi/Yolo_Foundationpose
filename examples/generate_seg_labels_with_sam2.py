#!/usr/bin/env python3
"""Generate reviewable YOLO segmentation labels from box labels with SAM2.

This script never edits the source dataset.  It uses each existing YOLO
detection box as a SAM2 prompt, writes polygon labels to a separate dataset,
and creates an overlay plus a CSV quality report for manual review.

Pilot run (recommended before generating all labels):
    conda activate yolov8
    python examples/generate_seg_labels_with_sam2.py \
        --output-root datasets/custom_segmentation_sam2_pilot --limit 30

Full run, only after reviewing the pilot previews:
    python examples/generate_seg_labels_with_sam2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import yaml

from depth_mask_utils import clip_box, mask_to_yolo_polygon, overlay_mask, yolo_box_to_xyxy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "configs/datasets"
DEFAULT_OUTPUT = REPO_ROOT / "datasets/custom_segmentation_sam2"
DEFAULT_MANIFEST = REPO_ROOT / "datasets/custom_detection/split_manifest.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "models/sam2/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = IMAGE_SUFFIXES | {".npy"}
COLORS = ((70, 190, 255), (80, 220, 80), (255, 150, 60), (220, 80, 220))


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
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


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class list not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found: {path}")
    return names


def load_detection_labels(path: Path | None, class_count: int) -> list[tuple[int, list[float]]]:
    if path is None:
        return []
    objects: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 detection fields, got {len(fields)}")
        try:
            values = [float(value) for value in fields]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric value") from error
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class ID {values[0]}")
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"{path}:{line_number}: box coordinates must be normalized to [0, 1]")
        if values[3] <= 0.0 or values[4] <= 0.0:
            raise ValueError(f"{path}:{line_number}: box width and height must be positive")
        objects.append((class_id, values[1:]))
    return objects


def canonical_frame_key(stem: str) -> tuple[str, int] | None:
    """Treat legacy ``frame_0000081`` and corrected ``frame_000081`` as frame 81."""
    match = re.fullmatch(r"(.*?)(\d+)", stem)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def split_from_manifest(stems: list[str], path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train = set(manifest.get("train", []))
    val = set(manifest.get("val", []))

    # A repaired RGB/depth collection can contain corrected filenames while an
    # older manifest and copied detection split still use the old spelling.
    # Also include the actual split directories, which cover files copied into
    # the detection set after its manifest was first written.
    for split, destination in (("train", train), ("val", val)):
        image_split = path.parent / "images" / split
        if image_split.is_dir():
            destination.update(file.stem for file in image_split.iterdir() if file.is_file())
    overlap = train & val
    if overlap:
        raise ValueError(f"Split manifest has {len(overlap)} train/val overlaps")

    canonical_splits: dict[tuple[str, int], set[str]] = {}
    for split, split_stems in (("train", train), ("val", val)):
        for split_stem in split_stems:
            key = canonical_frame_key(split_stem)
            if key is not None:
                canonical_splits.setdefault(key, set()).add(split)

    resolved: dict[str, str] = {}
    corrected_names = 0
    missing: list[str] = []
    for stem in stems:
        if stem in train:
            resolved[stem] = "train"
            continue
        if stem in val:
            resolved[stem] = "val"
            continue
        choices = canonical_splits.get(canonical_frame_key(stem), set())
        if len(choices) == 1:
            resolved[stem] = next(iter(choices))
            corrected_names += 1
        elif len(choices) > 1:
            raise ValueError(f"Corrected stem '{stem}' maps to both train and val in the old split")
        else:
            missing.append(stem)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Split manifest is missing {len(missing)} image stems, e.g. {sample}")
    if corrected_names:
        print(f"Resolved {corrected_names} corrected filenames against the legacy detection split")
    return resolved


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep and fill the largest external SAM component for a YOLO polygon."""
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(binary.shape, dtype=bool)
    contour = max(contours, key=cv2.contourArea)
    cleaned = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(cleaned, [contour], -1, 1, thickness=cv2.FILLED)
    return cleaned.astype(bool)


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int]) -> float:
    if first is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def mask_metrics(mask: np.ndarray, prompt_box: tuple[int, int, int, int]) -> dict[str, float | int]:
    x1, y1, x2, y2 = prompt_box
    mask_pixels = int(np.count_nonzero(mask))
    box_area = max(1, (x2 - x1) * (y2 - y1))
    inside_pixels = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    return {
        "mask_pixels": mask_pixels,
        "mask_box_ratio": mask_pixels / float(box_area),
        "outside_box_ratio": (mask_pixels - inside_pixels) / float(max(1, mask_pixels)),
        "mask_bbox_iou": box_iou(mask_bounds(mask), prompt_box),
    }


def choose_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    prompt_box: tuple[int, int, int, int],
    bbox_iou_weight: float,
) -> tuple[int, np.ndarray, dict[str, float | int], list[dict[str, float | int]]]:
    candidate_rows: list[dict[str, float | int]] = []
    cleaned_masks: list[np.ndarray] = []
    for candidate_index, (mask, score) in enumerate(zip(masks, scores)):
        cleaned = clean_mask(mask)
        metrics = mask_metrics(cleaned, prompt_box)
        rank = float(score) + bbox_iou_weight * float(metrics["mask_bbox_iou"])
        candidate_rows.append(
            {"candidate": candidate_index, "sam_score": float(score), "rank": rank, **metrics}
        )
        cleaned_masks.append(cleaned)
    selected = max(range(len(candidate_rows)), key=lambda index: float(candidate_rows[index]["rank"]))
    return selected, cleaned_masks[selected], candidate_rows[selected], candidate_rows


def review_reasons(metrics: dict[str, float | int], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if float(metrics["sam_score"]) < args.review_score:
        reasons.append(f"SAM score<{args.review_score:.2f}")
    if float(metrics["mask_bbox_iou"]) < args.review_bbox_iou:
        reasons.append(f"mask-box IoU<{args.review_bbox_iou:.2f}")
    if float(metrics["mask_box_ratio"]) < args.review_min_area_ratio:
        reasons.append(f"mask/box area<{args.review_min_area_ratio:.2f}")
    if float(metrics["mask_box_ratio"]) > args.review_max_area_ratio:
        reasons.append(f"mask/box area>{args.review_max_area_ratio:.2f}")
    if float(metrics["outside_box_ratio"]) > args.review_outside_ratio:
        reasons.append(f"outside box>{args.review_outside_ratio:.2f}")
    return reasons


def format_polygon(class_id: int, polygon: list[float]) -> str:
    return str(class_id) + " " + " ".join(f"{coordinate:.6f}" for coordinate in polygon)


def write_dataset_metadata(
    output_root: Path,
    names: list[str],
    source_root: Path,
    manifest: Path,
    processed_stems: list[str],
    args: argparse.Namespace,
) -> None:
    data = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": None,
        "names": dict(enumerate(names)),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    metadata = {
        "generator": "SAM2 prompted by existing ground-truth detection boxes",
        "source_root": str(source_root),
        "source_split_manifest": str(manifest),
        "sam2_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "sam2_config": args.model_config,
        "processed_stems": processed_stems,
        "manual_review_required_before_training": True,
    }
    (output_root / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def draw_preview(
    image: np.ndarray,
    instances: list[tuple[int, np.ndarray, tuple[int, int, int, int], dict[str, float | int], list[str]]],
    names: list[str],
) -> np.ndarray:
    preview = image.copy()
    for class_id, mask, box, metrics, reasons in instances:
        color = COLORS[class_id % len(COLORS)]
        preview = overlay_mask(preview, mask, color)
        x1, y1, x2, y2 = box
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        state = "REVIEW" if reasons else "OK"
        text = (
            f"{names[class_id]} {state} sam={float(metrics['sam_score']):.3f} "
            f"boxIoU={float(metrics['mask_bbox_iou']):.2f}"
        )
        cv2.putText(
            preview,
            text,
            (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def generate(args: argparse.Namespace) -> None:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.split_manifest.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}\n"
            "Choose a new --output-root; this safeguard protects reviewed labels from being overwritten."
        )

    image_dir = source_root / args.image_dir
    label_dir = source_root / args.label_dir
    depth_dir = source_root / args.depth_dir
    images = index_files(image_dir, IMAGE_SUFFIXES)
    labels = index_files(label_dir, {".txt"})
    labels.pop("classes", None)
    depths = index_files(depth_dir, DEPTH_SUFFIXES) if depth_dir.is_dir() else {}
    names = load_classes(label_dir / "classes.txt")
    if not images:
        raise ValueError(f"No RGB images found: {image_dir}")
    missing_labels = sorted(set(images) - set(labels))
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} RGB images have no detection label")
    if depths:
        missing_depth = sorted(set(images) - set(depths))
        extra_depth = sorted(set(depths) - set(images))
        if missing_depth or extra_depth:
            raise ValueError(
                f"RGB/depth stems must match exactly; missing={len(missing_depth)}, extra={len(extra_depth)}"
            )

    all_stems = sorted(images)
    split_for_all = split_from_manifest(all_stems, manifest)
    stems = all_stems
    if args.stems:
        requested = list(dict.fromkeys(Path(value).stem for value in args.stems))
        unknown = [stem for stem in requested if stem not in images]
        if unknown:
            raise ValueError(f"Unknown --stems: {', '.join(unknown)}")
        stems = requested
        print(f"Selected pilot mode: processing {len(stems)} requested images")
    elif args.limit is not None:
        stems = stems[: args.limit]
        print(f"Pilot mode: processing the first {len(stems)} of {len(images)} images")

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise RuntimeError("SAM2 is not installed in this Python environment") from error

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    print(f"Loading SAM2: config={args.model_config}, checkpoint={checkpoint}, device={args.device}")
    model = build_sam2(args.model_config, str(checkpoint), device=args.device, apply_postprocessing=True)
    predictor = SAM2ImagePredictor(model)
    output_root.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(output_root, names, source_root, manifest, stems, args)

    rows: list[dict[str, object]] = []
    image_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    review_images: set[str] = set()
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        for image_index, stem in enumerate(stems, start=1):
            image_path = images[stem]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            height, width = image_bgr.shape[:2]
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            objects = load_detection_labels(labels[stem], len(names))
            lines: list[str] = []
            preview_instances: list[
                tuple[int, np.ndarray, tuple[int, int, int, int], dict[str, float | int], list[str]]
            ] = []
            for object_index, (class_id, box_xywhn) in enumerate(objects):
                box_float = yolo_box_to_xyxy(box_xywhn, width, height)
                box = clip_box(box_float, width, height)
                prompt = np.asarray(box, dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompt,
                    multimask_output=not args.single_mask,
                )
                selected, mask, metrics, candidates = choose_candidate(
                    np.asarray(masks), np.asarray(scores), box, args.bbox_iou_weight
                )
                polygon = mask_to_yolo_polygon(mask, epsilon_ratio=args.polygon_epsilon)
                reasons = review_reasons(metrics, args)
                if polygon is None:
                    reasons.append("no valid polygon")
                else:
                    lines.append(format_polygon(class_id, polygon))
                if reasons:
                    review_images.add(stem)
                class_counts[class_id] += 1
                preview_instances.append((class_id, mask, box, metrics, reasons))
                rows.append(
                    {
                        "stem": stem,
                        "split": split_for_all[stem],
                        "object_index": object_index,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "status": "REVIEW" if reasons else "OK",
                        "reasons": "; ".join(reasons),
                        "selected_candidate": selected,
                        "candidate_count": len(candidates),
                        "sam_score": f"{float(metrics['sam_score']):.6f}",
                        "mask_bbox_iou": f"{float(metrics['mask_bbox_iou']):.6f}",
                        "mask_box_ratio": f"{float(metrics['mask_box_ratio']):.6f}",
                        "outside_box_ratio": f"{float(metrics['outside_box_ratio']):.6f}",
                        "mask_pixels": int(metrics["mask_pixels"]),
                        "candidate_metrics_json": json.dumps(candidates, ensure_ascii=False),
                    }
                )

            # Reject an incomplete label rather than silently teaching a missed
            # object as background.  The CSV still records why it was rejected.
            if len(lines) != len(objects):
                print(f"WARNING: {stem} has an invalid SAM polygon and was not added to the dataset")
                continue
            split = split_for_all[stem]
            for kind in ("images", "labels", "previews"):
                (output_root / kind / split).mkdir(parents=True, exist_ok=True)
            detection_label_dir = output_root / "detection_labels" / split
            detection_label_dir.mkdir(parents=True, exist_ok=True)
            if depths:
                (output_root / "depth" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_root / "images" / split / image_path.name)
            shutil.copy2(labels[stem], detection_label_dir / labels[stem].name)
            if depths:
                shutil.copy2(depths[stem], output_root / "depth" / split / depths[stem].name)
            (output_root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            preview = draw_preview(image_bgr, preview_instances, names)
            if not cv2.imwrite(str(output_root / "previews" / split / f"{stem}.jpg"), preview):
                raise OSError(f"Could not write preview for {stem}")
            image_counts[split] += 1
            if image_index % 10 == 0 or image_index == len(stems):
                print(f"Processed {image_index}/{len(stems)} images")

    fieldnames = [
        "stem",
        "split",
        "object_index",
        "class_id",
        "class_name",
        "status",
        "reasons",
        "selected_candidate",
        "candidate_count",
        "sam_score",
        "mask_bbox_iou",
        "mask_box_ratio",
        "outside_box_ratio",
        "mask_pixels",
        "candidate_metrics_json",
    ]
    with (output_root / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    review_queue = output_root / "review_queue"
    flagged_pairs = sorted({(str(row["split"]), str(row["stem"])) for row in rows if row["status"] == "REVIEW"})
    for split, stem in flagged_pairs:
        review_queue.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_root / "previews" / split / f"{stem}.jpg", review_queue / f"{stem}.jpg")
    summary = {
        "processed_source_images": len(stems),
        "accepted_train_images": image_counts["train"],
        "accepted_val_images": image_counts["val"],
        "images_flagged_for_review": len(review_images),
        "review_stems": sorted(review_images),
        "instances_per_class": {names[index]: class_counts[index] for index in range(len(names))},
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SAM2 dataset written to: {output_root}")
    print(f"train={image_counts['train']}, val={image_counts['val']}, review images={len(review_images)}")
    print(f"Inspect overlays: {output_root / 'previews'}")
    print(f"Quality report: {output_root / 'review.csv'}")
    print("Do not train until the masks have been manually reviewed/corrected.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--label-dir", default="labels")
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="pilot mode: process only the first N sorted images")
    parser.add_argument("--stems", nargs="+", help="pilot mode: process only these image stems")
    parser.add_argument("--single-mask", action="store_true", help="ask SAM2 for one mask instead of three candidates")
    parser.add_argument("--bbox-iou-weight", type=float, default=0.35, help="candidate rank bonus for matching the GT box")
    parser.add_argument("--polygon-epsilon", type=float, default=0.003)
    parser.add_argument("--review-score", type=float, default=0.80)
    parser.add_argument("--review-bbox-iou", type=float, default=0.65)
    parser.add_argument("--review-min-area-ratio", type=float, default=0.20)
    parser.add_argument("--review-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--review-outside-ratio", type=float, default=0.10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.stems:
        parser.error("--limit and --stems are mutually exclusive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.bbox_iou_weight < 0.0 or args.polygon_epsilon < 0.0:
        parser.error("--bbox-iou-weight and --polygon-epsilon must be nonnegative")
    for name in ("review_score", "review_bbox_iou", "review_min_area_ratio", "review_max_area_ratio", "review_outside_ratio"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.review_min_area_ratio >= args.review_max_area_ratio:
        parser.error("--review-min-area-ratio must be less than --review-max-area-ratio")
    try:
        generate(args)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
