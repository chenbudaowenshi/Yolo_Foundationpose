#!/usr/bin/env python3
"""Create an object mask from an aligned depth map and a 2D detection box."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthMaskResult:
    """Result returned by :func:`depth_box_mask`."""

    mask: np.ndarray
    box: tuple[int, int, int, int]
    reference_depth_m: float | None
    tolerance_m: float | None
    area_pixels: int
    area_ratio: float
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.reason is None and self.area_pixels > 0


def clip_box(box_xyxy: np.ndarray | tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clip an ``xyxy`` box to an image and keep at least one pixel per side."""
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    x1_i = max(0, min(width - 1, int(np.floor(x1))))
    y1_i = max(0, min(height - 1, int(np.floor(y1))))
    x2_i = max(x1_i + 1, min(width, int(np.ceil(x2))))
    y2_i = max(y1_i + 1, min(height, int(np.ceil(y2))))
    return x1_i, y1_i, x2_i, y2_i


def yolo_box_to_xyxy(box_xywhn: list[float] | tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a normalized YOLO ``xywh`` box to pixel ``xyxy`` coordinates."""
    cx, cy, box_width, box_height = box_xywhn
    return (
        (cx - box_width / 2.0) * width,
        (cy - box_height / 2.0) * height,
        (cx + box_width / 2.0) * width,
        (cy + box_height / 2.0) * height,
    )


def _component_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep the connected component best supported by the center seed."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(candidate, dtype=bool)

    image_center = np.array([candidate.shape[1] / 2.0, candidate.shape[0] / 2.0])
    choices: list[tuple[tuple[float, ...], int]] = []
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(seed & (labels == label_id)))
        distance = float(np.linalg.norm(centroids[label_id] - image_center))
        choices.append(((float(overlap > 0), float(overlap), float(area), -distance), label_id))
    selected = max(choices)[1]
    return labels == selected


def _fill_external_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=bool)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _empty_result(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    reason: str,
    reference_depth_m: float | None = None,
    tolerance_m: float | None = None,
) -> DepthMaskResult:
    return DepthMaskResult(
        mask=np.zeros(shape, dtype=bool),
        box=box,
        reference_depth_m=reference_depth_m,
        tolerance_m=tolerance_m,
        area_pixels=0,
        area_ratio=0.0,
        reason=reason,
    )


def depth_box_mask(
    depth_raw: np.ndarray,
    box_xyxy: np.ndarray | tuple[float, float, float, float],
    *,
    depth_scale: float,
    color_bgr: np.ndarray | None = None,
    seed_ratio: float = 0.35,
    depth_tolerance_m: float = 0.045,
    max_tolerance_m: float = 0.12,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    min_seed_pixels: int = 20,
    min_mask_pixels: int = 150,
    min_area_ratio: float = 0.04,
    max_area_ratio: float = 0.97,
    grabcut_iterations: int = 3,
) -> DepthMaskResult:
    """Estimate a full-resolution instance mask inside a detection box.

    The center of the detection box provides a robust depth seed. Pixels near
    that depth are grouped by connectivity. When RGB is supplied, GrabCut uses
    the depth component as a foreground seed to recover object pixels with
    missing/noisy depth near silhouettes and reflective labels.
    """
    if depth_raw.ndim != 2:
        raise ValueError(f"depth_raw must be HxW, got {depth_raw.shape}")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if not 0.0 < seed_ratio <= 1.0:
        raise ValueError("seed_ratio must be in (0, 1]")

    height, width = depth_raw.shape
    if color_bgr is not None and color_bgr.shape[:2] != (height, width):
        raise ValueError("RGB and depth must already be aligned and have identical dimensions")

    box = clip_box(box_xyxy, width, height)
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    if box_area < min_mask_pixels:
        return _empty_result((height, width), box, "box is too small")

    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    valid_depth = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)

    margin_x = int(round((x2 - x1) * (1.0 - seed_ratio) / 2.0))
    margin_y = int(round((y2 - y1) * (1.0 - seed_ratio) / 2.0))
    sx1, sy1 = x1 + margin_x, y1 + margin_y
    sx2, sy2 = max(sx1 + 1, x2 - margin_x), max(sy1 + 1, y2 - margin_y)
    seed_values = depth_m[sy1:sy2, sx1:sx2]
    seed_valid = valid_depth[sy1:sy2, sx1:sx2]
    values = seed_values[seed_valid]

    if values.size < min_seed_pixels:
        # A reflective label can remove most center depth. Expand once, but do
        # not silently use the whole frame or an unrelated nearby object.
        ex_margin_x = int(round((x2 - x1) * 0.15))
        ex_margin_y = int(round((y2 - y1) * 0.15))
        sx1, sy1 = x1 + ex_margin_x, y1 + ex_margin_y
        sx2, sy2 = max(sx1 + 1, x2 - ex_margin_x), max(sy1 + 1, y2 - ex_margin_y)
        seed_values = depth_m[sy1:sy2, sx1:sx2]
        seed_valid = valid_depth[sy1:sy2, sx1:sx2]
        values = seed_values[seed_valid]
    if values.size < min_seed_pixels:
        return _empty_result((height, width), box, f"only {values.size} valid seed depth pixels")

    q1, q3 = np.percentile(values, [25.0, 75.0])
    trimmed = values[(values >= q1) & (values <= q3)]
    if trimmed.size < min_seed_pixels:
        trimmed = values
    reference_depth = float(np.median(trimmed))
    mad = float(np.median(np.abs(trimmed - reference_depth)))
    adaptive_tolerance = max(float(depth_tolerance_m), 3.0 * 1.4826 * mad)
    tolerance = min(float(max_tolerance_m), adaptive_tolerance)

    in_box = np.zeros((height, width), dtype=bool)
    in_box[y1:y2, x1:x2] = True
    seed = np.zeros((height, width), dtype=bool)
    seed[sy1:sy2, sx1:sx2] = True
    depth_candidate = in_box & valid_depth & (np.abs(depth_m - reference_depth) <= tolerance)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_u8 = cv2.morphologyEx(depth_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel3)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel5, iterations=2)
    component = _component_touching_seed(candidate_u8.astype(bool), seed)
    component = _fill_external_contour(component)
    if np.count_nonzero(component) < min_mask_pixels:
        return _empty_result(
            (height, width),
            box,
            "depth component is too small",
            reference_depth,
            tolerance,
        )

    final_mask = component
    if color_bgr is not None and grabcut_iterations > 0:
        gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
        gc_mask[component] = cv2.GC_PR_FGD
        sure_foreground = cv2.erode(component.astype(np.uint8), kernel3, iterations=2).astype(bool)
        if np.count_nonzero(sure_foreground) >= min_seed_pixels:
            gc_mask[sure_foreground] = cv2.GC_FGD
            background_model = np.zeros((1, 65), dtype=np.float64)
            foreground_model = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(
                    color_bgr,
                    gc_mask,
                    None,
                    background_model,
                    foreground_model,
                    int(grabcut_iterations),
                    cv2.GC_INIT_WITH_MASK,
                )
                refined = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)) & in_box
                refined = _component_touching_seed(refined, sure_foreground)
                refined = _fill_external_contour(refined)
                refined_area = int(np.count_nonzero(refined))
                refined_ratio = refined_area / float(box_area)
                if min_mask_pixels <= refined_area and min_area_ratio <= refined_ratio <= max_area_ratio:
                    final_mask = refined
            except cv2.error:
                # The depth-only component remains a deterministic fallback.
                pass

    final_mask = _fill_external_contour(final_mask & in_box)
    area_pixels = int(np.count_nonzero(final_mask))
    area_ratio = area_pixels / float(box_area)
    if area_pixels < min_mask_pixels:
        return _empty_result((height, width), box, "final mask is too small", reference_depth, tolerance)
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        return _empty_result(
            (height, width),
            box,
            f"mask/box area ratio {area_ratio:.3f} is outside [{min_area_ratio:.3f}, {max_area_ratio:.3f}]",
            reference_depth,
            tolerance,
        )

    return DepthMaskResult(
        mask=final_mask,
        box=box,
        reference_depth_m=reference_depth,
        tolerance_m=tolerance,
        area_pixels=area_pixels,
        area_ratio=area_ratio,
    )


def mask_to_yolo_polygon(mask: np.ndarray, *, epsilon_ratio: float = 0.005) -> list[float] | None:
    """Convert the largest external contour into normalized YOLO-seg points."""
    if mask.ndim != 2 or not np.any(mask):
        return None
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    approximated = cv2.approxPolyDP(contour, max(0.5, perimeter * epsilon_ratio), closed=True).reshape(-1, 2)
    if len(approximated) < 3:
        return None
    normalized = approximated.astype(np.float64)
    normalized[:, 0] = np.clip(normalized[:, 0] / float(width), 0.0, 1.0)
    normalized[:, 1] = np.clip(normalized[:, 1] / float(height), 0.0, 1.0)
    return normalized.reshape(-1).tolist()


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask and its contour on a BGR image."""
    output = image_bgr.copy()
    color_array = np.asarray(color, dtype=np.float32)
    output[mask] = np.clip(
        output[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha,
        0,
        255,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, thickness=2)
    return output
