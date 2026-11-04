#!/usr/bin/env python3
"""Gravity-axis and constrained-pose helpers for upright bottle tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return None
    return value / norm


def axis_tilt_degrees(first: np.ndarray, second: np.ndarray, *, undirected: bool = True) -> float:
    """Smallest angle between two axes; optionally treat +axis and -axis alike."""
    first_unit = normalize_vector(first)
    second_unit = normalize_vector(second)
    if first_unit is None or second_unit is None:
        return float("inf")
    cosine = float(np.dot(first_unit, second_unit))
    if undirected:
        cosine = abs(cosine)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def relative_rotation_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    relative = np.asarray(previous, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        current, dtype=np.float64
    ).reshape(3, 3)
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rotation_with_axis_aligned(
    rotation: np.ndarray,
    target_axis: np.ndarray,
    *,
    object_axis_index: int = 2,
) -> np.ndarray:
    """Minimally rotate an object axis onto a target camera-frame axis.

    The target is treated as an undirected vertical axis. The minimal
    camera-frame rotation preserves as much of the registered heading as an
    accelerometer (which cannot observe yaw about gravity) permits.
    """
    value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(value)):
        raise ValueError("rotation must contain finite values")
    if object_axis_index not in (0, 1, 2):
        raise ValueError("object_axis_index must be 0, 1, or 2")
    current = normalize_vector(value[:, object_axis_index])
    target = normalize_vector(target_axis)
    if current is None or target is None:
        raise ValueError("rotation axis and target axis must be nonzero and finite")
    if float(np.dot(current, target)) < 0.0:
        target = -target

    cosine = float(np.clip(np.dot(current, target), -1.0, 1.0))
    cross = np.cross(current, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-9:
        return value.copy()

    x, y, z = cross / sine
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    camera_rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    aligned = camera_rotation @ value

    # Remove tiny numerical non-orthogonality before returning a pose rotation.
    u, _, vt = np.linalg.svd(aligned)
    aligned = u @ vt
    if np.linalg.det(aligned) < 0.0:
        u[:, -1] *= -1.0
        aligned = u @ vt
    return aligned


def pose_with_center_position(
    pose_original_cad: np.ndarray,
    center_position_camera: np.ndarray,
    inverse_to_origin: np.ndarray,
) -> np.ndarray:
    """Keep CAD rotation and move its centered bounding box to a 3D position."""
    pose = np.asarray(pose_original_cad, dtype=np.float64).reshape(4, 4).copy()
    center = np.asarray(center_position_camera, dtype=np.float64).reshape(3)
    centered_offset_in_original = np.asarray(inverse_to_origin, dtype=np.float64).reshape(4, 4)[:3, 3]
    pose[:3, 3] = center - pose[:3, :3] @ centered_offset_in_original
    return pose


def pose_with_upright_axis(
    pose_original_cad: np.ndarray,
    target_axis_camera: np.ndarray,
    inverse_to_origin: np.ndarray,
) -> np.ndarray:
    """Align CAD Z to vertical while preserving the current 3D box center."""
    pose = np.asarray(pose_original_cad, dtype=np.float64).reshape(4, 4).copy()
    inverse = np.asarray(inverse_to_origin, dtype=np.float64).reshape(4, 4)
    center_position = (pose @ inverse)[:3, 3].copy()
    pose[:3, :3] = rotation_with_axis_aligned(pose[:3, :3], target_axis_camera)
    return pose_with_center_position(pose, center_position, inverse)


@dataclass(frozen=True)
class GravitySnapshot:
    axis_camera: np.ndarray | None
    acceleration_norm_mps2: float | None
    accepted_samples: int
    rejected_samples: int

    def metadata(self) -> dict[str, float | int | list[float] | None]:
        return {
            "axis_camera": (
                None if self.axis_camera is None else [float(value) for value in self.axis_camera]
            ),
            "acceleration_norm_mps2": self.acceleration_norm_mps2,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
        }


class GravityAxisFilter:
    """Exponential smoothing for a camera-frame accelerometer gravity axis."""

    def __init__(
        self,
        *,
        smoothing: float = 0.10,
        minimum_acceleration_mps2: float = 7.0,
        maximum_acceleration_mps2: float = 12.5,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("gravity smoothing must be in (0, 1]")
        if minimum_acceleration_mps2 <= 0 or maximum_acceleration_mps2 <= minimum_acceleration_mps2:
            raise ValueError("invalid accepted accelerometer magnitude range")
        self.smoothing = float(smoothing)
        self.minimum_acceleration_mps2 = float(minimum_acceleration_mps2)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self._axis: np.ndarray | None = None
        self._last_norm: float | None = None
        self._accepted = 0
        self._rejected = 0

    def update(self, acceleration_camera_mps2: np.ndarray | None) -> GravitySnapshot:
        if acceleration_camera_mps2 is None:
            return self.snapshot()
        acceleration = np.asarray(acceleration_camera_mps2, dtype=np.float64).reshape(-1)
        if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            self._rejected += 1
            return self.snapshot()
        norm = float(np.linalg.norm(acceleration))
        self._last_norm = norm
        if not self.minimum_acceleration_mps2 <= norm <= self.maximum_acceleration_mps2:
            self._rejected += 1
            return self.snapshot()
        unit = acceleration / norm
        if self._axis is None:
            self._axis = unit
        else:
            # Accelerometer sign should be stable, but sign alignment makes the
            # filter robust to a driver/profile convention change.
            if float(np.dot(unit, self._axis)) < 0.0:
                unit = -unit
            blended = (1.0 - self.smoothing) * self._axis + self.smoothing * unit
            self._axis = normalize_vector(blended)
        self._accepted += 1
        return self.snapshot()

    def snapshot(self) -> GravitySnapshot:
        return GravitySnapshot(
            axis_camera=None if self._axis is None else self._axis.copy(),
            acceleration_norm_mps2=self._last_norm,
            accepted_samples=self._accepted,
            rejected_samples=self._rejected,
        )
