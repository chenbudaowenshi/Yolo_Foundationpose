#!/usr/bin/env python3
"""Persistent multi-instance RGB-D tracks and nearest-target locking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from rgbd_kalman_tracker import ConstantVelocity3DTracker, TrackerConfig, TrackerSnapshot


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(0.0, upper - lower)))
    first_area = float(np.prod(np.maximum(0.0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0.0, second[2:] - second[:2])))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def normalized_box_motion(current: np.ndarray, previous: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float64).reshape(4)
    previous = np.asarray(previous, dtype=np.float64).reshape(4)
    current_size = np.maximum(1.0, current[2:] - current[:2])
    previous_size = np.maximum(1.0, previous[2:] - previous[:2])
    current_center = (current[:2] + current[2:]) * 0.5
    previous_center = (previous[:2] + previous[2:]) * 0.5
    translation = float(
        np.linalg.norm(current_center - previous_center)
        / max(np.linalg.norm(previous_size), 1.0)
    )
    scale = float(np.max(np.abs(current_size - previous_size) / previous_size))
    return max(translation, scale)


@dataclass(frozen=True)
class Detection3D:
    box: np.ndarray
    confidence: float
    position_m: np.ndarray | None
    covariance: np.ndarray | None
    payload: Any = None


@dataclass(frozen=True)
class TrackView:
    track_id: int
    box: np.ndarray
    confidence: float
    tracker: TrackerSnapshot
    payload: Any
    detection_hits: int
    detection_misses: int
    stable_box_frames: int
    confirmed: bool
    distance_m: float | None

    @property
    def observed_this_frame(self) -> bool:
        return self.payload is not None and self.detection_misses == 0

    def metadata(self, *, selected: bool = False) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "selected": selected,
            "confirmed": self.confirmed,
            "observed_this_frame": self.observed_this_frame,
            "box_xyxy": [float(value) for value in self.box],
            "confidence": self.confidence,
            "distance_camera_m": self.distance_m,
            "detection_hits": self.detection_hits,
            "detection_misses": self.detection_misses,
            "stable_box_frames": self.stable_box_frames,
            "tracker": self.tracker.metadata(),
        }


@dataclass
class _Track:
    track_id: int
    tracker: ConstantVelocity3DTracker
    box: np.ndarray
    confidence: float
    payload: Any
    detection_hits: int
    detection_misses: int
    stable_box_frames: int
    last_timestamp_s: float

    def view(self, confirmation_hits: int) -> TrackView:
        snapshot = self.tracker.snapshot()
        distance = None
        if snapshot.position_m is not None and snapshot.status != "lost":
            distance = float(np.linalg.norm(snapshot.position_m))
        return TrackView(
            track_id=self.track_id,
            box=self.box.copy(),
            confidence=self.confidence,
            tracker=snapshot,
            payload=self.payload,
            detection_hits=self.detection_hits,
            detection_misses=self.detection_misses,
            stable_box_frames=self.stable_box_frames,
            confirmed=self.detection_hits >= confirmation_hits,
            distance_m=distance,
        )


class MultiObjectRGBDTracker:
    """Associate same-class detections and maintain one KF3D per instance."""

    def __init__(
        self,
        tracker_config: TrackerConfig,
        *,
        confirmation_hits: int = 3,
        maximum_detection_misses: int = 30,
        maximum_3d_distance_m: float = 0.15,
        minimum_2d_iou: float = 0.02,
        stable_box_motion: float = 0.025,
    ) -> None:
        if confirmation_hits <= 0 or maximum_detection_misses < 0:
            raise ValueError("invalid track confirmation/lifetime settings")
        if maximum_3d_distance_m <= 0 or not 0.0 <= minimum_2d_iou <= 1.0:
            raise ValueError("invalid association thresholds")
        self.tracker_config = tracker_config
        self.confirmation_hits = int(confirmation_hits)
        self.maximum_detection_misses = int(maximum_detection_misses)
        self.maximum_3d_distance_m = float(maximum_3d_distance_m)
        self.minimum_2d_iou = float(minimum_2d_iou)
        self.stable_box_motion = float(stable_box_motion)
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    def _predicted_position(self, track: _Track, timestamp_s: float) -> np.ndarray | None:
        snapshot = track.tracker.snapshot()
        if snapshot.position_m is None or snapshot.status == "lost":
            return None
        dt = float(np.clip(timestamp_s - track.last_timestamp_s, 0.0, 0.25))
        velocity = np.zeros(3) if snapshot.velocity_mps is None else snapshot.velocity_mps
        return snapshot.position_m + velocity * dt

    def _association_cost(
        self,
        track: _Track,
        detection: Detection3D,
        timestamp_s: float,
    ) -> float | None:
        iou = box_iou(track.box, detection.box)
        predicted = self._predicted_position(track, timestamp_s)
        if predicted is not None and detection.position_m is not None:
            distance = float(np.linalg.norm(predicted - detection.position_m))
            if distance > self.maximum_3d_distance_m:
                return None
            if iou < self.minimum_2d_iou and distance > self.maximum_3d_distance_m * 0.45:
                return None
            return distance / self.maximum_3d_distance_m + 0.65 * (1.0 - iou)
        if iou < max(0.10, self.minimum_2d_iou):
            return None
        return 1.25 * (1.0 - iou)

    def update(self, detections: list[Detection3D], timestamp_s: float) -> list[TrackView]:
        if not np.isfinite(timestamp_s):
            raise ValueError("timestamp must be finite")
        track_ids = list(self._tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if track_ids and detections:
            large = 1e6
            costs = np.full((len(track_ids), len(detections)), large, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                for column, detection in enumerate(detections):
                    cost = self._association_cost(track, detection, timestamp_s)
                    if cost is not None:
                        costs[row, column] = cost
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] >= large:
                    continue
                track_id = track_ids[int(row)]
                detection_index = int(column)
                track = self._tracks[track_id]
                detection = detections[detection_index]
                motion = normalized_box_motion(detection.box, track.box)
                track.stable_box_frames = (
                    track.stable_box_frames + 1 if motion <= self.stable_box_motion else 1
                )
                track.box = np.asarray(detection.box, dtype=np.float64).copy()
                track.confidence = float(detection.confidence)
                track.payload = detection.payload
                track.detection_hits += 1
                track.detection_misses = 0
                track.last_timestamp_s = float(timestamp_s)
                track.tracker.step(
                    detection.position_m,
                    timestamp_s,
                    detection.covariance,
                )
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

        for track_id in track_ids:
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.payload = None
            track.detection_misses += 1
            track.stable_box_frames = 0
            track.last_timestamp_s = float(timestamp_s)
            track.tracker.step(None, timestamp_s)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            tracker = ConstantVelocity3DTracker(self.tracker_config)
            tracker.step(detection.position_m, timestamp_s, detection.covariance)
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                tracker=tracker,
                box=np.asarray(detection.box, dtype=np.float64).copy(),
                confidence=float(detection.confidence),
                payload=detection.payload,
                detection_hits=1,
                detection_misses=0,
                stable_box_frames=1,
                last_timestamp_s=float(timestamp_s),
            )

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.detection_misses > self.maximum_detection_misses
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return [
            self._tracks[track_id].view(self.confirmation_hits)
            for track_id in sorted(self._tracks)
        ]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1


class NearestTargetSelector:
    """Lock one confirmed target; switch only after it expires or is excluded."""

    def __init__(self, *, lost_grace_frames: int = 15, distance_mode: str = "euclidean") -> None:
        if lost_grace_frames < 0:
            raise ValueError("lost_grace_frames must be nonnegative")
        if distance_mode not in ("euclidean", "z"):
            raise ValueError("distance_mode must be euclidean or z")
        self.lost_grace_frames = int(lost_grace_frames)
        self.distance_mode = distance_mode
        self.selected_track_id: int | None = None

    def _distance(self, view: TrackView) -> float:
        assert view.tracker.position_m is not None
        if self.distance_mode == "z":
            return float(view.tracker.position_m[2])
        return float(np.linalg.norm(view.tracker.position_m))

    def select(
        self,
        views: list[TrackView],
        *,
        excluded_track_ids: set[int] | None = None,
    ) -> TrackView | None:
        excluded = excluded_track_ids or set()
        by_id = {view.track_id: view for view in views}
        current = None if self.selected_track_id is None else by_id.get(self.selected_track_id)
        if (
            current is not None
            and current.track_id not in excluded
            and current.detection_misses <= self.lost_grace_frames
        ):
            return current

        candidates = [
            view
            for view in views
            if view.track_id not in excluded
            and view.confirmed
            and view.tracker.position_m is not None
            and view.tracker.status != "lost"
            and view.observed_this_frame
        ]
        if not candidates:
            self.selected_track_id = None
            return None
        selected = min(candidates, key=self._distance)
        self.selected_track_id = selected.track_id
        return selected

    def release(self) -> None:
        self.selected_track_id = None
