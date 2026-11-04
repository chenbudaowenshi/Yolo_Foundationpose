"""Observation provenance and output validation, independent of target selection."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.spatial import cKDTree


class VisionStatePublisher:
    """Atomically publish snapshots; lifecycle transitions bypass rate limiting."""

    def __init__(self, path: Path, rate_hz: float) -> None:
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("vision output rate must be positive and finite")
        self.path = path.expanduser().resolve()
        self.period_s = 1.0 / rate_hz
        self.last_publish_s = float("-inf")
        self.session_id = str(uuid4())
        self.last_state = {}

    def publish(self, state: dict, now_s: float, *, force=False) -> None:
        if not force and now_s - self.last_publish_s < self.period_s:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{self.session_id}.tmp")
        snapshot = dict(state, session_id=self.session_id)
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(snapshot, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write("\n")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_state = snapshot
        self.last_publish_s = now_s

    def invalidate(self, status: str, *, reason: str = "") -> None:
        state = dict(self.last_state)
        state.update(schema_version=2, wall_time=datetime.now().isoformat(timespec="milliseconds"),
                     status=status, processing_state=status, pose_valid=False,
                     observation_state="unavailable", observation_stamp_ns=None,
                     observation_sequence=None, observation_age_s=None,
                     timestamp_semantics="unavailable", pose_age_s=None,
                     registration_age_s=None, selected_track_id=None, selected_target=None,
                     pose_camera_object=None, pose_quality=None, registration_busy=False,
                     imu_enabled=True, imu_required=True, imu_fresh=False, ros2_imu_fresh=False,
                     imu_alignment={"required": True, "ready": False, "reason": status},
                     tracks=[], exit_reason=reason)
        self.publish(state, time.perf_counter(), force=True)


class ObservationHistory:
    """A selected target's last accepted RGB-D observation, never its write time."""

    def __init__(self):
        self.track_id = None
        self.stamp_ns = None
        self.sequence = None
        self.timestamp_semantics = "unavailable"
        self.registered_s = None

    def select(self, track_id):
        if track_id != self.track_id:
            self.track_id = track_id
            self.stamp_ns = self.sequence = self.registered_s = None
            self.timestamp_semantics = "unavailable"

    def observe(self, *, stamp_ns, sequence, semantics):
        if type(stamp_ns) is not int or stamp_ns <= 0 or self.track_id is None:
            return
        self.stamp_ns = stamp_ns
        self.sequence = sequence
        self.timestamp_semantics = semantics

    def metadata(self, now_ns, now_s):
        age = None if self.stamp_ns is None else (now_ns - self.stamp_ns) / 1e9
        return {
            "observation_stamp_ns": self.stamp_ns,
            "observation_sequence": self.sequence,
            "observation_age_s": age,
            # v2: backward-compatible spelling now explicitly means observation age.
            "pose_age_s": age,
            "pose_age_semantics": "last_accepted_rgbd_observation",
            "registration_age_s": None if self.registered_s is None else max(0., now_s - self.registered_s),
            "timestamp_semantics": self.timestamp_semantics,
        }


def output_status(processing_state, *, has_pose, measured, observation_age_s,
                  quality_valid, maximum_age_s, imu_ready):
    """Output-only state: keeping a prediction does not make it a measurement."""
    if not imu_ready:
        return "waiting_for_imu", "unavailable", False
    if not has_pose:
        return processing_state, "unavailable", False
    if not measured:
        return "pose_predicted", "predicted", False
    if observation_age_s is None or not 0 <= observation_age_s <= maximum_age_s:
        return "pose_stale", "stale", False
    if not quality_valid:
        return "pose_degraded", "measured", False
    return "pose_valid", "measured", True


class DepthSurfaceValidator:
    """Compare raw masked depth with a sampled CAD surface in model coordinates.

    No pose correction or target-selection feedback. This is a geometry
    consistency gate, not an estimate of absolute localization accuracy.
    """

    def __init__(self, vertices, faces, *, tolerance_m=.015, minimum_support=.6,
                 minimum_depth_ratio=.2, maximum_points=1024):
        vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        rng = np.random.default_rng(0)  # Do not change inference's random state.
        sampled = triangles[rng.choice(len(triangles), 12000, p=area / area.sum())]
        uv = rng.random((len(sampled), 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        points = sampled[:, 0] + uv[:, :1] * (sampled[:, 1] - sampled[:, 0]) + uv[:, 1:] * (sampled[:, 2] - sampled[:, 0])
        self.tree = cKDTree(np.concatenate((vertices, points)))
        self.tolerance_m = tolerance_m
        self.minimum_support = minimum_support
        self.minimum_depth_ratio = minimum_depth_ratio
        self.maximum_points = maximum_points

    def assess(self, K, depth_m, mask, pose):
        mask, depth = np.asarray(mask, bool), np.asarray(depth_m)
        valid = mask & np.isfinite(depth) & (depth > 0)
        count = int(valid.sum())
        ratio = count / max(1, int(mask.sum()))
        result = {"valid": False, "reason": "insufficient raw depth", "valid_depth_ratio": ratio,
                  "sample_count": 0, "support_fraction": None, "median_distance_m": None,
                  "p90_distance_m": None, "tolerance_m": self.tolerance_m,
                  "minimum_support": self.minimum_support,
                  "semantics": "raw_depth_to_cad_surface_consistency"}
        if count < 20 or ratio < self.minimum_depth_ratio:
            return result
        vv, uu = np.nonzero(valid)
        step = max(1, int(np.ceil(count / self.maximum_points)))
        vv, uu = vv[::step], uu[::step]
        pixels = np.column_stack((uu, vv, np.ones(len(uu))))
        xyz = (pixels @ np.linalg.inv(K).T) * depth[vv, uu, None]
        pose = np.asarray(pose, float)
        if not np.isfinite(pose).all():
            return dict(result, reason="nonfinite pose")
        local = (xyz - pose[:3, 3]) @ pose[:3, :3]
        distances = self.tree.query(local)[0]
        support = float(np.mean(distances <= self.tolerance_m))
        accepted = support >= self.minimum_support
        return dict(result, valid=accepted, reason="ok" if accepted else "raw depth disagrees with CAD surface",
                    sample_count=len(distances), support_fraction=support,
                    median_distance_m=float(np.median(distances)),
                    p90_distance_m=float(np.percentile(distances, 90)))
