"""TargetManager: turn a stream of raw object poses into one validated target in the planning frame.

ROS-free on purpose.  The node wrapper feeds it samples and a transform callable; everything below is pure
so the rejection rules can be tested without a running system.

Pipeline per sample (README 3.1 / 3.4):

    ObjectPoseSample -> TargetValidator (freshness, ordering, confidence, covariance, jump)
                     -> transform into the planning frame (callable supplied by the caller)
                     -> low-pass filter (pass-through when tau = 0)
                     -> TargetSnapshot

Locking follows the grasp state machine (README 3.2): before APPROACH the target may keep updating; from
APPROACH on the target is locked, and ``snapshot()`` keeps returning the frozen pose even while new samples
arrive.  New samples are still validated while locked, so the diagnostics keep telling the truth.

A rejected sample never replaces the last good one: the previous value is held and the reason is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from shuangbi_grasp.contracts import (
    DEFAULT_JUMP_ORIENTATION_RAD, DEFAULT_JUMP_POSITION_M, DEFAULT_MAX_AGE_S, DEFAULT_MAX_OBSERVATION_DELAY_S,
    ObjectPoseSample, RejectReason, TargetSnapshot,
)

#: (position, orientation) in the planning frame -> transformed (position, orientation), or None if unavailable.
TransformFn = Callable[[ObjectPoseSample], "tuple[np.ndarray, np.ndarray] | None"]


def quaternion_angle(a, b) -> float:
    """Smallest rotation angle [rad] between two (w, x, y, z) quaternions."""
    dot = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return 2.0 * math.acos(min(1.0, dot))


def bottle_axis_angle(a, b) -> float:
    """Unoriented object-Z angle; yaw and reversed cylinder axes are equivalent."""
    def axis(q):
        q = np.asarray(q, dtype=float)
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)])
    return math.acos(float(np.clip(abs(np.dot(axis(a), axis(b))), 0.0, 1.0)))


def quaternion_slerp(a, b, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two (w, x, y, z) quaternions."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:                                   # nearly parallel: linear is accurate and has no 0/0
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - t) * theta) * a + math.sin(t * theta) * b) / sin_theta


@dataclass
class ValidatorConfig:
    max_age: float = DEFAULT_MAX_AGE_S                     # nothing ARRIVED in this long -> timeout
    max_observation_delay: float = DEFAULT_MAX_OBSERVATION_DELAY_S   # arrived, but describes too old a moment
    min_confidence: float = 0.2
    max_position_sigma: float = 0.05               # metres; 0 disables the covariance gate
    jump_position: float = DEFAULT_JUMP_POSITION_M
    jump_orientation: float = DEFAULT_JUMP_ORIENTATION_RAD
    jump_window: float = 0.5                       # only compare against a previous sample this recent
    jump_confirm_samples: int = 3                  # consecutive samples agreeing on the new pose to accept a jump
    object_id: str | None = None                   # when set, samples for other objects are rejected
    orientation_mode: str = "bottle_axis"          # "quaternion" for asymmetric objects

    def __post_init__(self) -> None:
        if self.orientation_mode not in {"bottle_axis", "quaternion"}:
            raise ValueError("orientation_mode must be bottle_axis or quaternion")
        if self.max_age <= 0.0 or self.jump_window <= 0.0:
            raise ValueError("max_age and jump_window must be positive")
        if self.max_observation_delay <= 0.0:
            raise ValueError("max_observation_delay must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.jump_position <= 0.0 or self.jump_orientation <= 0.0:
            raise ValueError("jump thresholds must be positive")
        if self.jump_confirm_samples < 1:
            raise ValueError("jump_confirm_samples must be at least 1")


class TargetValidator:
    """Stateful checks over a stream of samples; ``check`` returns None when the sample is acceptable."""

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()
        self.last_accepted: ObjectPoseSample | None = None
        self.counts: dict[str, int] = {}
        self._pending: ObjectPoseSample | None = None      # candidate new pose after a jump
        self._pending_count = 0

    def reset(self) -> None:
        self.last_accepted = None
        self._pending = None
        self._pending_count = 0
        self.counts.clear()

    @property
    def pending_jump(self) -> int:
        """How many consecutive samples currently agree on a jumped-to pose (0 = none pending)."""
        return self._pending_count

    def _reject(self, reason: str) -> str:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return reason

    def check(self, sample: ObjectPoseSample, now: float) -> str | None:
        cfg = self.config
        if cfg.object_id is not None and sample.object_id != cfg.object_id:
            return self._reject(RejectReason.WRONG_OBJECT)
        if sample.age(now) > cfg.max_age:
            return self._reject(RejectReason.STALE)
        if sample.valid_until is not None and sample.stamp > sample.valid_until:
            return self._reject(RejectReason.STALE)
        if sample.observation_delay is not None and sample.observation_delay > cfg.max_observation_delay:
            return self._reject(RejectReason.DELAYED)
        previous = self.last_accepted
        if previous is not None:
            # Ported from target_pose_cache.py: a sequence or timestamp that goes backwards means a replayed
            # or duplicated message, never new information.
            if sample.sequence and previous.sequence and sample.sequence <= previous.sequence:
                return self._reject(RejectReason.OUT_OF_ORDER)
            if sample.stamp < previous.stamp:
                return self._reject(RejectReason.OUT_OF_ORDER)
        if sample.confidence < cfg.min_confidence:
            return self._reject(RejectReason.LOW_CONFIDENCE)
        if cfg.max_position_sigma > 0.0 and sample.position_sigma > cfg.max_position_sigma:
            return self._reject(RejectReason.HIGH_COVARIANCE)
        if previous is not None and (sample.received - previous.received) <= cfg.jump_window:
            if sample.frame_id == previous.frame_id:     # a frame change is not a jump, it is a re-reference
                if self._is_jump(sample, previous):
                    # A jump is a question, not a verdict: a single wild sample is noise or a mis-detection and
                    # must not move the target, but an object that really WAS moved keeps reporting the new
                    # pose.  So hold the old value until `jump_confirm_samples` consecutive samples agree with
                    # each other on the new pose; anything else restarts the count.  Without this the manager
                    # either freezes forever after one bad frame, or (with a plain time-out) silently accepts
                    # the jump once the stale previous sample falls outside the window.
                    if self._pending is not None and not self._is_jump(sample, self._pending):
                        self._pending_count += 1
                    else:
                        self._pending, self._pending_count = sample, 1
                    if self._pending_count >= cfg.jump_confirm_samples:
                        self._pending, self._pending_count = None, 0
                        return None                      # confirmed: the object really moved
                    self._pending = sample
                    return self._reject(RejectReason.JUMP)
        self._pending, self._pending_count = None, 0     # a normal sample clears any pending jump
        return None

    def _is_jump(self, sample: ObjectPoseSample, reference: ObjectPoseSample) -> bool:
        moved = float(np.linalg.norm(sample.position - reference.position))
        angle = bottle_axis_angle if self.config.orientation_mode == "bottle_axis" else quaternion_angle
        turned = angle(sample.orientation, reference.orientation)
        return moved > self.config.jump_position or turned > self.config.jump_orientation

    def accept(self, sample: ObjectPoseSample) -> None:
        self.last_accepted = sample


class TargetManager:
    """Validated, frame-corrected, optionally filtered and lockable view of one object's pose."""

    def __init__(self, planning_frame: str, *, config: ValidatorConfig | None = None,
                 transform: TransformFn | None = None, filter_tau: float = 0.0) -> None:
        if not planning_frame:
            raise ValueError("planning_frame must not be empty")
        if filter_tau < 0.0:
            raise ValueError("filter_tau must not be negative")
        self.planning_frame = planning_frame
        self.validator = TargetValidator(config)
        self.transform = transform
        self.filter_tau = float(filter_tau)
        self._snapshot: TargetSnapshot | None = None
        self._locked: TargetSnapshot | None = None
        self._filtered_position: np.ndarray | None = None
        self._filtered_orientation: np.ndarray | None = None
        self._last_filter_time: float | None = None
        self.accepted = 0
        self.rejected = 0
        self.last_reject: str | None = None

    # ---- state -----------------------------------------------------------------------------------
    @property
    def locked(self) -> bool:
        return self._locked is not None

    def lock(self) -> TargetSnapshot | None:
        """Freeze the current target; returns the frozen snapshot (None when there is nothing to freeze)."""
        if self._snapshot is not None and self._locked is None:
            self._locked = TargetSnapshot(**{**self._snapshot.__dict__, "locked": True})
        return self._locked

    def unlock(self) -> None:
        self._locked = None

    def reset(self) -> None:
        self.validator.reset()
        self._snapshot = self._locked = None
        self._filtered_position = self._filtered_orientation = self._last_filter_time = None
        self.accepted = self.rejected = 0
        self.last_reject = None

    def snapshot(self, now: float | None = None) -> TargetSnapshot | None:
        """The pose the pipeline should use: the locked one if locked, else the latest valid one."""
        if self._locked is not None:
            return self._locked
        if self._snapshot is None:
            return None
        if now is not None and self._snapshot.age(now) > self.validator.config.max_age:
            return None                              # held value has itself gone stale: report "no target"
        return self._snapshot

    def timed_out(self, now: float) -> bool:
        """True when there is no usable target because nothing fresh has arrived (locked targets never expire)."""
        if self._locked is not None:
            return False
        return self._snapshot is None or self._snapshot.age(now) > self.validator.config.max_age

    # ---- ingestion -------------------------------------------------------------------------------
    def update(self, sample: ObjectPoseSample, now: float | None = None) -> str | None:
        """Feed one sample.  Returns None when accepted, otherwise the RejectReason.

        A rejected sample never replaces the held value; the caller decides what to do with the reason.
        """
        now = sample.received if now is None else float(now)
        reason = self.validator.check(sample, now)
        if reason is not None:
            self.rejected += 1
            self.last_reject = reason
            return reason
        placed = sample
        if sample.frame_id != self.planning_frame:
            if self.transform is None:
                self.rejected += 1
                self.last_reject = RejectReason.NO_TRANSFORM
                return RejectReason.NO_TRANSFORM
            result = self.transform(sample)
            if result is None:
                self.rejected += 1
                self.last_reject = RejectReason.NO_TRANSFORM
                return RejectReason.NO_TRANSFORM
            placed = sample.with_pose(result[0], result[1], self.planning_frame)
        position, orientation, filtered = self._filter(placed, now)
        self.validator.accept(sample)                # jump detection compares RAW samples, before any filtering
        self.accepted += 1
        self.last_reject = None
        self._snapshot = TargetSnapshot(
            object_id=placed.object_id, frame_id=self.planning_frame, position=position, orientation=orientation,
            stamp=placed.stamp, received=placed.received, confidence=placed.confidence, source=placed.source,
            sequence=placed.sequence, locked=False, filtered=filtered)
        return None

    def _filter(self, sample: ObjectPoseSample, now: float):
        """First-order low pass; tau = 0 is a pass-through.  An EKF would replace this method."""
        if self.filter_tau <= 0.0 or self._filtered_position is None:
            self._filtered_position = np.asarray(sample.position, float).copy()
            self._filtered_orientation = np.asarray(sample.orientation, float).copy()
            self._last_filter_time = now
            return self._filtered_position, self._filtered_orientation, False
        dt = max(now - float(self._last_filter_time), 0.0)
        alpha = 1.0 - math.exp(-dt / self.filter_tau) if dt > 0.0 else 0.0
        self._filtered_position = (1.0 - alpha) * self._filtered_position + alpha * np.asarray(sample.position, float)
        self._filtered_orientation = quaternion_slerp(self._filtered_orientation, sample.orientation, alpha)
        self._last_filter_time = now
        return self._filtered_position.copy(), self._filtered_orientation.copy(), True

    # ---- diagnostics -----------------------------------------------------------------------------
    def diagnostics(self, now: float) -> dict:
        snapshot = self.snapshot(now)
        return {
            "planning_frame": self.planning_frame,
            "has_target": snapshot is not None,
            "timed_out": self.timed_out(now),
            "locked": self.locked,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "last_reject": self.last_reject,
            "reject_counts": dict(self.validator.counts),
            "pending_jump": self.validator.pending_jump,
            "age": None if snapshot is None else round(snapshot.age(now), 3),
            "filter_tau": self.filter_tau,
            "target": None if snapshot is None else snapshot.as_dict(),
        }


__all__ = ["TargetManager", "TargetValidator", "TransformFn", "ValidatorConfig",
           "quaternion_angle", "quaternion_slerp", "bottle_axis_angle"]
