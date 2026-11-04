"""Plain data contracts of the grasp pipeline: no ROS, no numpy-in-the-interface surprises.

``ObjectPoseSample`` is what an ObjectPoseProvider produces and what ``TargetManager`` consumes.  It carries
the observation time separately from the arrival time on purpose: a camera pipeline delivers a pose that was
true some tens of milliseconds ago, and both the TF lookup and the freshness check must use the observation
time, while out-of-order detection uses the arrival time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

#: No sample has ARRIVED within this window -> the stream is dead (target timeout).
DEFAULT_MAX_AGE_S = 0.25
#: The OBSERVATION behind a sample may not be older than this when it arrives (pipeline latency).
DEFAULT_MAX_OBSERVATION_DELAY_S = 0.30
#: Position / orientation change between consecutive samples that counts as a jump rather than motion.
DEFAULT_JUMP_POSITION_M = 0.05
DEFAULT_JUMP_ORIENTATION_RAD = math.radians(30.0)


def _readonly(array, shape, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=float).reshape(shape).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ObjectPoseSample:
    """One 6D observation of a task object, in the frame it was observed in."""

    object_id: str
    frame_id: str
    position: np.ndarray                 # (3,) metres
    orientation: np.ndarray              # (4,) quaternion (w, x, y, z), normalised on construction
    stamp: float                         # observation time [s], the sender's clock
    received: float                      # arrival time [s], the receiver's monotonic clock
    confidence: float = 1.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    valid_until: float | None = None     # absolute time on the sender's clock; None = use the freshness window
    source: str = "unknown"
    sequence: int = 0
    #: How old the OBSERVATION was when it arrived [s], measured on one clock by the receiver
    #: (ROS now minus header.stamp).  A camera pipeline keeps publishing at full rate while every frame
    #: describes a moment 100-200 ms in the past, so "the stream is alive" and "the pose is recent" are two
    #: different questions and both have to be asked.  None = the sender gave no usable stamp.
    observation_delay: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly(self.position, (3,), "position"))
        quat = np.asarray(self.orientation, dtype=float).reshape(4)
        norm = float(np.linalg.norm(quat))
        if not norm > 1.0e-9 or not np.all(np.isfinite(quat)):
            raise ValueError("orientation must be a non-zero finite quaternion")
        object.__setattr__(self, "orientation", _readonly(quat / norm, (4,), "orientation"))
        object.__setattr__(self, "covariance", _readonly(self.covariance, (6, 6), "covariance"))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        for name in ("stamp", "received"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not self.object_id:
            raise ValueError("object_id must not be empty")

    def age(self, now: float) -> float:
        """Seconds since arrival, on the receiver's clock."""
        return float(now) - float(self.received)

    @property
    def position_sigma(self) -> float:
        """Largest translational standard deviation [m]; 0 when no covariance was supplied."""
        diag = np.diag(self.covariance)[:3]
        return float(np.sqrt(max(diag.max(), 0.0)))

    def with_pose(self, position, orientation, frame_id: str) -> "ObjectPoseSample":
        return replace(self, position=np.asarray(position, float), orientation=np.asarray(orientation, float),
                       frame_id=frame_id)


@dataclass(frozen=True)
class TargetSnapshot:
    """What the rest of the pipeline is allowed to use: a validated pose in the planning frame."""

    object_id: str
    frame_id: str
    position: np.ndarray
    orientation: np.ndarray
    stamp: float
    received: float
    confidence: float
    source: str
    sequence: int
    locked: bool = False
    filtered: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly(self.position, (3,), "position"))
        object.__setattr__(self, "orientation", _readonly(self.orientation, (4,), "orientation"))

    def age(self, now: float) -> float:
        return float(now) - float(self.received)

    def as_dict(self) -> dict:
        return {"object_id": self.object_id, "frame_id": self.frame_id,
                "position": [round(float(v), 4) for v in self.position],
                "orientation": [round(float(v), 4) for v in self.orientation],
                "confidence": round(float(self.confidence), 3), "source": self.source,
                "sequence": int(self.sequence), "locked": bool(self.locked), "filtered": bool(self.filtered)}


class RejectReason:
    """Why a sample was not accepted.  Values are stable strings: they go into diagnostics and tests."""

    STALE = "stale"                      # nothing arrived within the freshness window / past valid_until
    DELAYED = "delayed"                  # it arrived, but it describes a moment too far in the past
    OUT_OF_ORDER = "out_of_order"        # sequence or stamp went backwards
    LOW_CONFIDENCE = "low_confidence"
    HIGH_COVARIANCE = "high_covariance"
    JUMP = "jump"                        # moved further than a real object could between samples
    NO_TRANSFORM = "no_transform"        # the frame could not be resolved to the planning frame
    WRONG_OBJECT = "wrong_object"
    MALFORMED = "malformed"


__all__ = ["DEFAULT_JUMP_ORIENTATION_RAD", "DEFAULT_JUMP_POSITION_M", "DEFAULT_MAX_AGE_S",
           "DEFAULT_MAX_OBSERVATION_DELAY_S", "ObjectPoseSample", "RejectReason", "TargetSnapshot"]
