from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from async_pose_worker import LatestPoseRegistrationWorker
from multi_object_rgbd_tracker import Detection3D, MultiObjectRGBDTracker, NearestTargetSelector
from rgbd_kalman_tracker import TrackerConfig


def detection(x: float, z: float, box_x: float, payload: str) -> Detection3D:
    return Detection3D(
        box=np.array([box_x, 100.0, box_x + 60.0, 300.0]),
        confidence=0.9,
        position_m=np.array([x, 0.0, z]),
        covariance=np.diag([0.005**2, 0.005**2, 0.010**2]),
        payload=payload,
    )


def test_multi_tracks_keep_ids_and_selector_locks_nearest() -> None:
    manager = MultiObjectRGBDTracker(
        TrackerConfig(),
        confirmation_hits=3,
        maximum_detection_misses=10,
        maximum_3d_distance_m=0.20,
    )
    views = []
    for frame in range(3):
        # Reverse detector order to prove IDs do not depend on list order.
        items = [detection(-0.10, 0.55, 100, "near"), detection(0.12, 0.85, 400, "far")]
        if frame == 1:
            items.reverse()
        views = manager.update(items, frame * 0.1)
    assert len(views) == 2
    assert all(view.confirmed for view in views)
    payload_to_id = {view.payload: view.track_id for view in views}
    assert payload_to_id["near"] != payload_to_id["far"]

    selector = NearestTargetSelector(lost_grace_frames=2)
    selected = selector.select(views)
    assert selected is not None
    assert selected.payload == "near"
    locked_id = selected.track_id

    # Even if detector confidence/order changes, a live selection remains locked.
    views = manager.update(
        [detection(0.12, 0.50, 400, "far"), detection(-0.10, 0.56, 100, "near")],
        0.4,
    )
    assert selector.select(views).track_id == locked_id


def test_selector_switches_after_selected_track_grace_expires() -> None:
    manager = MultiObjectRGBDTracker(
        TrackerConfig(),
        confirmation_hits=1,
        maximum_detection_misses=10,
        maximum_3d_distance_m=0.20,
    )
    views = manager.update(
        [detection(-0.10, 0.55, 100, "near"), detection(0.12, 0.85, 400, "far")],
        0.0,
    )
    selector = NearestTargetSelector(lost_grace_frames=2)
    first = selector.select(views)
    assert first is not None and first.payload == "near"
    for frame in range(1, 4):
        views = manager.update([detection(0.12, 0.85, 400, "far")], frame * 0.1)
    replacement = selector.select(views)
    assert replacement is not None and replacement.payload == "far"


def test_async_worker_is_bounded_and_returns_result() -> None:
    def work(value: int) -> int:
        time.sleep(0.03)
        return value * 2

    worker = LatestPoseRegistrationWorker(work)
    try:
        assert worker.submit(21)
        assert not worker.submit(22)
        deadline = time.perf_counter() + 1.0
        results = []
        while time.perf_counter() < deadline and not results:
            results = worker.poll()
            time.sleep(0.005)
        assert len(results) == 1
        assert results[0].value == 42
        assert results[0].error is None
    finally:
        worker.close()
