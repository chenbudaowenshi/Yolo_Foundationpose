from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from d435i_capture import next_frame_index, prepare_session, save_frame


K = np.array([[608.155, 0.0, 429.478], [0.0, 607.853, 247.833], [0.0, 0.0, 1.0]])


def test_frame_numbers_start_at_600_and_continue_across_sessions(tmp_path: Path) -> None:
    assert next_frame_index(tmp_path / "missing") == 600

    def add_frame(session: str, index: int) -> None:
        image_dir = tmp_path / session / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"frame_{index:06d}.png").touch()

    add_frame("old_copy", 42)
    assert next_frame_index(tmp_path) == 600
    add_frame("shelf_a", 600)
    add_frame("shelf_b", 642)
    (tmp_path / "shelf_b" / "image" / "preview.png").touch()
    assert next_frame_index(tmp_path) == 643


def test_save_frame_is_lossless_and_never_overwrites(tmp_path: Path) -> None:
    prepare_session(tmp_path, K, {"schema_version": 1})
    color = np.random.default_rng(0).integers(0, 256, (480, 848, 3), dtype=np.uint8)
    depth = np.zeros((480, 848), dtype=np.uint16)
    depth[100:200, 300:400] = 7923

    stem = save_frame(tmp_path, 5, color, depth, {"mode": "manual"})

    assert stem == "frame_000005"
    restored_depth = cv2.imread(str(tmp_path / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    assert restored_depth.dtype == np.uint16
    np.testing.assert_array_equal(restored_depth, depth)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "image" / f"{stem}.png")), color)
    record = json.loads((tmp_path / "frames.jsonl").read_text(encoding="utf-8"))
    assert record["stem"] == stem and record["mode"] == "manual"
    assert record["depth_valid_ratio"] == pytest.approx(10000 / depth.size, abs=1e-4)
    with pytest.raises(FileExistsError):
        save_frame(tmp_path, 5, color, depth, {"mode": "manual"})


def test_resumed_session_requires_same_intrinsics_and_writes_no_labels(tmp_path: Path) -> None:
    session = tmp_path / "session"

    prepare_session(session, K, {"schema_version": 1})
    prepare_session(session, K + 1e-5, {"schema_version": 1})

    assert sorted(path.name for path in session.iterdir()) == ["K.txt", "depth", "image", "session.json"]
    np.testing.assert_allclose(np.loadtxt(session / "K.txt"), K)
    other_resolution = K.copy()
    other_resolution[0, 2] = 319.5
    with pytest.raises(ValueError):
        prepare_session(session, other_resolution, {"schema_version": 1})
