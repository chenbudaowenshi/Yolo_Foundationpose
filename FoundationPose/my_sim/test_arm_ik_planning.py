"""Geometry and pose-parsing tests for the arm_ik reach pipeline.

These cover the parts that need no MuJoCo model, so they run in any environment
that has numpy.  The kinematics themselves are exercised by running
``my_sim/arm_ik.py`` against a pose record.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from my_sim import arm_ik


def _pose_record(translation=(2.0, 0.5, 1.0), **overrides):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    record = {
        "T_world_object": {"matrix": matrix.tolist()},
        "target": {"class_name": "bottle"},
        "symmetry": {"yaw_observable": False},
        "pose_quality": {"valid": True, "reason": "ok"},
    }
    record.update(overrides)
    return record


def _write(tmp_path, record):
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_load_object_pose_reads_translation(tmp_path):
    path = _write(tmp_path, _pose_record(translation=(1.5, -0.25, 0.9)))
    pose = arm_ik.load_object_pose(path)
    assert pose["translation"] == pytest.approx((1.5, -0.25, 0.9))
    assert pose["class_name"] == "bottle"
    assert pose["yaw_observable"] is False


def test_load_object_pose_requires_world_pose(tmp_path):
    record = _pose_record()
    del record["T_world_object"]
    path = _write(tmp_path, record)
    with pytest.raises(RuntimeError, match="no 'T_world_object'"):
        arm_ik.load_object_pose(path)


def test_load_object_pose_rejects_invalid_quality(tmp_path):
    record = _pose_record()
    record["pose_quality"] = {"valid": False, "reason": "depth_mismatch"}
    path = _write(tmp_path, record)
    with pytest.raises(RuntimeError, match="depth_mismatch"):
        arm_ik.load_object_pose(path)


def test_missing_pose_file_names_the_upstream_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="get_pose_mujoco"):
        arm_ik.load_object_pose(tmp_path / "absent.json")


def test_grasp_point_lifts_above_the_reported_origin():
    pose = {"translation": (1.0, 2.0, 0.5)}
    grasp = arm_ik.grasp_point_from_pose(np, pose, 0.10)
    assert grasp == pytest.approx([1.0, 2.0, 0.60])


def test_standing_target_sits_on_the_line_to_the_object():
    grasp = np.array([3.0, 4.0, 1.0])  # planar radius is exactly 5.0
    stance = arm_ik.standing_target(np, grasp, 1.0)
    assert np.linalg.norm(stance) == pytest.approx(4.0)
    # Stance, origin and object stay collinear so the object ends up in front.
    assert stance == pytest.approx([2.4, 3.2])


def test_standing_target_clamps_when_already_inside_the_standoff():
    grasp = np.array([0.30, 0.40, 1.0])  # planar radius 0.5, standoff larger
    stance = arm_ik.standing_target(np, grasp, 0.9)
    assert stance == pytest.approx([0.0, 0.0])


def test_approach_frame_points_from_the_base_to_the_object():
    yaw, axis = arm_ik.approach_frame(np, np.array([1.0, 1.0]), np.array([2.0, 1.0, 1.1]))
    assert yaw == pytest.approx(0.0)
    assert axis == pytest.approx([1.0, 0.0, 0.0])

    yaw, axis = arm_ik.approach_frame(np, np.array([0.0, 0.0]), np.array([0.0, 2.0, 1.0]))
    assert yaw == pytest.approx(math.pi / 2.0)
    assert axis == pytest.approx([0.0, 1.0, 0.0])


def test_approach_frame_rejects_a_vertical_line():
    with pytest.raises(RuntimeError, match="directly under"):
        arm_ik.approach_frame(np, np.array([1.0, 1.0]), np.array([1.0, 1.0, 2.0]))


def test_stance_correction_restores_the_intended_standoff():
    grasp = np.array([2.0, 0.0, 1.0])
    # The gait parked 0.10 m short of a 0.45 m standoff.
    walked = np.array([1.45, 0.0])
    corrected, correction = arm_ik.stance_position_for_reach(np, walked, grasp, 0.45)
    assert np.linalg.norm(grasp[:2] - corrected) == pytest.approx(0.45)
    assert correction == pytest.approx(0.10)


def test_stance_correction_keeps_the_walked_heading():
    grasp = np.array([1.0, 1.0, 1.0])
    walked = np.array([0.2, 0.1])
    corrected, _ = arm_ik.stance_position_for_reach(np, walked, grasp, 0.4)
    before = grasp[:2] - walked
    after = grasp[:2] - corrected
    # Same direction, only the distance changes.
    cosine = float(
        before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
    )
    assert cosine == pytest.approx(1.0)


def test_wrap_angle_folds_into_a_single_turn():
    assert arm_ik.wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert arm_ik.wrap_angle(-1.5 * math.pi) == pytest.approx(0.5 * math.pi)
    assert arm_ik.wrap_angle(0.4) == pytest.approx(0.4)


def test_standoff_default_stays_inside_the_measured_arm_reach():
    assert arm_ik.DEFAULT_STANDOFF_DISTANCE < arm_ik.ARM_REACH_LIMIT
