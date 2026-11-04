import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomous_state import (DepthSurfaceValidator, ObservationHistory,
                              VisionStatePublisher, output_status)
from foundationpose_pose_ros_node import load_pose_state, pose_state_timestamp

EPOCH = 1_800_000_000


def valid_state():
    return dict(schema_version=2, status="pose_valid", pose_valid=True,
                observation_state="measured", observation_stamp_ns=EPOCH * 10**9,
                observation_sequence=1, timestamp_semantics="realsense_exposure_midpoint",
                selected_track_id=1, requested_class_id=0, requested_class_name="fenda",
                selected_target=dict(track_id=1, class_id=0, class_name="fenda", observed_this_frame=True),
                camera_frame="d435i_color_optical_frame", pose_camera_object=np.eye(4).tolist(),
                pose_quality=dict(valid=True, depth_surface=dict(valid=True)),
                pose_age_s=.01, registration_age_s=25., imu_fresh=True, imu_enabled=True, imu_required=True,
                imu_alignment=dict(ready=True,image_stamp_ns=EPOCH*10**9,imu_stamp_ns=EPOCH*10**9-5_000_000,
                                   camera_frame="d435i_color_optical_frame"))


class ObservationTests(unittest.TestCase):
    def test_switch_resets_both_ages_but_same_target_keeps_them(self):
        h = ObservationHistory()
        h.select(1)
        h.observe(stamp_ns=EPOCH*10**9, sequence=5, semantics="realsense_exposure_midpoint")
        h.registered_s = 4
        h.select(1)
        metadata = h.metadata(EPOCH*10**9 + 300_000_000, 30)
        self.assertEqual(metadata["pose_age_s"], .3)
        self.assertEqual(metadata["registration_age_s"], 26)
        h.select(2)
        metadata = h.metadata(EPOCH*10**9, 30)
        self.assertIsNone(metadata["pose_age_s"])
        self.assertIsNone(metadata["registration_age_s"])

    def test_paused_detection_never_refreshes_capture_stamp(self):
        h = ObservationHistory()
        h.select(1)
        h.observe(stamp_ns=EPOCH*10**9, sequence=5, semantics="realsense_exposure_midpoint")
        for ns in (100_000_000, 300_000_000, 1_000_000_000):
            metadata = h.metadata(EPOCH*10**9 + ns, 4)
            status, mode, valid = output_status("registering_motion-refresh", has_pose=True,
                measured=False, observation_age_s=metadata["observation_age_s"], quality_valid=True, maximum_age_s=.5, imu_ready=True)
            self.assertEqual((status, mode, valid), ("pose_predicted", "predicted", False))
            self.assertEqual(metadata["observation_sequence"], 5)
        self.assertEqual(metadata["observation_age_s"], 1)

    def test_current_measurement_requires_freshness_and_quality(self):
        for age, quality, expected in ((.02, True, "pose_valid"), (.6, True, "pose_stale"),
                                        (-.01, True, "pose_stale"), (None, True, "pose_stale"),
                                        (.02, False, "pose_degraded")):
            with self.subTest(age=age, quality=quality):
                status, _, valid = output_status("pose_valid", has_pose=True, measured=True,
                    observation_age_s=age, quality_valid=quality, maximum_age_s=.5, imu_ready=True)
                self.assertEqual(status, expected)
                self.assertEqual(valid, expected == "pose_valid")


class StateReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "latest.json"

    def tearDown(self):
        self.directory.cleanup()

    def read(self, state, now=EPOCH+.1):
        self.path.write_text(json.dumps(state))
        # Simulate the producer rewriting the file just before the read.
        os.utime(self.path, (now-.001, now-.001))
        return load_pose_state(self.path, now_epoch_s=now, maximum_state_age_s=.5)

    def test_capture_time_passed_to_pose_instead_of_mtime(self):
        result = self.read(valid_state())
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.source_stamp_s, EPOCH)
        self.assertAlmostEqual(result.observation_age_s, .1, places=5)
        self.assertEqual(result.registration_age_s, 25)

    def test_rewritten_file_cannot_refresh_old_capture(self):
        result = self.read(valid_state(), now=EPOCH+1)
        self.assertFalse(result.valid)
        self.assertIn("observation is stale", result.reason)

    def test_capture_timestamp_preserves_every_nanosecond(self):
        state = valid_state()
        state["observation_stamp_ns"] += 12345679
        state["imu_alignment"]["image_stamp_ns"] = state["observation_stamp_ns"]
        result = self.read(state)
        self.assertTrue(result.valid)
        stamp = pose_state_timestamp(result)
        self.assertEqual(stamp.sec * 10**9 + stamp.nanosec, state["observation_stamp_ns"])

    def test_prediction_rejected_even_with_old_observed_flag_and_valid_quality(self):
        state = valid_state()
        state.update(status="pose_predicted", pose_valid=False, observation_state="predicted")
        self.assertFalse(self.read(state).valid)

    def test_stop_rejected_immediately_even_if_fields_inconsistent(self):
        state = valid_state()
        state["status"] = "stopped"
        result = self.read(state)
        self.assertFalse(result.valid)
        self.assertIn("stopped", result.reason)

    def test_no_capture_or_future_capture_is_rejected(self):
        for stamp in (None, 0, True, (EPOCH+1)*10**9):
            with self.subTest(stamp=stamp):
                state = valid_state()
                state["observation_stamp_ns"] = stamp
                self.assertFalse(self.read(state).valid)

    def test_independent_depth_required(self):
        for depth in (None, dict(valid=False)):
            state = valid_state()
            state["pose_quality"]["depth_surface"] = depth
            self.assertFalse(self.read(state).valid)

    def test_legacy_replay_retains_explicit_legacy_time_semantics(self):
        state = valid_state()
        state["schema_version"] = 1
        result = self.read(state)
        self.assertTrue(result.valid)
        self.assertEqual(result.timestamp_semantics, "state_file_mtime_unix_system_time")

    def test_shutdown_bypasses_rate_limit_clears_pose_and_target(self):
        publisher = VisionStatePublisher(self.path, rate_hz=.001)
        publisher.publish(valid_state(), 10)
        publisher.invalidate("stopped", reason="test")
        state = json.loads(self.path.read_text())
        self.assertEqual(state["status"], "stopped")
        for key in ("pose_camera_object", "pose_quality", "selected_target", "selected_track_id",
                    "pose_age_s", "registration_age_s", "observation_stamp_ns"):
            self.assertIsNone(state[key], key)
        self.assertFalse(self.read(state).valid)


class SurfaceQualityTests(unittest.TestCase):
    def setUp(self):
        vertices = np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])
        self.validator = DepthSurfaceValidator(vertices, [[0,1,2],[0,2,3]])
        self.K = np.array([[100.,0,10],[0,100.,10],[0,0,1]])
        self.depth = np.full((20,20), .7)
        self.mask = np.ones((20,20), bool)
        self.pose = np.eye(4)
        self.pose[2,3] = .7

    def test_consistent_surface_passes(self):
        result = self.validator.assess(self.K, self.depth, self.mask, self.pose)
        self.assertTrue(result["valid"], result)
        self.assertLess(result["median_distance_m"], .005)

    def test_wrong_depth_fails_despite_matching_box_and_self_center(self):
        result = self.validator.assess(self.K, self.depth+.12, self.mask, self.pose)
        self.assertFalse(result["valid"])
        self.assertGreater(result["median_distance_m"], .1)

    def test_wrong_pose_fails_independently(self):
        self.pose[2,3] += .12
        self.assertFalse(self.validator.assess(self.K, self.depth, self.mask, self.pose)["valid"])

    def test_absent_or_sparse_depth_never_valid(self):
        for depth in (np.zeros_like(self.depth), np.full_like(self.depth, np.nan)):
            self.assertFalse(self.validator.assess(self.K, depth, self.mask, self.pose)["valid"])


if __name__ == "__main__":
    unittest.main()
