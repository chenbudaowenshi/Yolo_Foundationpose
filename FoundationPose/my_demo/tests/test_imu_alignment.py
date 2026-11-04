import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_alignment import ImuAlignment
from camera_contract import aligned_imu_samples
from autonomous_state import output_status
from d435i_autonomous_vision import build_parser
from run_realsense_viewer import build_argument_parser
import test_autonomous_state as state_helpers
valid_state, EPOCH = state_helpers.valid_state, state_helpers.EPOCH

STAMP = EPOCH * 10**9
FRAME = 'head_d435i_optical_frame'


class ImuAlignmentTests(unittest.TestCase):
    def setUp(self): self.buffer = ImuAlignment()

    def add(self, stamp=STAMP, frame=FRAME, vector=(0.,-9.81,0.)):
        return self.buffer.add(stamp, frame, vector, now_ns=STAMP+100_000_000)

    def match(self, image=STAMP, now=STAMP+100_000_000, **kw):
        return self.buffer.match(image, FRAME, now_ns=now, **kw)

    def test_capture_time_alignment(self):
        self.add(STAMP+5_000_000)
        match = self.match()
        self.assertTrue(match.ready)
        self.assertEqual(match.skew_ms,5.)
        np.testing.assert_allclose(match.axis_camera,[0,-1,0])

    def test_no_sample_blocks_even_if_a_topic_exists(self):
        self.assertFalse(self.match().ready)

    def test_fresh_arrival_with_old_stamp_cannot_refresh_data(self):
        self.add(STAMP-1_000_000_000)
        self.assertEqual(self.match().reason,'imu_stale')

    def test_new_imu_for_other_frame_is_not_current_gravity(self):
        self.add(STAMP+100_000_000)
        self.assertEqual(self.match().reason,'imu_image_time_mismatch')

    def test_future_timestamp_blocks(self):
        self.add(STAMP+200_000_000)
        self.assertEqual(self.match().reason,'imu_timestamp_in_future')

    def test_duplicates_do_not_refresh_age_or_filter(self):
        self.add()
        first=self.match()
        self.assertFalse(self.add(vector=(9.81,0,0)))
        second=self.match(now=STAMP+200_000_000)
        self.assertEqual(first.gravity.accepted_samples,second.gravity.accepted_samples)
        np.testing.assert_array_equal(first.axis_camera,second.axis_camera)
        self.assertFalse(self.match(now=STAMP+600_000_000).ready)

    def test_invalid_nearest_sample_cannot_fall_back_to_previous_axis(self):
        self.add(STAMP-10_000_000)
        self.assertTrue(self.match().ready)
        self.add(vector=(0,0,0))
        self.assertEqual(self.match().reason,'invalid_gravity_norm')

    def test_invalid_vectors_and_frames(self):
        for frame, vector in (('',(0,-9.81,0)),(FRAME,(float('nan'),0,0)),(FRAME,(1,2))):
            with self.subTest(frame=frame,vector=vector):
                b=ImuAlignment();b.add(STAMP,frame,vector,now_ns=STAMP+100_000_000)
                self.assertFalse(b.match(STAMP,FRAME,now_ns=STAMP+10_000_000).ready)

    def test_missing_timestamp_clears_old_samples(self):
        self.add();self.buffer.add(0,FRAME,[0,-9.81,0],now_ns=STAMP+100_000_000)
        self.assertFalse(self.match().ready)

    def test_different_frame_requires_tf_at_sample_stamp(self):
        self.add(frame='imu_link',vector=(9.81,0,0))
        self.assertFalse(self.match().ready)
        calls=[]
        def rotate(vector,source,target,stamp):
            calls.append((source,target,stamp))
            return np.array([[0,1,0],[-1,0,0],[0,0,1]])@vector
        match=self.match(rotate=rotate)
        self.assertTrue(match.ready)
        self.assertEqual(calls,[('imu_link',FRAME,STAMP)])
        np.testing.assert_allclose(match.axis_camera,[0,-1,0])

    def test_tf_failure_does_not_use_an_old_axis(self):
        self.add();self.assertTrue(self.match().ready)
        self.add(STAMP+10_000_000,frame='imu_link')
        def fail(*args): raise ValueError('missing historical TF')
        self.assertFalse(self.match(image=STAMP+10_000_000,rotate=fail).ready)

    def test_dropout_and_recovery(self):
        self.add();self.assertTrue(self.match().ready)
        self.assertFalse(self.match(image=STAMP+1_000_000_000,now=STAMP+1_100_000_000).ready)
        self.buffer.add(STAMP+1_000_000_000,FRAME,[0,-9.81,0],now_ns=STAMP+1_100_000_000)
        self.assertTrue(self.match(image=STAMP+1_000_000_000,now=STAMP+1_100_000_000).ready)

    def test_valid_pose_cannot_bypass_imu_gate(self):
        kwargs=dict(has_pose=True,measured=True,observation_age_s=.02,quality_valid=True,maximum_age_s=.5)
        self.assertEqual(output_status('pose_valid',imu_ready=False,**kwargs),('waiting_for_imu','unavailable',False))
        self.assertEqual(output_status('pose_valid',imu_ready=True,**kwargs),('pose_valid','measured',True))

    def test_camera_default_and_no_imu_cli_rejected(self):
        self.assertTrue(build_argument_parser().parse_args([]).use_imu)
        for source in ('off','reference'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(['--upright-source',source])

    def test_camera_selects_previous_native_sample_nearer_exposure(self):
        old, latest = np.array([0,-9.81,0]), np.array([1,-9.7,0])
        result=aligned_imu_samples({STAMP-8_000_000:old,STAMP+25_000_000:latest},
                                    {STAMP-10_000_000:np.zeros(3),STAMP+23_000_000:np.ones(3)},STAMP)
        self.assertEqual(result['accel_stamp_ns'],STAMP-8_000_000)
        self.assertEqual(result['gyro_stamp_ns'],STAMP-10_000_000)
        np.testing.assert_array_equal(result['accel'],old)

    def test_camera_cannot_invent_missing_or_badly_timed_motion(self):
        self.assertIsNone(aligned_imu_samples({STAMP:np.zeros(3)},{},STAMP))
        self.assertIsNone(aligned_imu_samples({STAMP+40_000_000:np.zeros(3)},
                                            {STAMP+40_000_000:np.zeros(3)},STAMP))


class RequiredImuReaderTests(unittest.TestCase):
    setUp = state_helpers.StateReaderTests.setUp
    tearDown = state_helpers.StateReaderTests.tearDown
    read = state_helpers.StateReaderTests.read
    # Exercise the real reader with current-file mtime and invalid IMU provenance.
    def test_cannot_disable_required_imu_with_old_boolean(self):
        from foundationpose_pose_ros_node import load_pose_state
        import json,os
        state=valid_state();state['imu_fresh']=False
        self.path.write_text(json.dumps(state));os.utime(self.path,(EPOCH+.1,EPOCH+.1))
        result=load_pose_state(self.path,now_epoch_s=EPOCH+.1,require_imu_fresh=False)
        self.assertFalse(result.valid)

    def test_wrong_imu_image_or_frame_binding_rejected(self):
        for key,value in (('image_stamp_ns',STAMP-1),('camera_frame','wrong_optical'),('imu_stamp_ns',STAMP-100_000_000)):
            state=valid_state();state['imu_alignment'][key]=value
            self.assertFalse(self.read(state).valid)

    def test_missing_required_alignment_rejected(self):
        state=valid_state();state.pop('imu_alignment')
        self.assertFalse(self.read(state).valid)


if __name__ == '__main__': unittest.main()
