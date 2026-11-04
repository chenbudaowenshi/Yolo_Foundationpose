import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'my_demo'))
sys.path.insert(0, str(ROOT / 'sb_vision_handoff_20260910'))
from camera_contract import capture_stamp_ns, ros_distortion, depth_mm_image
from sb_pose_contract import (RGBDFrame, RGBDSynchronizer, PublicationGate, decode_image,
                              load_object_profile, assess_quality, quaternion_xyzw, resize_pose_inputs)
from sb_vision_pipeline import PosePipeline
from shuangbi_grasp.target_manager import TargetValidator, ValidatorConfig, bottle_axis_angle
from shuangbi_grasp.contracts import ObjectPoseSample

EPOCH = 1_800_000_000_000_000_000
FRAME = 'head_d435i_optical_frame'


def header(ns=EPOCH):
    return NS(frame_id=FRAME, stamp=NS(sec=ns//10**9, nanosec=ns%10**9))


def image(values, encoding, ns=EPOCH, padding=0):
    values = np.asarray(values)
    h, w = values.shape[:2]
    rows = [row.tobytes() + bytes(padding) for row in values]
    return NS(header=header(ns), height=h, width=w, step=len(rows[0]), encoding=encoding,
              is_bigendian=values.dtype.byteorder == '>', data=b''.join(rows))


def info(ns=EPOCH, width=2, height=2):
    return NS(header=header(ns), width=width, height=height, k=[100.,0,1,0,100,1,0,0,1],
              d=[0.]*5, distortion_model='plumb_bob')


def scene(ns=EPOCH):
    depth = np.zeros((100,100), np.float32)
    depth[38:62,46:54] = .68
    mask = depth > 0
    detection = NS(mask=mask, box=np.array([45.,37,55,63]), confidence=.95)
    frame = RGBDFrame(ns, ns, FRAME, np.zeros((100,100,3), np.uint8), depth,
                      np.array([[100.,0,50],[0,100,50],[0,0,1]]))
    pose = np.array([[1.,0,0,0],[0,0,-1,.07975],[0,1,0,.7],[0,0,0,1]])
    return frame, detection, pose


class FakeBackend:
    bounds = np.array([[-.032534,-.032561,0],[.032534,.032561,.1595]])
    def __init__(self):
        _, detection, self.pose = scene()
        self.detections = [detection]
        self.registers = self.tracks = 0
    def detect(self, frame): return self.detections
    def register(self, frame, detection, full=False):
        self.registers += 1
        return self.pose.copy()
    def track(self, frame):
        self.tracks += 1
        return self.pose.copy()
    def reset(self): pass
    def checkpoint(self): return self.pose.copy()
    def restore(self, state): self.pose = state.copy()


class CameraContractTests(unittest.TestCase):
    def test_epoch_midpoint(self):
        stamp, basis = capture_stamp_ns(EPOCH/1e6, 'global_time', {'frame_timestamp':20000,'sensor_timestamp':15000})
        self.assertEqual(stamp, EPOCH-5000000)
        self.assertEqual(basis, 'exposure_midpoint')
    def test_device_clock_is_not_ros_time(self):
        with self.assertRaises(ValueError): capture_stamp_ns(12345, 'hardware_clock')
    def test_unknown_epoch_is_rejected(self):
        with self.assertRaises(ValueError): capture_stamp_ns(12345, 'global_time')
    def test_distortion_not_fabricated(self):
        self.assertEqual(ros_distortion('brown_conrady', [.1,0,0,0,0])[1][0], .1)
        with self.assertRaises(ValueError): ros_distortion('inverse_brown_conrady', [.1,0,0,0,0])
    def test_depth_unit_conversion_invalids(self):
        np.testing.assert_array_equal(depth_mm_image([0, 1000, 2000], .0005), [0,500,1000])
        np.testing.assert_array_equal(depth_mm_image([float('nan'), -1, 1e9], .001), [0,0,0])
    def test_padding_color_channel_order(self):
        msg=image(np.array([[[1,2,3],[4,5,6]]],np.uint8),'rgb8',padding=5)
        np.testing.assert_array_equal(decode_image(msg), [[[3,2,1],[6,5,4]]])
    def test_big_endian_depth(self):
        msg=image(np.array([[123,456]],dtype='>u2'),'16UC1',padding=2)
        np.testing.assert_array_equal(decode_image(msg), [[123,456]])


class SyncTests(unittest.TestCase):
    def feed(self, sync, rgb_ns=EPOCH, depth_ns=EPOCH, encoding='16UC1'):
        sync.add('rgb',image(np.zeros((2,2,3),np.uint8),'rgb8',rgb_ns))
        values=np.full((2,2), 700, np.uint16) if encoding=='16UC1' else np.full((2,2),.7,np.float32)
        sync.add('depth',image(values,encoding,depth_ns))
        sync.add('rgb_info',info(rgb_ns))
        sync.add('depth_info',info(depth_ns))
    def test_capture_pair_is_one_use(self):
        sync=RGBDSynchronizer(); self.feed(sync)
        frame=sync.pop_latest()
        self.assertEqual(frame.stamp_ns,EPOCH)
        np.testing.assert_allclose(frame.depth_m,.7)
        self.assertIsNone(sync.pop_latest())
        self.feed(sync)
        self.assertIsNone(sync.pop_latest())
    def test_metre_float_depth_not_scaled_again(self):
        sync=RGBDSynchronizer(); self.feed(sync,encoding='32FC1')
        np.testing.assert_allclose(sync.pop_latest().depth_m,.7)
    def test_unpaired_rgb_depth_not_consumed(self):
        sync=RGBDSynchronizer(); self.feed(sync,depth_ns=EPOCH+30000000)
        self.assertIsNone(sync.pop_latest())
    def test_info_must_match_capture(self):
        sync=RGBDSynchronizer(); self.feed(sync)
        sync.buffers['rgb_info'].clear()
        sync.add('rgb_info',info(EPOCH-1))
        self.assertIsNone(sync.pop_latest())
    def test_native_depth_calibration_rejected(self):
        sync=RGBDSynchronizer(); self.feed(sync)
        sync.buffers['depth_info'][EPOCH].k[0]=110
        with self.assertRaisesRegex(ValueError,'aligned'): sync.pop_latest()
    def test_unknown_frame_rejected(self):
        sync=RGBDSynchronizer(); msg=info(); msg.header.frame_id='camera_link'
        with self.assertRaises(ValueError): sync.add('rgb_info',msg)
    def test_latest_pair_drops_backlog(self):
        sync=RGBDSynchronizer()
        self.feed(sync)
        self.feed(sync,EPOCH+100000000,EPOCH+100000000)
        self.assertEqual(sync.pop_latest().stamp_ns,EPOCH+100000000)
        self.assertIsNone(sync.pop_latest())


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.profile=load_object_profile(ROOT/'my_demo/config/bottle_current.json')
        self.backend=FakeBackend()
        self.pipeline=PosePipeline(self.backend,self.profile)
    def test_model_center_and_axis_preserved(self):
        frame,_,pose=scene(); result=self.pipeline.process(frame)
        self.assertEqual(result.status,'valid')
        np.testing.assert_allclose(result.pose_center[:3,3],[0,0,.7])
        np.testing.assert_allclose(result.pose_center[:3,:3],pose[:3,:3])
        self.assertTrue(np.all(result.covariance==0))
    def test_fresh_tracking_after_long_registration(self):
        self.pipeline.process(scene()[0])
        result=self.pipeline.process(scene(EPOCH+1500000000)[0])
        self.assertEqual(result.mode,'tracking')
        self.assertEqual(self.backend.registers,1)
    def test_brief_occlusion_stops_output_without_discarding_seed(self):
        self.pipeline.process(scene()[0]); self.backend.detections=[]
        self.assertIsNone(self.pipeline.process(scene(EPOCH+100000000)[0]).pose_center)
        self.backend.detections=[scene()[1]]
        result=self.pipeline.process(scene(EPOCH+200000000)[0])
        self.assertEqual(result.mode,'tracking')
    def test_long_occlusion_requires_reregistration(self):
        self.pipeline.process(scene()[0]); self.backend.detections=[]
        self.assertIsNone(self.pipeline.process(scene(EPOCH+400000000)[0]).pose_center)
        self.backend.detections=[scene()[1]]
        self.assertEqual(self.pipeline.process(scene(EPOCH+500000000)[0]).mode,'reregistration')
    def test_bad_pose_rejected_with_high_detector_confidence(self):
        frame,detection,pose=scene(); pose[2,3]+=.15
        ok,_,_=assess_quality(frame,detection,pose,self.backend.bounds)
        self.assertFalse(ok)
    def test_failed_track_does_not_poison_next_seed(self):
        class OnceBad(FakeBackend):
            def track(self, frame):
                self.tracks += 1
                if self.tracks == 1: self.pose[2,3] += .2
                return self.pose.copy()
        backend=OnceBad(); pipeline=PosePipeline(backend,self.profile)
        pipeline.process(scene()[0])
        self.assertIsNone(pipeline.process(scene(EPOCH+100000000)[0]).pose_center)
        np.testing.assert_allclose(backend.pose,scene()[2])
        recovered=pipeline.process(scene(EPOCH+200000000)[0])
        self.assertEqual(recovered.mode,'tracking')
        self.assertEqual(recovered.status,'valid')
    def test_ambiguous_bottles_stop_output(self):
        self.pipeline.process(scene()[0])
        self.backend.detections=[scene()[1],scene()[1]]
        self.assertIsNone(self.pipeline.process(scene(EPOCH+100000000)[0]).pose_center)
    def test_duplicate_capture_does_not_reenter_inference(self):
        frame=scene()[0]; self.pipeline.process(frame)
        self.assertIsNone(self.pipeline.process(frame).pose_center)
        self.assertEqual(self.backend.registers,1)
    def test_timestamp_not_refreshed_on_slow_result(self):
        gate=PublicationGate(); frame=scene()[0]
        self.assertIsNone(gate.admit(frame,EPOCH+400000000))
        self.assertEqual(gate.sequence,0)
    def test_sequence_survives_target_reset(self):
        gate=PublicationGate(); self.assertEqual(gate.admit(scene()[0],EPOCH),1)
        self.pipeline.reset_target()
        self.assertEqual(gate.admit(scene(EPOCH+100000000)[0],EPOCH+100000000),2)
        self.assertIsNone(gate.admit(scene(EPOCH+100000000)[0],EPOCH+110000000))
    def test_future_timestamp_rejected(self):
        self.assertIsNone(PublicationGate().admit(scene()[0],EPOCH-10000000))
    def test_quaternion_roundtrip(self):
        from shuangbi_grasp.target_manager import quaternion_angle
        from scipy.spatial.transform import Rotation
        R=Rotation.from_euler('xyz',[1.2,-.7,2.8]).as_matrix()
        q=quaternion_xyzw(R)
        np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(),R,atol=1e-12)

    def test_resize_preserves_projected_optical_geometry(self):
        frame, detection, _ = scene()
        reduced, mask = resize_pose_inputs(frame, 50, detection.mask)
        point = np.array([.02, -.04, .7])
        uv = frame.K @ point; uv = uv[:2]/uv[2]
        new_uv = reduced.K @ point; new_uv = new_uv[:2]/new_uv[2]
        np.testing.assert_allclose(new_uv, (uv+.5)*.5-.5)
        self.assertEqual(reduced.stamp_ns, frame.stamp_ns)
        self.assertEqual(mask.shape, reduced.depth_m.shape)
        np.testing.assert_allclose(reduced.depth_m[mask], .68)


class ValidatorTests(unittest.TestCase):
    def sample(self,q,sequence):
        return ObjectPoseSample('bottle',FRAME,[0,0,.7],q,sequence*.05,sequence*.05,sequence=sequence)
    def test_yaw_and_axis_sign_do_not_trigger_jump(self):
        v=TargetValidator(); first=self.sample([1,0,0,0],1); v.accept(first)
        self.assertIsNone(v.check(self.sample([0,0,0,1],2),.1))
        self.assertIsNone(v.check(self.sample([0,1,0,0],3),.15))
    def test_real_axis_tilt_triggers_jump(self):
        v=TargetValidator(); v.accept(self.sample([1,0,0,0],1))
        q=[math.cos(math.pi/8),math.sin(math.pi/8),0,0]
        self.assertEqual(v.check(self.sample(q,2),.1),'jump')
    def test_asymmetric_mode_retains_yaw_gate(self):
        v=TargetValidator(ValidatorConfig(orientation_mode='quaternion'))
        v.accept(self.sample([1,0,0,0],1))
        self.assertEqual(v.check(self.sample([0,0,0,1],2),.1),'jump')


if __name__ == '__main__': unittest.main()
