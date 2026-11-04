import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

os.environ["D435I_FOPOSE_REEXEC"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ros_rgbd_input import RosRGBDInput, resolve_camera_arguments
from d435i_foundationpose import build_parser
from d435i_autonomous_vision import build_parser as autonomous_parser
from rgbd_transport import encode_color_jpeg, encode_depth_png
from test_sb_handoff import header, image, info, EPOCH


class NodeStub:
    context = NS(get_domain_id=lambda: 0)
    def __init__(self): self.subscriptions = []; self.logs = []
    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions.append((kind.__name__, topic, callback))
        return self.subscriptions[-1]
    def get_logger(self): return self
    def info(self, text): self.logs.append(text)
    def warning(self, text): self.logs.append(text)


class RosRGBDInputTests(unittest.TestCase):
    def args(self, *argv):
        return resolve_camera_arguments(build_parser().parse_args(list(argv)))

    def test_both_demo_defaults_match_current_camera(self):
        for parser in (build_parser(), autonomous_parser()):
            args = resolve_camera_arguments(parser.parse_args(['--ros2']))
            node = NodeStub(); RosRGBDInput(node, args)
            self.assertEqual([(t[0],t[1]) for t in node.subscriptions], [
                ('CompressedImage','/head_camera/color/image_raw/compressed'),
                ('CompressedImage','/head_camera/depth/image_rect_raw/compressedDepth'),
                ('CameraInfo','/head_camera/color/camera_info'),
                ('CameraInfo','/head_camera/depth/camera_info'),
                ('Imu','/head_camera/imu')])
            self.assertEqual(args.imu_topic, '/head_camera/imu')

    def test_raw_and_custom_namespace_and_explicit_topic(self):
        args = self.args('--image-transport','raw','--camera-prefix','/wrist',
                         '--rgb-topic','/custom/rgb','--ros-domain-id','0')
        node = NodeStub(); RosRGBDInput(node,args)
        self.assertEqual(node.subscriptions[0][:2],('Image','/custom/rgb'))
        self.assertEqual(args.depth_topic, '/wrist/depth/image_rect_raw')
        self.assertEqual(args.depth_camera_info_topic, '/wrist/depth/camera_info')
        self.assertEqual(args.ros_domain_id,0)

    def feed(self, receiver, *, compressed=True, depth=None, depth_stamp=EPOCH+5_000_000):
        rgb = np.full((2,2,3),128,np.uint8)
        depth = np.full((2,2),700,np.uint16) if depth is None else depth
        if compressed:
            fmt, payload = encode_color_jpeg(rgb)
            rgb_msg = NS(header=header(),format=fmt,data=payload)
            fmt, payload = encode_depth_png(depth)
            depth_msg = NS(header=header(depth_stamp),format=fmt,data=payload)
        else:
            rgb_msg = image(rgb,'bgr8')
            depth_msg = image(depth,'32FC1' if depth.dtype.kind=='f' else '16UC1',ns=depth_stamp,padding=4)
        receiver.receive('rgb_info',info())
        receiver.receive('depth_info',info(ns=depth_stamp))
        receiver.receive('depth',depth_msg)
        receiver.receive('rgb',rgb_msg)

    def test_compressed_decode_capture_stamps_calibration_and_one_use(self):
        receiver = RosRGBDInput(NodeStub(),self.args())
        self.feed(receiver)
        frame = receiver.take_latest()
        self.assertEqual(frame.stamp_ns,EPOCH)
        self.assertEqual(frame.depth_stamp_ns,EPOCH+5_000_000)
        self.assertEqual(frame.frame_id,'head_d435i_optical_frame')
        np.testing.assert_allclose(frame.depth_m,.7,atol=1e-7)
        np.testing.assert_array_equal(frame.K,np.array(info().k).reshape(3,3))
        self.assertIsNone(receiver.take_latest())
        self.feed(receiver)
        self.assertIsNone(receiver.take_latest())

    def test_float_depth_already_metres_not_scaled_twice(self):
        receiver = RosRGBDInput(NodeStub(),self.args('--image-transport','raw'))
        self.feed(receiver,compressed=False,depth=np.full((2,2),.7,np.float32))
        np.testing.assert_allclose(receiver.take_latest().depth_m,.7,atol=1e-7)

    def test_uint16_raw_depth_is_millimetres_without_scale_topic(self):
        receiver = RosRGBDInput(NodeStub(),self.args('--image-transport','raw'))
        self.feed(receiver,compressed=False)
        np.testing.assert_allclose(receiver.take_latest().depth_m,.7,atol=1e-7)

    def test_wrong_optical_frame_is_rejected(self):
        receiver = RosRGBDInput(NodeStub(),self.args('--camera-frame','other_optical'))
        self.feed(receiver)
        self.assertIsNone(receiver.take_latest())
        self.assertGreater(receiver.errors['metadata'],0)

    def test_mismatched_capture_times_do_not_form_a_pair(self):
        receiver = RosRGBDInput(NodeStub(),self.args())
        self.feed(receiver,depth_stamp=EPOCH+100_000_000)
        self.assertIsNone(receiver.take_latest())

    def test_invalid_domain_is_rejected(self):
        with self.assertRaises(ValueError): self.args('--ros-domain-id','-1')


if __name__ == '__main__': unittest.main()
