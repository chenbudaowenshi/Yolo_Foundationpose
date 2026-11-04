"""Codec and timestamp/CameraInfo regression tests, without ROS or hardware."""
from array import array
import copy
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'my_demo'))
from rgbd_transport import (encode_color_jpeg, encode_depth_png, decode_color_jpeg,
                            decode_depth_png, image_topics)
from sb_pose_contract import RGBDSynchronizer
from test_sb_handoff import EPOCH, FRAME, header, info, image


def message(encoded, ns=EPOCH):
    format_text, data = encoded
    return NS(header=header(ns), format=format_text, data=array('B', data))


class CodecTests(unittest.TestCase):
    def test_all_uint16_values_roundtrip_losslessly(self):
        depth = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        msg = message(encode_depth_png(depth))
        self.assertEqual(msg.format, '16UC1; compressedDepth png')
        # Independent ROS ABI check: enum INV_DEPTH=0 + two unused float32s.
        self.assertEqual(bytes(msg.data[:12]), bytes(12))
        self.assertEqual(bytes(msg.data[12:20]), b'\x89PNG\r\n\x1a\n')
        np.testing.assert_array_equal(decode_depth_png(msg), depth)

    def test_depth_endianness_and_strided_input(self):
        depth = np.arange(256, dtype='>u2').reshape(16, 16)[:, ::2]
        np.testing.assert_array_equal(decode_depth_png(message(encode_depth_png(depth))), depth)

    def test_depth_legacy_ros_format_and_unused_params(self):
        original = np.array([[0, 1, 999, 65000, 65535]], np.uint16)
        fmt, data = encode_depth_png(original)
        # Humble 16UC1 ignores the float parameters (some versions leave them unset).
        msg = message(('16UC1; compressedDepth', struct.pack('<iff', 0, 12.5, -3) + data[12:]))
        np.testing.assert_array_equal(decode_depth_png(msg), original)

    def test_jpeg_preserves_size_and_color_order(self):
        color = np.full((64, 80, 3), [20, 60, 200], np.uint8)
        decoded = decode_color_jpeg(message(encode_color_jpeg(color, 90)))
        self.assertEqual(decoded.shape, color.shape)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertLess(float(np.abs(decoded.astype(float) - color).mean()), 3)
        self.assertGreater(float(decoded[:, :, 2].mean()), float(decoded[:, :, 0].mean()) + 100)

    def test_depth_does_not_accept_float_quantization(self):
        with self.assertRaises(ValueError): encode_depth_png(np.ones((4, 4), np.float32))
        msg = message(encode_depth_png(np.ones((4, 4), np.uint16)))
        msg.format = '32FC1; compressedDepth png'
        with self.assertRaises(ValueError): decode_depth_png(msg)

    def test_rejects_truncated_or_wrong_depth_payload(self):
        for payload in (b'', bytes(12), struct.pack('<iff', 99, 0, 0) + b'\x89PNG\r\n\x1a\n',
                        bytes(12) + b'not png data'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_depth_png(message(('16UC1; compressedDepth png', payload)))
        _, png8 = cv2.imencode('.png', np.zeros((4, 4), np.uint8))
        with self.assertRaises(ValueError):
            decode_depth_png(message(('16UC1; compressedDepth png', bytes(12) + png8.tobytes())))

    def test_rejects_unsupported_color_codec_and_corrupt_jpeg(self):
        with self.assertRaises(ValueError): decode_color_jpeg(message(('jpeg', b'not a jpeg')))
        msg = message(encode_color_jpeg(np.zeros((4, 4, 3), np.uint8)))
        msg.format = 'rgb8; png compressed bgr8'
        with self.assertRaises(ValueError): decode_color_jpeg(msg)

    def test_compression_parameters_are_validated(self):
        for quality in (0, 101, 20.5):
            with self.assertRaises(ValueError): encode_color_jpeg(np.zeros((4, 4, 3), np.uint8), quality)
        for level in (-1, 10, 2.5):
            with self.assertRaises(ValueError): encode_depth_png(np.zeros((4, 4), np.uint16), level)

    def test_topic_selection(self):
        self.assertEqual(image_topics('/head_camera/', 'compressed'),
                         ('/head_camera/color/image_raw/compressed', '/head_camera/depth/image_rect_raw/compressedDepth'))
        self.assertEqual(image_topics('/head_camera', 'raw'),
                         ('/head_camera/color/image_raw', '/head_camera/depth/image_rect_raw'))


class CompressedSyncTests(unittest.TestCase):
    def make_inputs(self):
        color = np.full((16, 24, 3), [20, 60, 200], np.uint8)
        depth = np.arange(384, dtype=np.uint16).reshape(16, 24)
        cs, ds = EPOCH + 123456789, EPOCH + 127456789
        ci, di = info(cs, 24, 16), info(ds, 24, 16)
        inputs = dict(rgb=message(encode_color_jpeg(color), cs), depth=message(encode_depth_png(depth), ds),
                      rgb_info=ci, depth_info=di)
        return inputs, color, depth

    def test_capture_times_and_intrinsics_survive_compressed_pair(self):
        inputs, color, depth = self.make_inputs()
        original = copy.deepcopy(inputs)
        sync = RGBDSynchronizer()
        for key in ('depth_info', 'rgb', 'rgb_info', 'depth'): sync.add(key, inputs[key])
        frame = sync.pop_latest()
        self.assertEqual(frame.stamp_ns, EPOCH + 123456789)
        self.assertEqual(frame.depth_stamp_ns, EPOCH + 127456789)
        self.assertEqual(frame.frame_id, FRAME)
        np.testing.assert_array_equal(frame.K, np.asarray(inputs['rgb_info'].k).reshape(3, 3))
        np.testing.assert_allclose(frame.depth_m, depth.astype(np.float32) * .001)
        self.assertEqual(inputs['rgb_info'], original['rgb_info'])
        self.assertEqual(inputs['depth_info'], original['depth_info'])
        self.assertIsNone(sync.pop_latest())

    def test_camera_info_required_at_matching_capture_time(self):
        inputs, _, _ = self.make_inputs()
        sync = RGBDSynchronizer()
        for key in ('rgb', 'depth', 'depth_info'): sync.add(key, inputs[key])
        self.assertIsNone(sync.pop_latest())
        wrong = copy.deepcopy(inputs['rgb_info']); wrong.header = header(EPOCH)
        sync.add('rgb_info', wrong)
        self.assertIsNone(sync.pop_latest())
        sync.add('rgb_info', inputs['rgb_info'])
        self.assertIsNotNone(sync.pop_latest())

    def test_compressed_dimensions_must_match_camera_info(self):
        inputs, _, _ = self.make_inputs()
        inputs['rgb_info'].width = inputs['depth_info'].width = 25
        sync = RGBDSynchronizer()
        for key, msg in inputs.items(): sync.add(key, msg)
        with self.assertRaisesRegex(ValueError, 'CameraInfo size'): sync.pop_latest()

    def test_raw_mode_still_decodes_identical_depth(self):
        inputs, color, depth = self.make_inputs()
        inputs['rgb'] = image(color, 'bgr8', EPOCH + 123456789)
        inputs['depth'] = image(depth, '16UC1', EPOCH + 127456789)
        sync = RGBDSynchronizer()
        for key, msg in inputs.items(): sync.add(key, msg)
        result = sync.pop_latest()
        np.testing.assert_array_equal(result.color_bgr, color)
        np.testing.assert_allclose(result.depth_m, depth.astype(np.float32) * .001)


if __name__ == '__main__': unittest.main()
