#!/usr/bin/env python3
"""Loopback DDS integration for the shared demo receiver, without GPU/robot output."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'my_demo'),str(ROOT/'my_demo/tests'),str(ROOT/'my_demo/scripts')]
from d435i_foundationpose import build_parser
from ros_rgbd_input import RosRGBDInput
from run_realsense_viewer import RealSenseViewerNode, build_argument_parser
from test_ros_handoff import FakeCamera
from test_sb_handoff import scene

def main():
    rclpy.init()
    camera_args = build_argument_parser().parse_args([
        '--topic-prefix','/rgbd_compat_test','--image-transport','both','--width','100','--height','100',
        '--publish-fps','20'])
    camera = RealSenseViewerNode(camera_args, FakeCamera)
    nodes = [camera]
    inputs = {}; frames = {'raw':[], 'compressed':[]}
    try:
        for transport in frames:
            node=Node('test_demo_'+transport);nodes.append(node)
            args=build_parser().parse_args(['--camera-prefix','/rgbd_compat_test','--image-transport',transport])
            inputs[transport]=RosRGBDInput(node,args)
        end=time.monotonic()+5
        while time.monotonic()<end:
            for node in nodes: rclpy.spin_once(node,timeout_sec=.002)
            for transport,receiver in inputs.items():
                frame=receiver.take_latest()
                if frame is not None: frames[transport].append(frame)
        report={}
        for transport,received in frames.items():
            assert len(received)>10,(transport,len(received))
            for frame in received:
                assert frame.stamp_ns in FakeCamera.stamps
                assert frame.depth_stamp_ns==frame.stamp_ns
                np.testing.assert_allclose(frame.depth_m,scene()[0].depth_m,atol=1e-7)
                np.testing.assert_array_equal(frame.K,scene()[0].K)
            report[transport]={'frames':len(received),'original_stamps':True,'depth_metres_correct':True,
                               'K_correct':True,'counts':dict(inputs[transport].counts)}
        if len(sys.argv)>1: Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        camera.close()
        for node in nodes: node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__': main()
