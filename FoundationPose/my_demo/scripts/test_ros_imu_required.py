#!/usr/bin/env python3
"""Local DDS IMU alignment/dropout test on private camera topics; no robot output."""
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT/'my_demo/scripts'),str(ROOT/'my_demo/tests')]
from d435i_foundationpose import build_parser
from ros_rgbd_input import RosRGBDInput
from run_realsense_viewer import RealSenseViewerNode, build_argument_parser
from test_ros_handoff import FakeCamera
from autonomous_state import output_status


class Camera(FakeCamera):
    mode='normal'
    def get_frame(self):
        frame=super().get_frame()
        if self.mode=='missing': frame.pop('imu')
        elif self.mode in {'old','skew'}:
            offset=1_000_000_000 if self.mode=='old' else 100_000_000
            for key in ('accel_stamp_ns','gyro_stamp_ns'): frame['imu'][key]-=offset
        elif self.mode=='bad_norm': frame['imu']['accel']=np.zeros(3)
        return frame


def main():
    rclpy.init()
    camera=RealSenseViewerNode(build_argument_parser().parse_args(
        ['--topic-prefix','/imu_required_test','--width','100','--height','100','--publish-fps','20']),Camera)
    node=Node('imu_required_test_receiver')
    args=build_parser().parse_args(['--camera-prefix','/imu_required_test'])
    receiver=RosRGBDInput(node,args)
    report={}
    try:
        for mode in ('normal','missing','old','skew','bad_norm','normal_recovered'):
            Camera.mode='normal' if mode=='normal_recovered' else mode
            begin=time.monotonic(); matches=[]
            while time.monotonic()-begin<1.2:
                rclpy.spin_once(camera,timeout_sec=.001)
                for _ in range(8): rclpy.spin_once(node,timeout_sec=.001)
                frame=receiver.take_latest()
                if frame is None: continue
                match=receiver.imu.match(frame)
                if time.monotonic()-begin>.25:
                    matches.append(match)
                    _,_,valid=output_status('pose_valid',has_pose=True,measured=True,
                        quality_valid=True,observation_age_s=.01,maximum_age_s=.5,imu_ready=match.ready)
                    assert valid==match.ready
            assert len(matches)>=5,(mode,len(matches))
            wanted=mode in {'normal','normal_recovered'}
            assert all(m.ready==wanted for m in matches),(mode,[m.metadata() for m in matches])
            report[mode]={'frames':len(matches),'ready':sum(m.ready for m in matches),
                          'reasons':dict(Counter(m.reason for m in matches))}
        print(json.dumps(report,indent=2))
        if len(sys.argv)>1: Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
    finally:
        camera.close();camera.destroy_node();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
