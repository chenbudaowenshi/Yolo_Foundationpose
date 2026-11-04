#!/usr/bin/env python3
"""Subscribe only: verify live RGB-D/IMU capture times and optical-frame alignment."""
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import rclpy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ros_rgbd_input import RosRGBDInput, add_camera_arguments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--imu-timeout',type=float,default=.5)
    parser.add_argument('--gravity-smoothing',type=float,default=.1)
    parser.add_argument('--imu-frame')
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    rclpy.init(domain_id=args.ros_domain_id)
    node=rclpy.create_node('imu_alignment_check')
    receiver=RosRGBDInput(node,args);rows=[]
    try:
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.005)
            frame=receiver.take_latest()
            if frame is not None: rows.append(receiver.imu.match(frame).metadata())
        good=[r for r in rows if r['ready']]
        result={'imu_publishers':node.count_publishers(args.imu_topic),'imu_messages':receiver.imu.count,
                'rgbd_frames':len(rows),'aligned_frames':len(good),
                'reasons':dict(Counter(r['reason'] for r in rows)),
                'absolute_skew_p50_p95_max_ms':np.percentile([abs(r['skew_ms']) for r in good],[50,95,100]).tolist() if good else [],
                'imu_age_p50_p95_max_ms':np.percentile([r['age_s']*1000 for r in good],[50,95,100]).tolist() if good else [],
                'samples':rows}
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
        return 0 if len(good)>=5 and len(good)>=.9*len(rows) else 1
    finally: node.destroy_node();rclpy.shutdown()

if __name__=='__main__':raise SystemExit(main())
