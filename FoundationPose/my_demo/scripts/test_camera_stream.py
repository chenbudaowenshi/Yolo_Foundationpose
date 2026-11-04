#!/usr/bin/env python3
"""Bounded hardware-camera test: publish RGB-D and verify it through ROS subscriptions."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo,Image,CompressedImage
from std_msgs.msg import String
from run_realsense_viewer import RealSenseViewerNode,build_argument_parser,load_camera_class,acquire_lock,release_lock
from sb_pose_contract import RGBDSynchronizer
from rgbd_transport import image_topics


def main():
    parser=build_argument_parser()
    parser.add_argument('--duration',type=float,default=20)
    parser.add_argument('--report',type=Path,default=Path('camera_test_report.json'))
    args=parser.parse_args()
    if args.duration<=0: parser.error('duration must be positive')
    if not acquire_lock(): return 2
    rclpy.init()
    camera=monitor=None
    try:
        camera=RealSenseViewerNode(args,load_camera_class())
        monitor=Node('camera_stream_probe')
        sync=RGBDSynchronizer(args.frame_id,args.max_rgbd_skew_ms)
        frames=[]; errors=[]; arrivals=[]; ages=[]; payload_bytes=0; capture_status=[]
        def receive(key,msg):
            nonlocal payload_bytes
            if key in ('rgb','depth'): payload_bytes+=len(msg.data)
            try:
                sync.add(key,msg)
                frame=sync.pop_latest()
                if frame:
                    arrivals.append(time.monotonic())
                    ages.append((monitor.get_clock().now().nanoseconds-frame.stamp_ns)/1e9)
                    frames.append({'stamp_ns':frame.stamp_ns,'shape':list(frame.depth_m.shape),
                                   'valid_depth_ratio':float((frame.depth_m>0).mean())})
            except Exception as exc: errors.append(str(exc))
        prefix='/'+args.topic_prefix.strip('/')
        transport='raw' if args.image_transport=='raw' else 'compressed'
        rgb_topic,depth_topic=image_topics(prefix,transport)
        image_type=Image if transport=='raw' else CompressedImage
        for key,topic,kind in [('rgb',rgb_topic,image_type),('depth',depth_topic,image_type),
                             ('rgb_info',prefix+'/color/camera_info',CameraInfo),('depth_info',prefix+'/depth/camera_info',CameraInfo)]:
            monitor.create_subscription(kind,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)
        monitor.create_subscription(String,prefix+'/capture_status',lambda msg:capture_status.append(json.loads(msg.data)),10)
        started=time.monotonic()
        while rclpy.ok() and time.monotonic()-started<args.duration and not camera.fatal_error:
            rclpy.spin_once(camera,timeout_sec=.002)
            rclpy.spin_once(monitor,timeout_sec=.002)
        hz=(len(frames)-1)/(arrivals[-1]-arrivals[0]) if len(frames)>1 else 0
        gap=max(np.diff(arrivals)) if len(arrivals)>1 else None
        age_p95=float(np.percentile(ages,95)) if ages else None
        passed=bool(hz>=10 and gap is not None and gap<.25 and age_p95 is not None and age_p95<=.25
                    and not errors and not camera.fatal_error and all(age>=-.005 for age in ages))
        report={'status':'PASS' if passed else 'FAIL','serial':camera.camera.serial,'frames':len(frames),
                'image_transport':args.image_transport,'received_image_payload_bytes':payload_bytes,
                'received_image_payload_mbps':payload_bytes*8/max(time.monotonic()-started,1e-9)/1e6,
                'capture_status':capture_status[-1] if capture_status else None,
                'hz':hz,'max_gap_s':gap,'capture_age_p95_s':age_p95,'errors':errors,
                'capture_error':camera.fatal_error,'first_frame':frames[0] if frames else None,
                'K':camera.camera.K.tolist(),'D':camera.camera.D,'sdk_distortion_model':camera.camera.sdk_distortion_model,
                'scope':'single-host hardware camera + DDS; excludes cross-host latency and robot TF'}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return 0 if passed else 1
    finally:
        if camera: camera.close(); camera.destroy_node()
        if monitor: monitor.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        release_lock()

if __name__=='__main__': raise SystemExit(main())
