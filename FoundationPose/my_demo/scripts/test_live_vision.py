#!/usr/bin/env python3
"""Bounded real D435i -> real GPU -> ObjectPose local integration measurement."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'my_demo'))
import rclpy
from rclpy.node import Node
from sb_communication.msg import ObjectPose
from run_realsense_viewer import RealSenseViewerNode,build_argument_parser,load_camera_class,acquire_lock,release_lock
from sb_vision_node import make_node,build_parser
from sb_pose_contract import load_object_profile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=40)
    parser.add_argument('--serial',default='327122073383')
    parser.add_argument('--image-transport',choices=('raw','compressed'),default='compressed')
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--sequence-state-dir',type=Path,default=None)
    parser.add_argument('--sequence-start-after',type=int,default=None)
    parser.add_argument('--acceptance', action='store_true', help='run the supplied static 60-second interface acceptance after startup')
    args=parser.parse_args()
    if not acquire_lock(): return 2
    camera=vision=monitor=None
    acceptance=acceptance_log=None
    rclpy.init()
    out=args.output_dir or ROOT/'runs/sb_vision'/f'live_{args.image_transport}'
    out.mkdir(parents=True,exist_ok=True)
    try:
        ca=build_argument_parser().parse_args(['--serial',args.serial,'--image-transport',args.image_transport])
        camera=RealSenseViewerNode(ca,load_camera_class())
        vision_arguments=['--metrics-jsonl',str(out/'live_metrics.jsonl'),'--image-transport',args.image_transport,
                          '--clock-sync-mode','local']
        if args.sequence_state_dir is not None:
            vision_arguments.extend(['--sequence-state-dir',str(args.sequence_state_dir)])
        if args.sequence_start_after is not None:
            vision_arguments.extend(['--sequence-start-after',str(args.sequence_start_after)])
        va=build_parser().parse_args(vision_arguments)
        vision=make_node(va,load_object_profile(va.object_profile))
        monitor=Node('sb_live_probe')
        samples=[]
        def receive(msg):
            ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            samples.append(dict(stamp_ns=ns,arrival_s=time.monotonic(),delay_s=(time.time_ns()-ns)/1e9,
                 sequence=msg.sequence,position=[msg.pose.position.x,msg.pose.position.y,msg.pose.position.z],
                 quaternion_xyzw=[msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w],confidence=msg.confidence))
        monitor.create_subscription(ObjectPose,'/object_pose',receive,10)
        started=time.monotonic()
        while rclpy.ok() and (time.monotonic()-started<args.duration or (acceptance is not None and acceptance.poll() is None)) and not vision.fatal_error and not camera.fatal_error:
            for node in (camera,vision,monitor): rclpy.spin_once(node,timeout_sec=.001)
            if args.acceptance and acceptance is None and len(samples)>=15:
                acceptance_log=(out/'live_acceptance_static.txt').open('w')
                acceptance=subprocess.Popen([sys.executable,str(ROOT/'sb_vision_handoff_20260910/scripts/vision_interface_acceptance.py'),
                    '--static','--duration','60'],stdout=acceptance_log,stderr=subprocess.STDOUT)
        arrivals=[s['arrival_s'] for s in samples]
        hz=(len(samples)-1)/(arrivals[-1]-arrivals[0]) if len(samples)>1 else 0
        gap=float(max(np.diff(arrivals))) if len(samples)>1 else None
        delays=[s['delay_s'] for s in samples]
        position=np.array([s['position'] for s in samples])
        sigma=(position.std(axis=0)*1000).tolist() if len(samples)>1 else None
        report={'scope':'real local camera and GPU; camera-coordinate consistency only, no external position truth',
                'image_transport':args.image_transport,
                'minimum_requested_duration_s':args.duration,
                'observed_stream_span_s':arrivals[-1]-arrivals[0] if len(arrivals)>1 else 0,
                'received':len(samples),'hz':hz,'max_gap_s':gap,
                'delay_p95_s':float(np.percentile(delays,95)) if delays else None,
                'position_sigma_mm':sigma,'counts':dict(vision.counts),'last':vision.last_result,
                'vision_error':vision.fatal_error,'camera_error':camera.fatal_error,
                'K':camera.camera.K.tolist(),'D':camera.camera.D,'serial':camera.camera.serial}
        report['acceptance_exit_code']=acceptance.poll() if acceptance is not None else None
        report['status']='PASS' if hz>=10 and gap is not None and gap<.25 and max(sigma or [999])<10 and not vision.fatal_error and not camera.fatal_error else 'FAIL'
        if args.acceptance and report['acceptance_exit_code'] != 0: report['status']='FAIL'
        (out/'live_local.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'live_object_pose.json').write_text(json.dumps(samples,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return 0 if report['status']=='PASS' else 1
    finally:
        if acceptance is not None and acceptance.poll() is None:
            acceptance.terminate(); acceptance.wait(timeout=5)
        if acceptance_log: acceptance_log.close()
        if vision: vision.close()
        if camera: camera.close()
        for node in (camera,vision,monitor):
            if node: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        release_lock()

if __name__=='__main__': raise SystemExit(main())
