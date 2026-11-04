#!/usr/bin/env python3
"""Real camera + real GPU metrics, publishing only to private test pose/status topics."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sb_communication.msg import ObjectPose

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'my_demo'))
from sb_vision_node import make_node,build_parser
from sb_vision_pipeline import FoundationPoseBackend
from sb_pose_contract import load_object_profile
from run_realsense_viewer import RealSenseViewerNode,build_argument_parser,load_camera_class,acquire_lock,release_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=50)
    parser.add_argument('--target-class',type=int,default=0)
    parser.add_argument('--resets',type=int,default=3)
    parser.add_argument('--start-camera',action='store_true')
    parser.add_argument('--serial',default='327122073383')
    parser.add_argument('--force-full-search',action='store_true',help='benchmark the real conservative fallback on every registration')
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    locked=False;camera=vision=monitor=None
    rclpy.init()
    samples=[];reset_requests=[];next_reset=None;future=None
    try:
        if args.start_camera:
            if not acquire_lock(): raise RuntimeError('Camera is already owned; subscribe instead')
            locked=True
            camera=RealSenseViewerNode(build_argument_parser().parse_args(['--serial',args.serial]),load_camera_class())
        va=build_parser().parse_args([
            '--clock-sync-mode','local','--target-class',str(args.target_class),
            '--node-name','sb_metrics_test','--object-pose-topic','/sb_metrics_test/object_pose',
            '--status-topic','/sb_metrics_test/status','--metrics-source','live_camera',
            '--metrics-jsonl',str(out/'events.jsonl'),'--metrics-summary',str(out/'summary.json'),
            '--sequence-state-dir',str(out/'sequence_state'),'--sequence-start-after','0'])
        class FullSearchBackend(FoundationPoseBackend):
            def register(self, frame, detection, force_full=False):
                return super().register(frame,detection,force_full=True)
        vision=make_node(va,load_object_profile(va.object_profile),
                         backend_factory=FullSearchBackend if args.force_full_search else None)
        vision.performance.context.update(backend='FoundationPoseBackend',
            benchmark_force_full_search=args.force_full_search,
            benchmark_stimulus='manual target reset on stationary real object')
        vision._write_event({'record_type':'benchmark_context','context':vision.performance.context})
        monitor=Node('sb_metrics_test_observer')
        def receive(msg):
            samples.append({'arrival_mono_ns':time.monotonic_ns(),'stamp_ns':msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                            'sequence':msg.sequence,'position_m':[msg.pose.position.x,msg.pose.position.y,msg.pose.position.z]})
        monitor.create_subscription(ObjectPose,'/sb_metrics_test/object_pose',receive,10)
        client=monitor.create_client(Trigger,'/sb_metrics_test/reset_target')
        nodes=[n for n in (camera,vision,monitor) if n is not None]
        started=time.monotonic();saved=False
        while time.monotonic()-started<args.duration and not vision.fatal_error and not (camera and camera.fatal_error):
            for node in nodes: rclpy.spin_once(node,timeout_sec=.001)
            if not saved and vision.latest is not None:
                cv2.imwrite(str(out/'scene.png'),vision.latest.color_bgr);saved=True
            if samples and next_reset is None: next_reset=time.monotonic()+5
            if future is not None and future.done():
                response=future.result();reset_requests[-1]['accepted']=response.success;future=None
            if (next_reset is not None and time.monotonic()>=next_reset and len(reset_requests)<args.resets
                and future is None and client.service_is_ready()):
                reset_requests.append({'request_mono_ns':time.monotonic_ns()})
                future=client.call_async(Trigger.Request());next_reset=time.monotonic()+8
        vision.close()
        summary=json.loads((out/'summary.json').read_text())
        report={'scope':'live camera + real YOLO/FoundationPose; local ROS only; private pose topic, no robot motion',
                'published_samples':len(samples),'reset_requests':reset_requests,'vision_error':vision.fatal_error,
                'camera_error':camera.fatal_error if camera else None,
                'metrics_source':summary['context']['input_source'],
                'event_counts':{k:v['attempts'] for k,v in summary['operations'].items()},
                'gpu_memory':summary['gpu_memory']}
        report['pass']=bool(len(samples)>20 and summary['operations']['initial_registration']['attempts']==1
            and summary['operations']['reregistration']['attempts']>=args.resets
            and summary['operations']['tracking']['attempts']>20 and not vision.fatal_error
            and all(r.get('accepted') for r in reset_requests) and len(reset_requests)==args.resets)
        if args.force_full_search:
            executed=sum(summary['operations'][k]['full_search_executions'] for k in ('initial_registration','reregistration'))
            report['pass']=report['pass'] and executed>=1+args.resets
        (out/'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'published_poses.json').write_text(json.dumps(samples,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return 0 if report['pass'] else 1
    finally:
        if vision: vision.close()
        if camera: camera.close()
        for node in (camera,vision,monitor):
            if node: node.destroy_node()
        rclpy.shutdown()
        if locked: release_lock()

if __name__=='__main__':raise SystemExit(main())
