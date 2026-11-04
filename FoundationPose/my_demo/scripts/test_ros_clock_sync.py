#!/usr/bin/env python3
"""Local NTP + synthetic camera + ROS pose test; no robot and no OS clock changes."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time
import tempfile
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'my_demo'), str(ROOT/'my_demo/tests'), str(ROOT/'my_demo/scripts')]
import rclpy
from rclpy.node import Node
from sb_communication.msg import ObjectPose
from std_srvs.srv import Trigger
from clock_sync import ClockSyncMonitor
from local_ntp_test_server import LocalNtpTestServer
from test_ros_handoff import FakeCamera
from test_sb_handoff import FakeBackend
from run_realsense_viewer import RealSenseViewerNode, build_argument_parser
from sb_vision_node import make_node, build_parser
from sb_pose_contract import load_object_profile


class Backend(FakeBackend):
    slow_next = False
    slow_started = threading.Event()
    def __init__(self, profile, args): super().__init__()
    def register(self, frame, detection, full=False):
        if Backend.slow_next:
            Backend.slow_next = False
            Backend.slow_started.set()
            time.sleep(1.5)
        return super().register(frame, detection, full)


def main():
    sequence_directory = tempfile.TemporaryDirectory(prefix='sb_clock_sequence_', dir='/tmp')
    source = LocalNtpTestServer()
    rclpy.init()
    camera = vision = monitor = None
    report = {'scope':'loopback simulated NTP source; synthetic camera/pose backend; real CycloneDDS',
              'robot_connected':False,'os_clock_changed':False,'scenarios':[]}
    try:
        camera = RealSenseViewerNode(build_argument_parser().parse_args(
            ['--width','100','--height','100','--publish-fps','20']), FakeCamera)
        args = build_parser().parse_args(['--clock-sync-mode','test','--clock-server','127.0.0.1',
                                         '--clock-port',str(source.port), '--sequence-state-dir',sequence_directory.name,
                                         '--sequence-start-after','0'])
        def clock_factory(settings):
            return ClockSyncMonitor(replace(settings, interval_s=.1,timeout_s=.03,samples=3,
                                            max_status_age_s=.5,query_timeout_s=2))
        vision = make_node(args,load_object_profile(args.object_profile),Backend,clock_factory)
        monitor = Node('clock_sync_test_monitor')
        samples=[]
        monitor.create_subscription(ObjectPose,'/object_pose',samples.append,10)
        def spin(seconds):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                for node in (camera,vision,monitor): rclpy.spin_once(node,timeout_sec=.001)
                if vision.fatal_error: raise RuntimeError(vision.fatal_error)
        def until(predicate, timeout=5):
            deadline=time.monotonic()+timeout
            while not predicate() and time.monotonic()<deadline: spin(.02)
            assert predicate(), f'timed out; status={vision.clock_status}, samples={len(samples)}'
        def verify_paused():
            until(lambda:not vision.clock_guard.ready)
            spin(.15)
            count=len(samples)
            spin(.3)
            assert len(samples)==count,'pose emitted while clock was unsynchronized'
            return dict(vision.clock_status)
        until(lambda:len(samples)>=10)
        report['scenarios'].append({'offset_ms':0,'published':True,'clock':dict(vision.clock_status)})
        for offset in (20,-20,400,-400):
            source.offset_ms=offset
            bad=verify_paused()
            assert bad['reason']=='clock_error_bound_exceeded',bad
            assert abs(bad['offset_ms']-offset)<2,bad
            count=len(samples)
            source.offset_ms=0
            until(lambda:len(samples)>=count+5)
            report['scenarios'].append({'offset_ms':offset,'paused':True,'recovered':True,'clock':bad})
        source.synchronized=False
        bad=verify_paused()
        assert bad['reason']=='ntp_probe_failed',bad
        source.synchronized=True
        count=len(samples); until(lambda:len(samples)>=count+5)
        report['scenarios'].append({'source_unsynchronized':True,'paused':True,'recovered':True})
        source.respond=False
        bad=verify_paused()
        assert bad['reason']=='ntp_probe_failed',bad
        source.respond=True
        count=len(samples); until(lambda:len(samples)>=count+5)
        report['scenarios'].append({'source_unreachable':True,'paused':True,'recovered':True})
        # A job started before loss of alignment must not escape after recovery.
        client=monitor.create_client(Trigger,'/sb_foundationpose/reset_target')
        assert client.wait_for_service(timeout_sec=2)
        Backend.slow_next=True; Backend.slow_started.clear()
        future=client.call_async(Trigger.Request())
        until(lambda:Backend.slow_started.is_set())
        source.offset_ms=20
        until(lambda:not vision.clock_guard.ready)
        source.offset_ms=0
        until(lambda:vision.clock_guard.ready)
        until(lambda:vision.counts['old_clock_epoch']>=1)
        assert future.done() and future.result().success
        until(lambda:len(samples)>=count+10)
        sequences=[msg.sequence for msg in samples]
        assert all(b>a for a,b in zip(sequences,sequences[1:]))
        assert all(msg.header.stamp.sec*10**9+msg.header.stamp.nanosec in FakeCamera.stamps for msg in samples)
        report.update(status='PASS',received=len(samples),counts=dict(vision.counts),
                      capture_stamp_preserved=True,sequence_preserved=True,old_clock_epoch_discarded=True)
        output=ROOT/'runs/sb_vision/clock_sync_20260919'; output.mkdir(parents=True,exist_ok=True)
        (output/'local_ntp_ros_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return 0
    finally:
        if vision: vision.close(); vision.destroy_node()
        if camera: camera.close(); camera.destroy_node()
        if monitor: monitor.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        source.close()
        sequence_directory.cleanup()


if __name__=='__main__': raise SystemExit(main())
