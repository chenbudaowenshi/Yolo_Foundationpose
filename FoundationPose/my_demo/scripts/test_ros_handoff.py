#!/usr/bin/env python3
"""Local DDS integration test with synthetic camera/pose backend; never moves a robot."""
import argparse
import json
from pathlib import Path
import sys
import time
import tempfile
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT/'my_demo/tests')]
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from sb_communication.msg import ObjectPose
from std_srvs.srv import Trigger
from run_realsense_viewer import RealSenseViewerNode, build_argument_parser
from sb_vision_node import make_node, build_parser
from sb_pose_contract import load_object_profile
from test_sb_handoff import scene, FakeBackend
from rgbd_transport import decode_depth_png, decode_color_jpeg
from sb_pose_contract import decode_image, stamp_ns


class FakeCamera:
    stamps=set()
    visible=True
    slow=False
    def __init__(self, **kwargs):
        self.fx=self.fy=100.; self.cx=self.cy=50.
        self.K=scene()[0].K; self.D=[0.]*5; self.distortion_model='plumb_bob'
        self.depth_scale=.0005; self.serial='SYNTHETIC_TEST'; self.sdk_distortion_model='none'
    def get_frame(self):
        stamp=time.time_ns()-5000000
        frame=scene(stamp)[0]
        self.stamps.add(stamp)
        color=frame.color_bgr.copy()
        color[:16,:16,:]=255 if not self.visible else (100 if self.slow else 0)
        return dict(color=color,depth=np.rint(frame.depth_m/self.depth_scale).astype(np.uint16),
                    color_stamp_ns=stamp,depth_stamp_ns=stamp,depth_scale=self.depth_scale,
                    imu={'accel':np.array([0.,-9.81,0.]),'gyro':np.zeros(3),
                         'accel_stamp_ns':stamp,'gyro_stamp_ns':stamp},
                    color_metadata={'timestamp_domain':'global_time','timestamp_basis':'synthetic'},depth_metadata={})
    def stop(self): pass


class Backend(FakeBackend):
    def __init__(self, profile,args): super().__init__()
    def detect(self, frame):
        return [] if frame.color_bgr[4,4,0]>200 else self.detections
    def register(self, frame, detection, full=False):
        if 80<frame.color_bgr[4,4,0]<120: time.sleep(.45)
        return super().register(frame,detection,full)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-transport',choices=('raw','compressed','both'),default='compressed')
    parser.add_argument('--report',type=Path,default=None)
    options=parser.parse_args()
    sequence_directory=tempfile.TemporaryDirectory(prefix='sb_ros_sequence_',dir='/tmp')
    rclpy.init()
    cam_args=build_argument_parser().parse_args(['--width','100','--height','100','--publish-fps','20',
                                                '--image-transport',options.image_transport])
    receiver_transport='raw' if options.image_transport=='raw' else 'compressed'
    args=build_parser().parse_args(['--image-transport',receiver_transport,'--clock-sync-mode','local',
                                   '--sequence-state-dir',sequence_directory.name,'--sequence-start-after','0'])
    camera=RealSenseViewerNode(cam_args,FakeCamera)
    vision=make_node(args,load_object_profile(args.object_profile),Backend)
    monitor=Node('sb_handoff_test_monitor')
    samples=[]; infos=[]; depth_infos=[]; compressed_depths=[]; compressed_colors=[]; raw_depths=[]
    monitor.create_subscription(ObjectPose,'/object_pose',samples.append,10)
    monitor.create_subscription(CameraInfo,'/head_camera/color/camera_info',infos.append,qos_profile_sensor_data)
    monitor.create_subscription(CameraInfo,'/head_camera/depth/camera_info',depth_infos.append,qos_profile_sensor_data)
    if receiver_transport=='compressed':
        monitor.create_subscription(CompressedImage,'/head_camera/depth/image_rect_raw/compressedDepth',compressed_depths.append,qos_profile_sensor_data)
        monitor.create_subscription(CompressedImage,'/head_camera/color/image_raw/compressed',compressed_colors.append,qos_profile_sensor_data)
    if options.image_transport=='both':
        monitor.create_subscription(Image,'/head_camera/depth/image_rect_raw',raw_depths.append,qos_profile_sensor_data)
    nodes=[camera,vision,monitor]
    def spin(seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            for node in nodes: rclpy.spin_once(node,timeout_sec=.002)
            if vision.fatal_error: raise RuntimeError(vision.fatal_error)
    try:
        spin(3)
        assert len(samples)>=15, f'only {len(samples)} messages received'
        for msg in samples:
            stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            assert stamp in FakeCamera.stamps, 'timestamp not preserved'
            assert msg.header.frame_id=='head_d435i_optical_frame'
            np.testing.assert_allclose([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z],[0,0,.7],atol=1e-6)
            assert .2<=msg.confidence<.95
            assert list(msg.covariance)==[0.]*36
        assert infos and infos[-1].k[0]==100
        if receiver_transport=='compressed':
            assert compressed_depths and compressed_colors
            color_info_by_stamp={stamp_ns(m.header):m for m in infos}
            depth_info_by_stamp={stamp_ns(m.header):m for m in depth_infos}
            common={stamp_ns(m.header) for m in compressed_depths}&set(depth_info_by_stamp)
            assert common, 'compressed depth has no matching CameraInfo'
            expected=np.rint(scene()[0].depth_m*1000).astype(np.uint16)
            for msg in compressed_depths:
                np.testing.assert_array_equal(decode_depth_png(msg),expected)
                assert stamp_ns(msg.header) in FakeCamera.stamps
            assert any(stamp_ns(m.header) in color_info_by_stamp for m in compressed_colors)
            assert decode_color_jpeg(compressed_colors[0]).shape==(100,100,3)
            for ci in infos+depth_infos:
                assert ci.header.frame_id=='head_d435i_optical_frame'
                np.testing.assert_array_equal(np.asarray(ci.k).reshape(3,3),camera.camera.K)
                assert list(ci.d)==camera.camera.D
            if options.image_transport=='both':
                raw_by_stamp={stamp_ns(m.header):m for m in raw_depths}
                compared=0
                for msg in compressed_depths:
                    raw=raw_by_stamp.get(stamp_ns(msg.header))
                    if raw is not None:
                        np.testing.assert_array_equal(decode_depth_png(msg),decode_image(raw))
                        assert raw.header==msg.header
                        compared+=1
                assert compared>0, 'both mode did not emit matching raw/compressed frames'
        FakeCamera.visible=False
        spin(.4); count=len(samples); spin(.4)
        assert len(samples)==count,'lost object still published'
        FakeCamera.visible=True
        spin(.4)
        assert len(samples)>count,'did not recover after occlusion'
        client=monitor.create_client(Trigger,'/sb_foundationpose/reset_target')
        assert client.wait_for_service(timeout_sec=2)
        FakeCamera.slow=True
        future=client.call_async(Trigger.Request())
        spin(1.8)
        assert future.done() and future.result().success
        assert vision.counts['stale_future_or_duplicate_result']>=1,'slow registration was not rejected'
        seq=[s.sequence for s in samples]
        assert all(a<b for a,b in zip(seq,seq[1:])), 'sequence reset or duplicated'
        result={'status':'PASS','kind':'synthetic camera and pose backend over real CycloneDDS',
                'image_transport':options.image_transport,
                'received':len(samples),'counts':dict(vision.counts),'checks':['capture stamp preserved','mm depth scaling',
                'geometric center','confidence/covariance','occlusion stops output','reacquisition','slow registration rejected',
                'reset service preserves sequence','RGB-D CameraInfo pairing','transport mode and depth values']}
        output=ROOT/'runs/sb_vision'; output.mkdir(parents=True,exist_ok=True)
        report=options.report or output/f'ros_smoke_{options.image_transport}.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        vision.close(); camera.close()
        for node in nodes: node.destroy_node()
        rclpy.shutdown()
        sequence_directory.cleanup()

if __name__=='__main__': main()
