#!/usr/bin/env python3
"""Read one frame from the running camera and save real YOLO detections."""
import json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT)]
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo,CompressedImage
from sb_pose_contract import RGBDSynchronizer


def main():
    rclpy.init(); node=Node('target_scene_inspector'); sync=RGBDSynchronizer(); captured=[]
    def receive(key,msg):
        sync.add(key,msg)
        frame=sync.pop_latest()
        if frame is not None: captured.append(frame)
    for key,topic,kind in [('rgb','/head_camera/color/image_raw/compressed',CompressedImage),
                          ('depth','/head_camera/depth/image_rect_raw/compressedDepth',CompressedImage),
                          ('rgb_info','/head_camera/color/camera_info',CameraInfo),
                          ('depth_info','/head_camera/depth/camera_info',CameraInfo)]:
        node.create_subscription(kind,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)
    try:
        until=time.monotonic()+15
        while not captured and time.monotonic()<until: rclpy.spin_once(node,timeout_sec=.1)
        if not captured: raise RuntimeError('No synchronized RGB-D from the running camera')
    finally:
        node.destroy_node(); rclpy.shutdown()
    frame=captured[-1]
    output=ROOT/'runs/sb_vision/target_lock_20260919'; output.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output/'scene_rgb.png'),frame.color_bgr)
    cv2.imwrite(str(output/'scene_depth_mm.png'),np.rint(frame.depth_m*1000).astype(np.uint16))
    np.savez_compressed(output/'scene_rgbd.npz',color_bgr=frame.color_bgr,depth_m=frame.depth_m,K=frame.K,
                        stamp_ns=frame.stamp_ns,depth_stamp_ns=frame.depth_stamp_ns,frame_id=frame.frame_id)
    os.environ['D435I_FOPOSE_REEXEC']='1'
    from ultralytics import YOLO
    import d435i_foundationpose as helpers
    args=helpers.build_parser().parse_args([]); args.target_class=1; args.conf=.30; args.imgsz=960
    model=YOLO(str(ROOT/'runs/yolov8/best.pt'))
    detections,_=helpers.find_targets(model,frame.color_bgr,frame.depth_m,1.,args)
    overlay=frame.color_bgr.copy(); items=[]
    for index,detection in enumerate(detections):
        overlay=helpers.annotate_target(overlay,detection)
        x,y=map(int,detection.box[:2]); cv2.putText(overlay,f'#{index}',(x,max(20,y-20)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
        valid=np.asarray(detection.mask,bool)&(frame.depth_m>0)
        items.append({'index':index,'class':detection.class_name,'confidence':detection.confidence,
                      'box':detection.box.tolist(),'valid_depth_pixels':int(valid.sum()),
                      'median_depth_m':float(np.median(frame.depth_m[valid])) if valid.any() else None})
    cv2.imwrite(str(output/'scene_detections.png'),overlay)
    result={'detections':items,'frame_id':frame.frame_id,'stamp_ns':frame.stamp_ns,
            'valid_depth_ratio':float((frame.depth_m>0).mean()),'K':frame.K.tolist()}
    (output/'scene_detection_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
