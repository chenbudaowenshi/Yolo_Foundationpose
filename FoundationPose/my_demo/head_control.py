#!/usr/bin/env python3
"""真机 ELF3 头部扫描和瓶子检测。"""
import argparse, math, threading, time
from pathlib import Path
import cv2, numpy as np, rclpy
from communication.msg import ActuatorCmds
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO

TOPIC = '/hardware/actuators_cmds_override'
CAMERA = '/hardware/head_depth_camera/color/image_raw'
MODEL = Path('/home/robot/FoundationPose/ultralytics/ultralytics/yolov8n-seg.pt')

def decode(msg):
    enc = str(msg.encoding).lower(); h,w,step = int(msg.height),int(msg.width),int(msg.step)
    if enc not in ('rgb8','bgr8'): raise ValueError(f'unsupported encoding: {msg.encoding}')
    a=np.frombuffer(msg.data,dtype=np.uint8,count=step*h).reshape(h,step)[:,:w*3].reshape(h,w,3)
    return np.ascontiguousarray(a[:,:,::-1] if enc=='rgb8' else a)

class HeadYolo(Node):
    def __init__(self,a):
        super().__init__('elf3_real_head_yolo'); self.a=a; self.yaw=0.; self.locked=False; self.frame=None; self.mu=threading.Lock(); self.stop=threading.Event(); self.hits=0; self.box=None
        self.pub=self.create_publisher(ActuatorCmds,TOPIC,qos_profile_sensor_data)
        self.sub=self.create_subscription(Image,CAMERA,self.image,qos_profile_sensor_data)
        self.timer=self.create_timer(1./a.rate,self.publish)
        self.model=YOLO(str(MODEL)); self.worker=threading.Thread(target=self.detect,daemon=True); self.worker.start()
        self.get_logger().info(f'YOLO camera={CAMERA}, override={TOPIC}; sweeping until bottle is detected')
    def image(self,msg):
        try:
            with self.mu: self.frame=decode(msg)
        except Exception as e: self.get_logger().warning(f'image decode failed: {e}')
    def publish(self):
        if not self.locked: self.yaw=self.a.amplitude*math.sin(2*math.pi*(time.monotonic()-self.a.t0)/self.a.period)
        m=ActuatorCmds(); m.actuators_name=['head_z_joint','head_y_joint']; m.pos=[self.yaw,self.a.pitch]; m.kp=[self.a.kp]*2; m.kd=[self.a.kd]*2; m.vel=[0.,0.]; m.torque=[0.,0.]; self.pub.publish(m)
    def detect(self):
        while not self.stop.wait(1./self.a.detect_rate):
            with self.mu: frame=None if self.frame is None else self.frame.copy()
            if frame is None: continue
            try:
                r=self.model.predict(frame,conf=self.a.confidence,imgsz=self.a.image_size,verbose=False)[0]; best=None
                for b in r.boxes:
                    if r.names[int(b.cls[0])].lower()!=self.a.target: continue
                    s=float(b.conf[0]); box=tuple(int(v) for v in b.xyxy[0].tolist())
                    if best is None or s>best[0]: best=(s,box)
                if best: self.hits+=1; self.box=best[1]
                else: self.hits=0; self.box=None
                if self.hits>=self.a.required and not self.locked:
                    self.locked=True; self.get_logger().info(f'detected {self.a.target}, locking yaw={self.yaw:+.3f} rad')
                if self.a.show:
                    v=frame.copy()
                    if self.box: cv2.rectangle(v,self.box,(0,255,0),3)
                    cv2.putText(v,f"{self.a.target} {'LOCKED' if self.locked else 'SCANNING'} yaw={self.yaw:+.2f}",(15,35),0,.8,(0,255,255),2)
                    cv2.imshow('ELF3 real head / YOLO',v)
                    if cv2.waitKey(1)&255 in (27,ord('q')): rclpy.shutdown(); return
            except Exception as e: self.get_logger().warning(f'YOLO failed: {e}')
    def close(self):
        self.stop.set(); self.worker.join(timeout=2.); [self.pub.publish(ActuatorCmds()) for _ in range(3)]; cv2.destroyAllWindows()

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--target',default='bottle'); p.add_argument('--amplitude',type=float,default=.45); p.add_argument('--period',type=float,default=4.); p.add_argument('--pitch',type=float,default=0.); p.add_argument('--kp',type=float,default=16.747); p.add_argument('--kd',type=float,default=1.066); p.add_argument('--rate',type=float,default=20.); p.add_argument('--detect-rate',type=float,default=8.); p.add_argument('--confidence',type=float,default=.5); p.add_argument('--required',type=int,default=2); p.add_argument('--image-size',type=int,default=640); p.add_argument('--duration',type=float,default=None); p.add_argument('--no-show',dest='show',action='store_false'); a=p.parse_args(argv); a.t0=time.monotonic()
    if abs(a.amplitude)>1.57 or abs(a.pitch)>.785: raise SystemExit('head target exceeds limit')
    rclpy.init(); n=None
    try:
        n=HeadYolo(a); end=None if a.duration is None else time.monotonic()+a.duration
        while rclpy.ok() and (end is None or time.monotonic()<end) and not n.locked: rclpy.spin_once(n,timeout_sec=.1)
        while rclpy.ok() and end is None and n.locked: rclpy.spin_once(n,timeout_sec=.1)
    except KeyboardInterrupt: pass
    finally:
        if n: n.close(); n.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return 0
if __name__=='__main__': raise SystemExit(main())
