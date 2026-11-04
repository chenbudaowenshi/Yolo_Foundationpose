#!/usr/bin/env python3
"""Exercise real YOLO/FP CUDA kernels and the current mesh on synthetic known geometry."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT)]
from sb_pose_contract import load_object_profile,RGBDFrame,assess_quality
from sb_vision_pipeline import FoundationPoseBackend
from sb_vision_node import build_parser


def main():
    args=build_parser().parse_args([])
    profile=load_object_profile(args.object_profile)
    backend=FoundationPoseBackend(profile,args)
    import cv2
    import torch
    from Utils import nvdiffrast_render
    import trimesh
    mesh=trimesh.load(profile['mesh_path'],process=False)
    K=np.array([[600.,0,319.5],[0,600,239.5],[0,0,1]])
    gt=np.array([[1.,0,0,0],[0,0,-1,.07975],[0,1,0,.65],[0,0,0,1]])
    def render(pose):
        rgb,depth,_=nvdiffrast_render(K=K,H=480,W=640,ob_in_cams=torch.as_tensor(pose[None],dtype=torch.float32,device='cuda'),
                                    glctx=backend.estimator.glctx,mesh=mesh,use_light=True)
        color=(rgb[0].detach().cpu().numpy()*255).astype(np.uint8)
        z=depth[0].detach().cpu().numpy().astype(np.float32)
        frame=RGBDFrame(time.time_ns(),time.time_ns(),'head_d435i_optical_frame',color[:,:,::-1].copy(),z,K)
        v,u=np.nonzero(z>0)
        detection=NS(mask=z>0,box=np.array([u.min(),v.min(),u.max()+1,v.max()+1],float),confidence=.95)
        return frame,detection
    frame,detection=render(gt)
    # A real saved photograph exercises the actual detector independently; it
    # lacks archived camera intrinsics, so no metric pose claim is made for it.
    real=cv2.imread(str(ROOT/'demo_data/bottle/image/frame_000001.png'))
    detector_report={}
    if real is not None:
        photo=RGBDFrame(time.time_ns(),time.time_ns(),'head_d435i_optical_frame',real,np.full(real.shape[:2],.65,np.float32),K)
        start=time.perf_counter(); observations=backend.detect(photo)
        detector_report={'seconds':time.perf_counter()-start,'detections':len(observations),
                         'classes':[d.class_name for d in observations],'confidence':[d.confidence for d in observations],
                         'note':'YOLO photograph smoke only; placeholder depth is not used for metric pose validation'}
    torch.cuda.synchronize()
    start=time.perf_counter(); pose=backend.register(frame,detection)
    torch.cuda.synchronize(); register_s=time.perf_counter()-start
    center=np.asarray(profile['T_model_object_center'])
    rows=[]
    for i in range(8):
        current=gt.copy(); current[0,3]+=.001*i
        sample,det=render(current)
        start=time.perf_counter(); tracked=backend.track(sample)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        ok,conf,quality=assess_quality(sample,det,tracked,backend.bounds)
        error=float(np.linalg.norm((tracked@center)[:3,3]-(current@center)[:3,3])*1000)
        axis=float(np.degrees(np.arccos(np.clip(abs(tracked[:3,2]@current[:3,2]),0,1))))
        rows.append(dict(track_s=elapsed,center_error_mm=error,axis_error_deg=axis,quality_pass=ok,quality=quality))
    report={'kind':'real GPU inference on synthetic current-bottle RGB-D; not real-camera accuracy',
            'gpu':torch.cuda.get_device_name(0),'register_s':register_s,
            'registration_center_error_mm':float(np.linalg.norm((pose@center)[:3,3]-(gt@center)[:3,3])*1000),
            'track_samples':rows,'detector_photo':detector_report,
            'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/2**20}
    report['status']='PASS' if all(r['quality_pass'] and r['center_error_mm']<20 for r in rows) else 'FAIL'
    out=ROOT/'runs/sb_vision'; out.mkdir(parents=True,exist_ok=True)
    (out/'gpu_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    cv2.imwrite(str(out/'synthetic_rgb.png'),frame.color_bgr)
    cv2.imwrite(str(out/'synthetic_depth_mm.png'),np.rint(frame.depth_m*1000).astype(np.uint16))
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
