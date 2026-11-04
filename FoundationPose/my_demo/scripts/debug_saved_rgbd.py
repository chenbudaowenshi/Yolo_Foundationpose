from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT)]
from sb_pose_contract import load_object_profile,RGBDFrame,assess_quality
from sb_vision_pipeline import FoundationPoseBackend
from sb_vision_node import build_parser
args=build_parser().parse_args([])
profile=load_object_profile(args.object_profile)
backend=FoundationPoseBackend(profile,args)
import cv2
out=ROOT/'runs/sb_vision'
K=np.array(json.loads((out/'camera_local.json').read_text())['K'])
color=cv2.imread(str(out/'camera_local_rgb.png'))
depth=cv2.imread(str(out/'camera_local_depth.png'),-1).astype(np.float32)*.001
frame=RGBDFrame(time.time_ns(),time.time_ns(),'head_d435i_optical_frame',color,depth,K)
detections=backend.detect(frame)
print('detector task',backend.model.task,'detections',len(detections))
records=[]
for i,d in enumerate(detections):
    pose=backend.register(frame,d)
    valid,confidence,quality=assess_quality(frame,d,pose,backend.bounds)
    points=backend.estimator.mesh_ori.vertices@pose[:3,:3].T+pose[:3,3]
    points=points[points[:,2]>.01]
    uv=points@K.T; uv=np.rint(uv[:,:2]/uv[:,2:3]).astype(int)
    overlay=color.copy()
    overlay[d.mask.astype(bool)]=(overlay[d.mask.astype(bool)].astype(float)*.6+np.array([60,0,80])).clip(0,255).astype(np.uint8)
    for u,v in uv[::10]:
        if 0<=u<overlay.shape[1] and 0<=v<overlay.shape[0]: cv2.circle(overlay,(u,v),1,(0,255,0),-1)
    cv2.imwrite(str(out/f'debug_registration_{i}.png'),overlay)
    cv2.imwrite(str(out/f'debug_mask_{i}.png'),d.mask.astype(np.uint8)*255)
    records.append(dict(box=d.box.tolist(),mask_source=d.mask_source,detector_confidence=d.confidence,
        valid=valid,confidence=confidence,quality=quality,pose_model=pose.tolist(),
        center=(pose@np.array(profile['T_model_object_center']))[:3,3].tolist()))
(out/'debug_saved_rgbd.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
