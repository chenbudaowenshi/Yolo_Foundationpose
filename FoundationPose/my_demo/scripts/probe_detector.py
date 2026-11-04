from pathlib import Path
import json,os,sys
os.environ['D435I_FOPOSE_REEXEC']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO
root=Path(__file__).resolve().parents[2]
model=YOLO(str(root/'runs/yolov8/best.pt'))
results=[]
for size in (640,960):
    r=model.predict(str(root/'runs/sb_vision/camera_local_rgb.png'),imgsz=size,conf=.10,device='0',verbose=False)[0]
    results.append(dict(imgsz=size,boxes=r.boxes.xyxy.cpu().tolist(),confidence=r.boxes.conf.cpu().tolist(),classes=r.boxes.cls.cpu().tolist()))
report={'class_names':model.names,'results':results}
(root/'runs/sb_vision/detector_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
