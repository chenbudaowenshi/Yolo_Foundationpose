#!/usr/bin/env python3
from pathlib import Path
import sys,time,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from RealSenseCamera import RealSenseCamera
from run_realsense_viewer import acquire_lock,release_lock
import cv2

if not acquire_lock(): raise SystemExit(2)
camera=None
try:
    camera=RealSenseCamera()
    rows=[]
    for i in range(35):
        begin=time.perf_counter(); frame=camera.get_frame()
        if frame is None: continue
        rows.append(dict(total=time.perf_counter()-begin,**frame['capture_timings_s'],
             age_ms=(time.time_ns()-frame['color_stamp_ns'])/1e6,stamp=frame['color_stamp_ns'],
             color_metadata=frame['color_metadata'],depth_metadata=frame['depth_metadata']))
    output=Path(__file__).resolve().parents[2]/'runs/sb_vision'
    cv2.imwrite(str(output/'camera_local_rgb.png'),frame['color'])
    cv2.imwrite(str(output/'camera_local_depth.png'),frame['depth'])
    result={'median':{key:float(np.median([r[key] for r in rows[5:]])) for key in ['total','wait','align_and_metadata','age_ms']},
            'last':rows[-1], 'rows':rows}
    (output/'camera_capture_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
finally:
    if camera: camera.stop()
    release_lock()
