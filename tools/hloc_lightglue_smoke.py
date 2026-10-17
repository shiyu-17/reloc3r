#!/usr/bin/env python3
"""HLoc-style SuperPoint+LightGlue pair audit (no GT)."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--query',type=Path,required=True); p.add_argument('--references',nargs='+',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    d=torch.device(a.device)
    extractor=SuperPoint(max_num_keypoints=2048).eval().to(d)
    matcher=LightGlue(features='superpoint').eval().to(d)
    q=load_image(a.query).to(d)
    with torch.inference_mode(): fq=extractor.extract(q)
    rows=[]
    for ref in a.references:
        r=load_image(ref).to(d)
        with torch.inference_mode():
            fr=extractor.extract(r)
            m=matcher({'image0':fq,'image1':fr})
        matches=m['matches'][0].detach().cpu().numpy()
        qxy=fq['keypoints'][0].detach().cpu().numpy()[matches[:,0]]
        rxy=fr['keypoints'][0].detach().cpu().numpy()[matches[:,1]]
        if len(matches)>=8:
            K=np.array([[382.3815308,0,326.0284119],[0,382.1210022,242.5746002],[0,0,1]],np.float64)
            E,mask=cv2.findEssentialMat(qxy,rxy,K,method=cv2.RANSAC,prob=.999,threshold=1.5)
            inliers=int(mask.sum()) if mask is not None else 0
        else: inliers=0
        rows.append({'reference':str(ref),'keypoints_query':int(len(fq['keypoints'][0])),'keypoints_reference':int(len(fr['keypoints'][0])),'matches':int(len(matches)),'essential_inliers':inliers,'gt_read':False})
        print(json.dumps(rows[-1]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main()
