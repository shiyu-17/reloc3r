#!/usr/bin/env python3
"""Post-hoc evaluator for AVL-004 recovery poses (GT is read only here)."""
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--gt-dir',type=Path,required=True); p.add_argument('--sequence',required=True); p.add_argument('--recovery',type=Path,required=True); p.add_argument('--memory-results',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    _,_,qt,_,qr=read_manifest(a.data_root,a.sequence); gt,_=tum_camera_poses(a.gt_dir/f'{a.sequence}_gt_poses.txt',qt,base_to_front_rgb_optical(a.data_root/a.sequence),.1)
    memory={}
    if a.memory_results:
        for line in (a.memory_results/'frame_results.jsonl').read_text().splitlines():
            x=json.loads(line); memory[x['query_index']]=x['methods']['memory3']
    rows=[]
    for line in a.recovery.read_text().splitlines():
        x=json.loads(line); pred=x.get('pose'); metric=None
        if pred is not None:
            t,r=pose_error(np.asarray(pred),gt[x['query_index']]); metric={'translation_error_m':t,'rotation_error_deg':r,'success_0.25m_5deg':bool(t<=.25 and r<=5)}
        rows.append({**x,'metric':metric,'memory3':memory.get(x['query_index']),'gt_used_only_for_metric':True})
    valid=[x['metric'] for x in rows if x['metric'] is not None]; success=[x['success_0.25m_5deg'] for x in valid]
    summary={'sequence':a.sequence,'frames':len(rows),'pose_available':len(valid),'coverage':len(valid)/len(rows) if rows else 0.0,'success_all_frames':sum(success)/len(rows) if rows else 0.0,'success_on_covered':sum(success)/len(valid) if valid else 0.0,'median_translation_error_m':float(np.median([x['translation_error_m'] for x in valid])) if valid else None,'median_rotation_error_deg':float(np.median([x['rotation_error_deg'] for x in valid])) if valid else None,'query_gt_used_by_recovery':False,'gt_used_only_for_metric':True}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'summary':summary,'frames':rows},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
