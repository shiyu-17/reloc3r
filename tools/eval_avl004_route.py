#!/usr/bin/env python3
"""Evaluate a no-GT Manager route after recovery poses are attached."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--gt-dir',type=Path,required=True); p.add_argument('--sequence',required=True); p.add_argument('--manager-trace',type=Path,required=True); p.add_argument('--recovery',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--start',type=int); p.add_argument('--end',type=int); a=p.parse_args()
    _,_,qt,_,qr=read_manifest(a.data_root,a.sequence); gt,_=tum_camera_poses(a.gt_dir/f'{a.sequence}_gt_poses.txt',qt,base_to_front_rgb_optical(a.data_root/a.sequence),.1)
    manager=[json.loads(x) for x in a.manager_trace.read_text().splitlines()]
    if a.start is not None: manager=[x for x in manager if int(x['query_index']) >= a.start]
    if a.end is not None: manager=[x for x in manager if int(x['query_index']) <= a.end]
    recovery={int(x['query_index']):x for x in (json.loads(line) for line in a.recovery.read_text().splitlines())}
    rows=[]
    for m in manager:
        qi=int(m['query_index']); action=m['action']; pred=None; source=None
        if action=='call_geometry':
            pred=recovery.get(qi,{}).get('pose'); source='geometry'
        elif action in ('accept_consensus','use_memory') and m.get('candidate_pose') is not None:
            pred=m['candidate_pose']; source=action
        metric=None
        if pred is not None:
            t,r=pose_error(np.asarray(pred),gt[qi]); metric={'translation_error_m':t,'rotation_error_deg':r,'success_0.25m_5deg':bool(t<=.25 and r<=5)}
        rows.append({'query_index':qi,'action':action,'source':source,'metric':metric,'evidence':m['evidence'],'query_gt_used_by_route':False})
    def stats(items):
        valid=[x['metric'] for x in items if x['metric'] is not None]; succ=[x['success_0.25m_5deg'] for x in valid]
        return {'frames':len(items),'pose_available':len(valid),'coverage':len(valid)/len(items) if items else 0.0,'success_all_frames':sum(succ)/len(items) if items else 0.0,'success_on_covered':sum(succ)/len(valid) if valid else 0.0,'median_translation_error_m':float(np.median([x['translation_error_m'] for x in valid])) if valid else None,'median_rotation_error_deg':float(np.median([x['rotation_error_deg'] for x in valid])) if valid else None}
    action_stats={a:stats([x for x in rows if x['action']==a]) for a in ('accept_consensus','use_memory','call_geometry','reject')}
    summary={'sequence':a.sequence,'route':stats(rows),'actions':action_stats,'query_gt_used_by_route':False,'memory_future_access':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'summary':summary,'frames':rows},indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
