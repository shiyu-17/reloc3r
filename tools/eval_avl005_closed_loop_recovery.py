#!/usr/bin/env python3
"""Evaluate an existing closed-loop action trace with a replacement no-GT recovery cache."""
import argparse, json
from pathlib import Path
import numpy as np
from tools.eval_lilocbench_reloc3r import base_to_front_rgb_optical, pose_error, read_manifest, tum_camera_poses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--gt-dir', type=Path, required=True)
    p.add_argument('--sequence', required=True); p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    _, _, times, _, root = read_manifest(a.data_root, a.sequence)
    gt, _ = tum_camera_poses(a.gt_dir / f'{a.sequence}_gt_poses.txt', times,
                             base_to_front_rgb_optical(root), .1)
    trace = [json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
    recovery = {int(x['query_index']): x for x in
                (json.loads(x) for x in a.recovery.read_text().splitlines() if x.strip())}
    rows=[]
    for row in trace:
        qi=int(row['query_index'])
        if row['action'] == 'call_geometry':
            # Preserve the original GT-free verification decision.  A
            # replacement recovery cache must not turn a rejected candidate
            # into an accepted pose merely because it contains a pose field.
            pred = recovery.get(qi,{}).get('pose') if row.get('geometry_verified', False) else None
        else:
            pred = row.get('pose')
        metric=None
        if pred is not None:
            t,r=pose_error(np.asarray(pred,dtype=np.float64),gt[qi])
            metric={'translation_error_m':t,'rotation_error_deg':r,
                    'success_0.25m_5deg':bool(t<=.25 and r<=5)}
        rows.append({'query_index':qi,'action':row['action'],'metric':metric,
                     'query_gt_used_by_route':False})
    valid=[x['metric'] for x in rows if x['metric'] is not None]
    report={'sequence':a.sequence,'route':{'frames':len(rows),'pose_available':len(valid),
      'coverage':len(valid)/len(rows),'success_all_frames':sum(x['success_0.25m_5deg'] for x in valid)/len(rows),
      'success_on_covered':sum(x['success_0.25m_5deg'] for x in valid)/len(valid) if valid else 0},
      'query_gt_used_by_route':False,'memory_future_access':False,'frames':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['route'],indent=2))

if __name__=='__main__': main()
