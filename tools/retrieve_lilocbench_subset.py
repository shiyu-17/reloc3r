#!/usr/bin/env python3
"""Compute mapping-only NetVLAD retrieval for an explicit query subset.

The subset is for AVL-004 reference re-ranking; no query GT is imported.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import read_manifest, retrieval_tensor
from reloc3r.image_retrieval.netvlad import NetVLAD

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--sequence',required=True); p.add_argument('--indices',nargs='+',type=int); p.add_argument('--indices-file',type=Path); p.add_argument('--topk',type=int,default=100); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    if (a.indices is None) == (a.indices_file is None):
        p.error('provide exactly one of --indices or --indices-file')
    if a.indices_file is not None:
        a.indices = [int(x) for x in a.indices_file.read_text().split()]
    d=torch.device(a.device); _,db,_,_,_=read_manifest(a.data_root,'mapping'); _,qf,_,_,_=read_manifest(a.data_root,a.sequence)
    ext=NetVLAD(NetVLAD.default_conf).eval().to(d)
    dbd=[]
    for f in tqdm(db,desc='mapping descriptors'): dbd.append(ext({'image':retrieval_tensor(f['path'],d)})['global_descriptor'].cpu().numpy()[0])
    qd=[]
    for i in a.indices: qd.append(ext({'image':retrieval_tensor(qf[i]['path'],d)})['global_descriptor'].cpu().numpy()[0])
    dbd=np.asarray(dbd,np.float32); qd=np.asarray(qd,np.float32); scores=qd@dbd.T; order=np.argsort(-scores,axis=1)[:,:a.topk]
    a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,mapping_indices=order.astype(np.int32),mapping_scores=np.take_along_axis(scores,order,axis=1).astype(np.float32),query_indices=np.asarray(a.indices,np.int32))
    a.output.with_suffix('.json').write_text(json.dumps({'sequence':a.sequence,'query_indices':a.indices,'topk':a.topk,'mapping_only':True,'query_gt_read':False},indent=2)+'\n')
    print(json.dumps({'frames':len(a.indices),'topk':a.topk,'output':str(a.output),'query_gt_read':False},indent=2))
if __name__=='__main__': main()
