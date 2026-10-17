#!/usr/bin/env python3
"""HLoc-style SuperPoint+LightGlue map recovery for AVL-004 (no query GT)."""
import argparse, json, sys, time
from itertools import combinations
from pathlib import Path
import cv2, numpy as np, torch, yaml
from scipy.spatial.transform import Rotation
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def read_manifest(root, sequence):
    seqroot = root / sequence; data = json.loads((seqroot / 'manifest.json').read_text())
    frames = [{'timestamp': float(x['timestamp']), 'path': seqroot / x['path']} for x in data['frames']]
    return data, frames, np.asarray([x['timestamp'] for x in frames]), None, seqroot

def base_to_front_rgb_optical(root):
    edges = yaml.safe_load((root / 'transformations.yaml').read_text()); graph = {}
    for e in edges:
        q = [e['rotation'][k] for k in ('x','y','z','w')]; T = np.eye(4); T[:3,:3] = Rotation.from_quat(q).as_matrix(); T[:3,3] = [e['translation'][k] for k in ('x','y','z')]
        graph.setdefault(e['parent'], []).append((e['child'], T)); graph.setdefault(e['child'], []).append((e['parent'], np.linalg.inv(T)))
    todo=[('base_link',np.eye(4))]; seen=set()
    while todo:
        f,T=todo.pop(0)
        if f=='camera_front_color_optical_frame': return T
        if f in seen: continue
        seen.add(f)
        for child,U in graph.get(f,[]):
            if child not in seen: todo.append((child,T@U))
    raise RuntimeError('camera extrinsic not found')

def tum_camera_poses(path, times, base_from_camera, max_delta):
    raw=np.loadtxt(path); out=[]; deltas=[]
    for t in times:
        right=np.searchsorted(raw[:,0],t); choices=[i for i in (right-1,right) if 0<=i<len(raw)]
        i=min(choices,key=lambda j:abs(raw[j,0]-t)); delta=abs(raw[i,0]-t)
        if delta>max_delta: raise RuntimeError('GT sync audit failed')
        T=np.eye(4); T[:3,:3]=Rotation.from_quat(raw[i,4:8]).as_matrix(); T[:3,3]=raw[i,1:4]; out.append(T@base_from_camera); deltas.append(delta)
    return np.asarray(out),np.asarray(deltas)

def load_relative(root, n):
    out=[None]*n
    for path in sorted(root.glob('relative_poses_shard_*_of_*.npz')):
        z=np.load(path)
        for j,qi in enumerate(z['query_indices']): out[int(qi)]={'indices':z['mapping_indices'][j].astype(np.int64),'relative':z['relative_poses'][j].astype(np.float64)}
    if any(x is None for x in out): raise RuntimeError('missing relative pose shard')
    return out

def K_from(path):
    d = yaml.safe_load(path.read_text())
    return np.asarray(d['K'], np.float64).reshape(3, 3), np.asarray(d.get('distortion_coefficients', []), np.float64).reshape(-1)

def P_from(c, K):
    R = c[:3, :3].T
    return K @ np.column_stack((R, -R @ c[:3, 3]))

def triangulate(Pa, Pb, a, b):
    h = cv2.triangulatePoints(Pa, Pb, a.T, b.T).T
    g = np.abs(h[:, 3]) > 1e-8
    X = h[g, :3] / h[g, 3:4]
    good = np.all(np.isfinite(X), 1) & (np.linalg.norm(X, axis=1) < 100.0)
    selected = np.zeros(len(a), bool)
    selected[np.flatnonzero(g)[good]] = True
    return X[good], selected

def pose_pnp(X, x, K, dist):
    if len(X) < 6:
        return None, 0, None
    ok, r, t, ins = cv2.solvePnPRansac(X.astype(np.float64), x.astype(np.float64), K, dist,
                                        iterationsCount=2000, reprojectionError=3.0,
                                        confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or ins is None or len(ins) < 6:
        return None, 0, None
    R, _ = cv2.Rodrigues(r)
    c = np.eye(4); c[:3, :3] = R.T; c[:3, 3] = (-R.T @ t).reshape(3)
    proj, _ = cv2.projectPoints(X[ins[:, 0]], r, t, K, dist)
    e = np.linalg.norm(proj[:, 0] - x[ins[:, 0]], axis=1)
    return c, int(len(ins)), float(np.median(e))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True); p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--mapping-gt', type=Path, required=True); p.add_argument('--sequence', required=True)
    p.add_argument('--indices', nargs='+', type=int); p.add_argument('--indices-file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--num-refs', type=int, default=10); p.add_argument('--select-refs', type=int, default=4)
    p.add_argument('--retrieval-npz', type=Path, help='optional mapping-only retrieval with query_indices')
    a = p.parse_args()
    if (a.indices is None) == (a.indices_file is None):
        p.error('provide exactly one of --indices or --indices-file')
    if a.indices_file is not None:
        a.indices = [int(x) for x in a.indices_file.read_text().split()]
    device = torch.device(a.device)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    _, mf, mt, _, mr = read_manifest(a.data_root, 'mapping')
    _, qf, qt, _, qr = read_manifest(a.data_root, a.sequence)
    map_pose, _ = tum_camera_poses(a.mapping_gt, mt, base_to_front_rgb_optical(mr), .1)
    K, dist = K_from(a.data_root / 'mapping/camera_front/color/intrinsics.yaml')
    rel = load_relative(a.results_root / a.sequence, len(qf))
    retrieval = None
    if a.retrieval_npz:
        retrieval = np.load(a.retrieval_npz)
        retrieval_lookup = {int(q): i for i, q in enumerate(retrieval['query_indices'])}
    rows = []
    for qi in a.indices:
        frame_start = time.perf_counter()
        query_extract_start = time.perf_counter()
        q = load_image(qf[qi]['path']).to(device)
        with torch.inference_mode(): fq = extractor.extract(q)
        query_extract_s = time.perf_counter() - query_extract_start
        rerank_start = time.perf_counter()
        entries = []
        if retrieval is None:
            refs = rel[qi]['indices'][:a.num_refs]
        else:
            refs = retrieval['mapping_indices'][retrieval_lookup[qi]][:a.num_refs]
        for ri in refs:
            r = load_image(mf[int(ri)]['path']).to(device)
            with torch.inference_mode():
                fr = extractor.extract(r); out = matcher({'image0': fq, 'image1': fr})
            ms = out['matches'][0].detach().cpu().numpy(); qids = ms[:, 0]
            qxy = fq['keypoints'][0].detach().cpu().numpy()[qids]
            rxy = fr['keypoints'][0].detach().cpu().numpy()[ms[:, 1]]
            ein = 0
            if len(ms) >= 8:
                _, mask = cv2.findEssentialMat(qxy, rxy, K, method=cv2.RANSAC, prob=.999, threshold=1.5)
                ein = int(mask.sum()) if mask is not None else 0
            entries.append({'ri': int(ri), 'qids': qids, 'qxy': qxy, 'rxy': rxy,
                            'matches': int(len(ms)), 'essential_inliers': ein})
        entries.sort(key=lambda z: (z['essential_inliers'], z['matches']), reverse=True)
        rerank_s = time.perf_counter() - rerank_start
        chosen = entries[:a.select_refs]; allX, allx, stats = [], [], []
        geometry_start = time.perf_counter()
        for u, v in combinations(range(len(chosen)), 2):
            left, right = chosen[u], chosen[v]; lookup = {int(k): i for i, k in enumerate(left['qids'])}
            qa, xa, xb = [], [], []
            for j, k in enumerate(right['qids']):
                if int(k) in lookup:
                    i = lookup[int(k)]; qa.append(left['qxy'][i]); xa.append(left['rxy'][i]); xb.append(right['rxy'][j])
            qa, xa, xb = np.asarray(qa, np.float32), np.asarray(xa, np.float32), np.asarray(xb, np.float32)
            if len(qa):
                X, good = triangulate(P_from(map_pose[left['ri']], K), P_from(map_pose[right['ri']], K), xa, xb)
                if len(X): allX.append(X); allx.append(qa[good])
            else: X = np.empty((0, 3))
            stats.append({'refs': [left['ri'], right['ri']], 'shared_tracks': int(len(qa)), 'triangulated': int(len(X))})
        if allX:
            X, x = np.concatenate(allX), np.concatenate(allx); pose, ins, reproj = pose_pnp(X, x, K, dist)
        else:
            X = np.empty((0, 3)); pose, ins, reproj = None, 0, None
        geometry_s = time.perf_counter() - geometry_start
        row = {'query_index': int(qi), 'retrieval_refs': [int(x) for x in refs],
               'selected_refs': [int(x['ri']) for x in chosen],
               'reference_scores': [{'index': x['ri'], 'matches': x['matches'], 'essential_inliers': x['essential_inliers']} for x in entries],
               'pose': None if pose is None else pose.tolist(), 'inliers': ins,
               'median_reprojection_px': reproj, 'triangulated_tracks': int(len(X)), 'pair_stats': stats,
               'query_extract_s': query_extract_s, 'rerank_s': rerank_s,
               'geometry_s': geometry_s,
               'elapsed_s': time.perf_counter() - frame_start,
               'query_gt_read': False, 'selection_uses_gt': False, 'tool': 'HLoc-style SuperPoint+LightGlue'}
        rows.append(row); print(json.dumps(row)[:1200], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text('\n'.join(json.dumps(x, sort_keys=True) for x in rows) + '\n')

if __name__ == '__main__':
    main()
