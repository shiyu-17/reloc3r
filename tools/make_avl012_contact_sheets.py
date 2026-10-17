#!/usr/bin/env python3
"""Make small evaluator-only audit contact sheets for AVL-012."""
from __future__ import annotations
import argparse, json, math
import sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_lilocbench_reloc3r import read_manifest

SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/lilocbench/prepared_1hz'))
    p.add_argument('--evidence-root', type=Path, default=Path('outputs/lilocbench/AVL-012'))
    p.add_argument('--threshold', type=float, default=1.5)
    args = p.parse_args()
    buckets = {'accepted_correct': [], 'accepted_wrong': [], 'rejected': []}
    for seq in SEQUENCES:
        _, frames, _, _, _ = read_manifest(args.data_root, seq)
        rows = [json.loads(x) for x in (args.evidence_root / seq / 'evaluator_rows.jsonl').read_text().splitlines() if x.strip()]
        split = int(math.ceil(len(rows) * 0.4))
        for row in rows[split:]:
            accept = bool(row['route_accepted'] and row['reliability_risk'] <= args.threshold)
            key = 'accepted_correct' if accept and row['success'] else 'accepted_wrong' if accept else 'rejected'
            if len(buckets[key]) < 16:
                buckets[key].append((seq, int(row['query_index']), frames[int(row['query_index'])]['path']))
    outdir = args.evidence_root / 'contact_sheets'; outdir.mkdir(parents=True, exist_ok=True)
    for name, items in buckets.items():
        if not items: continue
        thumb_w, thumb_h = 240, 180
        cols = 4; rows_n = math.ceil(len(items) / cols)
        canvas = Image.new('RGB', (cols * thumb_w, rows_n * (thumb_h + 28)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (seq, qi, path) in enumerate(items):
            try:
                im = Image.open(path).convert('RGB'); im.thumbnail((thumb_w, thumb_h))
                x = (i % cols) * thumb_w + (thumb_w - im.width) // 2
                y = (i // cols) * (thumb_h + 28)
                canvas.paste(im, (x, y)); draw.text(((i % cols) * thumb_w + 3, y + thumb_h + 3), f'{seq}:{qi}', fill='black')
            except Exception as exc:
                draw.text(((i % cols) * thumb_w + 3, (i // cols) * (thumb_h + 28) + 3), f'{seq}:{qi} ({exc})', fill='red')
        canvas.save(outdir / f'{name}_test_t{args.threshold:g}.jpg', quality=90)
        (outdir / f'{name}_test_t{args.threshold:g}.json').write_text(json.dumps([{'sequence':s,'query_index':q,'path':str(path)} for s,q,path in items], indent=2) + '\n')
    print(json.dumps({k: len(v) for k,v in buckets.items()}))

if __name__ == '__main__': main()
