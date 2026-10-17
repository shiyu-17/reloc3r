#!/usr/bin/env python3
"""Create representative contact sheets for memory-only/RoMa-only frames."""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEQUENCES = ["static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]
DATA_ROOT = Path("data/lilocbench/prepared_1hz")
MEM_ROOT = Path("outputs/lilocbench/AVL-003")
ROMA_ROOT = Path("outputs/lilocbench/AVL-005/unified_cost/eval_roma_seed0")
OUT = Path("outputs/lilocbench/AVL-007/audit")


def load_special():
    out = {"memory_only": [], "roma_only": []}
    for seq in SEQUENCES:
        manifest = json.loads((DATA_ROOT / seq / "manifest.json").read_text())
        mem = [bool(json.loads(x)["methods"]["memory3"].get("success_0.25m_5deg", False))
               for x in (MEM_ROOT / seq / "frame_results.jsonl").read_text().splitlines()]
        roma = [bool(x.get("metric") and x["metric"].get("success_0.25m_5deg", False))
                for x in json.loads((ROMA_ROOT / f"{seq}.json").read_text())["frames"]]
        for i, (m, r) in enumerate(zip(mem, roma)):
            kind = "memory_only" if m and not r else "roma_only" if r and not m else None
            if kind:
                out[kind].append({"sequence": seq, "query_index": i, "path": str(DATA_ROOT / seq / manifest["frames"][i]["path"]), "timestamp": manifest["frames"][i]["timestamp"]})
    return out


def select(items, per_sequence=12):
    chosen = []
    for seq in SEQUENCES:
        part = [x for x in items if x["sequence"] == seq]
        if len(part) <= per_sequence:
            chosen.extend(part)
        else:
            ids = [round(i * (len(part) - 1) / (per_sequence - 1)) for i in range(per_sequence)]
            chosen.extend([part[i] for i in ids])
    return chosen


def sheet(items, path, title):
    cell_w, cell_h, label_h = 260, 210, 35
    cols = 4
    rows = math.ceil(len(items) / cols)
    out = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(out)
    for j, item in enumerate(items):
        im = Image.open(item["path"]).convert("RGB")
        im.thumbnail((cell_w - 8, cell_h - 8))
        x = (j % cols) * cell_w + (cell_w - im.width) // 2
        y = (j // cols) * (cell_h + label_h) + (cell_h - im.height) // 2
        out.paste(im, (x, y))
        text = f"{item['sequence']} q{item['query_index']}"
        draw.text(((j % cols) * cell_w + 4, (j // cols) * (cell_h + label_h) + cell_h + 3), text, fill="black")
    out.save(path, quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    special = load_special()
    selected = {k: select(v) for k, v in special.items()}
    for kind, items in selected.items():
        sheet(items, OUT / f"{kind}_contact_sheet.jpg", kind)
    (OUT / "special_frames.json").write_text(json.dumps({"counts": {k: len(v) for k, v in special.items()}, "selected": selected}, indent=2) + "\n")
    print(json.dumps({"counts": {k: len(v) for k, v in special.items()}, "selected": {k: len(v) for k, v in selected.items()}}, indent=2))


if __name__ == "__main__":
    main()
