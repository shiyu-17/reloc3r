#!/usr/bin/env python3
"""Extract full-rate front-RGB streams and write deterministic manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def prepare(args, sequence):
    archive = args.archive_root / f"{sequence}.zip"
    target = args.output_root / sequence
    image_dir = target / "camera_front" / "color" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.startswith("camera_front/color/images/") and n.endswith(".png")]
        names.sort(key=lambda n: float(Path(n).stem))
        frames = []
        for name in names:
            out = target / name
            if not out.exists() or out.stat().st_size != zf.getinfo(name).file_size:
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            frames.append({"timestamp": Path(name).stem, "path": name,
                           "zip_crc32": f"{zf.getinfo(name).CRC:08x}",
                           "zip_bytes": zf.getinfo(name).file_size})
    intrinsics = args.prepared_root / "mapping/camera_front/color/intrinsics.yaml"
    target_intrinsics = target / "camera_front/color/intrinsics.yaml"
    target_intrinsics.parent.mkdir(parents=True, exist_ok=True)
    if not target_intrinsics.exists():
        shutil.copy2(intrinsics, target_intrinsics)
    gaps = [float(b["timestamp"]) - float(a["timestamp"]) for a, b in zip(frames, frames[1:])]
    manifest = {"sequence": sequence, "source_archive": str(archive), "camera": "camera_front/color",
                "sample_period_s": float(sorted(gaps)[len(gaps)//2]) if gaps else None,
                "frames": frames}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sequence": sequence, "frames": len(frames), "target": str(target),
                      "sample_period_s": manifest["sample_period_s"]}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", type=Path, default=Path("/data2/liusy/datasets/LILocBench"))
    p.add_argument("--prepared-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--sequences", nargs="+", required=True)
    args = p.parse_args()
    for seq in args.sequences:
        prepare(args, seq)


if __name__ == "__main__":
    main()
