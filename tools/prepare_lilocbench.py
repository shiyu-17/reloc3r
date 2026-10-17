#!/usr/bin/env python3
"""Extract a timestamp-uniform RGB subset of LILocBench individual-file archives.

The extractor deliberately keeps the source archives immutable and only writes the
front RGB frames and calibration needed by the Reloc3r RGB baseline.  It never
reads query ground truth; pose matching belongs to the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


RGB_MEMBER = re.compile(r"^camera_front/color/images/(\d+\.\d+)\.png$")
METADATA = (
    "transformations.yaml",
    "camera_front/color/intrinsics.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_members(archive: zipfile.ZipFile, sample_period_s: float):
    frames = []
    for member in archive.infolist():
        match = RGB_MEMBER.match(member.filename)
        if match:
            frames.append((float(match.group(1)), member))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise RuntimeError("Archive contains no camera_front RGB images")

    chosen = []
    next_timestamp = frames[0][0]
    for timestamp, member in frames:
        if timestamp + 1e-9 >= next_timestamp:
            chosen.append((timestamp, member))
            next_timestamp = timestamp + sample_period_s
    return chosen


def copy_member(archive: zipfile.ZipFile, member: str | zipfile.ZipInfo, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)


def extract_sequence(archive_path: Path, output_root: Path, sample_period_s: float, hash_archive: bool):
    sequence = archive_path.stem
    sequence_root = output_root / sequence
    manifest_path = sequence_root / "manifest.json"
    with zipfile.ZipFile(archive_path) as archive:
        selected = selected_members(archive, sample_period_s)
        manifest = {
            "sequence": sequence,
            "source_archive": str(archive_path.resolve()),
            "source_archive_bytes": archive_path.stat().st_size,
            "source_archive_sha256": sha256(archive_path) if hash_archive else None,
            "camera": "camera_front/color",
            "sample_period_s": sample_period_s,
            "frames": [
                {
                    "timestamp": f"{timestamp:.9f}",
                    "path": member.filename,
                    "zip_crc32": f"{member.CRC:08x}",
                    "zip_bytes": member.file_size,
                }
                for timestamp, member in selected
            ],
        }
        for metadata in METADATA:
            copy_member(archive, metadata, sequence_root / metadata)
        for _, member in selected:
            copy_member(archive, member, sequence_root / member.filename)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sequence, len(selected), manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True,
                        help="Archive stems, e.g. mapping static_0")
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--hash-archives", action="store_true")
    args = parser.parse_args()
    if args.sample_hz <= 0:
        raise ValueError("--sample-hz must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for sequence in args.sequences:
        archive = args.archive_root / f"{sequence}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        name, count, manifest = extract_sequence(
            archive, args.output_root, 1.0 / args.sample_hz, args.hash_archives
        )
        summary.append({"sequence": name, "frames": count, "manifest": str(manifest)})
        print(f"{name}: extracted {count} RGB frames -> {manifest}")
    (args.output_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
