#!/usr/bin/env python3
"""Verify the immutable F0/F2 Stage 3D baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/lilocbench/stage3d_v2_frozen_baselines_r1/freeze_manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = args.root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
        elif digest(path) != expected:
            failures.append({"path": relative, "error": "sha256_mismatch"})
    report = {"manifest": str(args.manifest), "verified": not failures,
              "artifacts": len(manifest["sha256"]), "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
