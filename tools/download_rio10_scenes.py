#!/usr/bin/env python3
"""Resumable downloader and pose audit for the official RIO10 archives.

The official RIO10 release stores each scene in three archives:
``seqNN.zip`` (RGB, depth, and public poses), ``modelsNN.zip``, and
``semanticsNN.zip``.  This script has no third-party dependencies and resumes
an interrupted download in place using HTTP Range requests.

Examples:
  # Download (or resume) the complete original data for scenes 01--03.
  python3 tools/download_rio10_scenes.py --scenes 01 02 03

  # Download only scenes 02 and 03, then audit their publicly released poses.
  python3 tools/download_rio10_scenes.py --scenes 02 03 --verify-poses

  # Validate already downloaded files without transferring anything.
  python3 tools/download_rio10_scenes.py --scenes 01 02 03 --verify-only

Archives are kept in <root>/raw/sceneNN/.  Existing partial ``*.zip`` files
are resumed directly, so this also continues older aria2/curl downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


BASE_URL = "https://campar.in.tum.de/public_datasets/RIO10/Dataset"
ARCHIVES = ("seq", "models", "semantics")
CHUNK_SIZE = 8 * 1024 * 1024
RETRIES = 8
POSE_MEMBER = re.compile(
    r"(?:^|/)(seq\d{2}_\d{2})/frame-(\d+)\.pose\.txt$"
)
RGB_MEMBER = re.compile(
    r"(?:^|/)(seq\d{2}_\d{2})/frame-(\d+)\.color\.jpg$"
)


def format_size(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def response_total(response, existing: int) -> int:
    """Return the full file length, requiring an unambiguous server response."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError("server did not return Content-Length or Content-Range")
    return existing + int(length) if response.status == 206 else int(length)


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "RIO10-resumer/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if response.status != 200 or length is None:
            raise RuntimeError(f"unexpected HEAD response for {url}: HTTP {response.status}")
        return int(length)


def download(url: str, destination: Path) -> None:
    """Resume ``destination`` safely, or validate it if it is already complete."""
    remote = remote_size(url)
    local = destination.stat().st_size if destination.exists() else 0
    if local > remote:
        raise RuntimeError(
            f"{destination} is larger than the official archive ({format_size(local)} > "
            f"{format_size(remote)}); move it aside before retrying."
        )
    if local == remote:
        print(f"already complete by size: {destination} ({format_size(local)})")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        local = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "RIO10-resumer/1.0"}
        if local:
            headers["Range"] = f"bytes={local}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if local and response.status != 206:
                    raise RuntimeError(
                        "server ignored the Range request; refusing to overwrite the partial archive"
                    )
                total = response_total(response, local)
                if total != remote:
                    raise RuntimeError(f"remote size changed during download: {total} != {remote}")
                started = time.monotonic()
                downloaded = local
                mode = "ab" if local else "wb"
                print(f"downloading {destination.name}: {format_size(local)} / {format_size(total)}")
                with destination.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.monotonic() - started, 0.001)
                        rate = (downloaded - local) / elapsed
                        percent = downloaded * 100 / total
                        print(
                            f"\r  {percent:6.2f}%  {format_size(downloaded)} / "
                            f"{format_size(total)}  {format_size(int(rate))}/s",
                            end="", flush=True,
                        )
                print()
            if destination.stat().st_size != remote:
                raise RuntimeError(
                    f"incomplete transfer: got {destination.stat().st_size}, expected {remote}"
                )
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
            if attempt == RETRIES:
                raise RuntimeError(f"download failed after {RETRIES} attempts: {error}") from error
            wait = min(60, 2 ** attempt)
            print(f"download error ({error}); retrying in {wait}s [{attempt}/{RETRIES}]", file=sys.stderr)
            time.sleep(wait)


def validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"invalid ZIP: {path}: {error}") from error
    if bad_member is not None:
        raise RuntimeError(f"CRC check failed in {path}: {bad_member}")


def parse_pose(raw: bytes, member: str) -> None:
    try:
        values = [float(value) for value in raw.decode("utf-8").split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"non-numeric pose {member}") from error
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid 4x4 pose {member}")
    if any(abs(value - expected) > 1e-5 for value, expected in zip(values[12:], (0, 0, 0, 1))):
        raise ValueError(f"invalid homogeneous row in {member}")
    rotation = [values[row * 4: row * 4 + 3] for row in range(3)]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 2e-3:
                raise ValueError(f"non-orthonormal rotation in {member}")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if determinant < 0.99:
        raise ValueError(f"improper rotation in {member}")


def audit_poses(path: Path) -> dict[str, object]:
    """Validate every public pose and report RGB/pose frame coverage per sequence."""
    poses: dict[str, set[str]] = defaultdict(set)
    rgb: dict[str, set[str]] = defaultdict(set)
    checked = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            pose_match = POSE_MEMBER.search(info.filename)
            if pose_match:
                sequence, frame = pose_match.groups()
                parse_pose(archive.read(info), info.filename)
                poses[sequence].add(frame)
                checked += 1
                continue
            rgb_match = RGB_MEMBER.search(info.filename)
            if rgb_match:
                sequence, frame = rgb_match.groups()
                rgb[sequence].add(frame)

    report: dict[str, object] = {"archive": str(path), "poses_checked": checked, "sequences": {}}
    sequences: dict[str, object] = report["sequences"]  # type: ignore[assignment]
    for sequence in sorted(set(rgb) | set(poses)):
        rgb_only = sorted(rgb[sequence] - poses[sequence])
        pose_only = sorted(poses[sequence] - rgb[sequence])
        sequences[sequence] = {
            "rgb_frames": len(rgb[sequence]),
            "pose_frames": len(poses[sequence]),
            "paired_frames": len(rgb[sequence] & poses[sequence]),
            "pose_usable": bool(poses[sequence]) and not rgb_only and not pose_only,
            "rgb_without_pose_count": len(rgb_only),
            "pose_without_rgb_count": len(pose_only),
            "rgb_without_pose_examples": rgb_only[:10],
            "pose_without_rgb_examples": pose_only[:10],
        }
    return report


def scene_id(value: str) -> str:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid scene: {value}") from error
    if not 1 <= number <= 10:
        raise argparse.ArgumentTypeError("scene must be between 01 and 10")
    return f"{number:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data2/liusy/datasets/RIO10"))
    parser.add_argument("--scenes", nargs="+", type=scene_id, default=["01", "02", "03"])
    parser.add_argument(
        "--archives", nargs="+", choices=ARCHIVES, default=list(ARCHIVES),
        help="archive categories to process (default: seq models semantics)",
    )
    parser.add_argument("--verify-only", action="store_true", help="do not download; validate existing archives")
    parser.add_argument("--verify-poses", action="store_true", help="validate every public pose in seqNN.zip")
    parser.add_argument("--report", type=Path, help="write pose-audit JSON here (requires --verify-poses)")
    args = parser.parse_args()

    pose_reports = []
    for scene in dict.fromkeys(args.scenes):
        for category in dict.fromkeys(args.archives):
            filename = f"{category}{scene}.zip"
            path = args.root / "raw" / f"scene{scene}" / filename
            url = f"{BASE_URL}/{filename}"
            print(f"\n== scene{scene}: {filename} ==")
            if args.verify_only:
                if not path.exists():
                    raise FileNotFoundError(f"missing archive: {path}")
            else:
                download(url, path)
            print("verifying ZIP CRC …", flush=True)
            validate_zip(path)
            print("ZIP CRC: OK")
            if category == "seq" and args.verify_poses:
                report = audit_poses(path)
                pose_reports.append(report)
                print(f"public poses checked: {report['poses_checked']}")
                for sequence, status in report["sequences"].items():
                    print(f"  {sequence}: paired={status['paired_frames']} usable={status['pose_usable']}")

    if args.report:
        if not args.verify_poses:
            parser.error("--report requires --verify-poses")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(pose_reports, indent=2) + "\n", encoding="utf-8")
        print(f"pose report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
