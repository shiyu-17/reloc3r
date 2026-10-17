#!/usr/bin/env python3
"""Freeze source-specific active-evidence verifier thresholds.

This utility consumes evaluator traces only after they have been produced by
live tools.  Ground truth is joined here, never in an online backend.  The
calibration split is selected by sequence name and the reported test split is
never used to choose a threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.eval_active_evidence_policies import load_query_gt, success_pose


SOURCES = ("reloc3r_online", "roma_online_geometry")
DEFAULT_SOURCES: Dict[str, Dict[str, float | int]] = {
    "reloc3r_online": {
        "min_references": 2, "max_dispersion_m": 1.5, "max_dispersion_deg": 15.0,
        "heldout_max_translation_m": 1.5, "heldout_max_rotation_deg": 20.0,
        "max_state_risk": 1.5,
    },
    "roma_online_geometry": {
        "min_pnp_inliers": 24, "max_reprojection_px": 3.0,
        "heldout_min_inliers": 24, "heldout_min_inlier_ratio": 0.10,
        "heldout_max_epipolar_px": 4.0, "max_state_risk": 1.5,
    },
    "memory": {"min_confidence": 0.5},
}


def source_rows(trace_root: Path, data_root: Path, gt_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((trace_root / "traces").glob("*_*.jsonl")):
        # Policy suffix may itself contain underscores; extract source from
        # each acquisition instead of relying on the filename.
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            sequence = str(record["sequence"])
            gt = load_query_gt(data_root, gt_root, sequence)
            frame_index = int(record["frame_index"])
            for acquisition in record.get("acquisitions", []):
                candidate = acquisition.get("candidate") or {}
                source = candidate.get("source")
                pose = acquisition.get("candidate_pose")
                verification = acquisition.get("verification") or {}
                if source not in SOURCES or pose is None or not verification:
                    continue
                rows.append({
                    "sequence": sequence, "frame_index": frame_index, "source": source,
                    "risk": float(verification.get("total_risk", float("inf"))),
                    "structural_pass": structural_pass(source, candidate.get("evidence", {}), verification),
                    "success": bool(success_pose(np.asarray(pose, dtype=np.float64), gt[frame_index])),
                    "trace": str(path),
                })
    return rows


def structural_pass(source: str, evidence: Mapping[str, Any], verification: Mapping[str, Any]) -> bool:
    """Apply all frozen source terms other than the calibrated risk cutoff."""
    cfg = DEFAULT_SOURCES[source]
    independent = verification.get("independent_verification", {})
    if source == "reloc3r_online":
        return (int(evidence.get("num_references", 0)) >= int(cfg["min_references"]) and
                float(evidence.get("translation_dispersion_m", float("inf"))) <= float(cfg["max_dispersion_m"]) and
                float(evidence.get("rotation_dispersion_deg", float("inf"))) <= float(cfg["max_dispersion_deg"]) and
                bool(independent.get("available", False)) and
                float(independent.get("translation_disagreement_m", float("inf"))) <= float(cfg["heldout_max_translation_m"]) and
                float(independent.get("rotation_disagreement_deg", float("inf"))) <= float(cfg["heldout_max_rotation_deg"]))
    return (int(evidence.get("inliers", 0)) >= int(cfg["min_pnp_inliers"]) and
            evidence.get("median_reprojection_px") is not None and
            float(evidence["median_reprojection_px"]) <= float(cfg["max_reprojection_px"]) and
            bool(independent.get("available", False)) and
            int(independent.get("inliers", 0)) >= int(cfg["heldout_min_inliers"]) and
            float(independent.get("inlier_ratio", 0.0)) >= float(cfg["heldout_min_inlier_ratio"]) and
            independent.get("median_epipolar_px") is not None and
            float(independent["median_epipolar_px"]) <= float(cfg["heldout_max_epipolar_px"]))


def metrics(rows: Iterable[Mapping[str, Any]], cutoff: float) -> Dict[str, Any]:
    eligible = [row for row in rows if row["structural_pass"] and row["risk"] <= cutoff]
    correct = sum(bool(row["success"]) for row in eligible)
    return {"candidates": len(list(rows)) if not isinstance(rows, list) else len(rows),
            "accepted": len(eligible), "correct": correct,
            "false_accepts": len(eligible) - correct,
            "coverage": len(eligible) / len(rows) if rows else 0.0,
            "accepted_accuracy": correct / len(eligible) if eligible else 0.0,
            "false_accept_rate": (len(eligible) - correct) / len(eligible) if eligible else 0.0}


def choose_cutoff(rows: List[Dict[str, Any]], max_far: float) -> tuple[float, Dict[str, Any]]:
    risks = sorted({float(row["risk"]) for row in rows if row["structural_pass"] and np.isfinite(row["risk"])})
    candidates = []
    for cutoff in risks:
        result = metrics(rows, cutoff)
        if result["accepted"] and result["false_accept_rate"] <= max_far:
            candidates.append((result["correct"], result["accepted"], cutoff, result))
    if not candidates:
        return 0.0, metrics(rows, 0.0)
    # Prefer correct coverage, then total coverage, then the stricter cutoff.
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    return float(candidates[0][2]), candidates[0][3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/lilocbench/gt"))
    parser.add_argument("--calibration-sequences", nargs="+", required=True)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.max_far < 1.0:
        parser.error("--max-far must be in [0, 1)")
    rows = source_rows(args.trace_root, args.data_root, args.gt_root)
    missing = [source for source in SOURCES if not any(row["source"] == source for row in rows)]
    if missing:
        raise RuntimeError("missing live calibration candidates for: " + ", ".join(missing))
    calibration_sequences = set(args.calibration_sequences)
    report: Dict[str, Any] = {"protocol": {
        "source_specific": True, "target_max_false_accept_rate": args.max_far,
        "calibration_sequences": sorted(calibration_sequences),
        "selection": "maximize correctly accepted candidates subject to validation FAR target; then stricter cutoff",
        "online_boundary": "GT joined only by this evaluator-side calibrator after live receipts",
    }, "sources": {}}
    config = {"schema_version": 1, "calibration_protocol": report["protocol"],
              "sources": {source: dict(values) for source, values in DEFAULT_SOURCES.items()}}
    for source in SOURCES:
        source_all = [row for row in rows if row["source"] == source]
        validation = [row for row in source_all if row["sequence"] in calibration_sequences]
        test = [row for row in source_all if row["sequence"] not in calibration_sequences]
        cutoff, selected = choose_cutoff(validation, args.max_far)
        config["sources"][source]["max_state_risk"] = cutoff
        report["sources"][source] = {
            "candidate_count": {"all": len(source_all), "calibration": len(validation), "test": len(test)},
            "chosen_max_state_risk": cutoff, "calibration": selected,
            "frozen_test": metrics(test, cutoff),
        }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
