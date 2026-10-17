#!/usr/bin/env python3
"""Risk--cost--accuracy curves for pre-Agent AVL-004 analysis."""
import argparse
import json
from pathlib import Path


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")
THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def load_jsonl(path):
    return {int(x["query_index"]): x for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def summary(rows, geometry_rate=None, total_frames=None):
    total_frames = len(rows) if total_frames is None else total_frames
    available = [x for x in rows if x.get("success") is not None]
    good = sum(bool(x["success"]) for x in available)
    return {"frames": total_frames, "selected_frames": len(rows), "pose_available": len(available), "coverage": len(available) / total_frames if total_frames else 0.0, "success_all_frames": good / total_frames if total_frames else 0.0, "selective_accuracy": good / len(available) if available else 0.0, "geometry_call_rate": geometry_rate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--risk-root", type=Path, required=True)
    p.add_argument("--always-route-dir", type=Path, required=True)
    p.add_argument("--always-roma-dir", type=Path, required=True)
    p.add_argument("--manager-route-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {"protocol": "risk scores are frozen no-GT evidence; GT is evaluator-only", "sequences": {}}
    for seq in SEQUENCES:
        risk = load_jsonl(args.risk_root / seq / "risk_rows.jsonl")
        roma = json.loads((args.always_route_dir / f"roma_{seq}.json").read_text())["frames"]
        roma = {int(x["query_index"]): x for x in roma}
        manager = json.loads((args.manager_route_dir / f"roma_{seq}.json").read_text())["frames"]
        manager = {int(x["query_index"]): x for x in manager}
        thresholds = []
        n = len(risk)
        geom_manager = sum(x["action"] == "call_geometry" for x in manager.values()) / n
        for threshold in THRESHOLDS:
            selective = []
            gated = []
            high = 0
            for qi, x in risk.items():
                low = float(x["risk_score"]) <= threshold
                if low and x["action_output"].get("available"):
                    selective.append({"success": bool(x["action_output"].get("success"))})
                if not low:
                    high += 1
                    metric = roma.get(qi, {}).get("metric")
                    gated.append({"success": bool(metric and metric["success_0.25m_5deg"]) if metric is not None else None})
                elif x["action_output"].get("available"):
                    gated.append({"success": bool(x["action_output"].get("success"))})
                else:
                    gated.append({"success": None})
            thresholds.append({"threshold": threshold, "reject_high_risk": summary(selective, 0.0, n), "risk_gated_roma": summary(gated, high / n, n), "high_risk_frames": high})
        result["sequences"][seq] = {"frames": n, "manager_geometry_call_rate": geom_manager, "thresholds": thresholds, "manager_roma": {"frames": n, "success": sum(bool(x.get("metric") and x["metric"]["success_0.25m_5deg"]) for x in manager.values()) / n, "geometry_call_rate": geom_manager}, "always_roma": {"frames": n, "success": sum(bool(x.get("metric") and x["metric"]["success_0.25m_5deg"]) for x in roma.values()) / n, "geometry_call_rate": 1.0}}
    total = sum(x["frames"] for x in result["sequences"].values())
    result["weighted_frames"] = total
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"weighted_frames": total, "sequences": list(result["sequences"])}, indent=2))


if __name__ == "__main__":
    main()
