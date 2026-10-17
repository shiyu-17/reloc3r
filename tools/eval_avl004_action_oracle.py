#!/usr/bin/env python3
"""Post-hoc action oracle for the AVL-004 route.

The oracle is an upper bound only.  It reads GT-derived evaluator outputs and
must never be used to select a reference, tune a threshold, or drive a route.
"""
import argparse
import json
from pathlib import Path


SEQUENCES = ("static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0")


def read_jsonl(path):
    return {int(x["query_index"]): x for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--avl003-root", type=Path, required=True)
    p.add_argument("--route-dir", type=Path, required=True)
    p.add_argument("--manager-route-dir", type=Path, help="Optional route directory used only for the deterministic-manager success baseline.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--always-route", action="store_true", help="Treat every HLoc/RoMa route row as an available geometry action.")
    args = p.parse_args()
    manager_route_dir = args.manager_route_dir or args.route_dir
    report = {"protocol": "post-hoc GT-only action upper bound; never used for route selection", "sequences": {}}
    for seq in SEQUENCES:
        avl003 = read_jsonl(args.avl003_root / seq / "frame_results.jsonl")
        # Route evaluation is JSON (summary + frames), unlike recovery JSONL.
        hroute = json.loads((args.route_dir / f"hloc_{seq}.json").read_text())["frames"]
        rroute = json.loads((args.route_dir / f"roma_{seq}.json").read_text())["frames"]
        manager_rroute = json.loads((manager_route_dir / f"roma_{seq}.json").read_text())["frames"]
        hroute = {int(x["query_index"]): x for x in hroute}
        rroute = {int(x["query_index"]): x for x in rroute}
        manager_rroute = {int(x["query_index"]): x for x in manager_rroute}
        rows = []
        for qi in sorted(avl003):
            base = avl003[qi]["methods"]
            values = {
                "consensus": bool(base["consensus3"]["success_0.25m_5deg"]),
                "memory": bool(base["memory3"]["success_0.25m_5deg"]),
                "hloc": bool((args.always_route or hroute.get(qi, {}).get("action") == "call_geometry") and hroute[qi].get("metric") and hroute[qi]["metric"]["success_0.25m_5deg"]),
                "roma": bool((args.always_route or rroute.get(qi, {}).get("action") == "call_geometry") and rroute[qi].get("metric") and rroute[qi]["metric"]["success_0.25m_5deg"]),
            }
            values["any"] = any(values.values())
            rows.append({"query_index": qi, "available": {"consensus": True, "memory": True, "hloc": qi in hroute, "roma": qi in rroute}, "success": values})
        n = len(rows)
        counts = {k: sum(x["success"][k] for x in rows) for k in ("consensus", "memory", "hloc", "roma", "any")}
        unique = {}
        for k in ("consensus", "memory", "hloc", "roma"):
            unique[k] = sum(x["success"][k] and not any(x["success"][j] for j in ("consensus", "memory", "hloc", "roma") if j != k) for x in rows)
        route_success = sum(bool(manager_rroute[qi].get("metric") and manager_rroute[qi]["metric"]["success_0.25m_5deg"]) for qi in manager_rroute)
        report["sequences"][seq] = {"frames": n, "success_counts": counts, "success_rates": {k: counts[k] / n for k in counts}, "unique_action_success": unique, "roma_manager_route_success": route_success, "manager_route_gap_to_oracle": counts["any"] - route_success, "rows": rows}
    total = sum(x["frames"] for x in report["sequences"].values())
    total_counts = {k: sum(x["success_counts"][k] for x in report["sequences"].values()) for k in ("consensus", "memory", "hloc", "roma", "any")}
    report["weighted"] = {"frames": total, "success_counts": total_counts, "success_rates": {k: total_counts[k] / total for k in total_counts}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"frames": total, "weighted": report["weighted"]}, indent=2))


if __name__ == "__main__":
    main()
