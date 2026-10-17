#!/usr/bin/env python3
"""Run the frozen AVL-008 VLM semantic probe from redacted requests."""
import argparse
import base64
import json
import mimetypes
import re
import time
from pathlib import Path

import httpx
from openai import OpenAI

OUT = Path("outputs/lilocbench/AVL-008")
ALLOWED = {
    "temporary_dynamic_occlusion", "motion_blur_or_weak_observation",
    "persistent_scene_change", "reference_semantic_mismatch",
    "normal_or_compatible", "uncertain",
}
JSON_ONLY = "Return exactly one valid JSON object matching the requested schema. No markdown, no prose."


def data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def content(request, retry=False):
    out = [{"type": "text", "text": request["prompt"] + ("\n" + JSON_ONLY if retry else "")}]
    out += [{"type": "text", "text": "CURRENT query image:"},
            {"type": "image_url", "image_url": {"url": data_url(request["current_path"])}}]
    for j, path in enumerate(request["reference_paths"], 1):
        out += [{"type": "text", "text": f"REFERENCE {j} (mapping image):"},
                {"type": "image_url", "image_url": {"url": data_url(path)}}]
    for j, path in enumerate(request["history_paths"], 1):
        out += [{"type": "text", "text": f"HISTORY {j} (earlier query frame):"},
                {"type": "image_url", "image_url": {"url": data_url(path)}}]
    return out


def parse(raw):
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError("no JSON object")
    obj = json.loads(match.group(0))
    if set(obj) != {"label", "confidence", "evidence"}:
        raise ValueError(f"wrong keys: {sorted(obj)}")
    if obj["label"] not in ALLOWED:
        raise ValueError(f"invalid label: {obj['label']}")
    conf = float(obj["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ValueError("confidence outside [0,1]")
    evidence = str(obj["evidence"]).strip()
    if len(evidence.split()) > 20:
        raise ValueError("evidence exceeds 20 words")
    return {"label": obj["label"], "confidence": conf, "evidence": evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen2.5-vl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    requests = [json.loads(x) for x in (OUT / "requests.jsonl").read_text().splitlines()]
    if args.limit is not None:
        requests = requests[:args.limit]
    result_path = OUT / "responses.jsonl"
    done = set()
    if result_path.exists() and not args.overwrite:
        done = {json.loads(x)["sample_id"] for x in result_path.read_text().splitlines() if x.strip()}
    elif args.overwrite:
        result_path.unlink(missing_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", http_client=httpx.Client(trust_env=False, timeout=300.0))
    with result_path.open("a") as f:
        for n, request in enumerate(requests, 1):
            if request["sample_id"] in done:
                continue
            record = {k: request[k] for k in ("sample_id", "sequence", "query_index", "timestamp")}
            record.update({"query_gt_read": False, "future_frame_read": False, "outcome_or_action_read": False,
                           "temperature": 0.0, "top_p": 1.0, "max_tokens": 96})
            start = time.monotonic()
            errors, parsed, raw = [], None, None
            for attempt in range(2):
                try:
                    response = client.chat.completions.create(
                        model=args.model, messages=[{"role": "user", "content": content(request, retry=attempt == 1)}],
                        temperature=0.0, top_p=1.0, max_tokens=96, seed=0,
                    )
                    raw = response.choices[0].message.content or ""
                    parsed = parse(raw)
                    record["retry_used"] = bool(attempt)
                    break
                except Exception as exc:
                    errors.append(f"attempt_{attempt}: {type(exc).__name__}: {exc}")
            record.update({"elapsed_s": time.monotonic() - start, "raw_response": raw, "parse_errors": errors,
                           "valid": parsed is not None})
            if parsed is not None:
                record.update(parsed)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps({"done": n, "sample_id": request["sample_id"], "valid": record["valid"],
                              "label": record.get("label"), "elapsed_s": round(record["elapsed_s"], 2)}), flush=True)
    (OUT / "vlm_run_manifest.json").write_text(json.dumps({
        "model": args.model, "base_url": args.base_url, "temperature": 0.0, "top_p": 1.0,
        "max_tokens": 96, "seed": 0, "allowed_labels": sorted(ALLOWED),
        "requests_attempted": len(requests), "requests_file": "requests.jsonl",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
