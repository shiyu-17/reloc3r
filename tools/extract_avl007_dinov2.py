#!/usr/bin/env python3
"""Extract frozen DINOv2 ViT-L/14 CLS embeddings for AVL-007.

No pose, GT, or Reloc3r output is opened here.  The model and preprocessing are
fixed so the resulting embeddings are a reusable, auditable cheap context
representation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


SEQUENCES = ["mapping", "static_0", "dynamics_0", "lt_changes_0", "lt_changes_dynamics_0"]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(repo: Path, weights: Path, device: torch.device):
    model = torch.hub.load(str(repo), "dinov2_vitl14", source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    msg = model.load_state_dict(state, strict=True)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(f"DINO state mismatch: {msg}")
    model.eval().to(device)
    return model


def image_paths(data_root: Path, seq: str):
    manifest = json.loads((data_root / seq / "manifest.json").read_text())
    return [data_root / seq / x["path"] for x in manifest["frames"]], manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/lilocbench/prepared_1hz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/lilocbench/AVL-007/embeddings"))
    ap.add_argument("--repo", type=Path, default=Path("/home/liusy/.cache/torch/hub/facebookresearch_dinov2_main"))
    ap.add_argument("--weights", type=Path, default=Path(".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("AVL-007 embedding extraction requires CUDA for the frozen ViT-L run")
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    t0 = time.perf_counter()
    model = load_model(args.repo, args.weights, device)
    model_load_s = time.perf_counter() - t0
    manifest_out = {
        "model": "DINOv2 ViT-L/14",
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "repo": str(args.repo.resolve()),
        "preprocess": "Resize(short=518,bicubic)->CenterCrop(518)->ImageNet normalize",
        "device": str(device),
        "batch_size": args.batch_size,
        "query_gt_read": False,
        "pose_or_reloc3r_read": False,
        "model_load_s": model_load_s,
        "sequences": {},
    }
    for seq in SEQUENCES:
        paths, manifest = image_paths(args.data_root, seq)
        vectors = []
        start = time.perf_counter()
        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i:i + args.batch_size]
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model.forward_features(batch)
                cls = features["x_norm_clstoken"] if isinstance(features, dict) else features
            vectors.append(cls.float().cpu().numpy())
        arr = np.concatenate(vectors, axis=0).astype(np.float32)
        arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        np.save(args.output / f"{seq}_cls.npy", arr)
        manifest_out["sequences"][seq] = {
            "frames": len(paths),
            "embedding_dim": int(arr.shape[1]),
            "elapsed_s": time.perf_counter() - start,
            "manifest_sha256": sha256(args.data_root / seq / "manifest.json"),
            "source_archive_sha256": manifest.get("source_archive_sha256"),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print(json.dumps(manifest_out, indent=2))


if __name__ == "__main__":
    main()
