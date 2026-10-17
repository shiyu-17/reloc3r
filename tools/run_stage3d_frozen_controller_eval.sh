#!/usr/bin/env bash
# Evaluate one frozen Stage 3D component controller on an arbitrary split.
# This is intentionally separate from the Geometry-advantage runner: it is
# the causal no-advantage comparator and must not load an advantage head.
set -euo pipefail

GPU_ID="$1"
SEQUENCE="$2"
VARIANT="$3"  # f0 or f2
ROOT="$4"
START_FRAME="${5:-120}"
MAX_FRAMES="${6:-20}"
OUT="${ROOT}/${VARIANT}/${SEQUENCE}"

case "${VARIANT}" in
  f0|f2) CHECKPOINT="outputs/lilocbench/stage3d_v2_controller_${VARIANT}_r1" ;;
  *) echo "variant must be f0 or f2" >&2; exit 2 ;;
esac
[[ -f "${OUT}/summary.json" ]] && exit 0

env "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
  "LD_PRELOAD=/data2/liusy/reloc3r/outputs/runtime_compat/libitt_compat_stub.so" \
  "HF_HOME=/data2/liusy/reloc3r/.cache/huggingface" \
  "TORCH_HOME=/data2/liusy/reloc3r/.cache/torch" \
  "PYTHONPATH=third_party/stage3c_xgboost:." \
  /data2/liusy/conda_envs/reloc3r/bin/python -m tools.collect_stage3b_dagger \
  --output-dir "${OUT}" \
  --learned-policy-checkpoint "${CHECKPOINT}" \
  --learned-policy-family stage3c6_utility \
  --data-root data/lilocbench/prepared_1hz --gt-root data/lilocbench/gt \
  --mapping-gt data/lilocbench/gt/mapping_gt_poses.txt \
  --retrieval-root outputs/lilocbench/AVL-002 \
  --verifier-config configs/active_evidence_verifier_v1.json \
  --utility-config configs/stage3a_terminal_utility_v1.json \
  --backend reloc3r --device cuda --learned-policy-device cpu \
  --stage3d-preaction-features --sequences "${SEQUENCE}" \
  --start "${START_FRAME}" --max-frames "${MAX_FRAMES}" --max-acquisitions 2 --oracle-depth 2 \
  --budget-ms 30000 --seed 1777
