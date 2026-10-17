#!/usr/bin/env bash
# Paired F2-vs-feedback latency replay on one GPU and one sequence.
# The two policies share the same backend process; policy order is deliberately
# counterbalanced across sequences by the caller.
set -euo pipefail

GPU_ID="$1"
SEQUENCE="$2"
ROOT="$3"
START_FRAME="$4"
MAX_FRAMES="$5"
ORDER="$6" # f2_first | rule_first
OUT="${ROOT}/${SEQUENCE}"
[[ -f "${OUT}/summary.json" ]] && exit 0

case "${ORDER}" in
  f2_first) POLICIES=(learned feedback_rule) ;;
  rule_first) POLICIES=(feedback_rule learned) ;;
  *) echo "ORDER must be f2_first or rule_first" >&2; exit 2 ;;
esac

env "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
  "LD_PRELOAD=/data2/liusy/reloc3r/outputs/runtime_compat/libitt_compat_stub.so" \
  "HF_HOME=/data2/liusy/reloc3r/.cache/huggingface" \
  "TORCH_HOME=/data2/liusy/reloc3r/.cache/torch" \
  "PYTHONPATH=third_party/stage3c_xgboost:." \
  /data2/liusy/conda_envs/reloc3r/bin/python -m tools.eval_active_evidence_sequential \
  --output-dir "${OUT}" --data-root data/lilocbench/prepared_1hz --gt-root data/lilocbench/gt \
  --mapping-gt data/lilocbench/gt/mapping_gt_poses.txt \
  --retrieval-root outputs/lilocbench/AVL-002 \
  --verifier-config configs/active_evidence_verifier_v1.json \
  --utility-config configs/stage3a_terminal_utility_v1.json \
  --backend reloc3r --device cuda --learned-policy-device cpu \
  --policies "${POLICIES[@]}" \
  --learned-policy-checkpoint outputs/lilocbench/stage3d_v2_controller_f2_r1 \
  --learned-policy-family stage3c6_utility --stage3d-preaction-features \
  --sequences "${SEQUENCE}" --start "${START_FRAME}" --max-frames "${MAX_FRAMES}" \
  --max-acquisitions 2 --oracle-depth 2 --budget-ms 30000 --skip-oracle --seed 1777
