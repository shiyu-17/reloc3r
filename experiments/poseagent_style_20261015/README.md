# PoseAgent-style feasibility archive

**Status:** retired experiment; not imported or executed by the active visual-localization Agent.

This directory freezes a short PoseAgent-inspired, no-Memory detour: candidate
profiling, rule/structured ranking, verification, fallback control, and a 7Scenes
feasibility backend.  It was isolated on 2026-10-15 so that the active code path
returns to the original dynamic indoor localization Agent: causal state,
active evidence acquisition, guarded historical Memory, and safe Accept/Defer.

## Evidence and decision

The comparison used the same 80 LILocBench frames (four sequences, 20 frames
each).  The initial static-ranker version achieved 42/80 success, versus 49/80
for the existing feedback-rule.  The discrepancy was an implementation-policy
mismatch: after a cheap candidate was verifier-rejected, the ranker could select
another cheap action rather than geometry.

After the fair, parameter-free repair (force same-reference Geometry after that
rejection), PoseAgent-style achieved 49/80 and feedback-rule 50/80.  The two
controllers selected the same first localize reference on 80/80 frames and the
same Geometry presence/reference on 80/80 frames.  The remaining one-frame
difference is not evidence of a new policy benefit.  Consequently, do not train
the ranker/PVN stack or use this route as the paper mainline.

The 7Scenes smoke also did not pass the value gate: its frozen
`redkitchen`/`stairs` Oracle smoke had zero accepted poses and a 0 pp Oracle
episode-success gap.  This is retained as a backend-coverage finding, not as a
negative result for the Agent question.

Raw receipts remain under:

- `outputs/lilocbench/poseagent_ablation_r1/`
- `outputs/lilocbench/poseagent_geometry_parity_smoke/`
- `outputs/lilocbench/poseagent_geometry_parity_r1/`
- `outputs/7scenes/`

## Contents

`tools/` and `configs/` are frozen source snapshots.  Their imports retain the
then-current repository layout and are deliberately not maintained as runnable
mainline entry points.  They may be restored only for forensic comparison; any
future reuse must begin with a fresh, independently justified experimental plan.
