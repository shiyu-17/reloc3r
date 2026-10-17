# Tools layout

The active research path is intentionally small after the Stage 3 closeout.

## Current mainline

- `active_evidence_protocol.py`: v2 shared legal-action and trace contract.
- `agentic_localization_env.py`: causal environment and guarded Memory writes.
- `lilocbench_agent_backend.py`: live Reloc3r/Geometry backend.
- `eval_active_evidence_sequential.py`: shared sequential evaluator.
- `stage3b_policy.py`, `stage3c6_utility_policy.py`, `stage3d_features.py`: frozen structured baselines.
- `audit_stage3f_f2_disagreements.py`, `close_stage3.py`: Stage 3 audit and closeout.
- `active_evidence_episodes.py`: failure-triggered episode mining and scene-disjoint split audit.
- `evaluate_active_evidence_value.py`: episode metrics and frozen realizable-value Go/No-Go gate.
- `test_agentic_localization_env.py`, `test_stage3b_terminal_control.py`: core causal regressions.

## Frozen historical experiments

Files named `avl*`, `stage3c*`, `stage3d_geometry*`, or `stage3e*` reproduce completed ablations.
They are not imported by the next-stage mainline unless a frozen baseline explicitly requires them.
Do not extend them with new policy variants. New active-evidence work should use the environment and
shared evaluator above.

The historical scripts remain in place because their outputs are cited by the experiment registry.
Generated `__pycache__`, one-off smoke outputs, and superseded shell wrappers are not part of the
research artifact and may be removed.  The retired PoseAgent-style and 7Scenes feasibility work is
preserved, but intentionally isolated under `experiments/poseagent_style_20261015/`; it is not a
mainline dependency.
