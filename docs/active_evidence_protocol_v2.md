# Active-evidence protocol v2

This protocol is the first common evaluation contract for the next-stage
localization Agent.  It replaces the earlier mixed controller interfaces only
for newly generated traces; historical Stage 3 artifacts remain unchanged.

## Online contract

Every controller receives `Observation + LegalActions` and returns one member
of `LegalActions`.  The environment executes that action and emits a receipt.
The first action space is:

- `localize(reference_group)` for each untried reference group;
- `geometry(reference_group)` for every reference group, once per frame;
- `accept` only after the common verifier accepts the current candidate;
- `defer` after at least one action, or when no acquisition is available.

Held-out verification remains a mandatory environment operation after every
pose-producing action.  It is charged as its own receipt cost, but it is not a
policy choice in v2.  The contract reserves a future parameterized
`verify(candidate, evidence_group)` action without exposing it now.

Memory is retained as causal state: accepted anchors, age, reject streak, and
the write ledger remain visible where appropriate.  `retrieve_memory` is
disabled in v2, so a historical pose cannot be used as a current-frame pose
proposal.

## Trace separation

Each v2 JSONL row contains non-sensitive routing metadata plus two namespaces:

```text
online:
  observation/decisions, legal action parameters, execution receipts,
  measured cost, terminal action, causal audit

evaluator:
  accepted pose, Memory write event, Oracle action values, terminal leaves,
  search configuration and evaluator wall time
```

Ground truth, pose error, success, and Oracle values must stay in `evaluator`.
Image paths and reference identifiers can be preserved in `online` for a later
VLM policy, but current structured policies read only the reference-ID-free
features in `tools/stage3b_policy.py`.

## Evaluation meaning

The finite-depth Oracle searches a legal action tree within one frame.  It is a
GT-only, cost-aware upper bound for *that frame's* executable program; it is
not an optimal planner for an entire recovery episode.  Episode metrics are
computed from the consecutive closed-loop trace.  Missing trace frames are
reported explicitly and right-censor the affected episode.  An unrecovered
episode is one that finishes without a correct accept and is not censored.

False acceptance and Memory pollution are distinct metrics.  A pollution event
requires both an incorrect accepted pose and an actual Memory write receipt.
An incorrect Accept does not terminate online execution, because the deployed
agent cannot know it is incorrect.  It remains a recorded safety event; the
next frame can recover from the resulting state.  Episode reports therefore
include both `episode_success` and `episode_has_false_accept`, rather than
silently treating a later recovery as if the earlier unsafe acceptance did not
happen.
