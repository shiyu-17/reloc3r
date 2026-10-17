# Stage-2 dataset readiness audit (2026-10-15)

This is an engineering readiness record, not a performance claim.

## Local inventory

| Dataset | Local evidence | Suitable use now | Prohibited conclusion |
| --- | --- | --- | --- |
| LILocBench | Four public query sequences and mapping GT are present. The episode manifest has 28 failure-triggered spans, all labelled `lilocbench_site_0`. | Protocol-v2 engineering audit and finite-depth action labels. | Scene-disjoint training, generalization, or multi-scene Agent result. |
| RIO10 `scene01` | `seq01_01` and `seq01_02` contain ordered RGB and per-frame `frame-*.pose.txt`; test `seq01_03`--`seq01_06` have RGB but **no local pose files**. `camera.yaml` and calibration are present. `metadata.json` defines train=`seq01_01`, val=`seq01_02`, and tests=`seq01_03`--`seq01_06`. | Mapping/validation coordinate-convention and sequence-order adapter check only. | Any local RIO10 test GT, training label, or scene-level performance result. |
| RIO10 `scene02`--`scene10` | Local raw folders are absent or partial (scene02 567 MB, scene03 226 MB, scenes04--10 effectively empty). `scene09/seq09.zip` fails `unzip -t` with no central directory. | None until complete, checksummed source assets and split/GT conversion are present. | Multi-scene evaluation or training labels. |
| 7Scenes | Multiple static scenes, image/pose sequence structure locally present. | Static backend and coordinate-adapter checks only. | Dynamic-environment Agent evidence. |
| 12Scenes | Twelve static indoor subscenes with train/test structure locally present. | Static backend and coordinate-adapter checks only. | Dynamic-environment Agent evidence. |

## Coordinate and split rules for RIO10 scene01

1. Treat every available `frame-*.pose.txt` as GT only in the evaluator until the pose convention has been verified against `camera.yaml` and a mapping/query registration check.  Local test sequences have no such files and cannot be labelled.
2. Preserve numeric frame order, rather than filesystem enumeration order.
3. Keep the published sequence role intact: mapping/train `seq01_01`, validation `seq01_02`, evaluation `seq01_03`--`seq01_06`.
4. No branch of a query episode may cross a split boundary.  A later scene-level learning split needs at least three *complete physical scenes*, not multiple sequences of `scene01`.

## Consequence

The current runnable Agent evidence is LILocBench audit-only.  RIO10 scene01 can validate a mapping/validation adapter but cannot satisfy the planned multi-scene gate; its local test GT is absent.  No RIO10-derived Oracle labels may enter a training set until complete query poses and the adapter's GT convention/mapping-query coordinate consistency are demonstrated.
