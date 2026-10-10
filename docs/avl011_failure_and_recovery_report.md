# AVL-011：Dynamic+LT 共同失败与强化 deterministic recovery

日期：2026-10-10
范围：不引入 Agent；所有在线动作只使用当前/过去信息。

## 1. 目的

本实验回答两个问题：

1. Dynamic+LT 中 Reloc3r/Memory 与 RoMa 同时失败时，失败更像是 reference 不可用、reference 选错、动态观测退化、长期结构变化，还是工具输出错误？
2. 能否把 `Reloc3r → reliability → Memory → RoMa → verification → Reject` 做成强的、可审计的非 Agent baseline？

## 2. 诊断协议

诊断使用 541 个 Dynamic+LT query 帧的 GT 后验结果，GT 只用于 evaluator，不参与任何在线动作。使用的输入包括：

- AVL-002 的 Reloc3r candidate dispersion、reference score 和图像变化 proxy；
- AVL-003 的 causal Memory-3 结果和内部一致性标记；
- AVL-004 full-frame RoMa 结果及独立 GT 误差；
- mapping-only Top-100 retrieval 的空间近邻检查。

`reference_unavailable_proxy`、`occlusion_or_motion_degradation_proxy`、`persistent_scene_change_proxy` 等名称是 operational proxy，不是人工语义真值。

## 3. Dynamic+LT 失败结构

| 结果类别 | 帧数 |
|---|---:|
| Memory 与 RoMa 均成功 | 326 |
| Memory-only | 14 |
| RoMa-only | 163 |
| both-fail | 38 |

空间 reference 检查结果：Top-100 中 516/541 帧存在空间上接近的 mapping reference，25 帧属于 Top-10 选错但 Top-100 有近邻；没有帧满足“Top-100 全部空间上不可用”的 proxy 条件。

在 38 个 both-fail 帧中：

- 32 帧有 reference mismatch proxy；
- 37 帧有 candidate conflict；
- 22 帧有 low-texture/blur proxy；
- 8 帧有 occlusion-or-motion proxy；
- 5 帧有 persistent-scene-change proxy；
- 38 帧均有 RoMa pose 输出，但后验位姿错误；
- 16 帧的 Memory candidate 在几何上“内部一致”，但后验仍错误。

因此，当前最重要的病因不是“地图完全没有覆盖”，而是：**空间上有近邻 reference，但外观/结构不兼容；候选会形成一致的错误簇；RoMa 的高 inlier/低 reprojection 也可能错误接受；Memory 的内部一致性不能单独证明可靠。**

代表帧和 contact sheets：

- [both-fail](../outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/both_fail.jpg)
- [Memory-only](../outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/memory_only.jpg)
- [RoMa-only](../outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/roma_only.jpg)
- [诊断摘要](../outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/summary.json)

## 4. 强化 deterministic baseline

规则只在 validation prefix 上选择，test 阈值冻结。最终规则：

1. candidate conflict 直接升级 geometry；
2. geometry 通过 pose/inlier/reprojection/pair-support 基本门槛才接受；
3. geometry 失败时优先使用当前可靠 Consensus，再使用过去的 Memory；
4. 没有可靠回退时 Reject；
5. 只有 accepted output 才能更新 Memory；
6. trace 明确记录 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。

### Test 结果（1016 帧）

| 条件 | Success | Geometry call rate | 最长失败 |
|---|---:|---:|---:|
| static | 91.62% | 45.38% | 4 |
| dynamic | 92.47% | 87.10% | 2 |
| LT | 90.12% | 69.17% | 4 |
| Dynamic+LT | 91.36% | 83.02% | 4 |
| 总体 | 91.24% | 67.13% | 4 |

总体平均 recovery latency 为 1.47 帧，coverage 为 100%。这已经接近当前 Always-RoMa 参考，但没有证明端到端速度优势：RoMa cache 的历史计时边界与 Reloc3r、rerank 不完全统一。

最终文件：

- [causal trace 与配置](../outputs/lilocbench/AVL-011/final_v1/)
- [独立评测](../outputs/lilocbench/AVL-011/final_v1/evaluation.json)
- [诊断实现](../tools/analyse_avl011_dynamic_lt_failures.py)
- [baseline 实现](../tools/run_avl011_strong_recovery.py)
- [评测实现](../tools/eval_avl011_recovery.py)

## 5. 科学结论

AVL-011 支持把主线表述从“sequence-based localization Agent”调整为：

> **Failure-aware Visual Localization in Dynamic Indoor Environments**

连续 RGB 仍然提供 Memory、failure streak、motion evidence 和 reference reliability，但连续 tracking 不再承担主要创新性。当前最值得解决的是错误 reference/一致性错误簇、RoMa 错误接受和 Memory 可靠性校准；这些问题在 deterministic baseline 中仍然存在，因此现在不进入 Agent。

限制：诊断类别不是语义真值；full-frame RoMa cache 的 seed/运行版本边界与 AVL-005 不完全相同；结果是 LILocBench 内部 calibration-shift test，不是独立场景泛化证据。

## AVL-014 jump-guard 配对消融（2026-10-10）

在相同 mapping/reference/RoMa cache 和所有非 jump 阈值下，将 geometry jump guard 从旧配置 `100m/180°` 收紧为 `2.5m/60°`。test coverage=`98.82%`、success=`87.50%`、最长失败=`14`、geometry call=`67.13%`；旧宽松配置为 `100%/91.24%/4`。因此 `91.24%` 应称为 cached deterministic route 结果，不能表述为严格运动验证后的最终可靠性。证据：`outputs/lilocbench/AVL-014/AVL-011_jumpguard/evaluation.json`。
