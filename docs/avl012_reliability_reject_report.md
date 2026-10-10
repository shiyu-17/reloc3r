# AVL-012：Localization Reliability & Reject

日期：2026-10-10。实验不引入 Agent，目标是检验系统能否识别“虽然有 pose，但证据不够可信”的帧。

## 做了什么

从 AVL-011 的因果闭环记录中，为每帧计算一个不看查询 GT 的风险分数。分数综合了：

- Reloc3r 候选的平移/旋转分散度；
- 当前帧与历史帧的外观/运动变化；
- RoMa 的 inlier、重投影误差、多 reference pair 支持；
- RoMa 与 consensus、Memory、上一可靠位姿之间的交叉检查。

验证集只用来选择阈值，测试集只用于画曲线。阈值网格为 `0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3`。

## 主要结果

验证集按“coverage 至少 50% 时选择性准确率最高”冻结阈值 `1.5`。测试集结果如下：

| 风险阈值 | Coverage | 被接受 pose 的准确率 | 错误接受率 |
|---:|---:|---:|---:|
| 0.50 | 5.91% | 96.67% | 3.33% |
| 0.75 | 22.64% | 96.52% | 3.48% |
| 1.00 | 37.50% | 96.33% | 3.67% |
| 1.25 | 48.23% | 96.12% | 3.88% |
| **1.50** | **55.22%** | **95.72%** | **4.28%** |
| 2.00 | 62.70% | 95.45% | 4.55% |
| 3.00 | 74.70% | 94.60% | 5.40% |

冻结阈值 `1.5` 在 1016 个测试帧中接受 561 帧：537 帧正确、24 帧错误，455 帧拒绝。接受的动作包括 geometry recovery 242、consensus 169、Memory 150 帧。

作为对照，AVL-011 不拒绝时 coverage 为 100%、success 为 91.24%。因此 AVL-012 的收益是减少错误接受，而不是提高全帧覆盖率；它牺牲了大量可用输出，尚不能直接部署。

按条件，阈值 1.5 的 test coverage / selective accuracy 为：static `70.81% / 93.88%`、dynamic `29.03% / 100.00%`、LT `55.73% / 96.45%`、dynamic+LT `45.68% / 97.30%`。dynamic 的低 coverage 说明当前风险分数偏保守。

## 结论与限制

1. 多 reference 支持、重投影、候选一致性和历史交叉检查确实能提供可用的可靠性排序；阈值越严格，错误接受率越低。
2. 当前分数无法同时保持高 coverage 和高选择性准确率，说明仅靠一个固定风险阈值还不够。
3. 本实验复用了 AVL-011 的 RoMa 缓存和已生成 trace，是 deterministic post-hoc calibration，不是新的在线端到端计时，也没有重新执行 Reject 后的状态回灌。
4. 所有 score trace 均保留 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`；查询 GT 只在独立 evaluator 中读取。

产物：

- 逐帧 evidence：`outputs/lilocbench/AVL-012/*/evidence.jsonl`
- 分数协议：`outputs/lilocbench/AVL-012/score_manifest.json`
- 验证/测试曲线：`outputs/lilocbench/AVL-012/evaluation.json`
- evaluator-only 误差行：各 sequence 的 `evaluator_rows.jsonl`
- 代表帧：`outputs/lilocbench/AVL-012/contact_sheets/`
- 实现：`tools/run_avl012_reliability.py`、`tools/eval_avl012_reliability.py`、`tools/make_avl012_contact_sheets.py`

决策：继续 deterministic verification / Reject 的在线闭环校准；不进入 Agent。下一步优先测试 Reject 后状态更新和统一成本，而不是直接构建 LLM Manager。

## AVL-014 selective-geometry 重校准（2026-10-10）

原 AVL-012 evidence branch 在未触发 geometry 的帧也读取了 Always-RoMa cache，因此旧阈值曲线不能直接代表 selective-geometry 在线策略。修正后只有 causal route 实际 `call_geometry` 的帧读取 geometry evidence；validation 仍选择 threshold=`1.5`。test coverage=`55.41%`、selective accuracy=`95.56%`、错误接受率=`4.44%`。该结果仍是缓存回放校准，不是完整 neural online latency。

证据：`outputs/lilocbench/AVL-014/AVL-012_selective/evaluation.json`。
