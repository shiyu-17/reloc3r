# AVL-013：Closed-loop Reject & Recovery

日期：2026-10-10。实验不引入 Agent，检验 Reject 是否能在连续序列中阻断错误状态传播，并在后续帧恢复。

## 在线闭环

每帧按以下顺序处理：

```text
当前候选/geometry → verification → risk threshold=1.5
    → Accept：写回 prediction/Memory
    → Reject：不写回，下一帧重新尝试
```

验证集前 40% 只用于确认阈值，测试集后 60% 冻结评估。所有状态更新只使用当前和过去信息。

## v1：严格陈旧状态交叉检查

如果 Reject 后仍把最后一个可靠 pose 无限期作为 jump reference，系统出现 recovery deadlock：test coverage 仅 `0.39%`，最长连续 Reject `991` 帧。该版本保留为 exploratory，说明“Reject 后不能继续用陈旧状态做无限期惩罚”。

## v2：stale-state recovery

连续 Reject 达到 3 帧后，暂时停止使用陈旧 previous-prediction 的 temporal/cross-check 惩罚，只依赖当前候选、当前观测和 geometry intrinsic evidence。只有通过阈值的 pose 才能重新写回状态。

### Test 结果（1016 帧）

| 方法 | Coverage | 最终全帧成功率 | 接受 pose 准确率 | 错误接受率 | 最长连续 Reject | 恢复到首个正确 pose |
|---|---:|---:|---:|---:|---:|---:|
| AVL-011 no-reject | 100.00% | 91.24% | 91.24% | 8.76% | 0 | — |
| AVL-013 v1 | 0.39% | 0.30% | 75.00% | 25.00% | 991 | 22 帧 |
| **AVL-013 v2** | **54.23%** | **51.57%** | **95.10%** | **4.90%** | **26** | **4.27 帧** |

v2 接受 `551/1016` 帧，其中正确 `524`、错误 `27`，拒绝 `465` 帧。错误状态写回从 no-reject 的 `89` 降到 `27`，连续错误状态传播 proxy 从 `30` 降到 `5`。

按条件，v2 test 的 coverage / 最终 success / accepted accuracy 为：

- static：`68.79% / 64.45% / 93.70%`
- dynamic：`29.03% / 27.96% / 96.30%`
- LT：`54.15% / 52.17% / 96.35%`
- dynamic+LT：`45.99% / 44.14% / 95.97%`

## 成本

本次 state machine replay 的实测 CPU 墙钟时间为 `101.91 s`，历史 RoMa cache 中对应 geometry 调用的实测 elapsed 合计为 `2634.97 s`。但 AVL-013 没有重新执行 Reloc3r/RoMa 神经推理，因此 `cost_complete_online_inference=false`；这些 neural 数字不能冒充本批次新的端到端在线 latency。完整成本账本见 `outputs/lilocbench/AVL-013/cost_ledger.json`。

随后登记了全帧 Reloc3r 在线计时尝试；由于启动时未设置本地 Hugging Face 离线缓存，进程尝试访问 Hub，随后 GPU driver 在并行任务中不可用，未生成有效 pose shard。该失败运行保留在 `outputs/lilocbench/AVL-013/online_cost/`，不进入任何结论。

## 结论

1. Reject 确实减少了错误 pose 写回和错误状态传播。
2. Reject 本身不能保证恢复；若继续依赖陈旧状态，系统会死锁。
3. stale-state recovery 能把恢复时间降到约 4.27 帧，但 coverage 只有 54.23%，最终 success 远低于 no-reject baseline。
4. 当前 deterministic Reject 已暴露出“何时重新信任全局 geometry”这一真实瓶颈，但还没有 Agent Go 证据。

下一步：补齐同一协议下的真实 Reloc3r/RoMa 在线计时，并研究 coverage–error–recovery 的联合阈值；继续暂缓 Agent。

产物：

- v1 trace：[closed_loop_t1.5](/data2/liusy/reloc3r/outputs/lilocbench/AVL-013/closed_loop_t1.5)
- v2 trace：[closed_loop_v2_t1.5](/data2/liusy/reloc3r/outputs/lilocbench/AVL-013/closed_loop_v2_t1.5)
- 评估：[evaluation_v2.json](/data2/liusy/reloc3r/outputs/lilocbench/AVL-013/evaluation_v2.json)
- 实现：[run_avl013_closed_loop.py](/data2/liusy/reloc3r/tools/run_avl013_closed_loop.py)、[eval_avl013_closed_loop.py](/data2/liusy/reloc3r/tools/eval_avl013_closed_loop.py)

## AVL-014 v3 修正重验（2026-10-10）

v3 使用 selective-geometry 重校准后的 threshold=`1.5`，并修正 state trace（记录更新前可靠性与上一接受帧）及 evaluator 的跨序列 run 聚合、右删失统计。test coverage=`54.23%`、全帧 success=`51.57%`、accepted-pose accuracy=`95.10%`、最长 Reject=`26`；119 个 Reject 段中 117 个恢复、2 个右删失，平均恢复到首个 accepted=`3.87` 帧、首个 correct=`4.21` 帧。核心 coverage/error 结论不变：Reject 减少错误 state 写回，但当前恢复证据过于保守。

证据：`outputs/lilocbench/AVL-014/AVL-013_v3/evaluation.json`。旧 v1/v2 结果保留为历史/探索性 artifact。
