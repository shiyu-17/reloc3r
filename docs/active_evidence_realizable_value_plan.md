# 下一阶段：主动取证的可实现价值验证

最后更新：2026-10-14

## 1. 阶段目标

本阶段只回答一个问题：

> 在动态室内定位失败时，允许策略主动选择下一份 map、geometry、held-out verification 或
> historical evidence，是否在相同工具、验证器和预算下产生固定顺序无法获得的恢复收益？

本阶段不训练 VLM，不优化语言解释，不把某个 controller 在 LILocBench 同场景 split 上的
小幅提升当作 Agent 贡献。只有可实现 Oracle gap 在多场景、未见 episode 上成立，才进入
learned sequential Agent。

## 2. 冻结边界

- 冻结当前 Reloc3r、RoMa/HLoc-style backend、source-aware verifier 和成功阈值；
- 冻结 F2 与 feedback-rule，作为开发期参考，不再在现有 220 帧上调参；
- 所有策略共享相同 Accept 安全门，legacy auto-Accept 只保留为历史对照；
- DPVO 只提供 motion/uncertainty evidence；Historical Memory 使用受保护的简单实现；
- evaluator 才能读取 GT，在线 trace 保持 `query_gt_read=false`、`uses_future=false`、
  `selection_uses_gt=false`。

## 3. Episode 与动作

从基础定位冲突、verifier 失败、连续 Reject、stale anchor、遮挡恢复和错误 Memory 注入等事件
触发 recovery episode。episode 从首次 uncertain/lost 开始，以正确 Accept、错误 Accept、预算
耗尽或右删失结束；不再以随机单帧作为主要学习单位。

第一版动作固定为：

1. `AlternativeReference`：排除当前 reference cluster，获取另一组地图证据；
2. `StrongGeometry`：对选定 reference 调用强几何；
3. `HeldoutVerify`：用未参与候选生成的 reference 验证竞争位姿；
4. `QueryExperience`：查询历史关键帧并重新估计当前 pose；
5. `Finish`：在共同 verifier 门控下 Accept、Reject 或 Defer。

动作必须返回新的、带 provenance 的 observation，并计入真实工具成本。共享 reference、anchor
或 matcher 的结果不得计为独立证据。

## 4. 决定性实验

在训练策略前，先对每个训练/验证 episode 展开深度 2--3 的合法 action tree，比较：

- best fixed action / fixed order；
- feedback-rule；
- geometry-greedy active acquisition；
- budget-matched finite-depth Oracle。

必须同时报告 episode success、false accept、未恢复比例、time-to-first-correct、Memory
pollution、tool calls 和同硬件端到端成本。统计以 scene/episode 为重采样单位。

## 5. Go / No-Go

进入 learned Agent 至少满足：

- Oracle 相对最佳规则提高不少于 5 pp episode success，或在成功率相当时降低至少 20% 成本；
- 至少 10%--15% recovery states 存在 action-dependent terminal outcome；
- 收益不能仅来自更多工具调用，且 false accept / Memory pollution 不恶化；
- gap 在多个 scene-disjoint 场景上方向一致；
- pre-action structured evidence 对最优动作具有可复现的预测信号。

若 Oracle gap 不成立，停止 Agent 训练，改进 action pool、reference grouping 或 independent
verification。若 Oracle gap 成立但 structured policy 不超过规则，研究可观测性与数据覆盖；
只有确认 raw visual evidence 能补足缺口后才引入 Qwen-VL。

## 6. 执行顺序

1. 构建 failure-triggered episode miner 与 scene-disjoint manifest；
2. 实现统一 terminal contract 和参数化 evidence actions；
3. 缓存/在线展开深度 2--3 action tree，审计 Oracle gap；
4. 通过 Go 条件后训练小型 structured sequential policy；
5. 做 policy-induced 数据聚合；
6. 最后才比较 Agentic SFT、Qwen-VL 与可选 RL；
7. 在独立长期变化数据和实机 lost-and-recovery 上冻结论文评测。

## 7. 实现进度（2026-10-14）

- `tools/active_evidence_episodes.py` 已实现只使用 policy-visible receipt 的 episode
  miner、scene-id split 和 fail-closed 审计。Stage 3G 的100--154帧 trace 产生28个 episode；
  四条 LILocBench 序列属于同一物理场景，因此明确标为 `audit_only`。
- `eval_active_evidence_sequential.py` 已支持 episode manifest、causal warm-up 和 episode-id
  分片，可在同一接口下运行 fixed、fixed-geometry、feedback-rule、geometry-greedy
  和 budget-matched finite-depth Oracle。
- `tools/evaluate_active_evidence_value.py` 已实现 episode success、FAR、未恢复率、
  time-to-first-correct、Memory pollution、tool cost、terminal-outcome dependency 和冻结 Gate。
- 一个2帧 dynamics episode 已通过 CPU 和真实 GPU Reloc3r/RoMa depth-2 smoke；所有
  online causal audit 通过。该 smoke 只验收实现，不用于科学结论。

当前的硬阻塞不是代码或 GPU，而是 LILocBench 只有一个 scene id。大规模 Gate 实验
启动前必须接入至少三个物理场景，否则只能得到同场景的工程审计。

## 8. 隔离决策（2026-10-15）

PoseAgent-style profiler/ranker/verifier/fallback 与 7Scenes feasibility code 已移入独立实验
档案，不再属于本阶段主线，也不得作为主线 Agent 的依赖。其在 LILocBench 80 帧上的公平
fallback 对照没有带来稳定 success 增益：动作计划与既有 feedback-rule 在 80/80 帧一致，
仅余一帧结果差异，不能归因为新方法。详细代码、评测口径和结果保存在
`experiments/poseagent_style_20261015/README.md`。
