# AVL-009：严格因果连续序列视觉定位报告

最后更新：2026-10-10

## 1. 目的

AVL-009 不引入 LLM、VLM、SFT 或 Agent，只检验连续 RGB 的时序信息是否能改善 LILocBench 定位。在线处理严格按时间顺序进行，每一帧只能访问当前帧和过去状态；query GT 只由本文的独立 evaluator 读取。

## 2. 协议与实现

- 数据为现有四条 LILocBench query：`static_0=578`、`dynamics_0=155`、`lt_changes_0=422`、`lt_changes_dynamics_0=541`，共 1696 帧；mapping-only 数据库为 785 帧。
- 用 Reloc3r-512 对相邻帧 `(I[t-1], I[t])` 生成 1692 个 temporal edges；四序列有效率均为 100%。方向固定为既有 Reloc3r 约定 `world_pose[t]=world_pose[t-1] @ relative[t]`。独立方向审计中，正向约定在四序列的平移/旋转误差均优于取逆，但长期仍出现明显漂移。
- 状态包括当前位姿、最后可靠帧、连续失败计数、短期可靠历史、最近 global anchor、reference reliability 和 memory hold 计数。只有当前预测能通过预注册几何一致性规则时才更新状态；GT 未写入状态。
- 预注册阈值：global candidate dispersion `1.5 m/15°`；global-temporal 一致性 `2 m/30°`；单步 jump guard `3 m/45°`；memory 最多 hold 3 帧；RoMa verification 为 `inliers≥100` 且 `median reprojection≤3 px`。
- 40% 时间前缀为 validation，60% 后缀为 test；阈值未用 test GT 调整。该划分是 same-scene temporal hold-out，不是跨场景泛化。

比较方法：原始 Top-10、consensus-3、causal memory-3、Always-RoMa、temporal-only、global+temporal、global+temporal+memory、global+temporal+memory+RoMa recovery。

## 3. 主要结果

### 3.1 Test 总体（四序列合计 1016 帧）

| 方法 | 成功率 | 最大连续失败 | RoMa 调用率 |
|---|---:|---:|---:|
| Reloc3r Top-10 | 27.85% | 37 | 0 |
| consensus-3 | 72.15% | 20 | 0 |
| memory-3 | 76.87% | 21 | 0 |
| temporal-only | 0.00% | 346 | 0 |
| global + temporal | 53.64% | 270 | 0 |
| global + temporal + memory | 53.64% | 270 | 0 |
| global + temporal + memory + RoMa | 53.74% | 270 | 0.10% |
| Always-RoMa | 92.62% | 6 | 100% |

AVL-009 全 1696 帧的对应 success 为 `28.48 / 70.64 / 76.71 / 0.29 / 52.77 / 52.77 / 52.83 / 91.51%`。

### 3.2 各场景 Test

| 场景 | memory-3 | global+temporal | recovery | Always-RoMa |
|---|---:|---:|---:|---:|
| Static | 91.04% | 84.10% | 84.10% | 93.64% |
| Dynamic | 56.99% | 52.69% | 53.76% | 93.55% |
| Long-term change | 85.77% | 79.45% | 79.45% | 92.49% |
| Dynamic + LT | 60.49% | 1.23% | 1.23% | 91.36% |

`global+temporal` 的 test 最大失败段分别为 `5 / 9 / 11 / 270` 帧；memory-3 为 `3 / 9 / 9 / 21` 帧。Dynamic+LT 中 temporal edge 长时间漂移，global candidate 不可靠时仍会连续传播错误状态。

### 3.3 轨迹跳变

Test 中 global+temporal 的 jump rate（误差超过 `1 m` 或 `20°`）为 static `4.1%`、dynamic `6.7%`、LT `2.8%`、Dynamic+LT `0.9%`。低 jump rate 不代表定位正确：Dynamic+LT 的主要问题是平滑但错误的长期漂移，而不是瞬时跳变。

## 4. Failure analysis 与问题回答

- **Q1：时序状态是否比独立帧更稳定？** 当前 Reloc3r temporal edge 不能证明这一点。temporal-only 在 test 上几乎完全失效；global+temporal 也低于 memory-3 和 Always-RoMa。
- **Q2：时序主要改善什么？** 在当前实现中没有稳定改善。它能产生平滑轨迹，但 1 Hz 相邻帧 Reloc3r 相对位姿误差会累积，平滑不等于正确。
- **Q3：能否跨过短时动态遮挡？** 没有得到可信的正证据。Dynamic 中 recovery 只比 global+temporal 多恢复少量帧；Dynamic+LT 仍形成 270 帧失败段。
- **Q4：长期变化是否仍需 global relocalization？** 是。LT 和尤其 Dynamic+LT 中，纯 temporal 不能长期维持，必须回到可靠 global 或强几何工具。
- **Q5：temporal 后 memory 还有多少额外收益？** 本协议下 global+temporal 与加 memory 的总体结果完全相同，说明当前时序候选一旦进入漂移状态，简单 hold memory 没有挽救能力；memory-3 单独相对 consensus-3 仍有明显收益。
- **Q6：何时必须调用 RoMa？** 当前最明确的触发证据是 global/temporal 同时不一致或 jump guard 失败。固定缓存 recovery 只触发 1/1696 帧，因此不能把这次小调用率结果解读为完整在线 RoMa 路由优势；Always-RoMa 仍是精度上界参考。

## 5. 成本与审计

- temporal Reloc3r edge 的实测 inference batch 时间合计 `457.43 s`；第一版 edge 运行没有单独记录 preprocessing，因此这是保守的 inference-only 计时。
- global Reloc3r 使用 AVL-005 八帧样本的冻结 Top-10 timing 做透明缩放估计 `3273.02 s`；不是本实验逐帧重新测量。
- retrieval `25.60 s`；recovery 固定缓存 RoMa 实际使用 1 次、`2.94 s`。
- 当前已知成本组件（global+temporal+retrieval+cached recovery+memory/verification proxy）合计 `3845.38 s`；其中 GPU 相关估计约 `1.04 GPU-hours`。reference reranking 未在 AVL-009 recovery replay 中重新计时，明确是混合“实测 + 继承估计”，不支持端到端加速结论。
- 所有最终 trace 均标记 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。方向审计为 evaluator-only，不改在线方向。

## 6. 无效运行与修正

- `outputs/lilocbench/AVL-009/invalid_fusion_v1/`：初版在 global 可靠但与 temporal 冲突时错误优先 temporal，违反预登记规则；已保留并排除。
- `outputs/lilocbench/AVL-009/invalid_recovery_v2/`：初版 recovery 只在 reject 触发，未覆盖低置信度 memory hold；已保留并排除。最终版加入 `reject_or_low_confidence` 触发。

## 7. 阶段结论与 Go/No-Go

AVL-009 建立了一个可审计的严格因果序列 baseline，但没有证明“用当前 Reloc3r 做 temporal edge”具有独立定位收益。当前最可信的解释是：Reloc3r 单对相对位姿在 1 Hz 间隔上误差累积，造成平滑漂移；简单状态融合和 hold memory 不能自动识别这种漂移。

因此：

1. 不进入完整 LLM/VLM Agent；Agent 继续 **Hold / No-Go**。
2. 不把本次 temporal 失败外推为所有 VO/SLAM 时序方法失败。
3. 若继续 Stateful 路线，下一步应先替换或补充真正的 temporal tracker（例如 DPVO/几何光流），并在同一 validation/test 协议下重新验证；随后才判断是否值得引入 Agent。

证据入口：`outputs/lilocbench/AVL-009/summary.json`、`validation/`、`test/`、`failure_analysis/condition_test_metrics.json`、`cost/summary.json`、`validation/temporal_direction_audit.json`。
