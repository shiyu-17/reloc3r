# AVL-010：Strong Temporal Tracker Baseline

日期：2026-10-10
状态：`completed with limits`，不引入 Agent

## 1. 要回答的问题

AVL-009 使用 Reloc3r 相邻帧 edge 时出现明显累积漂移。本实验只替换 temporal module，检验：

1. 连续时序是否本身有独立定位价值；
2. 失败是否主要来自 Reloc3r temporal edge 太弱；
3. DPVO 是否能跨过 Dynamic 的短时遮挡，并改善 LT / Dynamic+LT 的连续失败。

使用官方 [DPVO 实现](https://github.com/princeton-vl/DPVO)，commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`，权重 `third_party/DPVO/dpvo.pth`，SHA-256=`30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12`。

## 2. 协议与实现

- 评测帧、mapping-only 数据库、40% validation / 60% test、GT 阈值和 AVL-009 一致；仍是 same-scene temporal hold-out，不是新场景泛化。
- 原始 LILocBench RGB 流约 15 Hz：static `8975`、dynamic `2398`、LT `6535`、Dynamic+LT `8378` 帧。DPVO 读取完整流，最终只在与 AVL-009 相同的 1 Hz query 时间戳上评测。
- 单目尺度不直接与 metric map 比较。每个 causal state 维护局部 DPVO 轨迹；当出现经冻结 dispersion/jump 规则接受的 global anchor 时，使用至少两个 anchor 做 Sim(3) 尺度/位姿校正。局部基线小于 `0.05 m` 不更新，尺度限制为 `[0.1, 10]`。
- DPVO 只导出每帧在线 pose；没有调用 `DPVO.terminate()` 的最终批量优化。所有 online trace 的 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。
- 主要比较：`Memory-3`、`DPVO temporal-only`、`Global+DPVO`、`Global+DPVO+Memory`、`Global+DPVO+Memory+RoMa recovery`、`Always-RoMa`。

实现与结果：

- tracker：`tools/run_avl010_dpvo.py`
- full-rate 数据准备：`tools/prepare_avl010_full_rate.py`
- causal state：`tools/run_avl010_stateful.py`
- 独立 evaluator：`tools/eval_avl010_dpvo.py`
- 最终 trace：`outputs/lilocbench/AVL-010/stateful_full_rate_v3/`
- 最终评测：`outputs/lilocbench/AVL-010/eval_full_rate_v3/`

## 3. Test 结果

成功定义为平移 `≤0.25 m` 且旋转 `≤5°`。

| 方法 | 全部 test success | coverage | 最大连续失败 |
|---|---:|---:|---:|
| Memory-3 | **76.87%** | 100.00% | 21 |
| DPVO temporal-only | 41.54% | 100.00% | 324 |
| Global+DPVO | 40.55% | 53.54% | 346 |
| Global+DPVO+Memory | 40.55% | 88.39% | 346 |
| Global+DPVO+Memory+RoMa | 68.31% | 99.11% | 76 |
| Always-RoMa | **92.62%** | 100.00% | 6 |

按条件：

| 条件 | Memory-3 | DPVO temporal-only | Global+DPVO | +Memory+RoMa | Always-RoMa |
|---|---:|---:|---:|---:|---:|
| Static | 91.04% | 49.13% | 0.00% | 77.75% | 93.64% |
| Dynamic | 56.99% | 39.78% | 55.91% | 55.91% | 93.55% |
| LT | 85.77% | 84.98% | **87.75%** | **87.75%** | 92.49% |
| Dynamic+LT | 60.49% | 0.00% | 42.59% | 46.60% | 91.36% |

## 4. 结论

### 已支持的结论

1. **“时序信息完全没有价值”不能成立。** 在 LT 上，Global+DPVO 比 Memory-3 高 `1.98 pp`，最长失败从 `9` 降到 `8`；这说明连续 tracking 在部分长期变化片段有小的独立收益。
2. **AVL-009 的失败不只是 Reloc3r edge 太弱。** 换成完整 15 Hz 的官方 DPVO 后，DPVO temporal-only 在 test 仍只有 `41.54%`，Dynamic 为 `39.78%`，Dynamic+LT 为 `0%`；强 temporal tracker 没有普遍跨过动态遮挡或组合变化。
3. **Dynamic+LT 仍是主要瓶颈。** 即使允许 RoMa recovery，最终为 `46.60%`，明显低于 Memory-3 `60.49%` 和 Always-RoMa `91.36%`；Global+DPVO 的保守 jump guard 还会造成大段 reject/coverage 缺口。
4. **当前数据支持“状态化环境值得保留，但不能据此进入 Agent”。** DPVO 在 LT 片段提供局部增益，但不足以证明一个 sequence-first stateful system 已经优于 map-based Memory/geometry 路线。

### 不能声称的结论

- 不能声称所有 VO/SLAM 都无效；本实验只测试 DPVO 官方模型、当前相机和 1 Hz 评测协议。
- 不能声称 DPVO 的绝对速度优势；tracker 时间是实测的，但 Reloc3r/global 和 RoMa 部分沿用 AVL-005 的混合估计。
- 不能把这个 same-scene temporal hold-out 当作跨场景泛化。

## 5. 成本与审计

- DPVO 完整 RGB 流在线 tracking 总计 `698.88 s`，均值约 `26–27 ms/frame`，GPU 约 `0.194 h`。
- 继承的 global Reloc3r 估计 `3273.02 s`；本次 recovery trace 选择固定 seed RoMa `17` 次、缓存时间 `51.09 s`；已知组件合计 `4022.99 s`。该合计仍含继承估计，不能用来宣称端到端加速。
- causal audit：四序列 DPVO online pose 分别为 `8975/2398/6535/8378`，全部有效；四个 trace 均通过 GT/未来信息审计，`terminate_called=false`。审计文件：`outputs/lilocbench/AVL-010/eval_full_rate_v3/audit/causal_audit.json`。

## 6. 主线决策

AVL-010 不改变主线 **Failure-aware Agentic Visual Localization**，但把 sequence-first 假设定为：

- **保留为受限组件**：DPVO 可以作为短期 motion/tracking 信号，尤其在 LT 片段作为辅助；
- **不进入完整 Agent**：没有出现稳定的 dynamic / Dynamic+LT 独立收益，且整体明显落后 Memory-3 / Always-RoMa；
- **下一步**：继续 deterministic failure analysis，重点是 global anchor 的可靠性、Dynamic+LT 的共同失败和强几何 recovery；只有出现“数值风险相近但需要不同上下文动作”的可审计样本，才重新考虑 Agent。

早期 1 Hz-only DPVO、旧 Sim(3) anchoring 和 evaluator 版本均保留在 `outputs/lilocbench/AVL-010/`，不覆盖、不混入本报告主结果。

## AVL-014 修正重验（2026-10-10）

旧版状态回放把带尺度 Sim(3) 的 3×3 块直接乘到 SE(3) 位姿，导致 propagated pose 的旋转不再正交；旧版派生 success 指标作废。复用原始 DPVO full-rate trajectory 修正后，test 结果为：DPVO temporal-only `0.20%`、Global+DPVO `4.04%`、Global+DPVO+Memory `4.04%`、Global+DPVO+Memory+RoMa `65.26%`、Always-RoMa `92.62%`。修正 DPVO propagated 输出的最大行列式偏差约 `1.4e-6`、最大正交误差约 `2.0e-6`。当前结论收窄为：在本 DPVO、Sim(3) anchor 和 causal 规则下，没有观察到独立定位收益；不外推为所有 temporal tracker。

证据：`outputs/lilocbench/AVL-014/AVL-010_stateful_sim3fix/`、`outputs/lilocbench/AVL-014/AVL-010_eval_sim3fix/`。旧 trace 保留在 `outputs/lilocbench/AVL-010/stateful_full_rate_v3/`，不再作为主指标。
