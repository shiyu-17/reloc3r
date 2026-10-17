# Active Evidence Pilot：主动取证是否已经证明有价值

最后更新：2026-10-12

## 结论先行

在当前可执行的 CPU-ORB causal action pool 上，主动 reference 选择**尚未超过固定顺序**。在 304 个测试帧上，fixed 与 calibration-prefix 训练的 light policy 都得到 `81/304=26.6%` full-frame success；rule 为 `77/304=25.3%`，random 为 `69/304=22.7%`。light 与 fixed 的 evidence 调用数几乎相同，不能把结果写成主动取证收益。

这次结果比第一轮 pilot 更严格：历史关键帧不再直接复制旧 pose，而是建立局部 3D 并重新 PnP；verification 也不再复用候选自身的 inlier，而是用未参与候选生成的 held-out map references 做独立极线一致性检查。结论仍然是：**暂不训练 VLM/Agent；先在真实 GPU Reloc3r/RoMa 上复验强几何动作是否提供真正的 action complementarity。**

## 闭环补齐（2026-10-12，尚未产生新结果）

为修正上一轮“动作预排序”和不完整预算/Oracle 的问题，代码已增加以下**待运行协议**；旧 304 帧数值不应使用它们重新解释。

- `run_strong_geometry` 成为环境中的一等在线动作，和 reference action 一样强制返回 `executed=true`、`cache_hit=false`、当前 query ID、精确 references、执行阶段与因果审计。GPU `LILocBenchReloc3rBackend` 选择该动作时会实时执行 **RoMa dense matching → mapping-view triangulation → query PnP**，不读取 AVL-004 缓存；当前无 CUDA，故尚未运行。CPU backend 提供明确标为 `SIFT+triangulation+PnP CPU proxy` 的接口验收分支，不能替代 RoMa 论文结果。
- 新入口 `tools/eval_active_evidence_sequential.py` 在每次 acquire/verify 后读取候选的 intrinsic、held-out/state risk、剩余 reference、Memory 和已消耗时间，再选择下一组 reference、强几何、Memory 或 defer。`fixed` 是控制程序；`random`、`feedback_rule` 均在同一 action/state space 中运行，不再仅比较预先算好的 group order。
- 预算改为 backend receipt 的实测 `elapsed_s` 累加，仍报告 action count 作为安全上限；若单一动作执行后首次超预算，会显式记录 `budget_overshoot`，随后禁止新增 acquisition。accept/defer 的状态机开销不伪装为模型工具耗时。
- Oracle 改为 evaluator-only 的有限深度可执行树搜索：分支实际经过同一环境、同一 verifier gate、同一动作集合和同一成本预算；GT 只在叶节点选择正确的已验证程序。输出另记搜索计算量，绝不把 Oracle 搜索耗时写成部署延迟。

该补齐的最小单测现为 `10/10` 通过（含强几何 live receipt）。

### GPU 在线验收（2026-10-12）

- 在空闲 RTX 4090（物理 GPU 4）使用用户指定 `reloc3r` 环境完成 CUDA tensor smoke。该环境的 PyTorch 2.5.1 与 MKL 2025 缺少 legacy `iJIT_*` profiler 动态符号；`tools/itt_compat_stub.c` 只通过 `LD_PRELOAD` 关闭该 profiler，并不改动 CUDA、模型权重或推理算子。
- `outputs/lilocbench/AVL-agent-gpu-smoke/roma_static_0_frame000.json` 是首个真实强几何 receipt：`static_0/0` 对 mapping `[784,783,782]` 实时运行 RoMa dense matching、map-view triangulation、PnP，`cache_hit=false`，耗时 `14.065s`，`2839` PnP inliers、重投影中位数 `0.613px`，环境审计通过，未读 query GT/未来帧。
- `outputs/lilocbench/active_evidence_gpu_smoke/summary.json` 验收了完整的 GPU 闭环：`dynamics_0/0` 的 Reloc3r 3-pair action 实测约 `0.21–1.31s`，候选低风险并被 verifier 接受，因此 feedback rule 合理地停止而不调用 RoMa；fixed/random/feedback-rule 与 Oracle 的因果审计均通过。该单帧都成功，**不能**作为任何 policy 优劣结论。
- `eval_active_evidence_sequential.py` 现复用每个序列已加载的只读 model backend；不同 policy/Oracle 的 `AgenticLocalizationEnv` 状态和 Memory 仍完全隔离，模型初始化耗时不会进入 action budget。

下一步是以冻结的 GPU/模型/预算配置跑完四个同场景 query 序列，并报告每类动作的真实调用率、延迟、full-frame success、false-accept 和 recovery；这一步仍不是 scene-disjoint 泛化实验。在此之前，GPU smoke 只证明“动作真实可执行”，不证明 agent 有增益。

## GPU 全序列闭环结果（2026-10-12）

产物：`outputs/lilocbench/active_evidence_gpu_full_20261012/aggregate_summary.json`。在四个 LILocBench query 序列的全部 `1696` 帧上，四张 RTX 4090 并行运行真实 Reloc3r/RoMa；每帧最多两次 acquisition、工具预算 `30s`、Oracle 为 depth-1 的可执行且 verifier-gated 终端 GT 选择器。所有在线 policy 与四个 Oracle trace 的 causal audit 均通过（`query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`）。

| policy | full-frame success | accepted accuracy | false-accept rate | strong-geometry calls | measured tool cost / frame | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| fixed | `1104/1696 = 65.09%` | 94.12% | 5.88% (`69`) | 0.0% | 0.588s | 15.09pp |
| random | `958/1696 = 56.49%` | 87.25% | 12.75% (`140`) | 37.97% (`644`) | 1.177s | 23.70pp |
| feedback-rule | `1276/1696 = 75.24%` | 89.36% | 10.64% (`152`) | 35.02% (`594`) | 0.996s | 4.95pp |
| depth-1 Oracle | `1360/1696 = 80.19%` | 100% | 0% | 13.03% (`221`) | 0.492s selected-program cost | — |

**可成立的结论**：在同一真实工具、verifier、30s budget 和连续状态下，利用上一轮 verifier feedback 的规则能比 fixed 提高 `10.14pp` full-frame success（并把 Oracle gap 从 `15.09pp` 降至 `4.95pp`）；random 显著更差。因此，“主动取证动作存在可利用的条件互补”已经获得 GPU online 证据，不能再用 CPU-ORB pilot 的 No-Go 结论否定它。

**同样重要的代价**：feedback-rule 的调用强几何比例为 `35.02%`，平均工具耗时高 `69.5%`，错误接受率由 `5.88%` 升至 `10.64%`。所以当前结果不是“更可靠的 agent”，而是 **success–cost–false-accept trade-off 的可行动基线**；IROS 论文必须学习/校准选择与 verifier，以在相同 success 下少调强几何，或在相同 false-accept 约束下超过 fixed。

分序列中，feedback-rule success 为 static/dynamic/LT/combo=`91.00/50.32/73.93/66.54%`，对应 Oracle=`92.73/58.71/80.09/73.01%`；dynamic 与 dynamic+LT 的 geometry call rate 分别为 `62.58%/48.24%`，是下一轮 cost-aware policy 与 source-aware verifier 优先改进对象。

**Oracle 边界**：其 `0.492s` 是最终被 GT 选中的程序动作成本；为搜索候选分支实际平均花费约 `4.02s/frame`，不计入部署延迟，且 depth-1 不覆盖两步动作树。因此它仅是受限的后验上界，不能作为线上速度或策略性能主张。Oracle 的 100% accepted accuracy 是“GT 只选择正确叶，否则 defer”的定义性结果，不代表当前 verifier 能达到零 false accept。

## 受控协议

- 每帧从 mapping-only retrieval 的 Top-10 中构造 reference group，每组最多 3 个 reference。
- 每帧最多调用 2 组 evidence；每次调用重新执行当前 query 与所选 references 的 ORB matching、map-pose triangulation 和 PnP。
- 历史关键帧 action 从过去已接受帧的 RGB、global pose 和 map references 建立局部 3D；当前帧与历史帧匹配后重新做 PnP，不直接复制历史 pose。
- `verify_candidate` 从候选未使用的 retrieval references 中选择 held-out references，重新匹配并计算候选 pose 下的极线一致性；该证据独立于候选生成 inlier。
- 新 reference 会覆盖旧 candidate 并使旧 verification 失效；只有 verification accepted 的 candidate 才能写入 Memory。
- fixed 按 retrieval 顺序，random 使用固定种子的随机顺序，rule 按 retrieval score/map spread 排序，light 是只在 calibration prefix 的 evaluator labels 上训练的 logistic policy；测试时 action selection 不读取 GT。
- oracle 使用测试 GT 选择有限深度下的最佳结果，仅作为不可部署的上界。

在线审计结果：`query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`，所有在线 policy 的 environment audit 均通过。light policy 的 GT 只用于 calibration-prefix 离线标签，不进入测试时的选择。

连续帧 Memory 也在协议内：测试段内 accepted pose、历史关键帧和 reject state 会进入下一帧 observation，所有 policy 在 evidence budget 用尽后调用同一个 causal historical-memory action。为避免 calibration/test 泄漏，测试段从空 online Memory 冷启动；这不是 warm-start Memory ablation，也不是 learned memory agent。后续应同时报告 cold-start 和合法历史状态 warm-start。

## 最新结果

产物：`outputs/lilocbench/active_evidence_pilot128_heldout_v2/summary.json`。四个序列共 304 个测试帧，每帧最多两次 reference evidence action；held-out verification 包含在每次候选验证中，并计入总工具调用数。

| policy | accepted / frames | coverage | accepted accuracy | full-frame success | mean total tool calls |
|---|---:|---:|---:|---:|---:|
| fixed | 87 / 304 | 28.6% | 81 / 87 = 93.1% | 81 / 304 = 26.6% | 3.168 |
| random | 74 / 304 | 24.3% | 69 / 74 = 93.2% | 69 / 304 = 22.7% | 3.336 |
| rule | 83 / 304 | 27.3% | 77 / 83 = 92.8% | 77 / 304 = 25.3% | 3.211 |
| light | 87 / 304 | 28.6% | 81 / 87 = 93.1% | 81 / 304 = 26.6% | 3.158 |

表中最后一列是 reference、held-out verification 和 historical-memory 三类动作的总工具调用均值。fixed 和 light 完全并列；light 没有提高 success 或 accepted accuracy，只略微减少了总调用数（3.158 vs. 3.168）。rule 没有形成同成功率低成本优势，random 明显落后。历史 fixed RoMa cache 不能作为同成本 baseline，因为它使用不同的神经工具和离线缓存边界。

真实强几何尚未纳入本轮同成本结论：当前环境 `torch.cuda.is_available()=False`、device count 为 0，`nvidia-smi` 无法连接驱动；Reloc3r backend 会显式拒绝 CUDA 运行，不会把缓存结果伪装成在线执行。因此本轮严格结果证明的是“CPU action pool 下没有主动收益”，不是对 GPU Reloc3r/RoMa 主动策略的最终否定。

有限深度 oracle 仍明显高于可部署策略（static `81.6%`、dynamics `18.4%`、LT `34.2%`、dynamic+LT `11.8%`），但 oracle 使用测试 GT，仅说明当前候选池存在尚未被在线策略可靠识别的差异，不能作为 Agent 结果。

## 解释与边界

1. **当前 action pool 没有显示可学习的稳定分流。** 换 reference、查询历史关键帧和独立 held-out verification 已进入 causal 闭环，但 fixed 仍不低于 light。
2. **当前 CPU 结果不等于主动定位无效。** ORB 不是论文目标的强几何工具；真实 Reloc3r/RoMa 的候选质量、reference coverage 和 verifier calibration 尚未在同一在线成本边界下完成。
3. **不应把“多种方法 + 最后让 Agent 选择”当作贡献。** 只有当强几何动作产生稳定 only-action 集合，且 policy 在相同预算下超过 fixed/geometry-greedy，主动取证才具有论文价值。

## 进入训练前的必要复验

1. 在可用 GPU 上运行真实 Reloc3r/RoMa，记录每个 reference group 的真实 matching、pose、held-out verification 和 elapsed；禁止用历史 cache elapsed 代替在线执行。
2. 保持当前最小动作空间：换 reference、查询历史关键帧、调用强几何、独立验证、accept/reject/defer；分别记录 source provenance 和失败原因。
3. 使用 scene-disjoint 或跨数据集 validation/test，冻结 reference groups、threshold、tool cost 和 policy features。
4. 重新比较 fixed、random、geometry-greedy、light sequential policy 和 finite-depth oracle；至少要求主动策略在相同预算下提高 full-frame success/accepted accuracy，或在相同 success 下显著降低 geometry calls、latency 或 false accepts。
5. 只有出现稳定的 action-dependent context，才生成 tool-expanded trajectories，进入结构化 Agentic SFT；RL 继续后置。

## 复现入口

- `tools/eval_active_evidence_policies.py`
- `tools/cpu_orb_agent_backend.py`
- `tools/agentic_localization_env.py`
- `outputs/lilocbench/active_evidence_pilot128_heldout_v2/summary.json`

本报告是当前 pilot 的限制性结论，不覆盖 AVL-015 的固定动作 oracle、Historical Memory 负结果或已有 deterministic recovery 结果。
