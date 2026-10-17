# Stage 2.5：可靠训练环境冻结记录

本阶段的目标不是训练一个更复杂的 Agent，而是先冻结能够产生可信 observation、reward、Memory 标签和 action-value 的闭环环境。

## 阶段结论

阶段 2.5 已完成。系统已从“多个定位方法的静态路由”推进为一个可交互、可审计、可训练的连续视觉定位环境：策略可以观察当前帧、检索结果、候选证据、历史状态、Memory 和剩余预算，选择下一项定位或验证动作，并根据真实工具回执继续决策。当前尚未训练最终 Agent，但轻量阶段三的接口、监督信号和比较基线已经具备。

当前闭环为：

```text
连续 query 帧
  -> policy-visible observation + causal state + Memory
  -> Agent 选择 acquisition / verification / terminal action
  -> Reloc3r、RoMa 或 Memory 工具真实执行
  -> independent verifier 返回支持、冲突、风险和成本
  -> Agent 继续取证，或 Accept / Reject / Defer
  -> 只有 verified + accepted 候选更新 anchor 和 Memory
  -> 下一帧
```

与普通 router 的区别是：动作结果会改变当前候选、后续 observation、reject streak、anchor age、Memory 内容和下一帧状态。Agent 要学习的不是“Reloc3r 和 RoMa 哪个总体更好”，而是“在当前证据、历史状态和预算下，下一步获取什么新证据最有价值，以及何时停止”。

## 当前实现栈

### Causal environment

[agentic_localization_env.py](/data2/liusy/reloc3r/tools/agentic_localization_env.py) 是统一状态机入口，已经实现：

- 连续帧状态、当前候选、最近接受位姿、anchor age 和 reject streak；
- `localize_with_references`、`run_strong_geometry`、`retrieve_memory`、`verify_candidate`、`accept_candidate`、`reject_candidate`、`defer_frame`；
- 同帧多轮 acquisition 和每帧动作预算；
- verifier 通过后才允许 Accept；
- 只有 Accept 才能写入 Memory；
- action receipt、状态转移和 causal audit；
- 禁止 query GT、未来帧、事后 success 和 Oracle outcome 穿过在线边界；
- evaluator-only `fork()`，用于有限深度反事实搜索，但不暴露给在线 policy。

### Online pose tools

[lilocbench_agent_backend.py](/data2/liusy/reloc3r/tools/lilocbench_agent_backend.py) 提供论文路径上的 GPU backend：

- cheap action：对当前 query 和明确选择的 mapping references 实时运行 Reloc3r，并返回绝对位姿、参考数量、平移/旋转离散度及实际耗时；
- strong action：实时运行 RoMa dense matching、mapping-view triangulation 和 query PnP，返回 correspondences、inliers、reprojection、pair support 和实际耗时；
- Memory action：只读取过去帧的已接受记录，禁止未来 Memory；
- 所有 reference-conditioned action 必须报告 `executed=true`、`cache_hit=false`、当前 query frame 和精确 reference IDs；
- CUDA 不可用时显式失败，不退化为冻结候选池。

[cpu_orb_agent_backend.py](/data2/liusy/reloc3r/tools/cpu_orb_agent_backend.py) 提供 ORB/SIFT、triangulation 和 PnP 的 CPU smoke backend，用于验证交互语义和 Oracle 分支，不作为论文主结果。

### Memory

Memory 已经纳入状态机，但目前处于“接口和因果约束完成、主动策略尚未训练”的阶段：

- 只有 verified + accepted pose 能写入；
- 记录 frame、source、confidence、references 和 verification provenance；
- 当前帧只能读取更早帧记录；
- CPU backend 会使用历史 keyframe 与 map references 重新构建当前位姿，禁止直接复制旧位姿；
- 后续需要比较 no-memory、passive-memory 和 active-memory policy。

## 已冻结的在线边界

`tools/agentic_localization_env.py` 仍是唯一状态机入口。所有候选必须由当前 query 和明确 reference action 实时产生；接受前必须经过 verifier；只有接受的候选才写入 Memory。每个 acquisition 都要求 `executed=true`、`cache_hit=false`、当前 query frame 和精确 reference receipt。GPU 运行使用本地 Reloc3r/RoMa 权重，未读取 AVL-004 候选缓存。

## 1. Verifier

GPU Reloc3r 候选现在会对没有参与候选生成的 held-out mapping view 重新执行 Reloc3r 相对位姿，并计算候选位姿与独立位姿的平移/旋转不一致。RoMa 候选会在未用于三角化/PnP 的 held-out view 上重新执行 RoMa matching，并计算对称极线误差。两类独立证据都计入真实 verifier elapsed time。

分源校准由 `tools/calibrate_active_evidence_verifier.py` 完成，配置冻结在 [active_evidence_verifier_v1.json](/data2/liusy/reloc3r/configs/active_evidence_verifier_v1.json)，校准报告在 [stage25_calibration_report.json](/data2/liusy/reloc3r/outputs/lilocbench/stage25_calibration_report.json)。static_0 与 dynamics_0 用于选择风险阈值，lt_changes_0 与 lt_changes_dynamics_0 只用于冻结后测试。

校准结果：Reloc3r 的 calibration accepted candidates 为 27/27 正确，冻结 long-term test 为 32/32 正确；RoMa calibration false-accept rate 为 1/23=4.35%，低于预注册的 5% 上限，冻结 long-term test 为 36/36 正确。这个样本仍属于校准集规模，不应被表述为最终泛化证明。

## 2. 公平基线

`tools/eval_active_evidence_sequential.py` 已加入：

- `fixed_geometry`：GT-free 稳定哈希 schedule；
- `geometry_greedy`：每帧优先强几何，作为高成本上界；
- `uncertainty_trigger`：只根据 verifier risk 触发强几何；
- `one_shot_router`：只根据 retrieval score 在第一次 action 前选择；
- 原有 `fixed`、`random`、`feedback_rule`。

策略的 acquisition 上限默认是 2，所有成本来自 live receipt 的 elapsed time。校准用的 policy 门限记录在 [active_evidence_policy_v1.json](/data2/liusy/reloc3r/configs/active_evidence_policy_v1.json)。

冻结 verifier 下的 long-term held-out 48 帧结果在 [aggregate_summary.json](/data2/liusy/reloc3r/outputs/lilocbench/stage25_heldout_baselines/aggregate_summary.json)：

| policy | full-frame success | 实际 RoMa 调用率 | mean tool cost (s) | false accepts |
|---|---:|---:|---:|---:|
| fixed | 64.58% | 0.00% | 0.468 | 0 |
| fixed-geometry | 68.75% | 27.08% | 1.099 | 0 |
| uncertainty-trigger | 68.75% | 18.75% | 0.736 | 0 |
| one-shot router | 72.92% | 45.83% | 1.053 | 0 |
| random | 70.83% | 35.42% | 1.036 | 0 |
| feedback-rule | **79.17%** | **35.42%** | 1.083 | 0 |
| geometry-greedy | 79.17% | 100.00% | 1.740 | 0 |

因此，feedback-rule 在与 random 相同的实际强几何调用率下高 8.34 个百分点；它达到 geometry-greedy 的 success，但成本约为后者的 62%。one-shot router 尚未追平 feedback-rule，Agent 训练仍有可验证的研究空间。由于 long-term retrieval 分布发生偏移，预注册 35% 门限在测试集上的实际调用率会变化；最终论文应按实际调用率或实际耗时分桶比较，而不能只报告目标门限。

## 3. 两步 Oracle 与轨迹

Oracle 深度现在默认严格等于 `max-acquisitions`；若显式给出不一致值，评测器直接报错。每个 counterfactual state 都保存：policy-visible state、action、真实 action receipt、verification receipt、逐项成本、候选证据、累计成本，以及 evaluator 在 GT join 后生成的 terminal action-value。轨迹写入 `traces/<sequence>_oracle.jsonl`，不把 GT 传入在线 backend。

GPU 一帧 Oracle 冒烟输出在 [lt_changes_dynamics_0_oracle.jsonl](/data2/liusy/reloc3r/outputs/lilocbench/stage25_gpu_oracle_smoke/traces/lt_changes_dynamics_0_oracle.jsonl)：深度 2、13 个状态动作、10 个 terminal programs，且最终选择分支成功。另有四序列各 8 帧的 CPU 两步轨迹，在 [stage25_cpu_oracle_8](/data2/liusy/reloc3r/outputs/lilocbench/stage25_cpu_oracle_8) 共生成 32 帧、469 个状态动作和 375 个 terminal programs。CPU 10/10 causal environment tests 也通过。

## 为什么现在能够训练视觉定位 Agent

### 1. 已经存在真正的 sequential decision process

策略不是一次性选择 backend。一次 cheap localization 失败后，下一步可以换 reference、调用 strong geometry、查询 Memory 或 Defer；Accept/Reject 会改变未来 anchor 和 Memory。因此相同图像在不同历史状态和不同工具反馈下可以产生不同的最优动作。

### 2. observation 包含可学习的任务证据

当前 policy 可以使用：

- retrieval score 和 reference group；
- candidate source、reference provenance 和 pose availability；
- Reloc3r dispersion；
- RoMa/PnP inliers、reprojection 和 correspondence support；
- held-out pose disagreement 或 epipolar evidence；
- previous-state risk、anchor age、reject streak；
- Memory size、历史 confidence 和来源；
- 已用动作数、实际工具成本和剩余预算。

这些变量使 Agent 可以学习“继续取证的价值”，而不是只拟合一个数据集级方法排名。

### 3. reward、Memory 标签和训练标签有可信来源

Verifier 已从候选内部质量检查升级为 held-out evidence，并在 source-specific calibration 后冻结。GT 只在 evaluator 端用于 terminal success 和 action-value，不进入在线 observation。由此可以分别构造：

- Accept / Reject 标签；
- false-accept penalty；
- acquisition action-value；
- cost-aware target；
- Memory 写入标签；
- 恢复成功、合理 Defer 和工具失败轨迹。

### 4. 已有强规则基线和成本上界

Agent 不会只和弱固定顺序比较。当前已有同一环境内的 fixed、fixed-geometry、uncertainty-trigger、one-shot router、random、feedback-rule 和 geometry-greedy。尤其是 feedback-rule 与 random 在 held-out 48 帧上具有相同的实际 RoMa 调用率 `35.42%`，但 full-frame success 为 `79.17% vs 70.83%`，说明工具反馈本身具有可利用的信息；one-shot router 为 `72.92%`，尚未解释完反馈闭环的收益。

## 阶段三：学习证据价值的 Agent 路线

### Stage 3B 首轮：结构化 policy + value baseline（2026-10-12）

已基于 Stage 3A 的 `state_action_values.jsonl` 实现并训练两个严格 structured-only baseline：

- `MLPPolicyValue`：逐 action state-action scorer，同时输出 masked policy logit 和 `Q`；
- `TransformerPolicyValue`：将合法 action set 编码为 tokens，同时输出 masked policy logit 和 `Q`。

模型输入只包含 retrieval/reference-group statistics、candidate/verifier evidence、temporal/Memory/budget state 与
action type/reference-group statistics；不包含原始图像、reference IDs、query GT、Oracle result 或 future state。
环境生成并强制 action mask。为避免 Oracle branch 中失败后继状态占主导导致 reset 后立即零覆盖 `Defer`，Stage 3B
固定了一个 GT-free deployment constraint：每帧第一个 decision 必须执行一次合法 acquisition，后续才能 Defer。

训练使用每序列 0–63 帧的 state，64–79 帧为同序列时间后验 validation；在线 test 使用各序列 80 帧以后的未参与
Oracle collection 的帧（static/LT/combo 80 帧，dynamic 75 帧，共 315 帧）。因此这是 **temporal holdout、same-scene**
证据，不是最终的 scene-disjoint 泛化结果。

离线 validation：MLP top-1 Oracle-action accuracy=`59.59%`、mean Oracle regret=`0.1984`；Transformer 对应
`61.63%`、`0.1615`。Transformer 的离线 value/action 拟合略好，但这不保证 closed-loop 安全性。

真实 GPU closed-loop 结果如下（所有 online trace causal audit 通过；所有方法使用同一 Reloc3r/RoMa backend、
frozen verifier、两次 acquisition 上限和 30 s budget）：

| policy | full-frame success | false-accept rate | geometry calls/frame | measured cost (s/frame) |
|---|---:|---:|---:|---:|
| one-shot router | 47.94% | 4.43% | 0.00% | 0.83 (MLP run) / 0.68 (Transformer run) |
| feedback-rule | **55.24%** | **5.95%** | 51.75% | 1.75 (MLP run) / 1.57 (Transformer run) |
| learned MLP | 42.86% | 25.00% | 7.30% | 0.91 |
| learned Transformer | 33.02% | 34.59% | 8.89% | 0.74 |

因此 Stage 3B 首轮的结论是 **No-Go for the learned policy claim**：两种模型都没有超过 one-shot 或
feedback-rule，且 false accept 明显失控。它们的低 geometry call rate 不是有效效率收益，因为以 success 和安全性
下降为代价。不能把这轮模型称为最终 Agent，也不能以离线 action accuracy 代替 closed-loop result。

该失败精确定位了后续工作，而不是否定 Agent 设置本身：

1. Stage 3A Oracle 的 state 主要来自 Oracle-induced branches，无法覆盖 learned policy 实际诱导的错误状态，下一步必须执行 policy rollout → Oracle relabel 的 DAgger 数据聚合；
2. 当前 evaluator 对 verifier-passed candidate 自动 Accept，训练集缺少在 verifier-passed state 主动 `Accept/Reject/Defer` 的显式 action-value；必须补齐 terminal-control action 与 false-accept recovery label；
3. 数据规模仅 1277 states，且当前 four-sequence split 是 same-scene；扩展 GPU Oracle 时应主动采样 dynamic/high-risk/false-accept/Memory states，并报告 1k→5k→10k learning curve。

实现及产物：
[stage3b_policy.py](/data2/liusy/reloc3r/tools/stage3b_policy.py)、
[train_stage3b_policy.py](/data2/liusy/reloc3r/tools/train_stage3b_policy.py)、
[MLP output](/data2/liusy/reloc3r/outputs/lilocbench/stage3b_mlp_temporal_holdout_v1/)、
[Transformer output](/data2/liusy/reloc3r/outputs/lilocbench/stage3b_transformer_temporal_holdout_v1/)。

### Stage 3A 实现状态（2026-10-12）

Stage 3A 已完成代码冻结并开始真实 GPU coverage collection。统一终端效用配置位于
[stage3a_terminal_utility_v1.json](/data2/liusy/reloc3r/configs/stage3a_terminal_utility_v1.json)，由
`tools/eval_active_evidence_sequential.py` 在 evaluator-only 侧计算：

```text
U(tau) = success_reward * correct_accept
       - cost_penalty_per_s * measured_tool_cost_s
       - false_accept_penalty * false_accept
       - recovery_penalty_per_frame * recovery_time_frames
       - memory_pollution_penalty * memory_pollution
```

每个真实执行的 Oracle node 现在保存 `prior_cost_s`、live action/verification receipt、累计成本、
terminal utility、`Q*(s,a)`、最佳终端类型和最优动作标记。`tools/build_stage3a_oracle_dataset.py`
将这些 trace 转换为 `state_action_values.jsonl`，并输出 unique policy states、动作计数和状态覆盖标签。
数据转换会对旧 trace 从 receipt ledger 恢复 `frame_tool_cost_s` 与 `remaining_budget_s`，因此不会把预算状态
误当成缺失特征。

环境现在把每个 acquisition/verifier 的 backend `elapsed_s` 累计到当前帧的 policy-visible state；
Accept/Defer 后按帧清零，evaluator wall time 不会混入该状态。新增因果测试验证该成本状态正确累计和重置。

真实 GPU collection 已完成，输出根目录为
[stage3a_gpu_coverage_v1](/data2/liusy/reloc3r/outputs/lilocbench/stage3a_gpu_coverage_v1/)，四序列各采集 80 帧，
共 320 帧。所有 sequence trace 的 causal audit 均通过，最终汇总如下：

| sequence | frames | Oracle success | accepted accuracy | mean selected cost (s) | geometry call rate |
|---|---:|---:|---:|---:|---:|
| static_0 | 80 | 65.00% | 100% | 0.663 | 2.50% |
| dynamics_0 | 80 | 42.50% | 100% | 0.850 | 6.25% |
| lt_changes_0 | 80 | 58.75% | 100% | 0.703 | 7.50% |
| lt_changes_dynamics_0 | 80 | 61.25% | 100% | 0.884 | 12.50% |
| **aggregate** | **320** | **56.88%** | **100%** | **0.775** | **7.19%** |

最终 evaluator-only 数据集位于
[stage3a_gpu_dataset_v1](/data2/liusy/reloc3r/outputs/lilocbench/stage3a_gpu_dataset_v1/)。其中包含 **1277 unique
policy states、5419 个真实执行 state-actions、6696 个 state-action value labels**；每个 state 都有至少一个
`is_optimal_action=true`，`Q*` 范围为 `[-3.159, 1.000]`。覆盖标签包含 candidate 无输出/有输出、high/low risk、
连续 reject、Memory/source 类型、geometry 可用、correct accept、false accept 和 defer。该数据集现在可以作为
MLP/small Transformer 的第一版 policy/value training 输入，但 1277 states 仍是 Stage 3A 的第一批 coverage，
后续应继续按稀有状态 strata 扩增，而不能视为最终 IROS 规模。

阶段三的核心不预设 Qwen 一定优于小模型，而是学习在连续定位中获取和验证证据的价值。论文要让实验回答：复杂语言模型是否真的为这种序贯几何决策带来额外收益？因此 policy 比较顺序固定为：

```text
Rule -> MLP / small Transformer -> Qwen LoRA-SFT
```

如果 MLP 与 Qwen 表现相近，项目仍然是成立的 Agentic sequential visual localization；结论应是当前结构化状态下 LLM 不必要，而不是强行宣称 Qwen 更强。Qwen 是待验证的 policy 候选，不是预设答案。

### 3.1 第一优先级：扩大真实 GPU Oracle state coverage

当前 469 个 CPU state-action 和 GPU 单帧 smoke 仅证明协议可行，不足以训练正式 policy。首先应从真实 GPU 环境扩展有限深度反事实轨迹，并按状态类别主动采样，而非只追求帧数：

| 类别 | 必须覆盖的状态 |
|---|---|
| cheap localization | success、低/高风险 failure、无 pose |
| strong geometry | success、failure、有输出但错误、低支持 |
| verifier | correct accept/reject、false accept/reject |
| temporal state | fresh/stale anchor、连续 reject、lost 后恢复 |
| Memory | useful、无效、misleading、错误 anchor 依赖 |
| terminal | early accept、第二步恢复、合理 defer、错误 accept |
| budget | 预算充足、只剩一步、强几何已使用 |

每个采样状态都展开全部合法动作及其真实回执，例如不同 reference group 的 `Localize`、`StrongGeometry`、`RetrieveMemory` 和 `Defer`。随后以学习曲线而非主观数量决定数据量：至少报告 `1k -> 5k -> 10k` unique policy states 下的验证表现和状态覆盖率。

### 3.2 冻结 observation、action 和硬约束

第一版 observation 分为五组：

1. retrieval/reference：top-k score、reference group、diversity；
2. candidate evidence：source、dispersion、inliers、reprojection、held-out evidence、risk；
3. temporal state：anchor age、reject streak、last accepted source、localized/uncertain/lost；
4. Memory state：可用记录、age、confidence、source 和 provenance；
5. budget state：acquisition count、累计成本、strong geometry 是否已使用。

动作空间保持为：

```text
Localize(reference group)
StrongGeometry(reference group)
RetrieveMemory(memory IDs)
Accept
Defer
```

`Reject` 只保留为旧 trace/checkpoint 的兼容别名；在当前环境里它与 `Defer` 的状态转移完全相同，因此新的
policy action mask、Oracle label 和训练数据只暴露规范动作 `Defer`，不再把同一个负向终止决策拆成两个类别。

环境而非模型负责 action mask、reference/Memory ID 合法性、动作预算、verifier、Accept 权限和 Memory 写入。即使是 Qwen 也只能通过 constrained JSON/tool call 输出合法动作，不能绕过这些几何与因果约束。

### 3.3 Oracle utility、动作标签和 value target

不把成功轨迹简单加权为 SFT 样本。先定义 terminal utility：

```text
U(tau) = success_reward
       - lambda_cost * total_measured_cost
       - lambda_false_accept * false_accept
       - lambda_recovery * recovery_time
       - lambda_memory * memory_pollution
```

其中 false accept 的惩罚应显著大于一次合理 Defer。对于每个 state-action，从当前 Oracle 已保存的 terminal branches 反向计算：

```text
Q*(s, a) = max_{trajectory starts with a} U(trajectory)
a*(s)    = argmax_a Q*(s, a)
```

训练数据同时包含最优 action、全部合法 action 的 value、真实工具成本、success、false accept 和终端类型。这让 policy 学习“为什么某项新证据值得获取”，而不只是模仿一个动作。

### 3.4 先训练小模型，再检验 Qwen

所有模型使用相同 structured observation、action mask、训练 split 和 Oracle target。

| 模型 | 训练目标 | 作用 |
|---|---|---|
| MLP / small Transformer | masked action policy + Q-value regression | 判断问题是否已可由低维数值控制解决 |
| Qwen LoRA-SFT | constrained action generation + auxiliary value/success/cost prediction | 检验语言模型是否能更好利用多轮状态和 provenance |

共同损失为：

```text
L = L_policy(a*(s), action_logits)
  + alpha * L_value(Q*(s,a), Q_hat(s,a))
```

Qwen 的第一版只读取结构化 evidence，不读取原始图像；输出可以是 `action`、`action_confidence`、`predicted_success`、`predicted_cost_s` 和 `predicted_utility` 的严格 JSON。这样任何提升都更容易归因于 sequential policy，而非更强视觉 backbone。Qwen LoRA smoke 可以先运行以验证数据和约束解码管线，但不能在 state coverage 达标前作为正式性能结果。

### 3.5 Closed-loop rollout 与 DAgger-style relabeling

纯 behavior cloning 会遇到 covariate shift：policy 早期的错误动作会把系统带到专家轨迹较少出现的状态。正式训练采用：

```text
GPU Oracle dataset
  -> SFT / value fitting
  -> closed-loop policy rollout
  -> collect policy-induced states
  -> run Oracle relabeling on those states
  -> merge and retrain
```

至少进行两到三轮，并报告新增 unique states、专家动作分歧率、illegal-action rate、Oracle regret、false accept、RoMa 调用率和闭环 success。训练集必须保留直接接受、取证后恢复、合理 Reject/Defer、工具失败、错误 Memory 和无价值重复调用，不能只保留成功轨迹。

### 3.6 Active Memory、Qwen-VL 与 RL 的顺序

Active Memory 放在无 Memory learned policy 稳定之后：比较 no-memory、passive-memory 和 active-memory，并进一步检查 provenance/quarantine 是否抑制错误 anchor 传播。Memory 的有效收益可以是 success、更低 RoMa 调用、更短恢复时间或更少重复地图搜索。

Qwen-VL 不进入第一阶段。只有结构化 evidence 无法区分动态遮挡、长期结构变化或视觉别名时，才引入 raw image/crop，并与 structured-only policy 公平比较。

RL 最后再考虑，仅当 SFT + value learning + DAgger 仍无法处理长期 cost trade-off、自适应停止或 Memory 污染等长时序后果时使用。

## 阶段三验收标准

1. Oracle state coverage 覆盖主要 success/failure/Memory/budget strata，且所有 value 仅在 evaluator 侧用 GT 计算；
2. 在实际 RoMa 调用率或实际平均耗时匹配时，learned sequential policy 超过 one-shot router；
3. learned policy 超过 feedback-rule，或明确说明其未超过的原因及 Oracle gap；
4. false-accept rate 不高于冻结 verifier 的目标，且不能用近零 coverage 达成；
5. 相比 geometry-greedy，在接近 success 下显著降低成本；
6. MLP、small Transformer 和 Qwen 在同一数据/动作/成本协议下比较，由结果决定最终 policy；
7. active Memory 相比 no/passive Memory 具有独立闭环收益；
8. 所有结果报告实际调用率、实际 elapsed time、success、coverage、false accept、recovery time 和 Oracle regret。

## 当前边界与未完成事项

- 当前最终 held-out 基线是 long-term 两序列共 48 帧，用于验证机制和趋势，不是最终 IROS 规模；
- GPU 两步 Oracle 目前完成真实分支 smoke，四序列多帧轨迹仍以 CPU proxy 验证生成协议；后续需扩大 GPU Oracle 数据；
- one-shot router 在 calibration 上目标为 35% 调用率，但 long-term 测试发生分布偏移，实际为 45.83%；最终比较必须按实际 rate/cost 分桶；
- Memory 已有正确接口，但还没有证明 active Memory policy 的收益；
- 当前没有训练好的最终 Agent，不能把 feedback-rule 或 Oracle 包装成 Agent 论文结果；
- 论文前仍需四序列完整评测、多随机种子、rate-matched 重复实验、Memory/verifier ablation 和恢复时间统计。

## 阶段门槛判断

阶段 2.5 的工程前提已经具备：独立 held-out verifier、分源冻结阈值、同一预算的可执行基线、两步 Oracle 和可训练 action-value 轨迹均存在并经过 smoke/held-out 验证。

因此现在可以进入轻量阶段三。推荐的第一项正式实验是：使用现有 Oracle action-value 训练结构化两步 acquisition policy，在冻结 verifier 和完全相同的动作/成本协议下，对比 feedback-rule、random、one-shot router 和 geometry-greedy。Memory policy、复杂 RL 和论文最终结论应等待更大规模 GPU 轨迹及严格的 rate-matched 重复实验。

## Stage 3B-v2：Terminal supervision + DAgger（进行中）

Stage 3B-v1 的 deployment 结果不能作为 learned agent 的验收：MLP 的 false-accept rate 为 25%，strong-geometry rate 仅为 7.3%，低于 feedback-rule 的 55.24% success。根因不是模型容量，而是旧接口在 verifier passed 后由 evaluator 自动执行 Accept；因此训练集中没有下面这个对安全最重要的决策：

```text
verifier-passed candidate -> Accept | Reject | Defer | Continue acquisition
```

v2 将 verifier 定义为证据工具，而非 terminal controller。Oracle 现在在每一个 policy-visible state 枚举 `Accept`、`Defer` 及剩余 `Localize/StrongGeometry/RetrieveMemory`，并以同一个 terminal utility 标注全部合法动作的 `Q*(s,a)`。因此可出现且必须保留的反例是：candidate 虽然 verifier passed，但 geometry 的预期 utility 更高，最优动作就是继续取证而不是 Accept。环境仍负责合法性 mask 和记忆写入约束。

v2 的第一轮闭环协议为：

```text
terminal-Q Oracle data (frames 0--39)
 -> train terminal MLP/value model
 -> MLP rollout on disjoint frames 40--59
 -> fork each policy-induced state and run budget-matched two-step Oracle
 -> merge labels and retrain
 -> evaluate only on held-out frames 60--79
```

每一轮除 success/coverage/FAR/actual latency/RoMa rate 外，还必须输出：first-step action accuracy、second-step/off-expert action accuracy、每个所选动作的 `Q*` gap、premature-Accept 比例、`should geometry but Accept` 和 `should Defer but Accept` 比例。首轮的行为目标不是先追求超过 feedback-rule，而是让 FAR 从 v1 的 25% 明显下降，并让 geometry 调用从 7.3% 向合理的反馈触发水平恢复。若两至三轮 DAgger 后 FAR 仍高于 20%，应停止单纯扩充数据，审查 terminal utility、Accept permission 与 verifier-policy interface；此时不能把问题归因为 Qwen 或模型容量。

### Stage 3B-v2 第一轮实测状态

四序列 terminal Oracle（每序列 40 帧）已完成并通过因果审计：4,124 个 policy states、13,527 个 state-action labels，包含 1,635 个 verifier-passed、2,329 个 verifier-rejected 状态，以及 503 个 false-accept terminal 分支。terminal-aware MLP 的时间尾验证为 94.7% action accuracy、0.0253 mean Oracle regret。

第一轮 DAgger 在每序列 40--59 帧收集 160 个 policy-induced states，并新增 160 个 state labels。其关键诊断是：`static_0` 和 `lt_changes_dynamics_0` 各出现一次 premature Accept，且均同时属于 “should geometry but Accept” 和 “should Defer but Accept”；first-step accuracy 按序列为 0.50--0.75，second/off-expert accuracy 为 0.45--0.90。

最初记录的 held-out 结果（80 帧 success 5%、FAR 20%、coverage 6.25%）经复核后判定为**无效协议结果，不能用于评价 MLP**。learned evaluator 在达到第二次 acquisition 后，在把最新 verifier 结果交还给 policy 之前直接执行了 forced Defer。80 帧中有 33 帧走到该分支，其中 29 个最终 candidate 已通过 verifier、27 个实际定位正确。这解释了低 coverage，而不是模型主动学成了保守 terminal policy。

同时发现并修正了三个数据协议问题：

1. `Reject` 与 `Defer` 在环境中具有相同转移，却被作为两个监督类别；现在统一为 `Defer`；
2. DAgger Oracle 从 policy-induced state 展开时把已经花费的 acquisition/verifier cost 错置为 0；现在从 live `frame_tool_cost_s` 开始，并向 policy observation 暴露 `budget_s/remaining_budget_s`；
3. dataset builder 会用错误的 `prior_cost_s` 覆盖 live cost，并把 `memory_size` 状态误报为 memory empty；现在只为 legacy trace 补缺失字段，并正确统计 Memory。

新增端到端回归测试明确覆盖 `second acquisition -> verify -> terminal policy -> Accept`，并验证 DAgger root node 的 `prior_cost_s` 等于 live spent cost。环境、policy、Oracle 和 dataset tests 共 18 项通过。

### Stage 3B-v2 协议修复后的同 checkpoint 复测

保持 `stage3b_v2_dagger_policy_r1/mlp_policy_value.pt`、帧 60--79、verifier、两次 acquisition、30 s budget 和 seed 全部不变，仅修复上述协议后，在四张 RTX 4090 上重新执行真实闭环。四个 sequence 的 causal audit 均通过：

| sequence | success | coverage | false accept / accepted | geometry calls | mean tool cost |
|---|---:|---:|---:|---:|---:|
| dynamics_0 | 35% | 35% | 0 / 7 | 1 | 1.201 s |
| static_0 | 80% | 80% | 0 / 16 | 0 | 0.444 s |
| lt_changes_0 | 5% | 20% | 3 / 4 | 0 | 0.508 s |
| lt_changes_dynamics_0 | 45% | 45% | 0 / 9 | 0 | 0.546 s |
| **aggregate** | **41.25%** | **45%** | **3 / 36 = 8.33%** | **1 / 80 = 1.25%** | **0.675 s** |

相对无效旧协议，同一 MLP 的正确接受由 `4/80` 上升到 `33/80`，FAR 从 20% 降到 8.33%；终止动作全部由模型显式选择，`defer_fallback=0`。在三个已有完整 feedback-rule 对照的序列上，修复后 MLP 为 success 40%、FAR 11.11%、geometry 1.67%、cost 0.718 s/frame；feedback-rule 为 success 45%、FAR 10%、geometry 53.33%、cost 2.278 s/frame。MLP 已不再是 terminal-control 崩溃，但 success 仍低 5 个百分点，尚未通过 Gate 3。

当前不立即修改 utility，也不做“所有动作等量”的朴素 action-balanced 训练。修复后标签中 4,284 个 state 的唯一最优动作分布为：`Defer 2732 / Accept 1136 / Localize 275 / Memory 84 / Geometry 57`；Geometry 只在 956 个可用状态中的 57 个（5.96%）最优。强行平衡动作会把大量 utility 上不应调用 RoMa 的状态放大。FAR 已降到 8.33%，也没有证据支持现在降低 false-accept penalty。

下一步应先用修复后的协议执行新的 DAgger round，丢弃旧 DAgger Q label（旧 label 的 rollout observation 含真实 cost，但反事实 Q 从 cost=0 展开，不能通过离线改字段修复），再以相同 MLP 架构重训。采样采用 **Oracle-regret/rare-optimal targeted DAgger**：优先保留 `Geometry`、`Memory`、premature Accept 和较大 Q-gap 状态，但不改变原始 utility。若 1--2 轮修复后的 DAgger 仍持续漏掉 `Q*(Geometry)>Q*(Defer/Localize)` 的状态，再启用 action-balanced loss；只有在正确 cost 下 Oracle 本身系统性偏向错误终止时，才调整 utility。

前三个序列的旧 feedback-rule 对照为 success 45%、FAR 10%、geometry call rate 53.3%，平均实测工具成本约 2.28 s/frame。旧的长期动态序列运行曾因并行首次加载 RoMa/DINOv2 时发生缓存竞争而中断；该缺口现已通过下面的 Pareto baseline 重跑补齐，旧未审计 trace 不再进入 Gate。

### 三目标 Pareto Gate v1（2026-10-12）

Stage 3B 不再只用 `learned success > feedback-rule success` 的单点条件。版本化配置
`configs/stage3b_pareto_gate_v1.json` 冻结三个同时满足的 development 条件：

1. success 相对 feedback-rule 的非劣界为 `-2 pp`；
2. false-accept rate 不高于 `10%`；
3. mean measured tool cost 严格低于 `1.0 s/frame`，且严格低于 feedback-rule。

Gate 只在 learned/baseline 的相同 sequence/frame 上配对，要求四序列共至少 80 个共同帧；trace 必须在对应 evaluator summary 中存在 `causal audit passed=true`，文件存在但运行失败或缺少审计的 trace 会被排除。脚本同时报告 10,000 次 paired bootstrap 的 success、FAR、cost 差值区间，以及三维 `(Success, -FAR, -Cost)` Pareto dominance。development pass 只能授权下一轮实验，不能成为最终 held-out claim；最终盲测必须使用 policy 选择后才冻结的新帧或 scene-disjoint 数据，并可进一步开启 CI non-inferiority 硬约束。

为完成 Gate 验收，已在缓存预热后独立补跑四序列 feedback-rule，80 帧全部完成并通过 causal audit。当前 development Gate 为：

| policy | success | FAR | mean cost | geometry frame rate |
|---|---:|---:|---:|---:|
| corrected-protocol MLP | 41.25% | 8.33% | 0.675 s | 1.25% |
| feedback-rule | 48.75% | 7.14% | 2.533 s | 53.75% |

判定为 **2/3 objective pass，但 overall fail**：FAR safety 与 tool-cost efficiency 通过；success 差为 `-7.5 pp`，未达到 `-2 pp` 非劣界。paired bootstrap 的 learned-minus-rule success 差 95% 区间为 `[-13.75, -2.50] pp`，cost 差区间为 `[-2.818, -1.023] s/frame`。两者在三目标空间中为 `non_dominated_tradeoff`：MLP 没有被 rule 严格支配，但也没有通过预注册 Gate。该结果进一步支持下一步 corrected targeted DAgger，而不支持现在宣称 learned policy 已胜出。

实现与产物：`tools/evaluate_stage3b_pareto_gate.py`、`tools/test_stage3b_pareto_gate.py`、
`outputs/lilocbench/stage3b_v2_feedback_rule_pareto_baseline/`、
`outputs/lilocbench/stage3b_v2_protocol_fixed_eval_r1/pareto_gate_development.json`。

阶段结论：Stage 3B-v2 的 terminal-control 代码路径已经修复并通过真实 GPU 复测，但 learned policy 尚未超过 feedback-rule；因此仍暂不进入 Qwen。当前最有信息量的下一项实验是 corrected DAgger，而不是改模型容量或立即重写 utility。

### Corrected targeted DAgger r2（2026-10-12，完成但未通过 Gate）

在不改变 utility、MLP 架构、verifier 和动作预算的前提下，已对40--59帧重新运行四序列 live-cost Oracle。共获得80帧、208个 policy-induced states、893个 root action nodes；`893/893` 节点的 `prior_cost_s` 与 policy observation 中的 live `frame_tool_cost_s` 一致。该批 on-policy 最优/诊断标签包括：`Accept 32`、`Geometry 11`、`Memory 30`、high-regret 14、premature-Accept 2。

训练管线同时修正了一个旧 DAgger split 问题：过去按全数据时间尾切分会把40--59帧的全部 DAgger state 放入 validation，使其不产生训练梯度。现在 validation 只取原始 Oracle 的32--39帧；所有带 DAgger provenance 的状态进入 train。targeted loss 使用有上限的 state weight，组合取最大值而非相乘，validation 和闭环指标保持不加权。

本轮在同一60--79 development set 上得到：

| variant | success | FAR | cost | geometry rate |
|---|---:|---:|---:|---:|
| 进入本轮前 checkpoint | **41.25%** | 8.33% | **0.675s** | 1.25% |
| on-policy targeted / from scratch | 37.5% | 0% | 0.880s | 3.75% |
| + on-policy Accept-optimal weight | 37.5% | 3.23% | 0.777s | 2.50% |
| policy acquisition + value final terminal | 40.0% | 0% | 0.760s | 3.75% |
| value head for all decisions | **45.0%** | 0% | 2.352s | 55.0% |
| old policy warm-start targeted fine-tune | 32.5% | 7.14% | 0.991s | 5.0% |

没有 variant 同时达到 `success>=46.75% / FAR<=10% / cost<1s`。按预注册 Gate，应保留进入本轮前的 checkpoint，不用退化模型替换它。全 value head 能把 success 提到45%，说明结构化 observation/value 中存在恢复信号；但它以55% Geometry和2.35s成本实现，尚未学会 cost-sensitive acquisition。policy-head 版本成本安全但 success 不足；当时的“明显 under-accept”仅是待检验假设，随后 on-policy Q 审计已否定它对保留 checkpoint 的适用性。因此下一步不能继续在60--79帧调 sample weight；应先审查 value calibration 和 acquisition Q target 的成本标定，再决定新一轮数据或目标设计。60--79帧继续只作为 development，不产生最终 claim。

实现与产物：`tools/collect_stage3b_dagger.py`、`tools/build_stage3a_oracle_dataset.py`、`tools/train_stage3b_policy.py`、`configs/stage3b_targeted_dagger_v1.json`、`outputs/lilocbench/stage3b_v2_corrected_dagger_r2/round_summary.json`。

### 保留 MLP 的 on-policy Q 校准审计（2026-10-13，只读）

为区分“终止控制失效”与“取证价值失真”，对当前保留 checkpoint
`stage3b_v2_dagger_policy_r1/mlp_policy_value.pt` 在 corrected r2 的 208 个
policy-induced states（40--59 帧）作了只读复算。每一个数都以该状态的
live-cost、两步 Oracle `Q*(s,a)` 为目标；不会回写训练数据、utility 或 checkpoint。

- **Accept 假设不成立。** 在 79 个 verifier-passed 状态中，`Qhat(Accept)-Q*(Accept)`
  的均值为 `+0.087`（MAE `0.215`、Pearson `0.949`），不是系统性偏低。32 个
  Oracle-Accept-optimal 状态中，policy logit 和 Q-head 都没有漏选 Accept；两者各有
  2 个过早 Accept。Accept 相对最优非 Accept 的 margin 符号准确率为 `97.47%`。
  所以该 checkpoint 在这批 on-policy terminal states 没有 under-accept 证据；不能把
  development success 差直接归因为 terminal acceptance calibration。
- **Geometry 的绝对值并非单向严重高估，但排序不足。** `Qhat-Q*` 的平均偏差分别为
  Geometry `+0.049`、Localize `+0.054`、Memory `-0.020`。但 Geometry 相对
  `max(Localize, Memory, Defer)` 的 margin 与 Oracle 相关性仅 `0.216`，符号准确率
  `77.5%`；模型预测 Geometry 值得调用的比例为 `21.25%`，Oracle 为 `8.75%`。
  换言之，瓶颈是 acquisition **相对 value/ranking 的低分辨率**，而不是可由一个
  constant bias 修复的单动作 Q 偏移。
- **误差集中于已有候选/Memory 的恢复状态。** 当当前候选来自 Reloc3r 时，Geometry
  margin 的符号准确率仅 `47.1%`（17 states）；当前候选为 Memory 时为 `61.9%`
  （63 states）。无 candidate 的 root states 为 `96.3%`。长期变化/动态/静态三种
  sequence strata 的整体 margin 符号准确率分别为 `76.3% / 72.5% / 85.0%`，但各仅
  80/40/40 个 Geometry-action examples，不能将小样本差异写成泛化结论。stale anchor
  只有16个 Geometry examples，也不足以定论。
- **policy-logit 与 Q-head 是两个不同的控制器。** 它们在 `84/208=40.4%` 个 state
  选择不同 candidate/action；Q-head 的平均 Oracle regret `0.0880` 并不低于 policy 的
  `0.0878`，但其相对 Oracle 的平均工具成本多 `0.468s`（policy 为 `0.065s`）。典型
  分歧是 `defer→geometry` 17 次和 `localize→geometry` 14 次；后者 Q-head 虽降低
  utility regret（`0.214→0.077`），却平均多 `1.54s`，而 `defer→geometry` 的 Q-head
  同时增加 regret（`0.123→0.186`）和成本（`-0.176→+2.292s`）。这解释了全 value
  head 的 45% success/2.352s：它不是更可靠的 value controller，而是混入了昂贵且错误的
  Geometry 偏好。

因此下一次设计不应只是增大 Geometry sample weight，也不应先改 terminal utility。优先级应为：
将 acquisition value 拆成可校准的 success-gain 与 incremental-cost（或双头/advantage）目标，
在 candidate/Memory recovery state 上补充对比性 Oracle labels，并在**新的** development split
验证后才再次闭环。审计实现为 `tools/analyze_stage3b_q_calibration.py`；机器可读产物为
`outputs/lilocbench/stage3b_v2_q_calibration_audit_r1/q_calibration_audit.json`。

### Stage 3C：Recovery/Cost 分解与显式控制器（2026-10-13，首轮完成 / Gate fail）

本轮不再以单一 scalar-Q 作为 acquisition 的核心目标。对每个可执行
`Localize/Geometry/Memory` action，模型分别预测：

```text
P_recover(s, a) = finite-depth Oracle 下 action 后可正确接受的概率
C_hat(s, a)     = 从 live state 开始的新增实测工具成本
A(s, a)         = P_recover(s, a) - 0.05 * C_hat(s, a)
```

Defer 的显式基线为零新增收益/成本，controller 选取最大 `A` 的 acquisition；
terminal action 仍须经过 verifier。训练数据来自 corrected live-cost Oracle 与 targeted
DAgger 的 1,116 个 states / 4,650 个 acquisition labels，其中 876 个为已有
candidate 的 recovery state，且包含 956 个 Geometry、843 个 Memory action。
MLP 使用 BCE recovery、log-cost regression 和 pairwise ranking 三个损失；全部
on-policy DAgger state 只进入训练，四序列帧 32--39 的 192 states 作为时间尾验证。

离线验证表明数据中存在可学习排序信号，但这不是闭环性能主张：MLP 的 Geometry
margin sign accuracy 为 89.38%，`P_recover` Brier 为 0.1275、cost MAE 为 0.229s。
独立 XGBoost diagnostic 的显式 `P-C` controller / XGBRanker Geometry margin sign
分别为 76.25% / 95.63%；XGB 仅为 structured-evidence 的离线诊断，未被部署到
闭环 policy。

首个 rollout 暴露了一个重要安全接口 bug：Memory candidate 被 verifier 标为
`verified` 后，controller 会把它当成与 fresh Reloc3r/RoMa receipt 等价的 terminal
evidence。四序列中由此产生 20 次直接 Memory Accept，且全部错误；该版本
`r1` 已明确作废，不能作为方法指标。修复后的语义是 Memory 只能提供 retrieval
context，必须由 fresh `reloc3r_online` 或 `roma_online_geometry` evidence 才可 Accept；
此性质已加入回归测试。

在同一 checkpoint、同一预算、同一 verifier 和不调 controller 参数的条件下，修复后的
`r2` 四序列（帧 60--79、80 帧）真实 GPU 闭环结果为：

| policy | success | FAR | mean cost | Geometry frame rate |
|---|---:|---:|---:|---:|
| Stage 3C recovery/cost MLP | 42.5% | 8.11% | 1.165s | 1.25% |
| frozen feedback-rule | 48.75% | 7.14% | 2.533s | 53.75% |

因此三目标 Pareto Gate **未通过**：FAR 合格且 cost 比 rule 低 1.37s，但 success
差 `-6.25 pp`（目标非劣界为 `-2 pp`），同时 `1.165s > 1.0s` 的绝对成本阈值。
配对 bootstrap 的 success 差 95% CI 为 `[-12.5, -1.25] pp`。该模型仍为
non-dominated tradeoff，不应替换已保留的 Stage 3B 低成本 checkpoint，也不构成
Qwen/RL 的进入条件。

本轮的关键负诊断是：离线 Geometry margin 高并没有转化为 action selection；闭环中
只调用了 1 次 Geometry。原因至少包括 action distribution shift 与终止后可见的恢复
state 被直接接受，不能从当前离线 temporal-tail 数字推断为“排序已解决”。下一步应在
新的、未参与本轮选择的 development split 上，收集 *policy-induced verified-candidate*
states，单独衡量 `Geometry vs Localize/Defer` 的 ranking calibration，并部署真正的
XGB/MLP ranking controller 作公平闭环比较；在此之前不继续调 60--79 帧的权重、
不进入 Qwen 或 RL。

实现与产物：`tools/build_stage3c_recovery_dataset.py`、
`tools/train_stage3c_recovery_policy.py`、`tools/stage3c_recovery_policy.py`、
`tools/train_stage3c_xgboost_diagnostic.py`、`tools/test_stage3c_recovery.py`、
`configs/stage3c_recovery_policy_v1.json`、
`outputs/lilocbench/stage3c_recovery_dataset_r1/`、
`outputs/lilocbench/stage3c_recovery_policy_r1/`、
`outputs/lilocbench/stage3c_xgboost_diagnostic_r1/`、
`outputs/lilocbench/stage3c_recovery_eval_r2/pareto_gate.json`。

### Stage 3C.5：Fresh-split XGBRanker 与 Geometry opportunity audit（2026-10-13，completed / diagnostic negative）

为区分“MLP joint learning 没有将离线排序部署成在线决策”与“当前 evidence state/
coverage 对 recovery value 不足”，冻结 Stage 3C 的 XGBoost models（训练数据只覆盖
帧 0--59），不再训练 MLP，也不改 utility。新的评测段为四序列帧 80--99；它未参与
训练、DAgger 或此前的 60--79 development 选择。XGBRanker 只排序 acquisition
`Localize/Geometry/Memory`；因其 raw score 不是概率，终止仍由独立的
`P_recover-0.05*C_hat` 与冻结 verifier 执行，避免用 rank score 破坏 safety。

每个 policy 实际访问的 state 在**动作已经选定后** fork 出两步 live-cost Oracle，得到：

```text
Geometry opportunity recall
= #(Oracle Geometry-optimal 且 policy 选择 Geometry)
 / #(Oracle Geometry-optimal)
```

Oracle label、GT 和 terminal utility 从不进入线上 action selection。该审计同时记录
Geometry precision、missed-Geometry utility regret、recovery-state success，以及
`ΔP=P_G-P_bestcheap`、`lambda*ΔC` 和 `ΔP-lambda*ΔC` 的分布。

四序列 80 帧的真实 GPU 结果如下（所有 causal audit 通过）：

| policy | success | FAR | mean cost | Geometry frame rate |
|---|---:|---:|---:|---:|
| XGBRanker controller | 52.5% | 6.67% | 0.719s | 0% |
| feedback-rule | 57.5% | 9.80% | 2.272s | 41.25% |

Pareto Gate 仍未通过，原因是 success 差 `-5 pp`（95% paired-bootstrap CI
`[-10,-1.25] pp`）；FAR 与成本两项通过。该 fresh temporal split 未参与 policy
选择，但不是 scene-disjoint final paper test，故只作为独立诊断而非最终泛化主张。

更关键的是 opportunity audit：234 个 policy decisions 中，Oracle 有 5 个
Geometry-optimal opportunity（4 个为已有 candidate 的 recovery state），XGBRanker
调用 Geometry 0 次，故 recall 为 **0/5=0%**；5 次错过机会的平均 utility regret 为
`0.750`。在 160 个同时有 Geometry 与廉价替代的可比 state 中，预测 `ΔP` 的平均值
为 `+0.0217`，而平均 `lambda*ΔC` 为 `+0.0852`，显式 advantage margin 的平均值为
`-0.0635`；其中 `89/160=55.6%` 是 “`ΔP>0` 但被成本项压制”。不过线上实际用于
排序的 XGBRanker 仅在 `1/160` state 偏好 Geometry，显式 controller 也仅在 `13/160`
state 偏好它。

因此该实验排除了“只需把 MLP 换成 tree ranker”这一解释，但尚**不能**严格证明
structured evidence 信息理论上不足：它证明的是当前特征集、0--59 的 recovery-state
coverage 与 terminal/controller 接口，无法在 80--99 的 policy-induced states 上保留
Geometry value ranking。下一步不应上 Qwen；应补充可区分 Geometry 边际价值的可观察
特征（correspondence spatial distribution、matching concentration、dynamic-region/
appearance-change 与 temporal consistency），并在新的 split 重做相同 audit。只有这些
增强后仍无法提高 opportunity recall，才有充分证据引入视觉/VLM observation。

实现与产物：`tools/stage3c_xgboost_policy.py`、
`tools/analyze_stage3c_geometry_opportunities.py`、
`tools/test_stage3c_geometry_opportunities.py`、
`configs/stage3c5_xgbranker_fresh_split_v1.json`、
`outputs/lilocbench/stage3c5_xgbranker_fresh_eval_r1/geometry_opportunity_audit.json`、
`outputs/lilocbench/stage3c5_xgbranker_fresh_eval_r1/pareto_gate_fresh_split.json`。

### Stage 3D：Recovery Observability（2026-10-13，v1 implementation / diagnostic negative）

Stage 3C.5 把问题收敛到一个更具体的科学问题：在调用昂贵 Geometry 之前，当前
observation 是否包含足以预测其边际价值的证据？本阶段不先换大模型，而是冻结
XGBRanker/controller，定向扩大 recovery/opportunity state coverage，并建立只使用
Geometry **之前**可得到的信息的 feature ladder：

```text
F0: structured retrieval / verifier / budget / anchor state
F1: F0 + cheap correspondence 的空间覆盖率、熵、集中度、匹配半径
F2: F1 + temporal / anchor consistency
F3: F2 + appearance/change、dynamic/occlusion、viewpoint cues
```

`tools/stage3d_features.py` 对每个可选字段做 fail-closed 检查；RoMa、PnP、inlier、
reprojection、epipolar 等后验字段一旦出现在输入中就直接报错，不能泄漏到训练或决策。
`LILocBenchReloc3rBackend` 新增 `--stage3d-preaction-features` 开关：它在 cheap
Reloc3r action 内用 ORB/Hamming correspondence 计算 F1，并计算当前 query/reference
的亮度与外观变化 cue；这部分时间包含在该 action 的 execution receipt 和预算中，未引入
免费的额外分割网络。默认关闭以保持旧协议可复现。

Recovery targeted corpus 已从 Stage 3C live-cost Oracle 中选出 751 个 grouped states：
`geometry_positive_recovery=87`、`geometry_boundary_recovery=599`、
`memory_negative_recovery=6`、`recovery_background=4`、`root_background=55`。
正样本和 decision-boundary 样本只用于训练期 bounded weighting，最终评测仍使用自然分布。
配置和采样实现分别为 `configs/stage3d_recovery_observability_v1.json` 与
`tools/collect_stage3d_recovery_opportunities.py`。

在四序列 fresh temporal split（80--99，共80帧）上，先用现有 ladder 模型做冻结
XGBRanker 闭环诊断。该批旧训练记录中的可选 F1--F3 字段大多缺失，因此下面的结果是
feature-ladder/controller 基线，不应解释成“ORB appearance 已经带来提升”：

| level | success | FAR | cost/frame | Geometry call rate | Geometry opportunity recall |
|---|---:|---:|---:|---:|---:|
| F0 | 51.25% | 4.65% | 0.945s | 7.5% | 0/8 = 0% |
| F1 | 50.00% | 4.76% | 0.677s | 7.5% | 0/4 = 0% |
| F2 | 52.50% | 8.70% | 0.602s | 2.5% | 0/6 = 0% |
| F3 | 51.25% | 6.82% | 0.806s | 5.0% | 0/5 = 0% |
| feedback-rule | 57.50% | 9.80% | 2.272s | 41.25% | -- |

所有 ladder 版本均通过 FAR 和相对成本约束，但 success 尚未达到 feedback-rule 的
`-2 pp` 非劣界；更关键的是每个版本都漏掉了该 split 上全部 Oracle Geometry-optimal
机会。F0--F3 的离线 ranker margin 虽然较高，仍不能推出闭环 action selection 正确。
这说明当前失败仍可能来自 recovery-state coverage、terminal/controller 耦合或
cost-sensitive value 标定，尚不足以宣称 structured observation 的信息论瓶颈。

因此 Stage 3D 的下一条硬性实验路径已经固定为：

1. 用 `--stage3d-preaction-features` 在新的 policy-induced recovery split 真实采集
   ORB correspondence 与 appearance/change evidence；
2. 只在训练集重训 F1/F3，冻结 XGBRanker 和 terminal verifier；
3. 在全新 split 重做 opportunity recall、precision、missed-regret、margin sign 和
   Success--Cost--FAR Pareto Gate；
4. 只有当 structured ladder 已接近 observability ceiling 且 raw appearance/change
   明显增加 Geometry opportunity recall，才进入 Qwen-VL/视觉 Agent 比较。当前不进入
   Qwen、RL，也不把本轮结果写成最终泛化结论。

### Stage 3D v2：真实 pre-action ladder 闭环（2026-10-13，completed / diagnostic positive, feature gain negative）

在修复 Stage 3C.6 的同尺度 component utility、post-Geometry observation 清洗和
terminal/acquisition 协议后，使用 4,646 个 state 重训 F0--F3 component controller，
并在完全未用于训练和选参的四序列 100--119 帧（每层80帧）运行真实 GPU 闭环及两步
live-cost Oracle。F1--F3 的 ORB、temporal、appearance/change 字段均由在线 backend
实际产生，且其耗时计入 Localize receipt；RoMa/PnP/inlier 等后验量仍严格禁止进入策略。

| level | success | FAR | cost/frame | Geometry rate | Geometry recall | Geometry precision |
|---|---:|---:|---:|---:|---:|---:|
| F0 | **58.75%** | 2.08% | 1.553s | 41.25% | 9/17 = 52.94% | 9/33 = 27.27% |
| F1 | 56.25% | **0%** | 1.459s | 43.75% | 8/15 = 53.33% | 8/35 = 22.86% |
| F2 | 57.50% | **0%** | 1.298s | 41.25% | 8/15 = 53.33% | 8/33 = 24.24% |
| F3 | 52.50% | 2.33% | **1.125s** | 35.00% | 5/13 = 38.46% | 5/28 = 17.86% |

相较 v1 的 `0%` Geometry opportunity recall，v2 已恢复到 `38.5%--53.3%`，说明
此前主要是数据覆盖、utility/controller 协议和闭环分布问题，不能再把失败归因于
structured observation 的信息上限。四层 acquisition-admission recall 均为
`96.8%--99.0%`，terminal preemption 已不再是主要瓶颈；剩余误差集中在 acquisition
内部的 Geometry-vs-cheap 排序。Geometry precision 只有 `17.9%--27.3%`，表明策略
同时存在漏掉高价值 Geometry 和过度调用低价值 Geometry 两类校准错误。

新增特征没有形成单调增益：F1/F2 与 F0 基本持平，F3 的 success、Geometry recall、
precision 和 Oracle regret 均变差。因此本轮不支持“appearance/change proxy 提升
RoMa 边际价值预测”，也不支持现在切换到 Qwen-VL。当前 Pareto 候选是 F0（最高
success）与 F2（零 FAR、较低成本且 recall 相当）。由于 100--119 split 尚无同协议
feedback-rule 结果，正式三目标 Gate 暂记为 **pending baseline**，不能拿 80--99 的
旧 baseline 作严格验收。

下一步只做两项：首先在同一 100--119 split 运行 feedback-rule 并做 paired bootstrap；
随后冻结 F0/F2 observation，专门训练 Geometry-vs-best-cheap 的 pairwise advantage/
ranking calibration，报告 opportunity recall、precision、regret 与 Pareto frontier。
在该比较完成前不引入 Qwen、Qwen-VL 或 RL。

汇总产物：`outputs/lilocbench/stage3d_v2_fresh_collect_r1/aggregate_summary.json`；可复现
聚合器：`tools/summarize_stage3d_ladder.py`。

### Stage 3D v2.1：同 split feedback-rule 基线（2026-10-14，completed / Gate fail）

为消除 80--99 与100--119 temporal split 不可直接比较的问题，在与 F0/F2 完全相同的
四序列100--119帧、80共同帧、真实GPU backend、两次 acquisition、30s预算和
pre-action feature cost 计量协议下，独立运行 feedback-rule。每一个 feedback 选择后仍
按相同两步 live-cost Oracle 标注；Oracle 只用于事后审计，不参与在线规则决策。

| policy | success | FAR | cost/frame | Geometry rate |
|---|---:|---:|---:|---:|
| feedback-rule | **61.25%** | 2.00% | 1.597s | 52.50% |
| F0 | 58.75% | 2.08% | 1.553s | 41.25% |
| F2 | 57.50% | **0%** | **1.298s** | 41.25% |

逐帧 paired bootstrap（10,000次，95% CI）显示：F0 相对 rule 的 success 差为
`-2.50pp`（CI `[-6.25, 0]pp`），略低于 `-2pp` 非劣界；F0 成本仅低 `0.045s`
（CI跨0），FAR几乎持平。F2 的 success 差为 `-3.75pp`（CI `[-8.75, 0]pp`），但
成本低 `0.299s`（CI `[-0.523,-0.072]s`）且 FAR=0。两者均因 success 非劣与绝对
`<1s` 成本条件未同时满足而 Gate fail；它们与 feedback-rule 都是 non-dominated，不能
宣称 learned policy 已超越 rule。

这一步确认后续不应继续堆 F3 特征：F0/F2 的差距主要是 Geometry-vs-cheap acquisition
ranking，而非缺少同协议对照。下一项应在冻结 F0/F2 observation 下，以同一批
policy-induced states 训练/审计直接的 Geometry advantage pairwise head；目标是在不牺牲
FAR的条件下恢复至少约3--4个正确帧，或以显著更低成本建立可重复的 Pareto 前沿。

产物：`outputs/lilocbench/stage3d_v2_feedback_rule_fresh_baseline_r1/aggregate_summary.json`、
`outputs/lilocbench/stage3d_v2_fresh_collect_r1/{f0,f2}/pareto_gate_feedback_rule_same_split.json`、
`tools/evaluate_stage3d_pareto_gate.py`。

### Stage 3D F0/F2 baseline freeze（2026-10-14）

F0（最高 success）与 F2（零 FAR、较低成本）现被冻结为后续 Geometry-advantage 的仅读
comparison baselines。冻结同时锁定 100--119 的 feedback-rule、feature schema、verifier、
utility、checkpoint、训练摘要和同 split 聚合/Gate 结果；19个文件均记录 SHA-256，可由
`tools/verify_stage3d_frozen_baselines.py` 检查。后续工作不得在此 split 上调 F0/F2 特征、
阈值或 utility；任何新的 final split 都必须重新运行 feedback-rule。

### Stage 3E：recovery-targeted pairwise recovery/cost controller（2026-10-14，completed / diagnostic mixed）

本轮实现最终路线中的 recovery-targeted DAgger、`P_recover + C_hat`、pairwise ranking 与
单一显式 controller；terminal verifier、Accept/Reject/Defer 语义和 F2 observation 均冻结，
新模型只在允许 acquisition 且 Geometry 与廉价动作均可行时决定下一项证据。训练集含
1,106 个定向 state：Geometry-positive recovery=112、boundary=26、recovery-negative=723、
root-negative=219、Geometry-positive root=26；另有114个 policy-induced DAgger train-only
state。模型共享 state/action encoder，分别预测 correct recovery probability 与执行前成本；
以 Oracle value 的 regret 加权 pairwise loss 监督动作相对次序。在线唯一 score 是
`P_recover - lambda*C_hat`；Geometry margin 的单阈值仅在 validation 上校准，未保留旧版
classifier AND advantage-regressor 的双重门控。

全新 split 使用四序列共同可用的 140--154 帧（每序列15帧、共60帧）。因为
`dynamics_0` 在154后结束，不能用伪造的80帧门槛；三种方法均截断在同一共同前缀。每帧均
使用真实 GPU backend、两步 live-cost Oracle、两次 acquisition 与30秒 budget。聚合器
`tools/summarize_stage3d_ladder.py --require-aligned` 新增逐序列帧数校验；此前错误的
15-vs-20帧比较会被直接拒绝，最终结果通过该校验。

| policy | success | FAR | measured cost/frame | Geometry rate |
|---|---:|---:|---:|---:|
| Stage 3E recovery/cost ranker | 60.00% | **0%** | **3.562s** | 60.00% |
| frozen F2 controller | **63.33%** | **0%** | 4.961s | 50.00% |
| feedback-rule | 61.67% | 5.13% | 5.267s | 48.33% |

相对 F2，Stage 3E 少2个正确帧（`-3.33pp`，paired 95% CI `[-8.33,0]pp`），但实测成本低
`1.40s/frame`（CI `[-1.99,-0.82]s），属于 non-dominated trade-off，不能替换 F2。相对
feedback-rule，Stage 3E 少1个正确帧（`-1.67pp`，CI `[-8.33,+3.33]pp`）、FAR从5.13%降至0，
成本低`1.70s/frame`（CI `[-2.42,-0.98]s）。不过原 Pareto Gate 预注册80共同帧与`<1s`
绝对成本门槛均不满足，故该60帧 fresh split 只能作为独立诊断，不能宣称正式 Gate pass 或
learned policy 胜过 feedback-rule。不同批次的 GPU wall-time/receipt measurement 也不应
直接作为论文级绝对 latency 主张，后续需在固定硬件条件下重复。

更关键的 Geometry 审计显示，Stage 3E 的 conditional Geometry opportunity recall 为
`5/7=71.4%`，高于 feedback-rule的70.0%，且 acquisition ranking regret最低（0.0606，对
F2=0.0934、rule=0.1085）。但它调用36次 Geometry、precision仅`5/36=13.9%`，低于
F2的20.0%和 rule的24.1%。这解释了“离线/局部排序更好却没有更多最终 success”：当前
margin 校准为召回而放行过多低价值 Geometry，且两步 Oracle utility 与最终 terminal
coverage 并未完全对齐。下一步不增加特征、不上 Qwen；应在冻结 F2 observation 下专门
学习 `Delta P_recover(Geometry, best-cheap)`，以 recovery state 为主训练 pairwise
advantage，并用 precision/expected gain 约束校准单阈值。新训练只使用此前未用于选择的
split，最终仍需新的共同帧 split 与 feedback-rule 重跑。

产物：`tools/{build_stage3e_recovery_ranking_dataset,train_stage3e_recovery_ranking,stage3e_recovery_ranking_policy}.py`、
`tools/{run_stage3d_frozen_controller_eval,run_stage3d_feedback_rule_eval}.sh`（一次性 Stage 3E wrapper 已在收口清理中移除）、
`outputs/lilocbench/stage3e_recovery_cost_ranking_r1/{checkpoint,fresh_140_154/aggregate_summary_aligned.json}`、
`outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/{pareto_vs_frozen_f2,pareto_vs_feedback_rule}.json`。

### Stage 3F：冻结 F2 temporal stability 与同 GPU latency 复验（2026-10-14，completed）

为检验 Stage 3E fresh split 中 F2 看似同时胜过 feedback-rule 的信号是否稳定，本轮完全
冻结 F2 checkpoint、feature schema、utility、verifier 和所有阈值，不训练、不调参。success/
FAR 从三个互不重叠的 causal collector 时间块读取；每块均保持四序列与相同 seed。三个块的
F2-minus-rule success 分别为100--119=`-3.75pp`、120--139=`-1.25pp`、140--154=`+1.67pp`，
符号翻转。汇总220帧为 F2=`54.09% success / 0.83% FAR`，rule=`55.45% / 3.17% FAR`；
success差`-1.36pp`（frame bootstrap CI `[-3.64,+0.45]pp`）。因此 F2 **没有稳定的
success superiority**，最新60帧的正差不能作为方法选择依据。

历史多 GPU cost 仅作描述，不用于 latency 结论。为此新增
`tools/run_stage3f_f2_crossover.sh`：同一 sequence、同一 GPU、同一 backend process 内
顺序执行 F2/rule，且四序列交叉 policy order；随后对相同 `(time block, sequence)` 运行
反向 BA，按 AB/BA 均值消除 warm-up 偏置。首轮 order breakdown 证明位置偏差很大，必须
去偏。去偏后的 F2-minus-rule cost 为100--119=`-0.725s`、120--139=`-1.662s`、
140--154=`-0.098s`，三个块方向一致，12个 sequence-block 的 pooled 均值为`-0.828s/frame`。
这支持“F2 是稳定的效率候选”，但单次AB/BA仍不足以作为论文级绝对 latency claim；论文版
需要更多重复与固定 GPU clocks。

安全也只是总体正信号而非每块严格支配：F2 的 FAR 在三个块为`0%/2.78%/0%`，rule为
`2.00%/2.70%/5.13%`。结论应准确写为：**F2 是当前冻结的 safety--cost Pareto baseline，
success 与 rule 接近但不稳定，不能宣称 F2 全面优于 rule。**

因此下一步先做 F2-vs-rule paired disagreement audit（F2-only success、rule-only success、
共同失败，按 terminal/acquisition/Geometry opportunity 分层）。只有当 rule-only successes
主要来自 F2 错过的 Geometry terminal-success gain，才恢复专门的 precision-constrained
Geometry advantage learner；若差异主要在 terminal/memory，则不得再把问题错误归因于
Geometry head。

产物：`tools/run_stage3f_f2_crossover.sh`、`tools/aggregate_stage3f_f2_stability.py`、
`outputs/lilocbench/stage3f_f2_stability_r1/{temporal_stability,f2_stability_summary_order_debiased}.json`。

### Stage 3B--F 阶段性结论（2026-10-14，已被 Stage 3G 继承）

Stage 3B--F 已完成从协议修正、policy-induced DAgger、Recovery Observability feature
ladder、recovery/cost ranking 到冻结稳定性复验的一轮完整闭环。可靠的消极结论与正信号必须
同时保留：

- **已排除的失败解释**：旧版 terminal decision、最后一次 acquisition 后不可 Accept、
  Reject/Defer 混义、DAgger live-cost/budget 不一致、DAgger 被误划入 validation，以及
  不同帧数下直接比较的协议错误，均已修正或被对齐协议拒绝。
- **尚未解决的科学瓶颈**：F2 的 success 与 feedback-rule 接近却不稳定；Stage 3E 的更低
  acquisition regret 没有转化为更高终端 success，且其低 Geometry precision 表明不能把
  失败简化为“Geometry 调用太少”。这排除了继续仅靠调 DAgger 权重、换更大 policy 或无约束
  提升 Geometry rate 的路线。
- **当前保留物**：feedback-rule 是主成功率对照；冻结 F2 是 safety--cost Pareto baseline，
  Stage 3E 是“recovery/cost decomposition 可行、终端效用仍未闭合”的诊断基线。三者均须在
  后续 fresh split 上并列报告。
- **当时的直接下一步**：构建同一 `(sequence, frame)` 上的 F2-vs-rule paired
  disagreement audit。该审计及 terminal alignment 已在下方 Stage 3G 完成；其结果不支持
  启动 precision-constrained Geometry advantage learner。

### Stage 3G：成对分歧、terminal 对齐与阶段收口（2026-10-14，completed）

三个时间块共220个共同帧的 F2-vs-rule 成对结果为
`both-success/both-failure/F2-only/rule-only=118/97/1/4`。rule-only 中仅1帧是 F2 漏掉
Oracle Geometry opportunity，1帧是 terminal eligible 状态下错过 Oracle Accept，其余主要
是 Memory/localize acquisition 分歧。547个 F2 决策中有6次 Oracle Accept 优于所选动作，
但只有1次改变最终配对结果。

复核同时发现 legacy feedback-rule 在 verifier 通过后自动 Accept，而 F2 必须显式选择
Accept/Defer。固定 rule acquisition、仅改用冻结 F2 terminal controller 后，220帧结果为：

| policy | success | FAR | accepted/correct |
|---|---:|---:|---:|
| frozen F2 | 54.09% | 0.83% | 120/119 |
| legacy feedback-rule | 55.45% | 3.17% | 126/122 |
| terminal-aligned feedback-rule | 53.64% | 3.28% | 122/118 |

legacy rule 相对 F2 的正确帧优势从 `+3` 变为 `-1`，故旧 success 差主要不能归因于更好的
acquisition。直接移植 F2 terminal controller 也未改善 rule-induced states 的 FAR，说明
terminal calibration 存在状态分布依赖。本批成本没有同进程反平衡重跑，只作描述。

**Stage 3 正式收口。** 冻结 F2、feedback-rule、source-aware verifier 和当前 backend；
停止在现有220帧继续调 Geometry head、terminal threshold、DAgger 权重或扩大模型。下一阶段
转为 failure-triggered、scene-disjoint 的主动取证可实现价值验证，协议见
`docs/active_evidence_realizable_value_plan.md`。统一汇总为
`outputs/lilocbench/stage3f_terminal_aligned_feedback_r1/stage3_closeout.json`。
