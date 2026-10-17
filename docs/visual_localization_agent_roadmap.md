# Visual Localization Agent：研究定位、系统设计与 IROS 路线

最后更新：2026-10-14（Asia/Shanghai）

> **当前状态：Stage 3G 已收口。** 成对审计显示 F2 与 rule 在220帧中只有5帧最终结果不同；terminal-aligned 复验进一步证明 legacy rule 的微弱成功率优势主要混入了自动 Accept 接口差异。F2、rule、verifier 和现有工具现已冻结，不再在这些 split 上优化 Geometry/controller。下一阶段只验证“主动选择新证据是否存在可实现价值”，协议见 [主动取证可实现价值验证](/data2/liusy/reloc3r/docs/active_evidence_realizable_value_plan.md)。

## 1. 当前决策

项目继续以 **Failure-aware Agentic Visual Localization in Dynamic Indoor Environments** 为主线，但对“Agent”的定义作如下收敛：

> Agent 的核心不是在 Reloc3r、Memory、HLoc 和 RoMa 之间做一次性分类，而是在定位证据不足或相互冲突时，维护带来源和时效性的定位状态，主动选择下一项能够消除歧义的证据，观察工具返回后继续决策，最终 Accept、Reject 或 Defer。

项目可以从“Agent No-Go”转入**受控 Agent 原型研究**，但论文层面的 Agent 收益仍为未证明。当前优先级是先建立真实可交互的定位环境和主动取证基线，再进行 Agentic SFT；不能直接把 AVL-008 的 zero-shot VLM 或 AVL-015 的冻结回放包装成 Agent。

### 1.1 Stage 3 的最新事实与边界（2026-10-14）

真实 GPU 闭环已完成 terminal control、因果 Memory、live-cost Oracle、feedback-driven action
selection 和 policy-induced DAgger 的基础实现。它排除了先前的 terminal/budget/DAgger 划分
协议错误，但并没有自动得到可发表的 learned Agent：

- **Stage 3E** 把目标改为 recovery-state 下的 `P_recover(s,a)`、执行前可见的
  `C_hat(s,a)` 与 pairwise ranking。共同 60 帧 fresh split 上，它取得
  `60.00% success / 0% FAR / 3.562s`；冻结 F2 为 `63.33% / 0% / 4.961s`，
  feedback-rule 为 `61.67% / 5.13% / 5.267s`。样本量不足且绝对成本仍高，不能宣称 Gate
  pass。其 Geometry opportunity recall 为 `5/7=71.4%`，但 precision 仅
  `5/36=13.9%`：问题是过度调用低价值 Geometry，非简单漏掉强几何。
- **Stage 3F** 在三个互不重叠时间块（共 220 帧）冻结 F2 复验。F2 相对 rule 的 success
  差为 `-3.75/-1.25/+1.67pp`，合计 `-1.36pp`（95% CI `[-3.64,+0.45]pp`）；因此“F2
  在成功率上全面优于规则”的说法被否定。相同 GPU、同 backend 的 AB/BA 去偏复验则给出
  F2-rule cost `-0.725/-1.662/-0.098s`，合计 `-0.828s/frame`，支持效率优势的方向性证据，
  但尚不足以给出论文级绝对 latency claim。
- 因而当前准确表述是：**F2 是冻结的 safety--cost Pareto baseline，不是已经优于规则的
  Agent。** 规则仍是主成功率对照；F2 是后续每个 learned acquisition 改动必须同时比较的
  最强轻量 sequential policy。

这些结果也收窄了科学问题：当前缺口不能简单归因于“模型不够大”。Stage 3G 已排除
“继续学 Geometry/controller 就能带来稳定终局收益”这一当前主假设；下一问题是主动动作
能否生成原先不存在的独立证据，并改变 recovery episode 的终局。

建议的论文工作标题为：

> **Learning to Acquire and Validate Evidence for Visual Relocalization in Changing Environments**

核心科学问题是：

> 当地图定位、历史状态和几何结果相互矛盾时，系统应当相信哪些证据，以及下一步获取什么新证据，才能在受限成本下可靠地重新建立全局定位？

## 2. 现有实验对 Agent 设计的约束

现有 AVL-002–015 已经建立了以下事实：

1. 固定 Top-10 Reloc3r 在 static / dynamic / LT / dynamic+LT 上的成功率为 `39.79 / 13.55 / 24.88 / 23.48%`，reference/candidate reliability 是首要可操作问题。
2. consensus-3 和 memory-3 可显著恢复错误聚合，但 Memory 也会产生“内部一致、后验错误”的状态。
3. 强几何能恢复大量困难帧，Always-RoMa 全量为 `91.21%`；HLoc/RoMa 均存在有输出但错误的情况，inlier 和 reprojection 不能单独充当可靠性真值。
4. test 上 Consensus/Memory/HLoc/RoMa/Historical 分别为 `72.15/76.87/88.98/92.22/44.78%`，固定候选池五动作 Oracle 为 `96.95%`。相对 Always-RoMa 的理论增量只有约 `4.72 pp`，且 Oracle 使用 GT outcome，不能代表可部署策略。
5. 当前 Historical-keyframe Relocalization 的 test success 为 `44.78%`，相对全部其他动作只有 `2` 个 test 独占成功帧；专属 verifier 的 test selective accuracy 仅 `63.96%`。当前实现不能作为可靠的新恢复能力。
6. Reject 能减少错误状态写回，但 AVL-013 v3 的 test coverage / full-frame success 只有 `54.23/51.57%`；当前系统会“拒绝错误”，却缺少高覆盖恢复证据。
7. AVL-006/007 的轻量 router 和冻结 DINO 表征未可靠识别 Memory-only/RoMa-only；AVL-008 仅证明当前 zero-shot、one-shot VLM routing 没有额外信息，不能外推为任务训练后的多轮 Agent 无效。
8. 当前 DPVO/anchor 实现没有提供独立定位收益，DPVO 暂只保留为 motion/uncertainty evidence。

这些结果说明，论文价值不能只来自“增加多个 pose estimator 后学习选择”。Agent 必须能够通过动作改变 reference、验证证据、历史状态或后续观测，产生固定动作池中原先不存在的信息。

## 3. 三个必须先处理的实现问题

### 3.1 固定候选池限制了可证明的 Agent 价值

AVL-015 Pose Oracle 在保存的 Consensus/Memory/HLoc/RoMa/Historical 输出上事后计算 union。它能够回答“已有工具输出是否互补”，不能回答“主动选择新的 references 或验证证据是否会生成新的正确候选”。

因此后续 Agent 不能只在固定缓存结果中路由。动作至少要能改变 active reference set、证据来源或 Memory 查询范围；否则系统本质上仍是 estimator orchestration，理论上限也被现有固定候选池约束。

### 3.2 当前回放不是策略可交互环境

`tools/run_avl015_historical_memory.py` 从 AVL-013 既有 trace 中读取 `accepted && state_updated` 的历史帧；Agent 若在更早帧改变 Accept/Reject，历史库不会随之重建。`tools/eval_avl015_dynamic_retrust.py` 也明确只在冻结 candidate/evidence 上重算分数，不重新生成 pose 或 route。

因此现有缓存适合诊断、预训练和回归检查，不足以训练或评测多轮 Agent。下一版必须提供：

```text
reset(sequence, map)
step(action) -> observation, updated_state, measured_cost, terminal
```

策略改变 reference、Accept/Reject 或 Memory 写入后，后续候选、Memory、状态和工具观测必须因果地改变。缓存键至少包含 query、reference IDs、工具、工具参数、随机种子和影响输出的状态版本。

### 3.3 AVL-015 dynamic re-trust 的旧状态衰减实际未生效

`tools/run_avl012_reliability.py` 生成的 `geometry_risk` 已经取了 `geometry_vs_previous` 项的最大值；`tools/eval_avl015_dynamic_retrust.py` 随后在 `max()` 中保留未衰减的 `geometry_risk`，又对重复的 `geometry_vs_previous` 项乘 `state_trust`。未衰减的复合项始终覆盖后者。

对现有 `1108` 条 geometry-source evidence 做只读检查时，将状态从 fresh 改为极度 stale，最终 score 在 `1108/1108` 条记录上均不改变。因此 AVL-015 的 dynamic re-trust 数值仍可保留为该实现的回放结果，但不能支持“旧状态衰减无效”或“动态重新信任没有价值”的科学结论。

修复后已用 `outputs/lilocbench/AVL-012_componentized/` 重跑：fresh/stale score 在 `303/1108` 条 geometry-source evidence 上发生变化；componentized replay 的 test coverage=`50.00%`、accepted accuracy=`96.85%`、full-frame success=`48.43%`。这证明风险分解和衰减路径已经生效，但它仍是冻结候选 replay，不能替代后续真实 route regeneration。

修正要求：

- 保存不可约的原始风险项，不在上游提前把 intrinsic geometry、cross-source disagreement 和 previous-state disagreement 合并成一个不可拆分的 `max`；
- 只对 state-dependent 项施加 age/reject/motion trust；
- 保留 intrinsic geometry 和当前独立 cross-source evidence；
- 增加单元级不变量：当只有 previous-state disagreement 改变时，fresh 与 stale score 必须不同；
- 在修正后重新生成 causal route，而不只重放接受阈值。

## 4. 相关工作与论文边界

以下调研截至 2026-10-12。SpotAgent、PoseAgent、RIC-Loc、Active Exploration 和部分 2026 工作目前按预印本或近期论文处理，正式写作时需重新核对最终版本。

| 方向 | 代表工作 | 已解决的问题 | 对本项目的边界约束 |
|---|---|---|---|
| ReAct 与外部取证 | [ReAct](https://arxiv.org/abs/2210.03629)、[SpotAgent](https://arxiv.org/abs/2602.09463) | reasoning/action 交替；Web、地图、图像工具；SFT→agentic cold start→RL | 多轮工具调用和 SFT 本身不是新贡献；本项目需解决米制 6-DoF、连续状态和几何约束 |
| 估计器编排 | [PoseAgent](https://arxiv.org/abs/2609.38755) | learned ranker、pose verifier、失败后按排序 fallback | “profile→rank→run→verify→fallback”已有直接先例；本项目需主动改变证据而非只换 estimator |
| 选择性定位 | [RIC-Loc](https://arxiv.org/abs/2607.04722) | 保留 reference-induced hypotheses，用 dispersion/covariance 进行 failure ranking 和 abstention | candidate consensus、confidence、Reject 已有强相关工作；需要连续恢复、证据依赖和主动取证 |
| 主动地图搜索 | [Active Exploration for Sparse Visual Localization](https://lup.lub.lu.se/search/publication/7978aa9d-21d8-44d6-b304-adf39982b356) | 自适应探索地图区域，寻找更多有效 2D–3D 支持 | reference action 必须与几何驱动主动搜索比较 |
| 主动视角定位 | [Towards Accurate Active Camera Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5136_ECCV_2022_paper.php)、[ActLoc](https://proceedings.mlr.press/v305/li25b.html) | 基于定位不确定性选择视角、运动和停止时机 | 若增加转头/移动动作，需与 active viewpoint selection 比较 |
| 位姿验证 | [InLoc](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/1803.10368.pdf)、[FSNet](https://openaccess.thecvf.com/content/CVPR2023/html/Barroso-Laguna_Two-View_Geometry_Scoring_Without_Correspondences_CVPR_2023_paper.html) | view synthesis 或学习式 two-view geometry scoring | verifier 需要候选之外的证据，不能只重复候选生成时的 inlier/reprojection |
| 长期经验地图 | [Experience-based Navigation](https://ori.ox.ac.uk/news/experience-based-navigation) | 保存、检索和管理不同外观条件下的经验 | Memory 的存在不是新贡献；错误经验传播、可信写入和主动查询才是切入点 |
| 长期动态室内定位 | [RIO10](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520460.pdf)、[LILocBench](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/) | 长期结构/外观变化与动态室内 benchmark | 需要跨场景或隐藏 GT 评测，不能把单环境内部 split 写成泛化 |

与 PoseAgent 的关键差异应当明确为：PoseAgent 主要在一个 image pair 上顺序执行估计器；本项目以连续 localization state 为对象，动作会改变 reference、experience、验证证据和未来 Memory，研究的是“下一步需要获得什么证据”。

## 5. 拟议方法：Evidence-seeking Stateful Localization Agent

### 5.1 问题形式

将一次失锁或不确定区间定义为有限预算的 POMDP：

```text
history h_t
  = current observation
  + pose hypotheses/belief
  + evidence provenance
  + tool observations
  + memory state
  + remaining budget

policy pi(a_t | h_t)
  -> acquire evidence / estimate / verify / update / terminate
```

训练时可用 GT 计算动作价值和最终 reward；在线 observation、tool result、Memory update 和 decision 中禁止 query GT、未来帧和事后标签。

可用的效用形式为：

```text
U(a | h) = E[decision-loss reduction after a | h] - lambda * measured_cost(a)
```

错误接受的代价应显著高于短时 Reject，因为错误 pose 会污染后续状态和 Memory。

### 5.2 Localization State

Agent 观察的状态不应只是一张 RGB，而应包括：

- 当前 query、已请求的 map/history reference 缩略图或视觉 tokens；
- 少量当前 pose hypotheses，而非强制压成单个位姿；
- 每个 hypothesis 的来源、reference set、工具、时间和支持/冲突证据；
- 最近可靠 anchor、anchor age、reject streak、motion uncertainty；
- 当前处于 localized、uncertain、lost 或 recovery；
- reference 与 historical experience 的可靠性及依赖的 anchor；
- 已调用动作、返回结果、实际成本和剩余预算。

必须显式保存证据 provenance。共享同一 reference、同一 anchor 或同一 matcher 的两个结果不是两份独立支持；来源记录用于避免重复证据被错误计数，但不能自动证明统计独立。

### 5.3 Action 与 Observation

第一版动作保持少而参数化：

| Action | 主要参数 | 返回 Observation | 目的 |
|---|---|---|---|
| `RetrieveReferences` | pool size、排除的 reference group、appearance/region mode | reference IDs、scores、diversity、cost | 查找尚未检验的地图证据 |
| `EstimatePose` | reference IDs、backend | pose hypotheses、matching/geometry evidence、cost | 产生候选位姿 |
| `VerifyCandidate` | candidate ID、held-out evidence IDs | pose-conditioned verification、support/conflict、cost | 用未参与估计的证据验证候选 |
| `QueryExperience` | anchor group、appearance、age range | historical candidates、reliability/dependencies、cost | 查询历史经验，而非复制历史 pose |
| `UpdateMemory` | candidate/evidence IDs | staged/confirmed/quarantined status | 管理可信经验 |
| `Finish` | Accept(candidate)、Reject、Defer | terminal state | 自适应停止 |

Agent 的关键能力是**先行动再决定**。例如地图候选冲突后，Agent 可排除原 reference cluster 检索另一组证据，再用 held-out reference 验证两个竞争位姿，而不是只把 RoMa 换成 HLoc。

### 5.4 Geometry-grounded Verification

Verifier 保持为任务专用模块，Agent 读取其结果并决定下一步。可靠性证据分为：

1. **candidate intrinsic**：reprojection、inliers、spatial coverage、triangulation condition、degeneracy；
2. **fresh cross evidence**：未参与估计的 reference、另一 reference group、另一工具或后续新观测；
3. **state compatibility**：与 previous anchor、motion 和历史状态的一致性。

只有第三类证据随 anchor age、reject streak 和 motion uncertainty 衰减。长期 Lost 后，stale pose 不能否决具有高质量 fresh evidence 的新全局定位。接受阈值需要在 validation 上冻结，并报告 risk–coverage、false acceptance 和恢复时间。

### 5.5 Experience Memory

Memory 记录：

```text
image / descriptor / verified pose / pose covariance
references and evidence used / anchor dependency
timestamp and age / reliability history / status
```

写入采用 `staged -> confirmed` 两级机制；错误 anchor 相关的条目可降权或 quarantine。关键帧选择除外观相似度外还需考虑 viewpoint baseline、空间覆盖、历史可靠性和证据独立性。当前帧必须通过相对几何和独立 verifier 获得新 pose，禁止直接复制历史 pose。

Memory 不必以大量 only-action 帧证明价值；它也可以通过更低成本、更短恢复时间或减少重复地图搜索产生收益，但必须在闭环中测量。

## 6. 基本系统实现

建议将现有脚本重构为以下层次：

```text
LocalizationEnvironment
├── MapIndex / ReferenceGroups
├── PoseTools
│   ├── Reloc3r consensus
│   ├── rerank + Reloc3r
│   ├── rerank + RoMa
│   └── standard HLoc（公平强基线）
├── EvidenceStore
├── Source-aware / Pose-conditioned Verifier
├── ExperienceMemory
├── CausalStateMachine
└── CostLedger

Policy
├── Fixed order
├── Geometry-greedy evidence acquisition
├── Lightweight sequential policy
└── VLM/SFT ReAct policy
```

环境接口要求：

- `reset()` 后只暴露当前/历史可用信息；
- 工具结果只有在相应 action 调用后才进入 observation；
- 每个工具调用保存输入、输出、seed、elapsed、GPU/CPU boundary 和 provenance；
- Accept/Reject 改变后续 Memory 和状态；
- evaluator 单独读取 GT；
- 支持离线缓存，也支持未命中时真实执行；
- trace 能重放并审计 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。

Phase A 的第一版实现已落在 `tools/agentic_localization_env.py`：它提供
`reset()`、同帧证据 acquisition、`verify_candidate`、`accept/reject/defer` 和下一帧推进；
`tools/lilocbench_agent_backend.py` 将 reference action 接到真实的当前帧 Reloc3r 推理，
每次 reference 子集都会重新构造 pair batch 并执行模型，不接受 cache-hit 回执。只有 verifier
返回 accepted 的候选才能写入在线 Memory。`tools/run_agentic_localization_env.py` 是第一版
主动 reference batch baseline，尚不是 Agent 论文结果；环境还提供 `audit_trace()`，可直接审计每次
action 的 state transition，并核对 execution receipt 中的 query frame/reference IDs。在线 backend
在 CUDA 不可用时会显式失败，不会静默退化到冻结候选。

为验证环境本身不依赖 GPU，`tools/cpu_orb_agent_backend.py` 提供了明确标注的 CPU smoke backend：
每次 action 重新执行 ORB matching、map-pose triangulation、PnP 和 evidence verification。
它不是论文定位 baseline，只用于验收交互语义。已在 `static_0` 前 3 帧真实图像上跑通，生成
`outputs/lilocbench/AVL-agent-cpu-smoke/static_0/environment_trace.jsonl`，9 次 transition 的
environment audit 全部通过；`tools/run_agentic_reference_switch_smoke.py` 另在同一帧用两组不同
reference 完成真实重算，第二组候选必须重新 verification，audit 通过。
`tools/test_agentic_localization_env.py` 覆盖了执行回执、验证后写入、
拒绝写入和 evaluator 字段隔离四条因果约束。

第一版建议把 Agent 作为**事件触发的失锁恢复模块**。正常定位沿用轻量路径，只有 verifier 判为不确定或 lost 时进入多轮取证，避免过早承诺逐帧实时 VLM 推理。

## 7. Agentic Trajectory 与训练路径

### 7.1 数据生成

现有 1696 帧缓存可用于构造初始状态和部分反事实，但不足以直接生成完整交互数据。训练场景需对合法动作展开工具结果：

```text
O1 -> RetrieveReferences(alternative group)
O2 -> EstimatePose(RoMa, selected refs)
O3 -> VerifyCandidate(held-out refs)
O4 -> QueryExperience(other anchor group)
O5 -> Finish(Accept / Reject)
```

使用训练 GT 评价最终 pose、错误接受、恢复时间和成本，并通过有限深度搜索、beam search 或 cost-aware oracle 生成专家轨迹。轨迹集必须同时包含：

- 直接接受；
- 获取新证据后恢复；
- 新证据推翻旧 anchor；
- 合理 Reject/Defer；
- 工具失败、无效重复调用和错误 Memory 的纠正轨迹。

只保留成功轨迹会导致模型不会停止、拒绝和恢复。

### 7.2 训练阶段

1. **结构化 Agentic SFT**：训练 `state/evidence summary -> tool call or Finish`。主要监督结构化动作，不把长自然语言解释本身当作贡献。
2. **on-policy 数据聚合**：让 SFT policy 在训练环境运行，对它实际访问到的状态重新调用专家/搜索器标注，缓解序列误差累积；思想上对应 DAgger。
3. **可选 RL**：只有当 SFT 已能正确利用反馈，而长期成本、Memory 污染或自适应停止仍难以通过短轨迹学习时，再使用 RL。

可使用如下 reward：

```text
R = pose_success
    - lambda_cost * measured_tool_cost
    - lambda_false_accept * false_accept
    - lambda_delay * recovery_time
    - lambda_pollution * invalid_memory_update
```

其中 `lambda_false_accept` 和 `lambda_pollution` 应高于短期 Reject 的代价。

### 7.3 模型选择与公平对照

当前硬件为 8×RTX 4090 24 GB，支持小规模多图工具展开和 7B 级模型的参数高效微调；实际 batch、上下文和吞吐仍需测量。第一版先以相同结构化 observation 训练 MLP/small Transformer learned policy；随后将 Qwen（文本/结构化输入）的 LoRA-SFT 作为待验证候选，并使用相同 state、action mask、utility/value target 与测试协议进行公平比较。Qwen-VL 不进入第一阶段，只有结构化 evidence 已被证明不足以区分关键状态时，才引入 raw image/crop。

这一区分用于分别回答：

- 多轮交互是否比 one-shot routing 有价值；
- 较复杂的语言策略是否比结构化小模型提供额外价值；
- 若后续引入视觉语言输入，原始图像语义是否提供额外价值。

如果轻量 policy 更好，系统仍可作为 learned localization agent，但不能主张语言推理带来收益。

## 8. IROS 级别的实验设计

### 8.1 数据分层

1. **LILocBench**：继续作为问题发现、环境开发和隐藏 GT challenge 评测入口；当前四条件同属一个办公室，40/60 split 只能称 calibration-shift test。
2. **多场景训练/验证**：使用 ScanNet++ 等多场景 RGB 序列构造训练 episode。合成遮挡、reference corruption 和状态错误可用于训练，不代替真实变化测试。
3. **独立长期变化测试**：恢复 RIO10 或其他未参与设计的新场景评测，严格对齐 RGB/地图协议。
4. **实机恢复**：在未用于开发的环境中测试遮挡、重复走廊、家具变化、错误初始化或机器人搬移后的恢复。

### 8.2 主对照

- best fixed tool（Always-RoMa/标准 HLoc）；
- fixed-order + 同一 verifier；
- learned ranker + verification-guided fallback（PoseAgent-style baseline）；
- one-shot learned router；
- geometry-greedy active reference/evidence search；
- 与 Agent 相同 observation/action 的轻量 sequential policy；
- 完整 SFT ReAct Agent。

### 8.3 关键消融

- 去掉工具反馈，预先输出整条 action sequence；
- 保留反馈但固定 reference set；
- 去掉 evidence provenance/dependency；
- 去掉 anchor aging 和 stale-state re-trust；
- 去掉 staged Memory / invalidation；
- 固定策略，只替换更强 verifier；
- 相同工具预算下比较随机、固定、贪心、轻量 policy 和 VLM policy。

### 8.4 指标与统计

- full-frame success、coverage、accepted accuracy；
- false accepts / accepted outputs 与 false accepts / all frames，明确分母；
- error episode 数、错误状态持续时间、错误 Memory update；
- time-to-first-accepted 与 time-to-first-correct；
- 未恢复/右删失 episode 比例；
- tool calls、matching 数量、总 latency、P95 latency、GPU-hours、Memory storage；
- 在固定 false-accept 或固定 cost 下的 success–coverage–cost frontier。

置信区间以 scene/episode 为重采样单位，不能把高度相关的逐帧结果当作独立样本。正常定位与 Agent recovery 的成本分别报告。

## 9. 可发表的贡献与最低证据

理想的论文贡献由三部分组成：

1. **Evidence-aware state**：显式表示 pose hypotheses、证据来源/依赖、时效和 Memory 污染风险；
2. **Active evidence acquisition policy**：根据当前歧义选择能够区分竞争假设的 map/history/verification evidence；
3. **Causal reliability and recovery evaluation**：在动态长期变化、统一工具预算和真实状态回灌下评价错误接受与恢复。

最低 Go 条件：

- 相同工具、verifier 和预算下，主动取证稳定超过 fixed-order 与 one-shot router；
- 收益来自新证据或更少成本，而非单纯增加工具调用；
- learned policy 在 scene-disjoint test 上超过 geometry-greedy/轻量基线，或以同等成功率显著降低成本/错误接受；
- false acceptance、Memory pollution 和 recovery censoring 均得到报告；
- 至少一个独立长期变化数据集或实机环境复现主要结论。

如果 learned Agent 不能超过确定性主动取证，仍可保留一篇 failure-aware active localization 系统论文，但不应把贡献写成 VLM Agent。若连主动取证都不能超过固定强工具，则应冻结 Agent 主张，转向 verifier/reference selection 或 benchmark 分析。

## 10. 建议执行顺序

### Phase A：修正与环境化

- 拆分 risk 原始项并修复 dynamic re-trust（代码已完成，必须重跑 AVL-012/015）；
- 建立 `reset/step` causal environment（代码已完成，需在 LILocBench GPU 数据上运行）；
- 使 reference、Accept/Reject 和 Memory 写入真正影响后续状态（smoke test 已覆盖）；
- 补齐统一在线成本边界，并验证每个 reference action 的真实 matching/pose elapsed。

### Phase B：验证主动取证是否有价值

- 固定 Reloc3r/RoMa；
- 允许策略选择下一组 reference 或 held-out verification evidence；
- 比较 fixed、random、geometry-greedy 和有限深度 oracle；
- 主要观察相同 matching/tool budget 下是否更快、更可靠地消除歧义。

2026-10-12 已完成更严格的 CPU-ORB causal pilot（见 `docs/active_evidence_pilot_report.md`）。历史关键帧 action 已改为局部 3D + 当前帧 PnP，verification 改为未参与候选生成的 held-out reference 极线一致性。304 个测试帧上 fixed/light 均为 `81/304=26.6%` full-frame success，rule=`77/304=25.3%`，random=`69/304=22.7%`；fixed 与 light 的 accepted accuracy 都为 `93.1%`。因此当前实验仍**未证明主动取证有价值**，也未达到进入 VLM/Agent 训练的门槛。该结果不能外推为主动定位无效，因为真实 GPU Reloc3r/RoMa 强几何尚未接入；在真实 backend、独立验证动作和 scene-disjoint split 复验前保持 Agent No-Go。

该 pilot 没有忽略连续帧 Memory：accepted pose、历史关键帧和 reject state 会因果地传给下一帧，所有策略使用同一个 historical-memory action；但为了隔离 calibration/test，测试段从空 Memory 冷启动。它是 deterministic causal memory，不是已经训练好的“记忆 Agent”。正式连续帧评测还应补充合法历史状态 warm-start，并分别报告两种设置。

### Phase C：训练最小 Agent

- 生成多场景 tool-expanded trajectories；
- Agentic SFT；
- 在 policy-induced states 上补数据；
- 与相同 observation 的轻量 sequential policy 对照；
- 只有在必要时才进入 RL。

Stage 3G 已完成 paired disagreement 与 terminal alignment。结果不支持恢复 Geometry
advantage learner，也不支持继续在当前220帧校准 terminal head。当前主线改为先在
failure-triggered、scene-disjoint episodes 上测量 fixed/rule/greedy 与有限深度 Oracle 的
可实现差距。只有 action-dependent terminal outcome 足够密集且跨场景稳定，才生成 Agentic
trajectory 并训练 learned sequential policy；Qwen、Qwen-VL 和 RL 继续后置。

### Phase D：冻结论文评测

- scene-disjoint / cross-dataset；
- 标准 HLoc 与强固定工具；
- 实机 lost-and-recovery；
- 完整 ablation、成本、错误接受和恢复统计。

当前最值得继续执行的最小复现实验是：

> 固定 Reloc3r/RoMa 和 verifier，只开放“选择下一组 reference / held-out 验证证据”的动作，检验在相同匹配预算下，主动取证能否比固定顺序更可靠地消除定位歧义。

该实验能直接决定后续 Agentic SFT 和 Memory 投入是否有科学价值。当前 CPU pilot 已给出限制性 No-Go：只有真实强几何和独立 verifier 复验显示稳定 action-dependent context 后，才进入训练。
