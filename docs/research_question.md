# 面向动态室内环境的 Agentic Visual Localization：研究问题

最后整理：2026-10-12

## 1. 项目目标与核心研究问题

项目主线是 **Failure-aware Agentic Visual Localization in Dynamic Indoor Environments**。这里的 Agent 定义为一个多轮 evidence-seeking policy：系统以连续 RGB 序列为输入，维护带来源和时效性的定位状态；当证据不足或相互冲突时，主动决定下一步获取哪一项地图、历史或几何证据，根据工具返回更新定位假设，再决定继续取证、Accept、Reject 或 Defer。它不是对 Reloc3r/Memory/RoMa 做一次性分类，也不直接回归 pose。

系统最终应能在当前观测受到行人遮挡、长期场景变化或参考不匹配影响时：

1. 判断当前定位证据是否可靠；
2. 分析失败更可能来自候选冲突、动态遮挡、长期外观变化、历史失配或几何退化；
3. 判断当前缺少什么有区分力的证据，并在检索替代 reference、查询历史 experience、调用定位/验证工具和停止之间做预算受限的多轮选择；
4. 用独立证据验证候选位姿，并记录可审计的决策链。

核心研究问题是：在相同 reference database、可用信息和统一计算预算下，策略能否根据连续定位状态选择能够区分竞争 pose hypothesis 的新证据，比固定工具、固定调用顺序和 one-shot router 更可靠地恢复全局定位，并减少错误接受、状态污染和无效工具调用？

完整方法、相关工作、训练和 IROS 评测路线见 `docs/visual_localization_agent_roadmap.md`。

## 2. 核心假设与预期创新

### 核心假设

- **H1：参考可靠性是第一病因。** 固定 Top-k 聚合会把错误或不兼容的 reference 混入 pose，候选一致性和 reference 重排应能恢复一部分失败。
- **H2：历史状态具有受限恢复价值。** 只访问当前帧之前、且经过验证的 prediction/memory，可恢复重复视角下的短期连续失败；该收益不能来自未来帧或 query GT。
- **H3：强几何工具具有互补性。** 在 Reloc3r/Memory 仍失败的帧上，HLoc-style 或 RoMa 可能恢复部分结果，但两者都不会覆盖所有失败，输出必须继续验证或拒绝。
- **H4：主动取证比一次性路由更接近 Agent 的决策价值。** 检索、候选离散度、跳变、外观变化、历史可靠性和独立几何证据共同定义当前歧义；策略的动作应改变 active references、验证证据或历史查询，从而产生新的 observation，而不是只在冻结 pose 输出之间分类。
- **H5：连续时序状态比单帧语义更接近真实决策对象。** tracking/motion、当前位姿、历史可靠帧、连续失败长度和 reference reliability 的联合状态，应能区分“短暂观测变差但可由历史维持”和“状态已漂移、需要全局恢复”。
- **H6：证据来源、依赖和时效必须显式建模。** 共享同一 reference 或错误 anchor 的多个一致结果不能被当成独立支持；陈旧状态约束应随 anchor age/reject streak 衰减，而 fresh intrinsic/cross evidence 保持有效。

### 预期创新

预期创新是形成可复现的“连续观测—定位假设—主动取证—工具反馈—验证—状态/Memory 更新”闭环：动作、预算、reference、证据 provenance、历史依赖、验证结果和拒绝原因均可追溯，并在统一成本和严格时间隔离下评价。VLM/SFT policy 只有在相同 observation/action/budget 下超过确定性主动取证和轻量序列策略时，才构成语言/视觉推理的独立贡献。

## 3. 已证明的稳定结论

以下结论有对应的逐帧结果和审计产物支持，且不依赖 LLM Agent：

1. **Reloc3r 在 LILocBench 上存在明显条件相关失效。** 固定 mapping-only Top-10、Reloc3r-512 的成功率（static / dynamic / LT / dynamic+LT）为 `39.79% / 13.55% / 24.88% / 23.48%`。失败段最大长度为 `26 / 30 / 37 / 37` 帧，candidate translation dispersion 中位数为 `1.244 / 5.189 / 2.586 / 3.386 m`。static 本身也受 Top-k 冲突影响，不能把低性能全部归因于动态内容。
2. **候选选择和受限历史是有效的确定性动作。** consensus-3 成功率为 `82.70% / 55.48% / 75.12% / 58.60%`，memory-3 为 `91.18% / 58.06% / 81.52% / 62.85%`。这证明“选择/聚合”和短期历史具有恢复价值，但不等于已经证明 Agent 或节省了 Reloc3r 计算。
3. **reference 重排后调用强几何可以恢复一部分困难帧。** A.5 deterministic route 的 RoMa success 为 `1513/1696=89.21%`，HLoc-style 为 `1484/1696=87.50%`；几何调用池仍有 `97/952` 帧两种工具共同失败，且 geometry 输出仍有阈值外误接受。
4. **工具互补存在，但单一强工具不能解决全部问题。** 单次探索结果有 HLoc-only、RoMa-only 和共同失败帧；由于 RoMa seed 影响集合，精确互补计数只作 seed-specific/探索性证据。
5. **统一成本尚未支持 Manager 的效率优势。** seed=0 Always-RoMa 为 `1552/1696=91.51%`、约 `6622.564 s`；recovery-aware Manager 为 `1521/1696=89.68%`、coverage `1672/1696`、geometry call rate `46.17%`，统一成本估计 `6665.885 s`，比 Always-RoMa 高约 `0.65%`。因此“少调用 geometry 就更快”已被当前实现否定。
6. **廉价数值/冻结视觉表征目前不足以稳定路由。** AVL-006 的 LOSO logistic/tree 成功率为 `87.44% / 86.56%`，低于 Always-RoMa；AVL-007 最好树为 `87.62%`，memory-only / RoMa-only recall 为 `24.07% / 74.10%`，balanced accuracy `49.09%`，完整 embedding 特征反而更差。它们说明信号有弱排序信息，但不能独立识别互补动作。
7. **当前零样本 VLM 语义探针没有提供额外路由信息。** AVL-008 在四 cohort 的 216 个固定请求中有 200 个严格解析成功；有效样本的语义—outcome Cramér’s V=`0.2187`，低于预注册门槛。对 `memory-only/RoMa-only` 帧，VLM action balanced accuracy=`47.02%`，同帧 DINO=`47.64%`，差值=`−0.62 pp`。这否定了“只要加入零样本语义理解就足以进入 Agent”的当前解释，但不否定微调、不同输入表示或独立场景探针。
8. **AVL-009 建立了严格 causal 的连续序列 baseline，但当前 Reloc3r temporal edge 没有显示独立收益。** 四条序列相邻 edge 共 `1692/1692` 有效；test 上 temporal-only success=`0.00%`，global+temporal=`53.64%`，global+temporal+memory=`53.64%`，而 memory-3=`76.87%`、Always-RoMa=`92.62%`。Dynamic+LT 中 global+temporal 最大连续失败为 `270` 帧，说明相邻 Reloc3r 相对位姿会形成平滑但错误的长期漂移。该结论只针对当前 Reloc3r edge 实现，不外推为 DPVO/其他 VO。
9. **AVL-010 修正后不支持当前 DPVO 状态回放提供定位收益。** 原始 DPVO full-rate 轨迹本身有效，但旧 Sim(3) 回放把尺度错误地乘到了旋转矩阵，旧派生指标作废。修正后 test 上 DPVO temporal-only=`0.20%`、Global+DPVO=`4.04%`、Global+DPVO+Memory=`4.04%`、Global+DPVO+Memory+RoMa=`65.26%`，低于 memory-3=`76.87%` 和 Always-RoMa=`92.62%`；修正后的 DPVO propagated 输出旋转保持正交。该结果只说明当前 DPVO、anchor 规则和相机设置下没有可独立利用的定位收益，不外推为所有 VO/SLAM。
10. **AVL-011 进一步明确了主要病因和系统定位。** Dynamic+LT 的 541 帧中，Memory/RoMa 的 `both-success / Memory-only / RoMa-only / both-fail` 为 `326 / 14 / 163 / 38`。Top-100 mapping reference 在空间上覆盖 `516/541` 帧，另有 `25` 帧属于 Top-10 选错但 Top-100 有近邻；没有出现空间 reference 全部缺失的 proxy。38 个 both-fail 中，32 个有 reference mismatch proxy、37 个有 candidate conflict、22 个有低纹理/模糊 proxy，且 RoMa 均有输出但后验错误；16 个 Memory 内部一致但后验错误。类别是后验 operational proxy，不是真值语义标注。强化 deterministic baseline 在冻结 validation→test 协议下达到 test=`91.24%`、geometry call=`67.13%`、最长失败=`4`，说明 reliability→升级→回退→验证的非 Agent路线已接近当前 Always-RoMa 参考，但仍不能证明统一成本优势或 Agent 独立收益。
11. **AVL-012 证明可靠性分数有 Reject 校准信息，但还不能作为最终闭环策略。** 用 candidate dispersion、appearance/motion、RoMa inlier/reprojection、multi-reference support 和与 consensus/Memory/previous prediction 的 cross-check 构成 evidence-only risk，test 上形成单调 Accuracy–Coverage 曲线。validation 冻结阈值 `1.5` 后，test coverage=`55.22%`、selective accuracy=`95.72%`、错误接受率=`4.28%`；未拒绝的 AVL-011 route 为 coverage=`100%`、success=`91.24%`。因此风险证据能减少错误接受，但当前代价是大量 abstention，且尚未完成 Reject 后状态回灌与在线端到端计时。
12. **AVL-013 修正后仍显示 Reject 的 coverage bottleneck。** v1 的跨序列统计曾把最长 Reject 错报为 `991`，按序列重算的真实最大值为 `324`；deadlock 现象仍成立。v3 使用 selective-geometry 规则和修正 evaluator 后，test coverage=`54.23%`、最终 success=`51.57%`、accepted-pose accuracy=`95.10%`、最长 Reject=`26`；119 个 Reject 段中 117 个恢复、2 个右删失，恢复到首个 correct 的平均时间为 `4.21` 帧。错误 state update 从 no-reject 的 `89` 降至 `27`，但全帧 success 显著下降。Reject 能阻断错误写回，但当前恢复证据仍过于保守。
13. **AVL-014 的 selective-geometry 重校准未改变主要瓶颈。** AVL-012 改为只有实际调用 geometry 的帧使用 geometry evidence 后，validation 仍冻结 threshold=`1.5`；test coverage=`55.41%`、selective accuracy=`95.56%`。该风险分数仍不能同时保持高 coverage 和高可靠性。
14. **AVL-011 jump-guard 配对消融表明原始 91.24% 依赖宽松几何门。** 在相同 cache 和动作规则下，将 geometry jump guard 从 `100m/180°` 收紧至 `2.5m/60°`，test success=`87.50%`、coverage=`98.82%`、最长失败=`14`，低于无效 guard 的 `91.24%/100%/4`。因此原始结果应称 cached deterministic route，而不是严格运动验证后的最终可靠性。
15. **AVL-015 Q1 证明当前 pose action pool 存在互补与理论成本空间，但尚未证明可部署收益。** 加入因果 Historical-keyframe Relocalization 后，Consensus/Memory/HLoc-style/RoMa/Historical 的全量 success 为 `70.64/76.71/88.62/91.21/41.16%`，Pose Oracle union=`95.99%`；冻结 temporal test 分别为 `72.15/76.87/88.98/92.22/44.78%`，union=`96.95%`。Memory/RoMa 仍有 `56/302` 个 only-action 帧；Historical 相对全部其他 map/geometry action 只有 `4` 个 all-frame、`2` 个 test 的独占成功帧，未形成稳定新能力。固定动作的非支配 frontier 为 Memory=`76.71%/0/3397.3s`、HLoc=`88.62%/1696/2492.5s`、RoMa=`91.21%/1696/6622.6s`（success/geometry calls/latency）。统一 shared-retrieval 成本边界下，Historical 固定成本约 `3714.7s`；GT 后验 cost-aware oracle 在达到 RoMa action pool 成功率时约 `2461.5s`，达到五动作 union 时约 `2599.0s`。这些下界使用每帧 outcome，不能作为在线策略或 Agent 收益。
16. **AVL-015 Q2 支持 source-specific verification，但当前 coverage 仍是瓶颈。** 在旧 causal route 选择出的来源子集上，Consensus/Memory/Geometry 分别达到 `94.56%/93.10%/99.14%` selective accuracy，对应 coverage=`84.97%/50.58%/34.58%`。该结果没有包含 dynamic re-trust，也不是全帧 coverage；它只支持“不同来源不能共享一个未经校准的 reliability score”。
17. **AVL-015 Q2 dynamic re-trust 的首轮回放属于 implementation-limited 结果，已完成风险分解复核但仍需在线重验。** 旧回放报告 test coverage=`45.87%`、accepted accuracy=`96.57%`、false acceptance=`3.43%`、full-frame success=`44.29%`；复核发现其 `geometry_risk` 混入了 previous-state disagreement。修复后 componentized replay 中 fresh/stale score 在 `303/1108` 条 geometry-source evidence 上发生变化，test coverage=`50.00%`、accepted accuracy=`96.85%`、full-frame success=`48.43%`，且审计仍满足 `query_gt_read=false/uses_future=false/selection_uses_gt=false`。这只证明实现路径已生效，不能替代真实 route regeneration，也不能直接宣称 re-trust 带来 Agent 收益。
18. **AVL-015 Q3 表明当前 Historical Memory 实现没有稳定的独立恢复能力。** 原始因果候选在 test 上 coverage=`100%`、success=`44.78%`；加入 Memory 专属 verifier 后，冻结 test coverage=`21.85%`、selective accuracy=`63.96%`、full-frame success=`13.98%`，且 Historical-only 相对 Consensus/Memory/HLoc/RoMa 在 test 为 `0` 帧。所有 1696 帧 trace 均满足 `query_gt_read=false/uses_future=false/selection_uses_gt=false/historical_future_access=false`。因此当前实现更像高成本、低可靠的历史候选生成器，不应把它写成 Memory 新能力；后续若继续，只能改进历史关键帧选择/相对匹配和 verifier，而不是直接进入 Agent。

证据入口：`outputs/lilocbench/AVL-002/`、`AVL-003/summary_all.json`、`AVL-004/`、`AVL-005/unified_cost/summary.json`、`AVL-006/`、`AVL-007/summary.json`。

## 4. 尚未证明的结论

- LLM/VLM Agent 是否比确定性 Manager 有独立收益；当前没有 Go 证据。
- 在真正独立的未见场景或跨数据集上，reference/memory/geometry 调度能否泛化。LILocBench 四条件曾参与探索，内部 split 只能称 calibration-shift test。
- 标准 COLMAP-HLoc 3D localization 的收益；当前结果是 HLoc-style 2D/3D feasibility。
- 完整逐帧 Reloc3r 在线计时；统一成本中的 Reloc3r 项来自每序列 8 帧无 GT 样本缩放，必须标为估计。
- 动态、家具、门或照明变化的逐帧因果语义。当前 contact sheet 和 operational labels 只是人工审计证据，不是真值标签。
- Memory 收益是否超出同轨迹时间近邻的受限先验。
- 真正 temporal tracker（而非 Reloc3r 单对相对位姿）能否提供独立的动态遮挡恢复价值。

## 5. 被否定或失效的解释

- “固定 Top-10 在 static 上足够稳定”被 static 的 Top-k 分层和候选冲突结果否定。
- “调用 RoMa/HLoc 就能解决 Reloc3r 失败”被共同失败帧和阈值外错误 pose 否定。
- “geometry call rate 直接等价于 latency 节省”被统一 stage accounting 否定。
- “冻结 DINOv2 表征已经足以区分 memory-only 与 RoMa-only”被 AVL-007 LOSO 结果否定。
- “当前 Qwen zero-shot VLM 能把临时遮挡/长期变化稳定映射到 Memory/RoMa”被 AVL-008 的弱关联和配对 action 结果否定。
- “把 Reloc3r 相邻帧相对位姿直接累乘就能形成可靠连续定位”被 AVL-009 的 temporal drift 和 Dynamic+LT 长失败段否定。
- “Dynamic+LT 主要因为地图没有对应 reference”目前没有得到支持：空间近邻 proxy 在 Top-100 中覆盖绝大多数帧；更可能是外观/结构不兼容、Top-k 错选和一致性错误簇。
- “高 inlier/低 reprojection 的 RoMa 输出就可靠”被 38 个 both-fail 帧否定；几何输出必须结合 reference/历史/运动一致性并允许 Reject。
- “一次风险分数即可同时保持高 coverage 和高准确率”尚未成立：AVL-012 的选择性准确率提升伴随显著 coverage 损失，且分数尚未在 online reject→state update 闭环中验证。
- “Reject 本身就能改善最终定位”被 AVL-013 限定否定：Reject 减少错误 state 写回，但若 recovery evidence/状态老化处理不足，会显著降低 coverage 和全帧 success。
- 任何使用 query GT、未来帧、事后 GT reference 选择得到的数字，均不能作为在线策略性能结论。

## 6. 风险与替代解释

- 坐标系、外参、时间戳或 GT 对齐错误会使绝对位姿结论失效；当前分析保留 100 ms 主规则和严格 30 ms 敏感性子集。
- 四条 LILocBench query 轨迹/时段不同，dynamic+LT 总体不比 dynamic 更差，说明条件标签与难度不单调，必须避免只按总体均值作因果断言。
- reference 库规模、Top-100 重排、额外计算、预训练数据重叠和工具实现差异都可能伪造路由收益。
- 近邻 memory、同序列规则发现和 test-set adaptation 会高估泛化；所有未来实验必须冻结协议并单独记录。
- RoMa 运行的随机性、非零畸变下未先 undistort、HLoc-style 地图定义等实现限制，要求按 seed 和工具边界报告，不扩大结论。

## 7. 原型启动与论文 Go 条件

### 受控 Agent 原型可以立即开始

当前允许实现：

1. 策略可交互的 `reset/step` causal environment；
2. 能改变 reference/verification evidence 的参数化 action；
3. fixed/random/geometry-greedy/有限深度搜索基线；
4. 训练场景上的 tool-expanded trajectory 和最小 Agentic SFT；
5. 与相同 observation/action 的轻量 sequential policy 配对比较。

这一步是验证研究假设，不代表已经获得 Agent 论文结论。

### 主张 Agent/VLM 收益必须同时满足

1. 有独立 held-out、scene-disjoint 或跨数据集证据，且测试集未参与阈值、reference、轨迹或动作发现；
2. 有无 GT 在线 trace，证明决策只使用当前及过去信息，并且策略动作真正改变后续 observation、Memory 和 state；
3. 在冻结 test、统一端到端成本、reference/matching 预算和成功阈值下，主动取证相对 fixed-order 与最佳固定策略形成有意义的 Pareto 改进；
4. 任务特定 learned policy 在相同工具、verifier、输入和预算下稳定超过 one-shot router、geometry-greedy 与轻量 sequential policy；
5. source/pose-specific verification、Reject/Recovery 和 Memory invalidation 能降低错误接受与状态污染，而不是只通过大量 abstention 提高 selective accuracy；
6. Agent 只做证据获取、动作选择、信任更新和自适应停止，不直接预测 pose，不访问 query GT 或未来帧。

若主动取证不能超过固定强工具，则冻结 Agent 主张；若主动取证有效但 VLM 不超过轻量 policy，可保留 learned active localization system 的贡献，但不主张语言推理收益。

## 8. 当前主线决策

2026-10-14 Stage 3G 收口后，当前模块优化停止。F2 与 rule 的220帧最终结果只有5帧不同，
terminal alignment 又消除了 legacy rule 的表面成功率优势；继续调 Geometry head、terminal
threshold 或更大 policy 无法形成可信的 Agent 贡献。下一阶段先验证主动取证的可实现 Oracle
gap：只有在 failure-triggered、scene-disjoint episodes 上证明不同 evidence action 会稳定改变
terminal outcome，才恢复 learned Agent、SFT 或 VLM 路线。

主线不变，策略粒度调整为“连续状态上的多轮主动取证”：

```text
连续 RGB 序列
    ↓
Temporal Tracking / Motion
    ↓
Evidence-aware Localization State
    ├─ pose hypotheses / belief
    ├─ evidence provenance / dependency
    ├─ reliable anchors / Experience Memory
    ├─ reject streak / motion uncertainty
    └─ remaining tool budget
    ↓
Agent: decide which evidence is missing
    ↓
Retrieve / Estimate / Verify / Query Experience
    ↓
New Observation → Update State
    ├─ evidence insufficient → continue
    ├─ verified → Accept + Memory update
    └─ unresolved → Reject / Defer
```

当前已完成 AVL-009、AVL-011、AVL-012 与 AVL-013，并修正重验了 AVL-010：Reloc3r temporal edge 会漂移，修正后的 DPVO 状态回放没有显示独立定位收益；Dynamic+LT 的共同失败主要是 reference 外观/结构不兼容、候选一致性错误和强几何误接受，而非空间地图完全缺失。AVL-013 进一步证明 Reject 需要状态老化和恢复机制，否则会死锁；修正 evaluator 后 v3 仍能减少错误 state 写回，但 coverage 只有约54%。

因此底层环境收敛为 **Map Localization + Experience Memory + Geometry + Reliability**：

```text
Current RGB + Map References + Reliable Historical Keyframes
    → Pose Candidates
        ├─ Reloc3r consensus
        ├─ Historical-keyframe relocalization
        ├─ Rerank + Reloc3r
        └─ Rerank + RoMa / HLoc-style
    → Source-specific Verification
    → Accept / Reject / Recovery
    → Update State / Memory / Reference Reliability
```

DPVO 只保留为 motion/uncertainty evidence，不再作为主要 pose 分支；MASt3R-SLAM 暂不投入。AVL-015 已完成：原始工具有理论互补空间，但当前 Historical Memory 没有稳定独立恢复能力。2026-10-12 复核发现 dynamic re-trust 的旧状态衰减未实际改变 geometry score，因此该负结论需要重验。

当前进入受控原型阶段：先修复不可约风险项和 state trust，建立能因策略动作改变后续 Memory/observation 的环境，再验证“选择下一组 reference 或 held-out verification evidence”是否在相同预算下优于固定顺序。只有主动取证成立后，才进入多场景 Agentic SFT。Memory、Manager 和 Agent 均只能使用当前及历史信息，所有强几何结果必须经过独立验证或拒绝。
