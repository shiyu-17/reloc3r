# Codex 项目状态

最后更新：2026-10-14

## 当前主线与最高优先级任务

Stage 3G 已于2026-10-14收口：220帧 F2-vs-rule 成对审计只有5个最终分歧；terminal-aligned
复验显示 legacy rule 的正确帧优势从相对 F2 的 `+3` 变为 `-1`，旧比较混入 verifier 后
自动 Accept 的接口差异。现冻结 F2、rule、verifier 与工具，不再继续同 split 的 controller /
Geometry 调参。最高优先级切换为 failure-triggered、scene-disjoint 的主动取证可实现价值验证，
见 `docs/active_evidence_realizable_value_plan.md`。

主线不变：**Failure-aware Agentic Visual Localization in Dynamic Indoor Environments**。

当前阶段调整为 **Evidence-seeking Agent 的受控原型与环境建设**。允许开始 `reset/step` causal environment、主动 reference/verification evidence acquisition、trajectory generation 和最小 SFT prototype；论文层面的 Agent/VLM 独立收益仍未证明。AVL-008 只测试了 zero-shot one-shot routing，不能作为任务训练后的多轮 Agent 结论。

AVL-015 Reliability–Recovery Frontier 已完成并审核（`completed with limits`）。工具有理论互补空间，Historical Memory 当前没有稳定独立恢复能力。2026-10-12 复核发现旧 dynamic re-trust 的 previous-state 衰减被未拆分的 `geometry_risk` 覆盖；修复后 componentized replay 的 fresh/stale score 在 `303/1108` 条 geometry-source evidence 上发生变化。该结果证明实现路径已修复，但在线 route regeneration 和收益仍需重验。

当前底层工具与拟建交互闭环：

```text
Current RGB + Map References + Reliable Historical Keyframes
    → Evidence-aware Localization State
    → Agent selects missing evidence
    → Retrieve / Estimate / Verify / Query Experience
    → New Observation
    → Continue / Accept / Reject / Defer
    → Update State / Memory / Reference Reliability
```

DPVO 只作为 motion/uncertainty evidence，不再作为主要 pose 分支；MASt3R-SLAM 暂不继续投入。Agent 负责判断缺少哪类证据、调用工具、更新证据信任和自适应停止，不直接预测 pose。完整设计与 IROS 路线见 `docs/visual_localization_agent_roadmap.md`。

## 最新可信结论及证据路径

- **Reloc3r 体检（AVL-002）**：固定 mapping-only Top-10 的 `(0.25 m, 5°)` success（static / dynamic / LT / dynamic+LT）为 `39.79 / 13.55 / 24.88 / 23.48%`。失败段最大长度为 `26 / 30 / 37 / 37` 帧，candidate translation dispersion 中位数为 `1.244 / 5.189 / 2.586 / 3.386 m`。dynamic 与 LT 支持条件相关失效；combo 总体非单调，不能宣称必然最难。证据：`outputs/lilocbench/AVL-002/*/summary.json`。
- **确定性候选/历史（AVL-003）**：consensus-3 success=`82.70 / 55.48 / 75.12 / 58.60%`，memory-3=`91.18 / 58.06 / 81.52 / 62.85%`。Memory 只访问过去已产生的内部状态。证据：`outputs/lilocbench/AVL-003/summary_all.json`。
- **强几何恢复（AVL-004）**：在无 GT reference 重排后，RoMa route=`1513/1696=89.21%`，HLoc-style=`1484/1696=87.50%`；仍有共同失败和阈值外误接受。结果不能写成 LLM Agent 收益，也不能称标准 COLMAP-HLoc。证据：`outputs/lilocbench/AVL-004/full_route_a5/`、`always_route_eval/`。
- **RoMa seed 稳定性（AVL-005-A）**：相同 952 帧 geometry 池的三个 seed success=`86.76 / 86.97 / 87.18%`，均值=`86.97%`、波动约 `0.21 pp`；LT 共同失败集合仍有 seed 敏感性。证据：`outputs/lilocbench/AVL-005/multiseed_summary.json`。
- **统一成本（AVL-005-B）**：seed=0 Always-RoMa=`1552/1696=91.51%`、约 `6622.564 s`；recovery-aware Manager=`1521/1696=89.68%`、coverage=`1672/1696`、geometry call=`46.17%`，统一成本估计=`6665.885 s`，高约 `0.65%`。Reloc3r 成本是每序列 8 帧无 GT 样本缩放估计。证据：`outputs/lilocbench/AVL-005/unified_cost/summary.json`。
- **闭环回放（AVL-005-C）**：验证通过的 geometry pose 已回灌后续 prediction/memory；全 1696 帧 coverage=`1672/1696`，回放 success=`89.50%`。这是使用缓存工具结果的 deterministic closed-loop replay，不是重新在线测量端到端 geometry latency。证据：`outputs/lilocbench/AVL-005/closed_loop_manager/route_eval.json`。
- **GT-free 路由（AVL-006）**：LOSO logistic/tree 的 RoMa call/success 为 `42.92%/87.44%`、`45.70%/86.56%`，均未超过 Always-RoMa。事后 union=`1606/1696=94.69%` 是 oracle 上界，不是在线策略。证据：`outputs/lilocbench/AVL-006/early_router_cv/summary.json`、`break_even.json`。
- **冻结视觉表征（AVL-007）**：DINO-similarity tree 的 memory-only/RoMa-only recall=`24.07%/74.10%`、balanced accuracy=`49.09%`；全帧最好 success=`1485/1696=87.62%`，仍低于 Always-RoMa。证据：`outputs/lilocbench/AVL-007/summary.json`、`audit/semantic_audit.json`。
- **零样本语义探针（AVL-008）**：本地 Qwen2.5-VL 对四个 cohort 各 54 帧、共 216 个多图请求运行；200 个响应严格解析成功，16 个 schema invalid。有效样本 cohort—label Cramér’s V=`0.2187`，低于预注册的明显关系门槛；`memory-only/RoMa-only` 上 VLM action BA=`47.02%`，同帧 DINO=`47.64%`，差值=`−0.62 pp`，未提供额外路由信息。证据：`outputs/lilocbench/AVL-008/summary.json`。
- **连续序列 baseline（AVL-009）**：四序列相邻 temporal edge=`1692/1692` 有效；test（1016 帧）temporal-only=`0.00%`、global+temporal=`53.64%`、global+temporal+memory=`53.64%`、recovery=`53.74%`，低于 memory-3=`76.87%` 与 Always-RoMa=`92.62%`。Dynamic+LT 的 global+temporal 最大失败段为 `270` 帧，说明当前 Reloc3r edge 会平滑漂移。已知成本组件合计约 `3845.38 s`，reference reranking 未在本批次重新计时。证据：`outputs/lilocbench/AVL-009/summary.json`、`cost/summary.json`、`validation/temporal_direction_audit.json`、`docs/avl009_sequence_localization_report.md`。
- **强时序 tracker（AVL-010，修正后）**：原 Sim(3) 回放错误地把尺度乘到旋转矩阵，旧派生指标作废。使用同一 DPVO full-rate trajectory 修正后，test aggregate（1016 帧）DPVO temporal-only=`0.20%`、Global+DPVO=`4.04%`、Global+DPVO+Memory=`4.04%`、Global+DPVO+Memory+RoMa=`65.26%`、Always-RoMa=`92.62%`；所有输出旋转正交性通过审计。当前 DPVO/anchor 实现没有显示独立定位收益。证据：`outputs/lilocbench/AVL-014/AVL-010_stateful_sim3fix/`、`AVL-010_eval_sim3fix/`。
- **Dynamic+LT 共同失败诊断（AVL-011）**：541 帧中 `both-success / Memory-only / RoMa-only / both-fail = 326 / 14 / 163 / 38`。Top-100 空间近邻 proxy 覆盖 `516/541`，另有 `25` 帧为 Top-10 错选但 Top-100 有近邻；没有空间 reference 全部缺失的 proxy。38 个 both-fail 中 32 个有 reference mismatch、37 个 candidate conflict，且 RoMa 均有输出但后验错误；16 个 Memory 内部一致但后验错误。证据：`outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/summary.json`、`both_fail.jpg`。
- **强化 deterministic recovery（AVL-011）**：validation 冻结规则后，test（1016 帧）success=`91.24%`、coverage=`100%`、geometry call=`67.13%`、最长失败=`4`、平均恢复=`1.47` 帧；static/dynamic/LT/Dynamic+LT 分别为 `91.62/92.47/90.12/91.36%`。trace 全部 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。证据：`outputs/lilocbench/AVL-011/final_v1/evaluation.json`、`docs/avl011_failure_and_recovery_report.md`。
- **可靠性与 Reject（AVL-012，selective-geometry 重校准）**：geometry evidence 只在实际 `call_geometry` 帧使用；validation 仍冻结 threshold=`1.5`，test coverage=`55.41%`、selective accuracy=`95.56%`、错误接受率=`4.44%`。这仍是缓存回放校准，不是完整 online latency。证据：`outputs/lilocbench/AVL-014/AVL-012_selective/evaluation.json`。
- **闭环 Reject 与恢复（AVL-013 v3）**：修正状态 trace 和按序列 recovery evaluator 后，test coverage=`54.23%`、最终 success=`51.57%`、accepted accuracy=`95.10%`、最长 Reject=`26`；119 个 Reject 段中 117 个恢复、2 个右删失，平均恢复到首个 correct=`4.21` 帧。错误 state update 从 no-reject 的 `89` 降到 `27`，coverage/success 仍显著下降。所有 trace 审计均为 GT/future-free。证据：`outputs/lilocbench/AVL-014/AVL-013_v3/evaluation.json`。
- **AVL-011 jump-guard paired ablation**：相同 frozen inputs 下使用 `2.5m/60°` jump guard，test coverage=`98.82%`、success=`87.50%`、最长失败=`14`，相对旧宽松 guard 的 `100%/91.24%/4` 更差；原 91.24% 应表述为 cached deterministic route，不是严格运动验证后的可靠性提升。证据：`outputs/lilocbench/AVL-014/AVL-011_jumpguard/evaluation.json`。
- **架构决策**：后续不再把每帧独立看作一次 routing 问题，而是维护当前位姿、历史可靠帧/Memory、连续失败状态和 reference reliability。只有验证通过的恢复结果才能写回状态。该决定最初只用于确定性闭环；2026-10-12 起进一步允许在同一 failure-aware 主线上建设受控的主动取证 Agent 原型。决策记录：`docs/experiment_registry.md` 的“主线架构更新”与“Agent 主线与实现决策”。
- **AVL-015 Q1 最新证据**：Consensus/Memory/HLoc/RoMa/Historical 全量 success=`70.64/76.71/88.62/91.21/41.16%`，Pose Oracle=`95.99%`；temporal test 分别为 `72.15/76.87/88.98/92.22/44.78%`，union=`96.95%`。Historical 相对全部其他动作只贡献 `4` 个全量、`2` 个 test 独占成功帧。固定动作非支配 frontier 为 Memory=`76.71%/0/3397.3s`、HLoc=`88.62%/1696/2492.5s`、RoMa=`91.21%/1696/6622.6s`（success/geometry calls/latency）；五动作 union 的 GT 后验下界约 `2599.0s`，不是可部署策略收益。证据：`outputs/lilocbench/AVL-015/pose_oracle/summary.json`。
- **AVL-015 Q2/Q3 最新证据**：source-specific verifier 的旧 causal route 子集结果为 Consensus=`84.97%/94.56%`、Memory=`50.58%/93.10%`、Geometry=`34.58%/99.14%`（coverage/selective accuracy）。全帧 dynamic re-trust replay 曾报告 test coverage=`45.87%`、accepted accuracy=`96.57%`、false acceptance=`3.43%`、full-frame success=`44.29%`，但该结果受下述实现问题限制，不能用于判断 re-trust 是否带来全帧收益。Historical raw test success=`44.78%`；其专属 verifier test coverage=`21.85%`、selective accuracy=`63.96%`、full-frame success=`13.98%`。证据：`outputs/lilocbench/AVL-015/{source_verification,dynamic_retrust,historical_keyframe_memory}/`。
- **AVL-015 dynamic re-trust 复核**：旧上游 `geometry_risk` 已包含 previous-state disagreement，下游又保留该未衰减复合项并重复衰减 previous 项。拆分后 componentized replay 中 `303/1108` 条 geometry-source 记录的 fresh/stale score 发生变化；test coverage=`50.00%`、accepted accuracy=`96.85%`、full-frame success=`48.43%`。该结果仍是冻结候选 replay，不是在线 route regeneration。

## 已暂停、否定或尚未证明的路线

- **受控启动**：允许构建真正的交互环境、主动取证基线和最小 SFT Agent；在相同工具/verifier/budget 下超过规则与轻量 policy 之前，不主张 Agent/VLM 收益。跨数据集/独立新场景 held-out 与 RIO10 定位运行仍待恢复。
- **暂停**：单帧 VLM 直接决定 Memory/RoMa 的路线；AVL-008 未证明这种方式有额外信息。
- **否定**：固定 Top-10 在 static 上足够稳定；只调用 RoMa/HLoc 能解决全部失败；geometry call rate 可直接推导端到端加速；冻结 DINOv2 足以完成互补动作分类。
- **AVL-009 限定否定**：当前 Reloc3r 相邻帧相对位姿累乘不能作为可靠长期 tracking；这不外推为 DPVO 或其他真正 VO/SLAM 方法。
- **AVL-010 限定否定**：当前 DPVO 加 causal global anchor 在总体、Dynamic 和 Dynamic+LT 上没有超过 Memory-3；这不外推为所有 VO/SLAM，也不否定将 tracker 作为辅助状态信号。
- **AVL-011 限定否定**：Dynamic+LT 的共同失败不能简单归因于地图空间覆盖不足；当前更强证据指向 reference 外观/结构不兼容、候选一致性错误和 RoMa/Memory 的错误接受。诊断类别仍是 operational proxy，不是真值语义标签。
- **AVL-008 限定否定**：当前 Qwen zero-shot、多图像输入和冻结语义→action 映射未能稳定识别 RoMa-only 帧；这不外推为所有 VLM、微调方案或不同 temporal/reference 表示均无效。
- **尚未证明**：Agent 相对确定性 Manager 的独立收益；改进后的 Historical Memory 是否能超出时间近邻先验；标准 COLMAP-HLoc 的性能；完整逐帧 Reloc3r latency；动态/家具/门变化的语义因果标签。
- **AVL-015 Q3 限定否定**：当前 Historical-keyframe 实现不能作为可靠恢复分支；raw candidate 成功率低且专属 verifier 仍有高错误接受率。该结论不外推到更好的局部地图、关键帧选择或跨场景 Memory 设计。
- **保留限制**：A.5/A.7 的部分工具互补和共同失败计数受 RoMa seed 影响；A.8 是 proxy 审计，不是真值分类；旧计时边界不一致的速度比较无效。
- **AVL-010 成本限制**：DPVO 在线耗时为实测（总计 `698.877 s`），global Reloc3r、retrieval 与 RoMa recovery 沿用既有混合测量/估计；成本账本明确禁止据此宣称端到端加速。

## 代码、环境、数据和重要 artifact

- 数据与准备结果：`data/lilocbench/`；mapping=`785`，query static/dynamic/LT/combo=`578/155/422/541`。
- Reloc3r 权重：`.cache/huggingface/hub/models--siyan824--reloc3r-512/snapshots/55014aae1798ec830df51d7d2cd47c41f4388187/model.safetensors`。
- 核心评测：`tools/eval_lilocbench_reloc3r.py`、`tools/run_avl003_ablations.py`、`tools/run_avl004_*`。
- 成本/路由：`tools/aggregate_avl005_cost.py`、`tools/eval_avl006_early_router_cv.py`、`tools/analyse_avl006_break_even.py`。
- 表征路由：`tools/extract_avl007_dinov2.py`、`tools/run_avl007_representation_router.py`、`tools/audit_avl007_special_frames.py`。
- AVL-009 实现脚本：`tools/run_avl009_temporal_edges.py`、`tools/run_avl009_sequence_baseline.py`、`tools/eval_avl009_sequence.py`、`tools/audit_avl009_temporal_direction.py`；结果与 GT-free trace 位于 `outputs/lilocbench/AVL-009/`。
- AVL-010 实现脚本：`tools/prepare_avl010_full_rate.py`、`tools/run_avl010_dpvo.py`、`tools/run_avl010_stateful.py`、`tools/eval_avl010_dpvo.py`；DPVO 子仓库为 `third_party/DPVO`（commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`），权重为 `third_party/DPVO/dpvo.pth`；修正回放与评测位于 `outputs/lilocbench/AVL-014/AVL-010_stateful_sim3fix/`、`AVL-010_eval_sim3fix/`。
- AVL-010 报告：`docs/avl010_strong_temporal_tracker_report.md`；最终 causal 审计：`outputs/lilocbench/AVL-010/eval_full_rate_v3/audit/causal_audit.json`。
- AVL-011 诊断与 baseline：`tools/analyse_avl011_dynamic_lt_failures.py`、`tools/run_avl011_strong_recovery.py`、`tools/eval_avl011_recovery.py`；诊断产物为 `outputs/lilocbench/AVL-011/failure_diagnosis/`，最终 trace/evaluator 为 `outputs/lilocbench/AVL-011/final_v1/`；正式报告为 `docs/avl011_failure_and_recovery_report.md`。
- AVL-012 可靠性与 Reject：`tools/run_avl012_reliability.py`、`tools/eval_avl012_reliability.py`、`tools/make_avl012_contact_sheets.py`；修正 evidence、score manifest 和 evaluator 位于 `outputs/lilocbench/AVL-014/AVL-012_selective/`。
- AVL-013 闭环 Reject/Recovery：`tools/run_avl013_closed_loop.py`、`tools/eval_avl013_closed_loop.py`；修正 v3 trace/evaluator 位于 `outputs/lilocbench/AVL-014/AVL-013_v3/`；旧 v1/v2 仍保留且不覆盖。jump-guard 配对消融位于 `outputs/lilocbench/AVL-014/AVL-011_jumpguard/`。
- 语义探针：`tools/prepare_avl008_probe.py`、`tools/run_avl008_vlm.py`、`tools/eval_avl008_probe.py`；Qwen 本地模型目录为 `/data2/liusy/agent/vlm`，环境为 `/data2/liusy/conda_envs/agent`。
- AVL-008 产物：`outputs/lilocbench/AVL-008/{sample_manifest.json,requests.jsonl,responses.jsonl,summary.json,evaluator_rows.jsonl,vlm_run_manifest.json}`。
- AVL-015 产物：`tools/run_avl015_pose_oracle.py`、`tools/run_avl015_historical_memory.py`、`tools/eval_avl015_historical_memory.py`、`tools/eval_avl015_historical_verifier.py`、`tools/eval_avl015_source_verifiers.py`、`tools/eval_avl015_dynamic_retrust.py`；结果位于 `outputs/lilocbench/AVL-015/{pose_oracle,source_verification,dynamic_retrust,historical_keyframe_memory}/`。
- Agent 方法与 IROS 路线：`docs/visual_localization_agent_roadmap.md`。
- 审计文档：`docs/key_experiment_reaudit.md`、`docs/experiment_registry.md`、`docs/avl_experiment_report.md`。

## 当前运行任务与日志

当前运行任务：无。AVL-015 Q3 四类序列 causal replay、独立 evaluator、Historical verifier、Pose Oracle 重算均已完成并落盘。AVL-014 的四项修正回放、评测和审计均已落盘。AVL-005-D 新场景 held-out 仍按用户要求暂停。

## Git 状态

- 分支：`main`
- HEAD=`2cb9908`，分支 `main`；本轮 AVL-015 文档、脚本和结果尚未提交，工作树包含本轮修改和既有未跟踪工具/产物，未清理或覆盖历史 artifact。`croco` 子模块显示既有工作树状态。
- AVL-014 修正仍在 HEAD=`2cb9908` 中；AVL-015 本轮修改尚未提交。其他历史未跟踪工具和实验产物未清理或覆盖。

## 已知问题与下一步

1. Stage 3 已关闭；不再对 F0--F3、F2 terminal controller、Geometry advantage 或
   recovery/cost ranker 做同 split 迭代。它们仅作冻结 baseline 和历史复现。
2. 下一阶段先做 failure-triggered recovery episode miner 和 scene-disjoint manifest，然后展开
   budget-matched 的深度 2--3 action tree，比较 fixed order、feedback-rule、geometry-greedy
   和 finite-depth Oracle。
3. 只有 Oracle 相对最佳规则达到 `+5 pp` episode success，或等成功率下降低
   `20%` 成本，且至少 `10%--15%` 状态的终局依赖动作，才进入 learned Agent。
4. 完整 Reloc3r/RoMa online inference timing 需按统一边界重测；当前 ledger 的
   `cost_complete_online_inference=false` 保持有效。
5. LILocBench 内部 split 只能称 calibration-shift test；IROS 主张需要
   scene-disjoint/cross-dataset 和实机 lost-and-recovery 证据。

2026-10-12 主动取证 pilot 已升级为严格协议：`tools/eval_active_evidence_policies.py` 在四序列 304 个测试帧、每帧最多两组 reference 的 CPU-ORB causal 环境中比较 fixed/random/rule/light/oracle。历史关键帧 action 使用局部 3D + 当前帧 PnP，verification 使用未参与候选生成的 held-out references 做独立极线一致性，并将 reference/verification/memory 三类调用统一计入 budget。fixed/light 的 full-frame success 均为 `81/304=26.6%`，rule=`77/304=25.3%`，random=`69/304=22.7%`；fixed/light accepted accuracy 均为 `93.1%`；平均总工具调用为 fixed/random/rule/light=`3.168/3.336/3.211/3.158`。在线 policy 的 `query_gt_read/uses_future/selection_uses_gt` 全为 false，环境审计通过；light 的 GT 只用于 calibration-prefix 离线标签。该结果仍未证明主动取证价值，主要说明当前 CPU action pool 与真实强几何尚不足以支撑 Agent 学习。进入 VLM/SFT 前必须完成真实 GPU Reloc3r/RoMa、统一在线成本和 scene-disjoint 复验；报告见 `docs/active_evidence_pilot_report.md`。

连续帧 Memory 在该 pilot 中是包含的：每个 accepted pose、历史关键帧和 reject state 会进入同一序列的下一帧，所有 policy 共享 causal historical-memory action；但测试段为避免 split 泄漏从空 Memory 冷启动，所以尚未报告 warm-start 记忆收益，也没有 learned memory agent。下一轮应同时评估 cold-start 与合法历史状态 warm-start。

2026-10-12 Phase A 已有首版代码落点：`tools/agentic_localization_env.py` 提供 causal `reset/step` 状态机和 `audit_trace()`，`tools/lilocbench_agent_backend.py` 将 reference action 接到当前帧 Reloc3r 实时 pair inference，`tools/cpu_orb_agent_backend.py` 提供真实图像 CPU backend（历史关键帧局部 3D/PnP 与 held-out 极线 verifier），`tools/run_agentic_localization_env.py` 提供主动 reference batch baseline，`tools/run_agentic_reference_switch_smoke.py` 提供同帧 reference-switch 验收。环境强制 backend 返回 `executed=true/cache_hit=false`、当前 query frame、精确 reference IDs 和执行 stages 以及三项 false causal flags，拒绝 evaluator/outcome 字段，并把 anchor age/reject streak 传给 verifier；新 reference 会使旧 verification 失效，Memory 只在 verifier accepted 后写入；CUDA 不可用时 Reloc3r backend 显式失败，不会退化为冻结候选。CPU ORB 已完成真实图像 memory/held-out smoke，environment audit 通过；严格 pilot 的 304 帧 policy trace 也全部通过 causal audit，当前尚未完成 GPU Reloc3r 全序列、在线成本测量和真实强几何 route regeneration。
