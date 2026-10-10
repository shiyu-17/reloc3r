# Codex 项目状态

最后更新：2026-10-10

## 当前主线与最高优先级任务

主线不变：**Failure-aware Agentic Visual Localization in Dynamic Indoor Environments**。

当前阶段是 **Agent 前的时序状态建模与确定性恢复**，状态仍为 **Hold / No-Go for immediate Agent implementation**。AVL-008 已完成，但只测试了零样本 VLM 的语义能力，没有构建 LLM/多 Agent。AVL-009 已完成第一版严格 causal baseline；新场景 held-out 按用户要求暂缓。

最高优先级：在不引入 Agent 的前提下，继续校准 deterministic global-anchor / recovery 的 coverage–error–recovery trade-off。AVL-013 已完成 causal Reject→state update→recovery replay：Reject 能阻断错误写回，但 stale-state recovery 后 test coverage 仍只有 54.23%，且真实神经推理端到端计时尚未完成；当前仍不满足进入 Agent 的 Go 条件。

当前目标流程：

```text
连续 RGB → Temporal Tracking / Motion → Stateful Localization State
    → Reloc3r / Reference reranking / RoMa → Verification → 更新状态
```

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
- **架构决策**：后续不再把每帧独立看作一次 routing 问题，而是维护当前位姿、历史可靠帧/Memory、连续失败状态和 reference reliability。只有验证通过的恢复结果才能写回状态；该调整不改变 failure-aware 主线，也不等于开始构建 Agent。决策记录：`docs/experiment_registry.md` 的“主线架构更新”。

## 已暂停、否定或尚未证明的路线

- **暂停**：真正的 LLM/VLM Agent、跨数据集/独立新场景 held-out、RIO10 定位运行。
- **暂停**：单帧 VLM 直接决定 Memory/RoMa 的路线；AVL-008 未证明这种方式有额外信息。
- **否定**：固定 Top-10 在 static 上足够稳定；只调用 RoMa/HLoc 能解决全部失败；geometry call rate 可直接推导端到端加速；冻结 DINOv2 足以完成互补动作分类。
- **AVL-009 限定否定**：当前 Reloc3r 相邻帧相对位姿累乘不能作为可靠长期 tracking；这不外推为 DPVO 或其他真正 VO/SLAM 方法。
- **AVL-010 限定否定**：当前 DPVO 加 causal global anchor 在总体、Dynamic 和 Dynamic+LT 上没有超过 Memory-3；这不外推为所有 VO/SLAM，也不否定将 tracker 作为辅助状态信号。
- **AVL-011 限定否定**：Dynamic+LT 的共同失败不能简单归因于地图空间覆盖不足；当前更强证据指向 reference 外观/结构不兼容、候选一致性错误和 RoMa/Memory 的错误接受。诊断类别仍是 operational proxy，不是真值语义标签。
- **AVL-008 限定否定**：当前 Qwen zero-shot、多图像输入和冻结语义→action 映射未能稳定识别 RoMa-only 帧；这不外推为所有 VLM、微调方案或不同 temporal/reference 表示均无效。
- **尚未证明**：Agent 相对确定性 Manager 的独立收益；Memory 超出时间近邻先验的因果收益；标准 COLMAP-HLoc 的性能；完整逐帧 Reloc3r latency；动态/家具/门变化的语义因果标签。
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
- 审计文档：`docs/key_experiment_reaudit.md`、`docs/experiment_registry.md`、`docs/avl_experiment_report.md`。

## 当前运行任务与日志

当前无已登记的 AVL、Reloc3r、HLoc、RoMa、DPVO 或 VLM 运行任务；AVL-014 的四项修正回放、评测和审计均已落盘。AVL-005-D 新场景 held-out 仍按用户要求暂停。

## Git 状态

- 分支：`main`
- HEAD：`main`，本次 `Revalidate AVL temporal and recovery experiments` 提交已完成；精确哈希以 `git rev-parse --short HEAD` 为准。`croco` 子模块显示既有工作树状态。
- 本次 AVL-014 修正涉及的回放脚本、实验登记和项目状态已提交；其他历史未跟踪工具和实验产物未清理或覆盖。

## 已知问题与下一步

1. 完整 Reloc3r/RoMa online inference timing 暂缓：当前 deterministic route 刚完成 Sim(3)、selective evidence 和 jump-guard 修正，尚未形成稳定最终策略；待 recovery policy 冻结后再按统一串行边界测量。当前 cost ledger 明确 `cost_complete_online_inference=false`。
2. 继续优化 coverage–error–recovery 联合校准，重点解决 Dynamic/ Dynamic+LT 的低 coverage；不要把 AVL-013 v3 的 51.57% 全帧 success 解释为最终系统上限。
3. 暂不引入 Agent；Reject 已减少错误状态污染，但 deterministic baseline 尚未在统一成本和覆盖率下稳定优于 no-reject/Always-RoMa。
4. 若恢复泛化实验，再获取未参与规则发现的独立 held-out 数据；LILocBench 内部 split 只能报告为 calibration-shift test。
5. 若以后重启 VLM 探针，先修正 AVL-008 的 schema 解析率并登记新协议，不能事后修补当前主结果。
