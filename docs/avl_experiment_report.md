# Agentic Visual Localization 当前阶段完整实验报告

**日期**：2026-10-09（Asia/Shanghai）；2026-10-12 增补 Agent 路线说明
**项目主线**：Dynamic Indoor Environment 中的 failure-aware / Agentic Visual Localization。该报告只覆盖 AVL-001 至 AVL-004 A.5 的历史阶段证据；当时尚未实现 LLM/Agent，后续决策以文首索引的最新文档为准。

> 本文保留为 AVL-001–004 的阶段性实验报告。AVL-005–015、2026-10-12 的 dynamic re-trust 实现复核、相关工作、Evidence-seeking Agent 系统设计、训练路径与 IROS 评测方案，统一见 `docs/visual_localization_agent_roadmap.md`、`docs/experiment_registry.md` 和 `docs/codex_project_state.md`。当前允许受控 Agent 原型与环境建设，但尚未证明 Agent/VLM 的论文收益。

## 1. 研究目标与当前问题

目标不是单纯提高 Reloc3r 的平均精度，而是建立一个能完成以下闭环的定位系统：

`当前观测 → 判断定位证据是否可靠 → 分析失败原因 → 选择 reference / Memory / 强几何工具 → 验证候选 → 接受、升级或拒绝`。

当前首先回答的问题是：

> Reloc3r 在 static、dynamic、long-term change、dynamic+long-term change 下如何失败；哪些失败可以由无 GT 的 reference 选择、时序 memory、验证或强几何恢复；哪些失败仍无法恢复？

## 2. 数据、坐标和信息隔离

- 地图/reference：LILocBench `mapping` sequence 的前置 RGB；query 为 `static_0`、`dynamics_0`、`lt_changes_0`、`lt_changes_dynamics_0`。
- 采样：前置 RGB，1 Hz；四类 query 帧数分别为 578、155、422、541，共 1696 帧。
- 模型：Reloc3r-512，项目本地 Hugging Face cache；推理使用 `model.safetensors`。
- 基线：NetVLAD mapping-only retrieval，Reloc3r 相对位姿，motion averaging；Top-10 为原始 AVL-002 基线。
- GT：官方全局 `T_world_base_link`，结合各序列外参转成前置 RGB optical-frame camera pose。
- 同步：严格 30 ms 审计发现少量启动边界/时钟抖动帧超阈值；主分析统一采用 100 ms 最近邻并保留逐帧时间差，同时报告严格 30 ms 敏感性。严格子集没有改变主要结论。
- 泄漏限制：query GT 不进入 retrieval、Reloc3r、候选聚合、reference reranking、PnP、Manager 或 Memory；mapping 绝对 pose 仅作为地图资产；Memory 只访问当前帧之前的信息；GT 只在 post-hoc evaluator 中使用。

## 3. 实验状态总览

| 实验 | 内容 | 状态 | 结论用途 |
|---|---|---|---|
| AVL-001 | 原始 Reloc3r/7Scenes 评测完整性审计 | planned | 不作为当前 LILocBench 结论来源 |
| AVL-002 | Reloc3r 四条件失效体检 | completed | 建立失败模式与病因证据 |
| AVL-003 | consensus / verification / memory 无 Agent 消融 | completed | 建立 deterministic 恢复基线 |
| AVL-004 A.2/A.3 | Top-100 reference reranking + HLoc-style/RoMa 局部恢复 | completed | 验证 reference reliability 是前置瓶颈 |
| AVL-004 A.4 | 无 GT risk-coverage 与 Manager dry-run | completed | 验证风险信号可选择性分层，但暴露 reject 缺失 |
| AVL-004 A.5 | 四序列全 geometry-call route 回灌 | completed | 确认 deterministic geometry route 的收益和残余失败 |
| 真正 Agent | Context/Manager/Memory Agent | not started by design | 等 deterministic 分析和 Go/No-Go 决策 |

## 4. AVL-002：Reloc3r 四条件失效体检

固定 Top-10 Reloc3r 的主结果如下。成功定义为 `translation ≤0.25 m 且 rotation ≤5°`。

| 条件 | 帧数 | median t (m) | median r (°) | 成功率 | 最大连续失败 | jump `>1m 或 >20°` | candidate dispersion median (t/r) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static | 578 | 0.8967 | 0.4187 | 39.79% | 26 | 56.67% | 1.244 m / 1.327° |
| Dynamic | 155 | 3.9372 | 1.3013 | 13.55% | 30 | 82.47% | 5.189 m / 35.971° |
| Long-term change | 422 | 1.8915 | 0.6085 | 24.88% | 37 | 66.27% | 2.586 m / 3.668° |
| Dynamic + LT | 541 | 1.6810 | 0.7762 | 23.48% | 37 | 69.81% | 3.386 m / 7.269° |

### AVL-002 解释

1. Static 也不稳定：Top-k=3 成功率为 88.41%，显著高于 Top-k=10 的 39.79%，说明错误 reference 混入和 motion averaging 污染是首要问题。
2. Dynamic 相比 static 出现更大候选离散度、更高 jump 和更长失败段；人工 contact sheet 中可见人员、近景遮挡和运动模糊，但没有语义真值，因此只能称 operational evidence，不能作逐帧因果断言。
3. LT 出现持续失败和参考失配，代表帧中可见纸箱、办公物件、门/走廊外观变化；不能把原因唯一归于某一家具或门。
4. Dynamic+LT 的总体均值不比 Dynamic 更差，因此“条件叠加必然最难”被当前总体结果否定；轨迹、空间覆盖和检索可见性是替代解释，必须按时间/空间段分析。

## 5. AVL-003：无 Agent 恢复消融

在同一 Top-10 Reloc3r 候选上进行无 GT 的候选选择、验证和受限历史 memory：

| 方法 | Static | Dynamic | LT | Dynamic+LT | 最大失败段（四条件） |
|---|---:|---:|---:|---:|---:|
| Top-10 motion averaging | 39.79% | 13.55% | 24.88% | 23.48% | 26/30/37/37 |
| consensus-3 | 82.70% | 55.48% | 75.12% | 58.60% | 5/10/9/20 |
| verify-10 | 87.02% | 52.26% | 75.59% | 58.23% | 7/14/11/23 |
| memory-3 | 91.18% | 58.06% | 81.52% | 62.85% | 4/15/9/21 |

结论：候选一致性选择和短期可靠 memory 能显著减少错误候选污染与连续失败；但 memory-3 在 Dynamic+LT 仍留下 21 帧长失败段，说明需要 reference 重选、强几何、验证或拒绝动作。该实验不是 Memory Agent，也不证明 LLM 有收益。

## 6. AVL-004 A.2/A.3：reference reliability 与强几何

### 6.1 局部困难段

四类 Memory-3 最长失败段共 49 帧：Static 5–8、Dynamic 33–47、LT 413–421、Dynamic+LT 416–436。

冻结 Top-4 RoMa 成功率为 10/49（20.41%）；无 GT Top-100 NetVLAD → LightGlue/Essen­tial-RANSAC 重排 → 固定 4 refs 后：

- RoMa：28/49（57.14%）
- HLoc-style 2D/3D feasibility：29/49（59.18%）

逐段 RoMa route 成功率为 Static 3/4、Dynamic 9/15、LT 3/9、Dynamic+LT 13/21。LT 仅 3/9，说明 reference reranking 不是长期变化的充分条件。

### 6.2 关键因果证据

Dynamic+LT query 416 的 frozen references 可产生约 15.84 m / 157.19° 的错误 RoMa pose；GT 事后选择的兼容 map references 可达到约 0.0248 m / 1.19°，但这是 upper bound，不可部署。该对照说明：强几何失败的前置瓶颈经常是 reference 不可见或不兼容，而不是几何求解器本身。

## 7. AVL-004 A.4：deterministic Manager 风险审计

固定无 GT 风险分数由 candidate dispersion、jump、appearance/frame change 和历史可靠性组成。四序列共 1696 帧的 risk-coverage：

| 风险阈值 | coverage | 选择子集成功率 |
|---:|---:|---:|
| 1.0 | 15.09% | 96.09% |
| 1.5 | 27.36% | 94.61% |
| 2.0 | 35.38% | 91.33% |
| 3.0 | 43.69% | 89.34% |
| 5.0 | 57.19% | 87.22% |
| 全量 consensus candidate | 100% | 70.64% |

Manager 动作计数（accept / memory / geometry / reject）：

- Static：161 / 219 / 198 / 0，输出 coverage 65.74%
- Dynamic：3 / 22 / 130 / 0，输出 coverage 16.13%
- LT：67 / 112 / 243 / 0，输出 coverage 42.42%
- Dynamic+LT：26 / 134 / 381 / 0，输出 coverage 29.57%

结论：风险信号有选择性，但当前 Manager 是风险触发器，不是完成的恢复器；`reject=0` 暴露了安全边界，不能把 risk-coverage 结果写成 Agent 收益。

## 8. AVL-004 A.5：全序列 deterministic geometry route

### 8.1 协议

对 Manager 全部 952 个 `call_geometry` 帧：mapping-only Top-100；最多 100 个 reference 做 LightGlue/Essen­tial-RANSAC；固定选 4 refs；同一 refs 分别交给 HLoc-style 和 RoMa。RoMa 使用 `use_custom_corr=false` 原生 PyTorch fallback。两种工具不读取 query GT。

### 8.2 全路线结果

| 条件 | HLoc success | RoMa success | HLoc coverage | RoMa coverage |
|---|---:|---:|---:|---:|
| Static | 90.83% | 92.91% | 100% | 100% |
| Dynamic | 81.29% | 87.10% | 96.77% | 100% |
| LT | 86.73% | 86.97% | 99.76% | 100% |
| Dynamic+LT | 86.32% | 87.62% | 99.82% | 100% |
| **加权 1696 帧** | **87.50%（1484/1696）** | **89.21%（1513/1696）** | **99.59%（1689/1696）** | **100%** |

作为参照，AVL-003 memory-3 加权成功率为 1301/1696 = 76.71%。A.5 的提升主要来自 dynamic、LT 和 combo 的 geometry-call 帧；static 仅小幅变化，说明 geometry 不是普遍替代 Reloc3r/Memory 的路径。

### 8.3 geometry-call 池结果

在同一 952 帧触发池：

- memory-3：612/952 = 64.29%
- HLoc-style：795/952 = 83.51%
- RoMa：824/952 = 86.55%

两种工具均超过预注册的至少 5 个百分点恢复门槛。四类 geometry success（HLoc/RoMa）为：Static 84.85/90.91%、Dynamic 79.23/86.15%、LT 81.48/81.89%、Dynamic+LT 85.56/87.40%。

同一 selected references 的单次运行成功混淆为：both success 764、HLoc-only 31、RoMa-only 60、neither 97。复核发现 RoMa 未固定随机 seed；用另一既有 RoMa 运行复算时共同失败为 98 帧，原 97 帧集合仅有 88 帧重合。因此这里可作探索性互补证据，不能把精确集合当作稳定工具属性。

### 8.4 失败、误接受与计算成本

- RoMa route 最长失败段：Static/Dynamic/LT/combo = 4/5/5/5 帧。
- HLoc-style route 最长失败段：5/7/6/6 帧。
- RoMa geometry pose availability：952/952；HLoc-style：945/952。
- RoMa 被单独计时的 4-ref geometry 阶段中位数约 4.6–4.9 s，P95 约 6.1–6.8 s。
- HLoc-style 字段包含 Top-100 reranking 与 HLoc-style PnP，中位数约 18.8–20.2 s，P95 约 28.0–31.3 s；与上一项计时边界不同。
- RoMa geometry route false-accept 约 9.09–18.11%；HLoc-style 约 14.21–18.18%。有 pose 不等于 pose 已验证可靠。
- 8 路分片逐帧 elapsed 总和：RoMa 阶段 4506.7 s，reranking+HLoc-style 阶段 18921.0 s；输出文件时间跨度约 10 分钟与 33 分钟。二者既不能直接相互比较，也不能把并行 wall-clock 当单帧 latency。

Dynamic+LT 困难段 416–436 在全 route 中 HLoc/RoMa 均为 13/21（61.90%）；仍有 8 帧失败。

## 9. 数据泄漏、GT 和实现审计

- 所有 A.3/A.4/A.5 retrieval、reranking、matching、PnP、Manager trace 均标记 `query_gt_read=false`、`selection_uses_gt=false`。
- route evaluator 只在 post-hoc 阶段读取 query GT；recovery 工具本身不读取 GT。
- Memory 只读取过去帧；Manager 不以 GT 为输入。
- mapping pose 作为 metric-map asset 是协议允许的；标准 COLMAP reconstruction 尚未建立。
- 首次 RoMa 因未设置项目级 `TORCH_HOME` 而尝试默认缓存下载，结果无效；目录和日志保留在 `outputs/lilocbench/AVL-004/full_route_a5/roma_shards/`，不纳入结论。有效结果来自 `roma_shards_localcache/`。
- A.4 早期 coverage 分母错误的 `risk_coverage/` 保留为 invalid artifact；有效结果为 `risk_coverage_v2/`。
- 标准 HLoc 3D-PnP 尚未运行；当前称 HLoc-style 2D/3D feasibility。

## 10. 当前科学结论

### 已证明到什么程度

1. 固定 Top-10 Reloc3r 会被错误候选和 motion averaging 污染，static 也不稳定。
2. Dynamic 放大瞬时冲突和连续失败；LT 造成持续 reference mismatch；combo 难度不是总体均值单调增加。
3. 无 GT consensus、verification 和受限 Memory 能显著改善，但不能消除困难段。
4. 无 GT Top-100 reference reranking → 强几何是有效 deterministic 恢复动作；A.5 全路线对 geometry-call 帧有明显增益。
5. 保存结果中 RoMa 的单次运行准确率略高于 HLoc-style；工具有探索性互补且都有共同失败。端到端速度和互补集合稳定性尚未证明。

### 尚未证明

1. LLM/多 Agent 是否优于 deterministic Manager。
2. Context 信号是否能可靠区分遮挡、长期变化、reference mismatch 和几何退化。
3. 固定 geometry budget 下 Agent 是否能优于 always-geometry、固定阈值、reject 和工具组合策略。
4. Memory 是否带来脱离同轨迹近邻的因果收益。
5. 当前 HLoc-style 结果能否代表标准 HLoc 3D-PnP。

### 当前被否定或暂停的解释

- “Top-10 在 static 上足够稳定”已被否定。
- “调用 RoMa/HLoc 就能解决失败”未被支持：97/952 共同失败、LT 约 82%、困难段仍失败。
- “Dynamic+LT 总体必然最差”被当前总体均值否定。
- 标准 HLoc 3D-PnP 暂停，等待可审计 COLMAP map。

## 11. Agent 前的 Go / No-Go 决策

当前不进入 Agent 实现。必须先完成：

1. 固定 geometry budget 下的 always-geometry、reject、verification 和工具组合对照；
2. 97 个共同失败帧、LT 长失败段和 Dynamic+LT 416–436 的逐帧 reference/场景审计；
3. 公平核算 Top-100 matching、RoMa/HLoc-style、Reloc3r 和 Memory 的延迟与计算预算；
4. 预注册拒绝/验证阈值，禁止用 GT 调参。

只有当 deterministic 规则在这些审计后仍留下明确的上下文调度价值，才进入真正 Agent。主线仍然是 Agentic Visual Localization；当前停止是阶段性科学审查，不是主线变更。

## 12. 关键证据路径

- 长期科学问题与结论：[research_question.md](research_question.md)
- 实验协议与结果登记：[experiment_registry.md](experiment_registry.md)
- 当前项目状态：[codex_project_state.md](codex_project_state.md)
- AVL-002 结果：`outputs/lilocbench/AVL-002/*/summary.json`
- AVL-003 结果：`outputs/lilocbench/AVL-003/summary_all.json`
- AVL-004 A.4：`outputs/lilocbench/AVL-004/risk_coverage_v2/summary_all.json`
- AVL-004 A.5：`outputs/lilocbench/AVL-004/full_route_a5/recovery_analysis.json`

## 13. AVL-004 A.6：Always-HLoc / Always-RoMa

为排除 Manager 触发器本身造成的选择偏差，对四个 query 序列全部 1696 帧执行同一 mapping-only Top-100 检索、LightGlue/Essen­tial-RANSAC 重排和固定 4-reference 几何预算。恢复工具不读取 query GT；GT 只由独立 evaluator 读取。

| route | static | dynamic | LT | dynamic+LT | 加权 |
|---|---:|---:|---:|---:|---:|
| Always-HLoc-style success | 90.14% | 82.58% | 88.15% | 89.09% | 88.62% (1503/1696) |
| Always-RoMa success | 94.64% | 89.03% | 88.39% | 90.39% | 91.21% (1547/1696) |
| HLoc coverage | 99.65% | 96.77% | 99.76% | 99.82% | 99.47% |
| RoMa coverage | 100% | 100% | 100% | 100% | 100% |

加权 median translation/rotation 为 HLoc `0.0295 m / 0.437°`、RoMa `0.0271 m / 0.431°`。最长失败段（static/dynamic/LT/combo）HLoc 为 `10/7/7/6` 帧，RoMa 为 `9/4/5/5` 帧。Always 全帧 rerun 的单帧 elapsed 中位数约 HLoc `12.48 s`、RoMa `2.29 s`；但复核确认 HLoc elapsed 包含 Top-100 重排与 HLoc-style PnP，RoMa elapsed 只计复用 selected refs 后的 RoMa 阶段。这些值不是同阶段端到端对比，不能支持“RoMa route 更快”的结论。

## 14. AVL-004 A.7：Action Oracle

逐帧事后取 consensus-3、memory-3、Always-HLoc、Always-RoMa 的成功事件，计算上界：consensus `1198/1696=70.64%`，memory `1301/1696=76.71%`，HLoc `1503/1696=88.62%`，RoMa `1547/1696=91.21%`，任一候选成功的 oracle `1624/1696=95.75%`。A.5 deterministic Manager 的 RoMa route 为 `1513/1696=89.21%`，距 oracle 差 `6.54` 个百分点；Always-RoMa 距 oracle 差 `4.54` 个百分点。分序列 oracle success 为 static `98.62%`、dynamic `90.32%`、LT `95.26%`、dynamic+LT `94.64%`。

全量 HLoc/RoMa 成功混淆的单次运行值为 both `1466`、HLoc-only `37`、RoMa-only `81`、neither `112`。RoMa 未固定随机 seed；同一 952 个 Manager-call 帧的两次既有 RoMa 运行有 34 帧成功标签翻转。因此“存在互补”可作探索性结论，精确混淆计数和 Oracle 上界需固定 seed、多次重跑后确认。

## 15. AVL-004 A.8：97 个共同失败帧

A.5 geometry-call 池中 97 个 HLoc/RoMa 共同失败帧已保存 reference scores、selected refs、inliers、重投影、候选离散度、appearance/frame-change、GT 空间审计标签和 contact sheets：`outputs/lilocbench/AVL-004/common_failure_audit/`。按序列为 static/dynamic/LT/combo `16/18/28/35` 帧。

主要 operational multi-label 计数：trajectory jump `94`、both pose available but wrong `90`、appearance_change_proxy `79`、turn_motion `75`、candidate_conflict `69`、reference_mismatch `65`、low-texture/blur proxy `65`、low_texture_or_blur `60`、large reference distance proxy `54`、appearance_change `51`。73/97 帧含 reference mismatch/distance 证据，79/97 含 appearance proxy，65/97 含 blur/low-texture proxy，90/97 两工具均产生 pose 但仍错误。这些标签是可审计证据，不是逐帧遮挡、家具或门变化的语义真值；代表图像显示 dynamic 中近场行人/运动模糊、LT 中箱体/门/家具变化、combo 中两者共现。

## 16. AVL-004 A.9：Risk–cost–accuracy

风险分数和阈值在看到 GT 前已冻结。risk-gated RoMa 的加权 `geometry-call rate / coverage / success_all_frames / selective accuracy` 为：

| risk threshold | geometry rate | coverage | success (all frames) | selective accuracy |
|---:|---:|---:|---:|---:|
| 0.5 | 98.05% | 100% | 91.21% | 91.21% |
| 1.0 | 84.91% | 100% | 90.92% | 90.92% |
| 1.5 | 72.64% | 98.88% | 89.45% | 90.46% |
| 2.0 | 64.62% | 96.46% | 86.85% | 90.04% |
| 3.0 | 56.31% | 91.75% | 81.90% | 89.27% |
| 5.0 | 42.81% | 81.96% | 72.46% | 88.42% |

只保留低风险 action-output 的 post-hoc reject policy，按全局可用 pose 数正确汇总后，在阈值 0.5/1.0/1.5/2.0/3.0/5.0 的 coverage / all-frame success / selective accuracy 为 `1.95/1.95/100.00%`、`15.09/14.50/96.09%`、`26.24/24.88/94.83%`、`31.84/30.01/94.26%`、`35.44/33.25/93.84%`、`39.15/36.50/93.22%`。旧 selective accuracy 使用了错误的跨序列加权，已作废。该曲线证明风险分层有效，但低风险拒绝策略不是高覆盖定位器。A.5 Manager 的 geometry-call rate 为 `952/1696=56.13%`，RoMa route success `89.21%`，高于同等调用率附近的单阈值 route；这里比较的是 call-rate，不是统一 latency/cost。

## 17. Agent 前终止性判断

Agent 未实现。当前结果支持：固定 Top-10 Reloc3r 的主要可操作问题是 reference/candidate reliability；reference re-ranking→geometry 是有效 deterministic recovery；保存结果上的 RoMa 准确率略高于 HLoc-style，并存在探索性的工具互补和 Oracle gap。由于 RoMa seed、统一 latency、A.8 语义分类和闭环回灌未完成，A.6–A.9 不能再笼统称为全部终止性问题已完成。

因此当前结论是 **Hold / No-Go for immediate LLM implementation**，不是主线否定。下一步仍只做 deterministic Context/Manager、验证/拒绝和公平 geometry/latency 预算分析；只有建立“数值风险相近但视觉上下文明确分流动作”的独立、无 GT 证据，并在相同预算下超过 deterministic baseline，才进入 Agent 设计。主线仍保持 failure-aware Agentic Visual Localization。

## 18. 2026-10-10 关键实验复核勘误

完整复核见 `docs/key_experiment_reaudit.md`。AVL-002/003 的核心实现与数值成立，A.5/A.6 的 GT 事后准确率成立；但 RoMa 未固定随机种子、HLoc/RoMa 计时边界不同、A.5 geometry pose 未回写后续 Manager 状态，且四个 query 序列没有 held-out 隔离。故逐帧互补/共同失败/Oracle 精确值、端到端速度优势和 closed-loop Manager 表述均降级为暂定或不成立。当前主线与 Hold 决策不变。
