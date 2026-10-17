# 关键实验代码与结论复核报告（2026-10-10）

## 1. 复核范围与判定

本次只筛选直接支撑项目主线的实验：AVL-002 四条件 Reloc3r 体检、AVL-003 Reference/Consensus/Memory 消融、AVL-004 A.5 deterministic route、A.6 Always Geometry、A.7 Action Oracle、A.8 共同失败审计与 A.9 Risk–cost–accuracy。未重新运行 GPU 推理；复核包括代码路径、数据/GT 边界、位姿方向、逐帧产物、独立数值重算、确定性阶段重放和两次既有 geometry 运行交叉比较。

结论等级：

- **成立**：实现、产物和独立重算一致，可继续作为当前数据集内证据。
- **有条件成立**：数值本身正确，但只支持更窄的科学表述。
- **暂定/需重跑**：受随机性、预算口径或实验设计影响，不能作为稳定结论。
- **无效**：计算或汇总错误，立即停止引用旧值。

## 2. 总结

主线没有被否定。最稳健的结果仍是：固定 Top-10 聚合容易被候选污染；consensus 和仅访问过去的短期 memory 可大幅恢复；无 GT reference reranking 后的强几何能进一步恢复；仍需可靠性验证和拒绝机制。

但此前有三处过度结论必须修正：

1. **不能根据现有 elapsed 字段声称 RoMa 的端到端 route 比 HLoc-style 快，也不能声称 Manager 以约 56% 调用率节省约 44% 实际计算。** HLoc 的 elapsed 包含 Top-100 SuperPoint+LightGlue 重排及 HLoc-style PnP，RoMa 的 elapsed 只包含读取既有 selected refs 后的 RoMa 阶段，两者计时边界不同。
2. **RoMa 逐帧成功集合不是可复现的。** `model.sample` 使用 `torch.multinomial`，运行脚本未设置 torch/numpy/CUDA seed。同一 952 帧、同一 refs 的两次既有运行有 34 帧成功标签翻转。聚合成功数恰好同为 824/952，但逐帧工具互补、共同失败成员和 Oracle 精确值只能视为单次运行结果。
3. **A.5 是冻结决策的离线回灌，不是闭环 Manager。** geometry pose 没有回写到后续 `previous_prediction` 或 memory 状态，因此后续动作不会因 recovery 结果变化。

## 3. AVL-002：Reloc3r 四条件体检

### 复核结果：成立（限 LILocBench 当前采样协议）

- mapping database 为 785 帧；query 为 static/dynamic/LT/combo `578/155/422/541` 帧。mapping 与四个 query 序列之间未发现完全相同 RGB 哈希。
- mapping 与 query 的相机内参文件一致；query GT 从各自序列的 transformations/GT 读取。
- Reloc3r 返回的相对位姿方向与绝对位姿组合一致：query-to-reference 相对位姿按 `T_world_ref @ T_ref_query` 转成 query 绝对位姿。
- query GT 只在 evaluator/失败分析中读取，不参与 retrieval、Reloc3r 推理或候选聚合。
- 独立重算与保存汇总一致：Top-10 `(0.25 m, 5°)` 成功率 static/dynamic/LT/combo 为 `230/578=39.79%`、`21/155=13.55%`、`105/422=24.88%`、`127/541=23.48%`；median translation 为 `0.8967/3.9372/1.8915/1.6810 m`。

限制：当前所谓 1 Hz 实际为约 0.94 Hz（相邻采样约 1.06 s）；完整分析使用 100 ms GT 同步容差，严格 30 ms 敏感性已另报。四个 query 条件既用于发现 failure pattern，也用于后续阈值/策略设计，不能当作独立 held-out 泛化证据。

## 4. AVL-003：Consensus / Memory 消融

### 复核结果：成立（算法作用域需保持准确）

- 独立重放四序列 `frame_results.jsonl` 与原产物逐字节 SHA-256 相同。
- consensus-3 成功计数为 static/dynamic/LT/combo `478/86/317/317`；memory-3 为 `527/90/344/340`，与文档一致。
- memory 仅使用当前帧之前的内部预测；没有未来帧访问，也没有用 GT 决定更新。
- 该结果证明的是“从已计算的 Top-10 Reloc3r 候选中选择/聚合有价值”，不是检索前 Reference Agent，也不节省 Top-10 Reloc3r 推理成本。
- memory-3 是短期启发式历史，不是完整 Memory Agent；它可能保留内部一致但全局错误的状态。

## 5. AVL-004 A.5/A.6：Reference reranking 与强几何

### 5.1 准确率：成立

- A.5 Manager 触发 952 帧，计数 static/dynamic/LT/combo=`198/130/243/381`；两工具使用相同 selected refs。
- A.5 route success：HLoc-style `1484/1696=87.50%`，RoMa `1513/1696=89.21%`。
- A.6 Always：HLoc-style `1503/1696=88.62%`，RoMa `1547/1696=91.21%`；HLoc pose coverage 为 `1687/1696=99.47%`，不是旧文档中的 `1487/1696`。
- recovery/reference/Manager 均不读取 query GT；GT 只用于 route evaluator。
- HLoc-style 并非标准 COLMAP-HLoc 3D map localization，而是已知 mapping 相机位姿、局部匹配三角化、query PnP 的 feasibility pipeline。

### 5.2 RoMa 逐帧互补：暂定/需有种子重跑

RoMa 脚本未固定随机种子，而 RoMa `sample()` 使用两次 `torch.multinomial(..., replacement=False)`。A.5 与 A.6 两次既有运行在相同 952 个 Manager-call 帧上：

- refs 完全相同；HLoc 成功标签 952/952 一致。
- RoMa 聚合成功都为 `824/952`，但 34 帧成功/失败标签翻转。
- A.5 的 HLoc/RoMa 共同失败为 97 帧；用第二次 RoMa 运行替换后为 98 帧，交集 88、并集 107，Jaccard=`0.822`。
- 因此 `31 HLoc-only / 60 RoMa-only / 97 neither` 和 A.8 的具体 97 帧集合仅对 A.5 单次运行成立，不能解释为工具的稳定内在互补。

需要固定 Python/numpy/torch/CUDA/OpenCV seed，并至少报告 3 个 seed 的聚合均值、方差、逐帧一致率及共同失败集合稳定性。

### 5.3 延迟与计算成本：旧端到端解释无效

- HLoc `elapsed_s` 从 query SuperPoint 开始，包含 Top-100 reference 的 SuperPoint+LightGlue、Essential-RANSAC 排序、4-ref 三角化和 PnP。
- RoMa `elapsed_s` 在 selected refs 已从 HLoc JSONL 读入后开始，只包含 4-ref RoMa matching/三角化/PnP。
- 所以 A.5 的约 `18.8–20.2 s` 对 `4.6–4.9 s`、A.6 的 `12.48 s` 对 `2.29 s` 不是同阶段端到端对比；“RoMa 更快”只能描述其被单独计时的后半阶段。
- 作为不可用于正式结论的粗略上界代理，A.5 952 帧的 HLoc计时总和 `18921.0 s` 加 RoMa阶段 `4506.7 s` 为 `23427.7 s`；Always 1696 帧相应为 `21696.6+3914.2=25610.7 s`。该代理仅减少约 `8.5%`，而不是由调用率推导的 `43.9%`；并且它重复计入了 HLoc PnP，仍不是准确端到端成本。

## 6. AVL-004 deterministic Manager

### 复核结果：冻结离线路由成立；闭环结论不成立

- Manager 四序列重放的 trace 与原文件逐字节 SHA-256 相同，规则决策可复现。
- 输入确实不含 query GT；风险、appearance、jump 和历史可靠性均从图像或内部预测计算。
- `call_geometry` 时 `action_pose=None`；只有 accept/memory 的 pose 会更新 `previous_prediction`。A.5 evaluator 事后替换当前帧输出，但没有重放后续 Manager 状态。
- 因此 `89.21%` 是“冻结 Manager actions + 离线 geometry 回灌”的 route 分数，不是 recovery-aware closed-loop Manager 的分数。闭环实现后动作序列、memory、jump 与调用率都可能改变。

## 7. AVL-004 A.7：Action Oracle

### 复核结果：计算正确，但上界数值暂定

- Always 输出上的事后 oracle 计算逻辑正确；query GT 仅用于成功事件，未参与在线选择。
- 当前单次运行值 `1624/1696=95.75%` 可以作为这批保存结果的 upper bound。
- 因 RoMa 未固定 seed，oracle 的精确计数、HLoc-only/RoMa-only/neither 成员关系尚未证明稳定。可保留“存在工具互补和 oracle gap”的定性结论，不应把 95.75% 或 37/81/112 当作已建立置信区间的最终数字。
- `available` 元数据当前用 `query_index in route_dict`，而 route evaluator 为全部帧建行，因此在 A.5 非 geometry 帧也显示可用；该字段错误不影响 success（success 另以 action gate），但应在重跑时修正。

## 8. AVL-004 A.8：共同失败审计

### 复核结果：部分完成

- 对 A.5 单次运行定义的 97 个共同失败帧，artifact、contact sheet 和 operational proxy 统计存在且内部一致。
- 代码实际生成的是 `reference_mismatch`、appearance、blur、jump 等启发式多标签；其中 reference distance/空间字段使用 GT 仅作事后审计。
- 它没有完成预注册的语义/因果分类 `reference_visibility/mismatch`、`severe_occlusion_or_dynamic`、`long_term_structure_change`、`geometric_degradation`、`mixed/uncertain`，也没有语义标注真值。
- 因此 A.8 只能支持“多种 proxy 共现”，不能支持“已经识别 97 帧的真实病因”。具体集合还受 RoMa seed 影响。

## 9. AVL-004 A.9：Risk–cost–accuracy

### 9.1 准确率/调用率曲线：有条件成立

risk-gated RoMa 的每序列输出正确。全局按帧汇总后的主要值为：阈值 3.0 时 geometry rate `56.31%`、coverage `91.75%`、all-frame success `81.90%`、selective accuracy `89.27%`。它支持“单一冻结风险阈值在相近调用率下弱于当前冻结 Manager route”。

旧报告的 reject selective accuracy 是用序列总帧数错误加权各序列 selective accuracy，立即作废。正确全局值为：

| threshold | coverage | all-frame success | selective accuracy |
|---:|---:|---:|---:|
| 0.5 | 1.95% | 1.95% | 100.00% |
| 1.0 | 15.09% | 14.50% | 96.09% |
| 1.5 | 26.24% | 24.88% | 94.83% |
| 2.0 | 31.84% | 30.01% | 94.26% |
| 3.0 | 35.44% | 33.25% | 93.84% |
| 5.0 | 39.15% | 36.50% | 93.22% |

risk-gated threshold 5.0 的 selective accuracy 精确重算为 `88.42%`，不是旧文档的 `88.38%`。

### 9.2 cost/latency：未完成

A.9 代码只把 geometry call rate 作为 cost，没有读取实际 latency，也没有统一计算 reference reranking、Reloc3r/Memory、HLoc/RoMa 的 action cost。因此 A.9 完成的是 risk–call-rate–accuracy，而不是预注册的 risk–latency–accuracy。正式成本结论暂缓。

## 10. 其他实现限制

- 相机畸变系数非零。HLoc/RoMa 的 Essential-RANSAC 与三角化直接使用原始像素和 K，没有先 undistort；最终 PnP 使用了 distortion。该问题不改变已保存 pose 的 GT 事后评分，但可能限制几何精度，重跑应统一归一化/去畸变坐标。
- RoMa 的 query 坐标最近邻 intersection 不是严格一对一匹配；PnP RANSAC 可过滤部分噪声，但 inlier/reprojection 不能直接解释为校准可靠性。
- 同一四序列被用于 failure discovery、阈值制定、策略设计和最终报告。没有在线 GT 泄漏，但存在测试集适配；现有结论是 LILocBench 内部 feasibility，不是跨场景或 held-out 泛化结果。

## 11. 修正后的科学结论与下一步

### 可继续引用

1. Reloc3r Top-10 在 dynamic/LT/combo 明显退化，并出现连续失败、jump 和候选冲突。
2. consensus-3 和仅访问过去的 memory-3 在四条件上稳定改善，说明 candidate/reference reliability 是核心可操作问题。
3. mapping-only Top-100 reranking 后的 geometry 在当前数据集内显著提高准确率；单一候选或有 pose 不等于可靠。
4. 当前冻结风险阈值无法在相近 geometry call rate 下复现冻结 Manager route 的准确率。

### 暂停引用或必须加限定

1. RoMa 比 HLoc-style 端到端更快；Manager 已节省约 44% latency。
2. `31/60/97` 或全量 `37/81/112` 是稳定工具互补结构。
3. `95.75%` 是已稳定估计的 Oracle 上界。
4. 97 帧的真实失败原因已经分类完成。
5. A.5 已经是闭环自主 Manager。

### Agent 前必须补齐

1. 固定全部随机种子，对 RoMa/HLoc 重跑至少 3 seeds，并重新计算互补、共同失败和 Oracle 稳定性。
2. 拆分并统一计时：retrieval、Top-100 reranking、geometry、verification、基础 Reloc3r/Memory；报告串行 latency、GPU-hours 和 call-rate。
3. 实现 geometry pose 回灌后的 closed-loop deterministic Manager，再比较 always/risk-gated/oracle。
4. 使用独立 validation 冻结阈值，在 held-out 序列/场景评估；A.8 建立人工审计的语义标签或明确保持 proxy 研究。

当前决策仍为 **Hold / No-Go for immediate Agent implementation**。这是对证据标准的收紧，不是主线变更。
