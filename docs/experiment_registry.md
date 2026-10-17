# 实验登记册

> 规则：运行前登记协议；完成后只追加结果，不覆盖原协议。失败、无效与泄漏实验必须保留。日期采用 Asia/Shanghai。

## 状态定义

- `planned`：协议已预注册，尚未运行。
- `running`：已启动；须记录日志路径。
- `completed`：结果已附于该条目。
- `invalid`：存在数据泄漏、GT 错配或实现错误；不得用于科学结论。
- `paused`：保留原因与恢复条件。

## 整理后的当前索引（权威摘要）

本节只提供当前状态和可引用结果；下方 AVL-001–007 的协议与运行附录保持追加式保存，历史失败、无效尝试和修正记录均不删除、不覆盖。

| 编号 | 实验主题 | 状态 | 当前可引用结论 | 主要产物 |
|---|---|---|---|---|
| AVL-001 | Reloc3r 基线与评测审计 | `planned` | 原始基线协议保留，未作为 LILocBench 结论使用 | 见原协议 |
| AVL-002 | LILocBench 四条件 Reloc3r 体检 | `completed` | Top-10 success=`39.79/13.55/24.88/23.48%`；dynamic/LT 有更长失败和更高候选冲突；combo 非单调 | `outputs/lilocbench/AVL-002/*/summary.json` |
| AVL-003 | consensus / memory 确定性消融 | `completed` | consensus-3=`82.70/55.48/75.12/58.60%`；memory-3=`91.18/58.06/81.52/62.85%`；只使用过去状态 | `outputs/lilocbench/AVL-003/summary_all.json` |
| AVL-004 | reference 重排、HLoc-style/RoMa、验证与风险曲线 | `completed with limits` | RoMa route=`89.21%`、HLoc-style=`87.50%`；强几何可恢复但仍有共同失败和错误接受；不能称 Agent 收益 | `outputs/lilocbench/AVL-004/` |
| AVL-005-A | RoMa 多 seed（952 帧池） | `completed` | success=`86.76/86.97/87.18%`，均值=`86.97%`；LT 共同失败集合有 seed 敏感性 | `outputs/lilocbench/AVL-005/multiseed_summary.json` |
| AVL-005-B | 统一成本核算 | `completed` | Always-RoMa=`91.51%/6622.564s`；Manager=`89.68%/6665.885s`、call=`46.17%`；当前无成本优势 | `outputs/lilocbench/AVL-005/unified_cost/summary.json` |
| AVL-005-C | 无 GT 闭环 Manager 回放 | `completed` | coverage=`1672/1696`；缓存 recovery 回灌后 success=`89.50%`（未重新在线测量 geometry） | `outputs/lilocbench/AVL-005/closed_loop_manager/route_eval.json` |
| AVL-005-D | 独立 held-out | `paused` | 新场景按用户要求暂不运行；LILocBench 内部 split 只能称 calibration-shift test | `outputs/lilocbench/AVL-005/heldout_validation_static_dynamic_test_lt_combo_v2.json` |
| AVL-006 | break-even 与 GT-free pre-router | `completed` | LOSO logistic/tree=`87.44/86.56%`，均低于 Always-RoMa；oracle union 仅作上界 | `outputs/lilocbench/AVL-006/` |
| AVL-007 | 冻结 DINOv2 视觉表征路由 | `completed` | 最好 success=`87.62%`、balanced accuracy=`49.09%`；不能稳定识别 memory-only/RoMa-only | `outputs/lilocbench/AVL-007/summary.json` |
| AVL-008 | Semantic Context Probe（Qwen2.5-VL） | `completed with limits` | V=`0.219`，低于语义关系门槛；VLM action BA=`47.02%`，同帧 DINO=`47.64%`，未提供额外路由信息 | `outputs/lilocbench/AVL-008/summary.json` |
| AVL-009 | Causal sequence-based visual localization baseline | `completed with limits` | temporal-only 在 test 上 `0.00%`；global+temporal=`53.64%`，memory-3=`76.87%`，Always-RoMa=`92.62%`；当前 Reloc3r temporal edge 未显示独立收益，Agent 继续 No-Go | `outputs/lilocbench/AVL-009/summary.json`、`docs/avl009_sequence_localization_report.md` |
| AVL-010 | Strong temporal tracker（DPVO） | `revalidated with limits` | 修正 Sim(3) 后 temporal-only=`0.20%`、Global+DPVO=`4.04%`、+Memory+RoMa=`65.26%`；当前实现没有独立定位收益 | `outputs/lilocbench/AVL-014/AVL-010_eval_sim3fix/summary.json` |
| AVL-011 | Dynamic+LT 诊断与强化 deterministic recovery | `completed with limits` | test success=`91.24%`、coverage=`100%`、geometry call=`67.13%`；共同失败主要是 reference/一致性/错误接受 | `outputs/lilocbench/AVL-011/final_v1/evaluation.json` |
| AVL-012 | Localization Reliability & Reject | `revalidated with limits` | selective geometry 重校准后阈值=`1.5`；test coverage=`55.41%`、selective accuracy=`95.56%`、错误接受率=`4.44%` | `outputs/lilocbench/AVL-014/AVL-012_selective/evaluation.json` |
| AVL-013 | Closed-loop Reject & Recovery | `revalidated with limits` | v3 test coverage=`54.23%`、success=`51.57%`、accepted accuracy=`95.10%`、最长 Reject=`26`；117/119 Reject 段恢复，2 个右删失 | `outputs/lilocbench/AVL-014/AVL-013_v3/evaluation.json` |
| AVL-015 | Reliability–Recovery Frontier | `completed with limits` | Q1/Q2/Q3 已完成：动作存在理论互补与固定动作 Pareto 空间；source-specific verifier 仅提供有限校准证据；Historical Keyframe Relocalization 未形成稳定独立恢复能力；Agent 仍 Hold/No-Go | `outputs/lilocbench/AVL-015/{pose_oracle,source_verification,dynamic_retrust,historical_keyframe_memory}/` |
| AVL-014 | AVL-010/011/012/013 implementation revalidation | `completed with limits` | 修正 Sim(3)、selective geometry、recovery aggregation、state trace，并完成 jump-guard 配对消融；Agent 继续 No-Go | `outputs/lilocbench/AVL-014/` |
| Stage 3B--G | 真实 GPU 闭环、稳定性、成对分歧与 terminal alignment | `completed / closed` | 220 帧中只有5个最终分歧；legacy rule 相对 F2 的正确帧差在 terminal alignment 后由 `+3` 变为 `-1`。冻结 F2/rule/verifier/backends，停止同 split controller/Geometry 调参 | `outputs/lilocbench/stage3f_terminal_aligned_feedback_r1/stage3_closeout.json` |

> 2026-10-12 方法决策更新：项目由“立即实现完整 Agent 的 No-Go”调整为“允许受控 Agent 原型研究，论文收益仍未证明”。Agent 被重新定义为多轮主动取证策略，不是固定 pose 输出池上的一次性 estimator router。完整路线见 `docs/visual_localization_agent_roadmap.md`。

> 2026-10-14 Stage 4A 实现启动：已完成 failure-triggered episode miner、scene-id
> split 审计、episode-sharded sequential runner 和可实现价值 Gate evaluator。Stage 3G
> 100--154帧 trace 共生成28个 recovery episode，但 LILocBench 四条序列共用一个
> 物理场景，故 manifest 标记 `scene_disjoint_ready=false / audit_only`。CPU 和真实
> Reloc3r/RoMa 两帧 depth-2 smoke 已通过，不构成 Gate 结论。产物位于
> `outputs/lilocbench/active_evidence_value_v1/`。

### 当前统一判定

- 所有在线策略均禁止读取 query GT、未来帧或事后 GT reference；GT 只出现在独立 evaluator、审计和离线标签中。
- AVL-005-B 的成本结论以统一 stage accounting 为准；Reloc3r 项为每序列 8 帧样本缩放估计，不能写成逐帧实测。
- AVL-004 的 HLoc 结果只能称 **HLoc-style**，尚无审计过的标准 COLMAP-HLoc 3D map。
- AVL-007 至 AVL-015 的历史结果不支持直接宣称 Agent 收益。2026-10-12 起总决策调整为 **Go for controlled Agent prototype / No-Go for paper-level Agent claim**：允许建设交互环境和最小训练原型，但必须在同工具、同 verifier、同预算下超过规则与轻量策略后，才能提升论文 claim。
- 2026-10-14 Stage 3G 已完成成对分歧和 terminal-aligned 复验。learned policy
  没有在成功率上稳定超过 feedback-rule，而 rule 旧有的微弱优势主要受自动
  Accept 接口影响。F2 仅保留为 safety--cost Pareto baseline，Stage 3 正式收口；
  后续只进行 failure-triggered、scene-disjoint 的主动取证可实现价值验证。
- 后续所有新实验必须在本登记册追加新编号或明确的补充批次，不能覆盖以上协议、失败尝试或无效结论。

---

## AVL-001 — Reloc3r 静态基线与评测完整性审计

- **状态 / 登记日期**：`planned` / 2026-10-09
- **Claim**：在不引入 Agent、Memory、HLoc 或 RoMa 的条件下，当前 Reloc3r 评测可产生坐标系和 GT 一致、可重复的静态室内基线；仅作为后续动态实验的比较锚点。
- **数据与划分**：优先 7Scenes 官方 train database / test query；可用时以 Cambridge 官方 train / test 作跨域检查。不得将 query、测试序列或未来帧加入 database。结果中须追加数据版本、实际路径、场景与样本数。
- **方法 / baseline**：原始 `eval_visloc.py` 与 `Reloc3rVisloc.motion_averaging`；Top-k=10；Reloc3r-512；不修改权重。此固定链路是后续 Agent 的基线。
- **seed / 环境**：数据定义默认 seed=777（须记录实际传参）；结果中记录 Python、PyTorch、CUDA、GPU、checkpoint revision 与 Git commit。
- **指标**：每场景及总体的 median translation error（m）、median rotation error（deg）、明确阈值的成功率、每 query 端到端延迟及失败样本清单；不得只报中位数。
- **预注册判定标准**：合格需每个样本唯一匹配 GT、database/query 坐标系一致、两次运行逐样本结果在浮点容差内一致。发现 GT 作为推理输入、query/test 帧进入 database、错误引用 `pair_id` GT、样本静默跳过、或 checkpoint/数据不可追溯时，实验为 `invalid`，不得使用误差数值。
- **数据泄漏 / GT 审计**：检查 `pose-db` 仅来自 train/database GT，`pose-gt` 只用于评估；检查 query/database 路径、帧 ID、序列 ID 不交叉；检查 motion averaging 不读取 query GT；抽查至少 20 个样本的变换方向、坐标系与输出命名。单独记录 `GreatCourt seq=4 frame=73` 的无 GT 跳过及影响。
- **配置 / 日志 / 结果 / Git**：待运行。预登记 commit：`568a9d385a10c5458cb93ab345271a09bda909db`。命令、配置、标准输出、逐样本结果与审计表须在运行后追加实际路径。
- **结果、结论与替代解释**：尚无结果，不得推断基线性能。即便审计通过，静态成绩也不能支持动态鲁棒性或 Agent 有效性。
- **下一步决策**：合格后登记 AVL-002（动态遮挡/场景变化评测与无 GT Context 信号基线）；无效则先修复并登记新的运行版本。

---

## AVL-002 — LILocBench 上 Reloc3r 的四类失效体检

- **状态 / 登记日期**：`completed` / 2026-10-09
- **Claim**：固定的「mapping reference + NetVLAD Top-k retrieval + Reloc3r-512 + motion averaging」在 LILocBench 的 static、dynamics、long-term changes、long-term changes + dynamics 四种条件下，会呈现可量化且条件特异的失效形态。实验目的不是证明 Agent 有效，而是判定后续 Memory、Reference 路由和强几何工具各自是否有可解决的目标。
- **数据与顺序**：reference 始终为公开 `mapping` sequence 的 `camera_front/color`；query 依序为公开 GT 的 `static_0`、`dynamics_0`、`lt_changes_0`、`lt_changes_dynamics_0`。首阶段只运行 mapping → `static_0`；上一阶段的输入、GT 对齐和输出审计合格后才启动下一类。不得将任何 query RGB、query GT、query descriptors、query history 或未来 query 帧写入 database/检索索引。
- **本地可用性核对**：`/data2/liusy/datasets/LILocBench/` 有 `mapping.zip`、`static_0.zip`、`dynamics_0.zip`；`lt_changes_0` 与 `lt_changes_dynamics_0` 原始数据尚未发现。static 首跑所需公开 GT 已下载至 `data/lilocbench/gt/`：mapping 16204 行、SHA-256 `dfc8b3c4abf717abdd9826ba3ca67978cc00c8986ef038390cc1ac3ad73efd2a`；static_0 11929 行、SHA-256 `ff98cb1e423793a45bba7a3c95472a3e8daf6d0c54906b53c048cfd8d787acf6`。
- **相机与坐标协议**：仅用前置 RGB。官方 GT 是全局 `T_world_base_link`（TUM 格式）；由每个 sequence 的 `transformations.yaml` 合成固定 `T_base_link_camera_front_color_optical_frame`，得到评估相机位姿。mapping 的 GT 相机位姿只作为构建带全局坐标的 reference database；query GT 严禁进入检索、相对位姿、候选聚合或任何在线决策。
- **抽样 / 匹配 / 方法**：RGB 以 1 Hz 的时间戳均匀抽样，GT 以最近邻时间戳匹配，最大容许差 30 ms；超阈值样本显式列为缺失并使该阶段审计失败，不能静默丢弃。Top-k=10；固定 Reloc3r-512、NetVLAD 检索和既有 motion averaging；无 Agent、Memory、HLoc、RoMa、query-to-query 传播或学习式重排。随机种子 777；记录所有模型 revision、硬件和依赖版本。
- **主要指标**：逐帧平移误差（m）和旋转误差（deg）；主成功率为 `(t≤0.25 m 且 r≤5°)`，并报告 `(0.5 m,10°)` 与 `(1 m,15°)` 敏感性；失败连续段的帧数和秒数（均值、中位数、P95、最大）；trajectory jump 为相邻预测相对运动与相邻 GT 相对运动之差，报告平移/旋转分布及 `>1 m 或 >20°` 比率；Top-k 候选绝对位姿的成对平移/旋转离散度；k∈{1,3,5,10} 输出之间的位姿差及其成功率变化。结果必须按四类条件、序列和时间顺序报告，而非只报总体均值。
- **预注册模式判据**：static 若稳定而某非静态条件出现显著更长失败段、更多 jump 或更高候选冲突，仅作为「条件相关失效模式」报告，不作因果断言。动态中的短暂失败段是未来遮挡诊断/Reference 路由候选；LT 中空间集中的持续失败是未来 Memory/地图更新候选；Dynamic+LT 中高候选离散度并伴随错误是未来验证/强几何升级候选。只有这些模式在该条件内相对于 static 同时通过误差、时序和候选证据，才可作为下一阶段动作设计依据。
- **GT / 数据泄漏审计**：保存 database/query 清单及 SHA-256；确认数据库只含 mapping 帧，四个 query 集互不作为 reference；静态外参仅来自各自公开 YAML；记录每幅 RGB 与 GT 的时间差；检查所有推理输入与日志不含 query GT。用 20 个随机 mapping/query 样本独立复算 `T_world_camera`、相对位姿方向和最终绝对位姿；任何坐标系/时间错配使受影响阶段为 `invalid`。
- **配置 / 日志 / 结果 / Git**：预登记 commit `568a9d385a10c5458cb93ab345271a09bda909db`；输出根目录 `outputs/lilocbench/AVL-002/`。static 输入由 `tools/prepare_lilocbench.py` 以 1 Hz 从原始压缩包抽取：mapping 785 帧、archive SHA-256 `3f5d17cda7f043d2237271c9ff6c7c78034666c1fda4bea0de26c901d53f782b`；static_0 578 帧、archive SHA-256 `d260120a45fe6d3f7c9d2d10a26ba5ef2e4dbbe2f4f22155926ea015e50eecf9`。实现为 `tools/eval_lilocbench_reloc3r.py`；实际命令、环境、逐帧候选、结果表和日志将在阶段结束后追加。当前未产出定位结果。
- **运行附录（追加，不覆盖协议）**：2026-10-09 的 static retrieval 尝试在第一个 descriptor 前因实现错误停止：NetVLAD 接口收到 tensor 而非 `{"image": tensor}`。日志：`outputs/lilocbench/AVL-002/static_0/retrieval.log`。未生成 descriptor、检索、相对位姿或任何数值；该尝试无效且不支持结论。已修正接口，后续从头重跑。
- **运行附录（追加，不覆盖协议）**：修正后，mapping→static_0 检索在 GPU 1 完成，生成 `retrieval.npz`（578 query × Top-10 mapping reference）及 `retrieval_manifest.json`；查询 GT 未参与该阶段。GPU 0 的一次运行发生系统页锁等待、无 descriptor 输出，已中断并保留日志 `retrieval_rerun_1.log`；GPU 1 成功日志为 `retrieval_rerun_2_gpu1.log`。Reloc3r 位姿阶段尚未产生输出：默认 Hugging Face cache 只含 config、未发现完整 `siyan824/reloc3r-512` 权重；自动下载在用户要求复用既有权重时被中断。实验暂停，等待确认既有 checkpoint 路径或明确的模型变更；无定位数值或科学结论。
- **运行附录（追加，不覆盖协议）**：确认用户要求的是切换 Codex 对话模型，不是更换定位器；AVL-002 仍固定使用同一 Reloc3r-512。只读检查显示本机没有可复用的完整 checkpoint，因此恢复官方权重获取；不改变数据、模型或指标协议。
- **运行附录（追加，不覆盖协议）**：用户确认同一官方权重位于项目级缓存 `/data2/liusy/reloc3r/.cache/huggingface/hub/models--siyan824--reloc3r-512/snapshots/55014aae1798ec830df51d7d2cd47c41f4388187/`；`model.safetensors` 1,706,156,328 bytes（7Scenes 运行所用格式），`Reloc3r-512.pth` 1,706,323,002 bytes。此前“缺少权重”的判断仅因检查了默认 `~/.cache/huggingface`，已更正；不改变模型。设置 `HF_HOME=/data2/liusy/reloc3r/.cache/huggingface`、`HF_HUB_OFFLINE=1` 后，`Reloc3rRelpose.from_pretrained('siyan824/reloc3r-512')` 在 GPU 1 成功加载，参数量 426,523,916。static pose 已于 2026-10-09 启动 8 个分片（GPU 0–7，Top-10，batch=2，AMP），会话/日志为 `outputs/lilocbench/AVL-002/static_0/pose_shard_00.log`–`pose_shard_07.log`；尚无定位数值结论。
- **协议修订（2026-10-09，分析前追加）**：严格 30 ms 最近 GT 审计发现 RGB 与公开 20 Hz GT 的启动边界/时钟抖动导致 mapping 6/785、static_0 2/578 帧超阈值；最大差分别 97.1 ms、78.1 ms，中位差约 0.1 ms。原始 30 ms 规则下该阶段应判为审计失败。为保留完整 1 Hz 序列并显式量化同步影响，本次 static 分析采用统一 100 ms 最近 GT 容许差；所有 GT 仍只在分析阶段使用，逐帧 `gt_delta_s` 与超 30 ms 清单保留。后续四类条件沿用同一 100 ms 规则，并另报告严格 30 ms 子集敏感性；不得把放宽阈值当作模型性能收益。
- **static 结果（2026-10-09，追加）**：mapping→`static_0` 的 578/578 个 query pose 分片均完成，结果在 `outputs/lilocbench/AVL-002/static_0/summary.json`、`frame_diagnostics.jsonl`、`frame_metrics.csv`、`failure_runs.json`。100 ms 分析：median translation `0.8967 m`、median rotation `0.4187°`；成功率 `(0.25 m,5°)=39.79%`、`(0.5 m,10°)=44.46%`、`(1 m,15°)=50.17%`；失败段 69 个，median 4 帧、P95 12 帧、最大 26 帧/25.8 s；trajectory jump `>1 m or >20°` 比率 56.67%；Top-10 candidate dispersion median `1.2441 m / 1.3267°`。严格 30 ms query 子集 576 帧，Top-10 成功率 39.76%、median translation 0.9243 m，结论不由 2 个同步边界帧驱动。Top-k 敏感性：k=1/3/5/10 成功率分别 14.36%/88.41%/76.12%/39.79%；这显示 static 条件下 Top-10 motion averaging 会被候选冲突显著拉坏，不能描述为“Reloc3r 静态稳定”。
- **static 审计与替代解释（追加）**：pose 推理阶段未读取任何 GT；retrieval 仅使用 mapping reference，578×Top-10 输出为 `retrieval.npz`。一个已有 7Scenes pair 的本实现 q2d 与已生成 q2d 最大元素差 0.0235、平移差 0.0326 m，支持输入预处理/权重调用与既有链路一致。static 的低 Top-10 成功率首先支持“reference Top-k 稀释/候选冲突”失效模式；仍不能把它归因于动态或长期变化，也不能排除 LILocBench mapping/static 外观差异、NetVLAD 排名或 motion averaging 的鲁棒性不足。下一步先保留 Top-k 分层证据，再按登记顺序运行 dynamics。
- **dynamics 阶段启动附录（2026-10-09）**：官方公开 GT `gt_poses_public.zip` 已保存为 `data/lilocbench/gt_public/ref_data_public.zip`，SHA-256 `c869d05f6bb247635ae409f16400b34bdc6309771e3dff3180725d79e3e8f4a9`；其中 `dynamics_0/gt_poses.txt` SHA-256 `606f9392c1a992c9acdb5e680a9108ef7647ba8943d747435c90dfa56feaa38f`。`dynamics_0.zip` SHA-256 `0a1016e7e1759c1a67be8170f6d232ce0e302c22ec16c2bd9fc03dcca1a1b34c`，以同一 1 Hz/front RGB 规则准备出 155 帧。dynamic 阶段现已具备审计输入，下一步先运行 mapping-only NetVLAD Top-10 retrieval，再按同一 8-GPU Reloc3r-512 分片和 100 ms GT 分析规则执行；无 Agent/Memory/HLoc/RoMa。
- **dynamics 结果（2026-10-09，追加）**：`dynamics_0` 155/155 帧 pose 分片完成，结果在 `outputs/lilocbench/AVL-002/dynamics_0/summary.json`、`frame_diagnostics.jsonl`、`frame_metrics.csv`、`failure_runs.json`。100 ms 分析：median translation `3.9372 m`、median rotation `1.3013°`；成功率 `(0.25 m,5°)=13.55%`、`(0.5 m,10°)=15.48%`、`(1 m,15°)=22.58%`；失败段 10 个，median 11 帧、P95 29.55 帧、最大 30 帧/30.28 s；trajectory jump `>1 m or >20°` 比率 82.47%；Top-10 candidate dispersion median `5.1891 m / 35.9711°`。严格 30 ms query 子集 154 帧，Top-10 成功率 13.64%、median translation 3.9638 m，结论不由 1 个同步边界帧驱动。Top-k 成功率 k=1/3/5/10 为 10.32%/43.23%/29.68%/13.55%。相对 static，dynamic 的误差、连续失败和候选冲突均明显恶化，支持“行人/动态外观导致参考候选冲突和连续失效”的条件相关模式；尚不能区分遮挡、检索错误或 Reloc3r 几何退化的单独因果贡献。
- **LT 与 Dynamic+LT 阶段登记附录（2026-10-09，运行前）**：两份原始压缩包已下载并通过 `unzip -t` 完整性检查。`lt_changes_0.zip`：10,008,762,359 bytes，SHA-256 `32cfe42e5ae7cf73f2a40acdfae5c756a0351357002b86bc107b42ffd0e7040b`；`lt_changes_dynamics_0.zip`：12,940,694,854 bytes，SHA-256 `98c42aa5f22e839749c6221e3c9ed63ffd460a66b28a66b7003f29c186640ee9`。公开 GT：`lt_changes_0_gt_poses.txt` SHA-256 `32c1f451eced24dbd4891771c113aaf2dccfb9d67abf54036c3a2b0f42bcb461`；`lt_changes_dynamics_0_gt_poses.txt` SHA-256 `277699961867b9d5d3bcaa14de1e577bf7ef9e6737b6f971460231461ff1d2d0`。协议与前两类完全一致：mapping-only reference、front RGB、1 Hz、Top-10、Reloc3r-512、8-GPU pose shards、无 Agent/Memory/HLoc/RoMa；沿用 100 ms 最近 GT并报告严格 30 ms 子集。输出目录分别为 `outputs/lilocbench/AVL-002/lt_changes_0/` 与 `outputs/lilocbench/AVL-002/lt_changes_dynamics_0/`。
- **LT 与 Dynamic+LT 准备结果（追加）**：`tools/prepare_lilocbench.py` 已从上述压缩包抽取 1 Hz 前置 RGB：`lt_changes_0` 422 帧，`lt_changes_dynamics_0` 541 帧；各序列 manifest、标定和 source archive hash 保存在 `data/lilocbench/prepared_1hz/`。下一步先分别完成 mapping-only retrieval，再启动两组 8-GPU pose shards。
- **LT 结果（2026-10-09，追加）**：`lt_changes_0` 422/422 帧 pose 分片完成，结果在 `outputs/lilocbench/AVL-002/lt_changes_0/summary.json`、`frame_diagnostics.jsonl`、`frame_metrics.csv`、`failure_runs.json`。100 ms 分析：median translation `1.8915 m`、median rotation `0.6085°`；成功率 `(0.25 m,5°)=24.88%`、`(0.5 m,10°)=28.67%`、`(1 m,15°)=36.97%`；失败段 36 个，median 6 帧、P95 25.25 帧、最大 37 帧/37.23 s；trajectory jump `>1 m or >20°` 比率 66.27%；Top-10 candidate dispersion median `2.5856 m / 3.6681°`。严格 30 ms query 子集 421 帧，Top-10 成功率 24.70%、median translation 1.8921 m。Top-k 成功率 k=1/3/5/10 为 16.11%/69.67%/57.35%/24.88%。相对 static，LT 的持续失败和候选冲突明显增加；该模式支持“场景变化导致参考失配/候选冲突”，但仍不把原因唯一归于家具或门的具体类别。
- **Dynamic+LT 结果（2026-10-09，追加）**：`lt_changes_dynamics_0` 541/541 帧 pose 分片完成，结果在 `outputs/lilocbench/AVL-002/lt_changes_dynamics_0/summary.json`、`frame_diagnostics.jsonl`、`frame_metrics.csv`、`failure_runs.json`。100 ms 分析：median translation `1.6810 m`、median rotation `0.7762°`；成功率 `(0.25 m,5°)=23.48%`、`(0.5 m,10°)=30.31%`、`(1 m,15°)=37.52%`；失败段 50 个，median 3.5 帧、P95 29.75 帧、最大 37 帧/37.16 s；trajectory jump `>1 m or >20°` 比率 69.81%；Top-10 candidate dispersion median `3.3858 m / 7.2691°`。严格 30 ms query 子集 540 帧，Top-10 成功率 23.52%、median translation 1.6876 m。Top-k 成功率 k=1/3/5/10 为 19.96%/56.19%/45.84%/23.48%。组合条件没有比 `dynamics_0` 更差，反而其中位误差较低；这不是鲁棒性反证，因为四个 query 是不同轨迹/时段，说明条件标签与难度并非严格单调，必须按序列/空间分段解释，不能只用总体均值断言“叠加必然最难”。
- **AVL-002 四条件阶段结论（追加）**：四类数据均完成同一 mapping-only、Reloc3r-512、Top-10 和 100 ms/30 ms-sensitivity 流程，未发现 query GT 进入推理。固定 Top-10 在 static/LT/dynamics/combo 的成功率为 39.79%/24.88%/13.55%/23.48%；candidate translation dispersion 为 1.244/2.586/5.189/3.386 m；jump 比率为 56.67%/66.27%/82.47%/69.81%。因此已确认两类可服务病因：静态也存在 reference Top-k 冲突；dynamic 明显放大瞬时冲突与长失败段；LT 支持空间/外观变化造成持续参考失配。combo 结果的非单调性保留为替代解释，下一步应先做逐区域/时间段 failure 分析，再登记 AVL-003 的 Reference/Memory/验证动作消融。
- **AVL-002 failure 分析协议（2026-10-09，运行前追加）**：对四类已完成结果做事后解释，不修改原始 pose/GT，也不把 GT 输入 retrieval 或 pose。逐帧附加：GT 空间网格、相邻 GT 运动（转弯/速度代理）、Top-1/Top-k reference 的 GT 空间距离（仅审计标签）、candidate dispersion、jump、图像 Laplacian/边缘统计、相邻帧变化和 query-to-reference 外观差。输出连续失败段标签、空间热点、时间分段统计和可视化 contact sheets。标签只称为 operational evidence（`reference_mismatch`、`candidate_conflict`、`trajectory_jump`、`turn_motion`、`low_texture_or_blur`、`appearance_change`），不把启发式标签当作行人或家具的真值；需人工检查代表帧后再写科学结论。输出根目录：`outputs/lilocbench/AVL-002/failure_analysis/`。
- **阶段决策（已完成）**：四类体检和事后 failure analysis 已完成；已登记并执行 AVL-003。主线不改变，后续动作以 AVL-003 的可恢复子群和残余失败为依据。

- **AVL-002 失效分析结果（2026-10-09，追加）**：在不改动四类原始 pose 结果的前提下，完成逐帧 GT 空间网格、相邻运动代理、Top-1/Top-k reference 的事后空间距离、图像统计、候选离散度/trajectory jump 和失败段 contact sheet。产物位于 `outputs/lilocbench/AVL-002/failure_analysis/`，实现为 `tools/analyse_lilocbench_failures.py`；GT 只在分析阶段读取，所有标签均标为 operational evidence。失败帧比例 static/dynamics/LT/combo 为 60.21%/86.45%/75.12%/76.52%。最长失败段分别为 26/30/37/37 帧。dynamics 的代表性 contact sheet（如 `dynamics_0_failure_0051_0080.jpg`）可见多人经过、近景遮挡和运动模糊，并伴随高候选离散度；LT 的 `lt_changes_0_failure_0239_0275.jpg` 可见纸箱、办公物件和门/走廊外观变化，失败段中 reference distance 或 candidate conflict 持续升高；combo 的 `lt_changes_dynamics_0_failure_0261_0295.jpg` 同时出现人员、家具/箱体和门区外观变化。该视觉检查支持“动态遮挡是部分长失败段的可见伴随因素”和“LT 失败与可见布置/外观变化并存”，但没有语义标注，不能宣称每个失败由行人或特定家具因果造成。
- **AVL-002 失效分析解释边界（追加）**：static 的长失败段也多为 candidate conflict/trajectory jump，说明静态条件本身存在候选聚合问题；dynamics 的高 reference mismatch 与连续失败更突出；LT 的失败段在空间上集中于变化后区域；combo 的空间/时间窗口同时含人员与长期变化，但总体不比 dynamics 更差。heuristic 标签不能替代真实遮挡/场景变化标注，下一阶段只把它们用于预注册的动作假设，不作在线真值。

人工复核记录：`outputs/lilocbench/AVL-002/failure_analysis/scene_review.md`；其中只把可见人员、纸箱/办公物件和门区变化报告为场景共现证据，不当作逐帧语义因果标签。

## AVL-003 — 不引入 Agent 的 Reference / Verification / Memory 消融

- **状态 / 登记日期**：`completed` / 2026-10-09
- **Claim**：在完全复用 AVL-002 的 mapping-only Top-10 检索和 Reloc3r 相对位姿候选、不增加 GPU 推理的前提下，候选选择、几何一致性验证和受限时序先验能否缓解已观测的 reference conflict、candidate conflict 与连续失败；该实验用于界定未来 Agent 的最小动作边界，不宣称 Agent 收益。
- **数据 / 输入**：四个已完成 query 条件，mapping GT 只用于把已有相对位姿转成候选绝对位姿；读取 `outputs/lilocbench/AVL-002/*/retrieval.npz` 和 `relative_poses_shard_*`。query GT 仅在最终评估阶段读取，不能参与选择、阈值或 memory 状态。seed=777（本实验无随机采样）。
- **预注册方法**：`top1/top3/top5/top10` 原始检索前缀；`consensus3/5/10` 以候选绝对位姿的平移(1 m) + 旋转(10°)归一化距离选几何 medoid 邻域；`verify10_t1.0_r20` 保留距 medoid ≤1.0 m 且 ≤20° 的候选，少于 2 个时回退；`memory3` 仅使用该方法上一帧自身输出作为先验，在 ≤1.5 m 且 ≤30° 的候选中选最多 3 个，否则回退 consensus3，禁止 GT 和未来帧。`oracle_candidate` 和 `oracle_best_prefix` 显式只作 GT upper bound，不作为可部署方法。
- **指标 / 判定**：每序列及总体的 median translation/rotation、成功率 `(0.25 m,5°)`，另报告 fallback 率；按相同四条件比较 Top-10 基线。预注册支持标准：若某方法在不降低 static 成功率超过 5 个百分点的同时，使 dynamic/LT/combo 的成功率或最长失败段至少一项改善 ≥5 个百分点/帧，则作为候选动作进入 Agent 设计；否则只保留诊断，不将微小差异解释为收益。oracle 仅用于衡量候选生成上限。
- **验证 / 泄漏审计**：选择函数不接收 query GT；输出逐帧 `selection_uses_gt=false`。检查 mapping pose 仅来自 mapping GT，query GT 只进入评估分支的误差计算；oracle 单列并标记 `upper_bound_only=true`。配置、逐帧结果和摘要输出到 `outputs/lilocbench/AVL-003/`，脚本为 `tools/run_avl003_ablations.py`。
- **下一步决策**：结果用于决定 Reference 路由、独立候选验证和 Memory 哪些应进入最小闭环；若只有 oracle 有效而可部署选择无收益，则优先改进候选生成/强几何备用，不提前实现复杂 Agent。

- **运行结果（2026-10-09，追加）**：脚本 `tools/run_avl003_ablations.py` 复用 AVL-002 候选完成四序列评估，逐帧结果与摘要在 `outputs/lilocbench/AVL-003/{static_0,dynamics_0,lt_changes_0,lt_changes_dynamics_0}/`。所有可部署选择逐帧标记 `selection_uses_gt=false`；oracle 单列 `upper_bound_only=true`。成功率 `(0.25 m,5°)` 如下：

  | 方法 | static | dynamics | LT | dynamic+LT |
  |---|---:|---:|---:|---:|
  | Top-10 baseline | 39.79% | 13.55% | 24.88% | 23.48% |
  | consensus-3 | 82.70% | 55.48% | 75.12% | 58.60% |
  | verify-10 (1 m,20°) | 87.02% | 52.26% | 75.59% | 58.23% |
  | memory-3（仅内部一致历史） | **91.18%** | **58.06%** | **81.52%** | **62.85%** |
  | oracle candidate（GT upper bound） | 89.27% | 65.16% | 79.86% | 75.60% |

  `memory3` 的内部可靠历史门控只使用候选几何（选中候选 median dispersion ≤1.5 m、≤15°），无 GT；fallback 率为 10.0%/28.4%/17.5%/25.0%。Top-10 的最长失败段 26/30/37/37 帧，memory-3 降至 4/15/9/21 帧；总体加权成功率由 28.48% 提升至 77.30%。
- **AVL-003 结论（追加）**：固定 Top-10 的主要损失可被不增加 Reloc3r 推理的几何候选选择/验证大幅恢复，说明 reference/candidate conflict 是真实且可操作的病因；memory-3 在四类均优于 Top-10，并在 static/LT 显著优于纯 consensus，但 dynamic+LT 仍有 21 帧长失败段。oracle 与可部署方法仍有差距（尤其 dynamic+LT），保留强几何/更好 reference 作为后续 Agent 动作候选。该结果不是 Agent 整体收益证明，也不能把 heuristic memory 解释为语义记忆。
- **Agent 结构决策（追加）**：将候选/参考可靠性选择作为主动作，verification/拒绝作为安全门，内部一致的短期 memory 作为二级恢复动作，Manager 按 dispersion、jump、外观变化和历史可靠性调度；不先引入 LLM。由于 `consensus3` 复用了 Top-10 Reloc3r 候选，它证明的是聚合/选择价值，而非前置 reference 检索的计算节省；前置 reference 选择必须另做预算对齐实验。
- **实现修正（追加）**：初版 AVL-003 memory 版本未正确阻止不一致历史更新，结果未用于结论；修订后以内部候选离散度门控并从头重跑，当前 `summary_all.json` 为有效版本。

---

## 后续编号保留

- `AVL-004`：Recovery + deterministic Manager（本轮登记如下）。

## AVL-004 — Recovery + deterministic Manager

- **状态 / 登记日期**：`running` / 2026-10-09
- **Claim**：在 AVL-003 的 consensus/memory 仍不可靠的帧上，独立局部几何工具是否能恢复 Reloc3r 无法恢复的位姿；在此基础上，固定阈值 Manager 是否能在不读取 GT 的情况下调度 `accept_consensus`、`use_memory`、`call_geometry` 或 `reject`，并形成 Agent 的强 deterministic baseline。
- **阶段 A：强几何可行性**：优先测试 Dynamic+LT 的 memory-3 失败池，另报告四序列统一触发池。触发池必须由在线证据冻结（候选 dispersion、当前预测跳变、appearance/frame change、memory reliability），不能用 GT 选择要运行的帧；GT 只用于事后分层。HLoc 1.4 + pycolmap 使用 SuperPoint/LightGlue 或 RoMa 产生 2D 对应；若没有可审计的 3D reconstruction，则报告为“2D 几何 recovery feasibility”，不得冒充标准 HLoc 3D-PnP。RoMa 使用官方 `romatch` 包与固定权重；若权重/包不可用，记录环境 No-Go，不用未验证模型替代。
- **阶段 B：Manager baseline**：输入只包含当前候选 translation/rotation dispersion、候选 consensus/verification 结果、由预测位姿计算的历史 jump、图像 frame/appearance change、memory 是否由先前内部一致帧更新；输出四类动作。固定阈值在运行前冻结：consensus 内部 dispersion ≤1.5 m/15°；memory 距离门 ≤1.5 m/30°；high-risk jump 为预测相邻运动 >1 m 或 >20°；appearance/frame change 采用已有无 GT 图像差阈值 0.15。Manager 不接受 `query_gt_pose`、误差、成功标签、GT spatial label 或未来帧。
- **阶段 C：Recovery 路由**：先运行 Manager dry-run 生成 action trace；再只对 `call_geometry` 池运行强几何，并将几何候选回灌同一 evaluator。必须与 Top-10、consensus-3、memory-3 比较；报告成功率、median t/r、最长失败段、拒绝率、geometry 调用率、总时间和每帧预算。
- **判定标准**：强几何在 no-GT 触发池中相对 memory-3 至少恢复 5 个百分点成功率或将最长失败段减少 ≥5 帧，且 static 不降低超过 5 个百分点，才进入 Manager 路由；Manager 还必须给出 risk–coverage 曲线，不能只报全覆盖成功率。若 geometry 仅在 GT 选择的失败子集有效，标记为 oracle/分析结果，不作在线收益。
- **泄漏审计**：运行进程、工具输入、缓存和 action trace 中不得出现 query GT；Memory 只可读 `t'<t` 的已接受状态；阈值、触发池和工具权重在结果前冻结；mapping 绝对位姿作为地图资产可用，但不得把 query GT 转成 reference 或在线状态。保存环境 package lock、权重 hash、触发 frame IDs、action trace、失败日志和 Git 状态。
- **当前环境风险**：`/data2/liusy/conda_envs/hloc` 可导入 `hloc`/`pycolmap`/`lightglue`，但当前无 `romatch`；标准 HLoc 3D reconstruction 尚未从 LILocBench RGB 建立；这些是实验前置条件，不能静默绕过。
- **配置 / 输出**：预期输出根目录 `outputs/lilocbench/AVL-004/`；Manager 脚本为 `tools/run_avl004_manager.py`，强几何脚本/适配器须在环境审计通过后登记实际路径。
- **下一步决策**：先完成 romatch/HLoc 环境与权重审计，再做小规模 pair-level smoke；通过后冻结触发池并运行 recovery，最后评估 Manager。任何一阶段失败都保留 `invalid`/`paused` 记录，不修改 AVL-002/003 结论。

- **阶段 B 运行结果（2026-10-09，追加）**：`tools/run_avl004_manager.py` 在四类序列完成无 GT dry-run，阈值按协议冻结，逐帧 trace 在 `outputs/lilocbench/AVL-004/manager_dryrun/*/manager_trace.jsonl`，汇总为 `summary_all.json`。动作计数（accept / memory / geometry / reject）及覆盖率如下：static `161/219/198/0`、65.74%；dynamics `3/22/130/0`、16.13%；LT `67/112/243/0`、42.42%；dynamic+LT `26/134/381/0`、29.57%。`reject=0` 是当前规则在四序列上的实际触发结果，不是删除该动作；说明阈值仍偏向升级而不是拒绝。独立评估 `tools/eval_avl004_manager.py` 读取 GT 仅用于事后评分：覆盖帧的成功率分别为 93.95%、92.00%、93.85%、88.13%，但 geometry 尚未回灌位姿，因此全帧成功率为 61.76%、14.84%、39.81%、26.06%。该表只作为 risk–coverage dry-run，不宣称 Manager 已完成恢复收益。
- **阶段 A 环境审计与 smoke（2026-10-09，追加）**：`reloc3r` 环境安装官方 `romatch==0.1.2`，RoMa indoor 权重 `roma_indoor.pth` SHA-256 `4d3dca889ae1ef245123dc62aab914475c7bbf41f2c8002606450fb6cf2d91e6`（约 426 MB），DINOv2 权重 SHA-256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`（约 1.2 GB）；官方可选 `local_corr` CUDA 扩展缺失，故固定使用 `use_custom_corr=False` 的原生 PyTorch 实现，且记录 slow RoPE fallback。pair smoke 输出 `outputs/lilocbench/AVL-004/roma_smoke/combo_261.{json,npz}`：10,000 对应、confidence median=1.0、GT 未读取。HLoc 环境 `/data2/liusy/conda_envs/hloc` 为 HLoc 1.4 / pycolmap 0.3.0，可运行 SuperPoint+LightGlue；标准 HLoc 3D-PnP 仍因 LILocBench 没有审计过的 COLMAP 3D reconstruction 而不能直接声称已运行。
- **阶段 A recovery 结果（2026-10-09，追加）**：为避免把无 3D map 的标准 HLoc 误报为可用，先登记并运行“已知 mapping 相机位姿 + 2D 对应三角化 + query PnP”的 2D/3D feasibility。实现为 `tools/run_avl004_roma_recovery.py`（RoMa）和 `tools/run_avl004_hloc_recovery.py`（HLoc-style SuperPoint+LightGlue），后处理为 `tools/eval_avl004_recovery.py`；mapping pose 是地图资产，query GT 只在 evaluator 读取。Dynamic+LT 最大 21 帧 memory-3 失败段为 query `416–436`，其中 Manager trace 的 20 帧为 `call_geometry`、1 帧为 `use_memory`，memory-3 在 21 帧中为 0/21 成功。冻结 Top-10 retrieval 的前 4 个 reference 后，RoMa 输出 `outputs/lilocbench/AVL-004/roma_recovery/combo_416_436_top4_evaluation.json`：21/21 有 pose、9/21 成功（42.86%）、覆盖帧成功率同为 42.86%、median translation=1.0203 m、median rotation=15.3835°；在 20 个 geometry-call 帧中为 9/20，超过 memory-3 0/20，满足“恢复至少 5 个百分点”的局部触发池标准。HLoc-style 输出 `outputs/lilocbench/AVL-004/hloc_recovery/combo_416_436_evaluation.json`：19/21 有 pose、5/21 成功（23.81%）、covered success=26.32%、median translation=1.4629 m、median rotation=14.3035°。这些是触发池 feasibility 结果，不是四序列端到端收益，也不能替代标准 HLoc 3D-PnP。
- **阶段 A reference 因果审计（追加）**：query 416 的 frozen retrieval refs `[171,172,248,621]` 与该帧实际空间/朝向不兼容，RoMa recovery 误差为 15.84 m / 157.19°；仅作诊断的近邻 map refs `[31,32,33,34]`（由 GT 事后选取，明确 upper-bound，不可部署）可得 0.0248 m / 1.19°。该对照证明强几何恢复受 reference 选择强约束；不得把 GT 近邻结果计入 Manager。HLoc-style 在线 Top-10 内部匹配重排仍只恢复 5/21，显示需要独立的 Reference Agent/更大预算检索，而不是简单调用备用工具。
- **阶段 A 阶段结论与下一步（追加）**：RoMa 在冻结 retrieval reference 上相对 memory-3 的 21 帧触发池达到预注册的局部恢复门槛，但恢复仍依赖 reference 可见性，且当前只是 metric-map feasibility；HLoc 3D 路线暂停至建立可审计 reconstruction。下一步保持主线不变：将 reference re-selection 纳入 Manager 的预算动作，先用无 GT 的匹配/几何证据做 top-N 重排，再做四序列统一触发池；继续报告 reject/risk–coverage，不能把当前局部结果写成 Agent 总体收益。
- **阶段 A.2 Reference re-ranking（运行前登记，2026-10-09）**：Claim：在不读 query GT、只增加固定匹配预算的前提下，把 NetVLAD frozen Top-10 扩展到 Top-100，并用 SuperPoint+LightGlue 的匹配数/Essential-RANSAC 内点数重排，能否找到 Reloc3r Top-10 漏掉的可几何恢复 reference。数据先限定 Dynamic+LT query `416–436`，mapping-only reference；Top-100 由独立 NetVLAD retrieval 产生，重排最多读取 100 个 mapping RGB，最终只选 4 个 ref 做三角化+PnP。工具为 `tools/eval_lilocbench_reloc3r.py` 的 Top-100 retrieval 入口、`tools/run_avl004_hloc_recovery.py` 的无 GT 匹配重排和 `tools/eval_avl004_recovery.py` 的事后评分；不得使用 GT 选择/排序。
- **预算与判定**：记录每帧检索、匹配、PnP 时间和 reference 数；与 frozen Top-10→RoMa/HLoc-style 的 10/4 reference budget 对照。主要指标为 pose availability、成功率 `(0.25 m,5°)`、median t/r、错误高置信比例和最长失败段。若 Top-100 rerank 在同一 21 帧中相对 frozen Top-10 RoMa 的 9/21 成功率提升 ≥5 个百分点，且 trace 显示 `selection_uses_gt=false`，则把 re-ranking 作为 Manager 的候选动作；否则只作为诊断，不宣称 reference Agent 有效。
- **泄漏审计与状态**：query GT 不进入 retrieval、matching、排序、PnP 或 Manager；只允许 mapping pose 作为 metric map asset。保留完整 Top-100 indices/scores、每帧排序证据、失败日志和环境版本。该实验仅是 21 帧 feasibility，不代表四序列总体收益；完成后追加结果并决定是否扩展到四序列。
- **阶段 A.2 结果（2026-10-09，追加）**：`tools/retrieve_lilocbench_subset.py` 生成 `outputs/lilocbench/AVL-004/reference_rerank/combo_416_436_top100.npz`（21×100 mapping-only indices/scores，`query_gt_read=false`）。HLoc-style SuperPoint+LightGlue 在 100 个 reference 上按 Essential-RANSAC 内点数重排，最终固定 4 个 ref 三角化+PnP，输出 `combo_416_436_hloc_top100_evaluation.json`：21/21 有 pose，13/21 成功（61.90%），median translation `0.2042 m`、rotation `2.3844°`。相对 frozen Top-10 HLoc-style 的 5/21（23.81%）提高 38.10 个百分点；相对 frozen Top-10 RoMa 的 9/21（42.86%）提高 19.05 个百分点。
- **阶段 A.2 RoMa 回灌结果（追加）**：使用同一份无 GT 重排出的 4 个 reference，RoMa 三角化+PnP 输出 `combo_416_436_roma_reranked_evaluation.json`：21/21 有 pose，13/21 成功（61.90%），median translation `0.1109 m`、rotation `1.5055°`。Manager route 回灌 `combo_416_436_roma_route_evaluation.json`：21/21 coverage，13/21 success；20 个 `call_geometry` 帧中 13/20 success（65%），剩余 1 个 `use_memory` 帧失败。Recovery 进程和 route 均标记 `query_gt_used_by_recovery=false`、`query_gt_used_by_route=false`、`memory_future_access=false`。
- **阶段 A.2 结论与限制（追加）**：在该预注册 21 帧触发池中，无 GT reference re-ranking 的收益明显大于单纯更换几何工具；它把“强几何救不回”中的一部分转化为“reference 不可见”。结果仍是局部 feasibility：Top-100 匹配成本尚未与 Reloc3r/Manager 全序列预算对齐，HLoc 仍是 metric-map 2D/3D feasibility 而非标准 COLMAP HLoc 3D-PnP，不能直接宣称四序列 Agent 收益。下一步才扩展到各条件的统一触发池并记录时间/预算。
- **阶段 A.3 四条件长失败段扩展（运行前登记，2026-10-09）**：Claim：A.2 的无 GT reference re-ranking 是否能跨条件恢复 Memory-3 的最长失败段，而不是只适用于 Dynamic+LT。固定触发池为每序列 Memory-3 最长连续失败段：static `5–8`（4 帧）、dynamics `33–47`（15 帧）、LT `413–421`（9 帧）、dynamic+LT `416–436`（21 帧）。每帧先使用 mapping-only Top-100 NetVLAD 候选，再以 SuperPoint+LightGlue Essential-RANSAC 内点数无 GT 重排，最终 4 refs；同一 4 refs 另跑 RoMa。Query GT 只能进入 `tools/eval_avl004_recovery.py`/route evaluator。
- **预注册预算/判定**：Top-100、4 selected refs、每帧最多 6 个 RoMa pair、与 A.2 相同阈值和 PnP；记录 retrieval/matching/geometry elapsed time、pose coverage、成功率、median t/r、最长失败段。跨四段加权成功率相对 frozen Top-4 RoMa 提升 ≥5 个百分点且四段均保持 `query_gt_read=false`，才把“reference re-ranking→geometry”作为跨条件 Manager 动作；否则仅保留条件特异结论。该实验不改变 AVL-002/003，不实现 LLM Agent。
- **阶段 A.3 结果（2026-10-09，追加）**：四个最长 Memory-3 失败段均完成 Top-100 retrieval、无 GT LightGlue 重排和 RoMa recovery。逐段 RoMa route 成功率为 static `3/4=75.00%`、dynamics `9/15=60.00%`、LT `3/9=33.33%`、dynamic+LT `13/21=61.90%`；加权 `28/49=57.14%`。对应 frozen Top-4 RoMa 对照为 `0/4`、`1/15`、`0/9`、`9/21`，加权 `10/49=20.41%`，提升 `36.73` 个百分点；HLoc-style 重排+PnP 加权为 `29/49=59.18%`。各段 pose coverage（RoMa）均为 100%，route trace 和 recovery 均标记无 query GT、无未来 Memory 访问。
- **阶段 A.3 结果路径与预算**：汇总 `outputs/lilocbench/AVL-004/reference_rerank/a3_four_condition_summary.json`；逐帧 Top-100 排序、HLoc-style 和 RoMa JSONL 同目录。RoMa 单帧 geometry elapsed 约 1.6–2.0 s（native PyTorch fallback，具体逐帧记录在 JSONL）；Top-100 LightGlue 重排的逐帧时间也已写入 `elapsed_s`。NetVLAD retrieval、100 个匹配和 4-ref PnP 的额外成本尚未与完整 Reloc3r 端到端延迟对齐，故该结果是跨条件 recovery feasibility，不是最终实时性结论。
- **阶段 A.3 决策（追加）**：预注册的跨条件成功门槛已满足，reference re-ranking→geometry 进入 deterministic Manager 的候选动作；但必须先做预算对齐和全序列 risk–coverage，不能直接进入 LLM Agent 或宣称端到端总体提升。LT 段仅 3/9 成功，说明 reference re-ranking 不是所有长期变化的充分条件，仍需保留 reject/Memory/备用工具分工。
- **Manager route metadata（追加）**：`tools/run_avl004_manager.py` 已支持冻结的 `geometry_reference_policy=top100_lightglue_rerank`，并在新 trace `outputs/lilocbench/AVL-004/manager_dryrun_rerank/` 中逐帧记录 reference budget=100、工具候选和 `selection_uses_gt=false`；动作计数与原 dry-run 一致。该改动只明确 geometry route 的预算，不改变 Manager 阈值或动作选择，尚未声称全序列 recovery 已完成。

以上任务均须在实际运行前补齐环境、权重、预算和日志路径；Agent 本身暂不实现。

- **阶段 B/A.4 风险—覆盖率与 deterministic Manager 审计（运行前登记，2026-10-09）**：Claim：在不读取 query GT 的前提下，AVL-004 固定阈值 Manager 的证据风险是否能形成可审计的选择性定位基线，并明确当前 `reject`/`call_geometry` 分工，而不是把 dry-run 的低覆盖误报为定位收益。输入为 `manager_dryrun_rerank` 中四序列冻结 trace；新增 trace 字段 `consensus_pose`/`memory_pose` 仅保存无 GT 候选，不改变动作决策。实现为 `tools/eval_avl004_risk_coverage.py`。
- **固定风险协议**：逐帧风险分数为 `max(top10_t/1.5, top10_r/15, consensus_t/1.5, consensus_r/15, jump_t/1, jump_r/20, appearance/0.15, frame/0.15, 1-memory_unreliable_before)`；阈值预先固定为 `{0.5,1,1.5,2,3,5,+∞}`，不使用 GT 拟合或选择。对每个阈值仅保留 risk≤threshold 的 consensus candidate，报告 coverage、成功率、median t/r、false-accept；另按 Manager 实际 action 报告 `accept_consensus/use_memory/call_geometry/reject` 的输出与候选上限。GT 只在 evaluator 读取。
- **判定与限制**：风险—覆盖率曲线若显示低风险子集具有更高选择性成功率，作为 deterministic Manager 的证据门槛；不得将 consensus/memory 候选在 `call_geometry` 帧上的事后分数称为 route recovery。当前 geometry 仅记录 Top-100 LightGlue re-ranking + RoMa/HLoc-style 候选动作，完整四序列回灌仍另行登记。所有输出保存在 `outputs/lilocbench/AVL-004/risk_coverage/`，并记录 `selection_uses_gt=false`、`gt_used_only_for_posthoc_evaluation=true`。
- **A.4 无效运行记录（2026-10-09）**：初次运行使用的汇总函数把 risk 子集自身作为 coverage 分母，导致曲线 coverage 恒为 1；该实现错误不影响逐帧 pose score，但该输出目录 `outputs/lilocbench/AVL-004/risk_coverage/` 的聚合结果不用于结论，保留作为 invalid artifact。
- **A.4 有效结果（2026-10-09，追加）**：修正分母后，Manager 以无 GT trace 重跑至 `outputs/lilocbench/AVL-004/manager_dryrun_rerank_v2/`，四序列共 1696 帧，所有 trace 均 `selection_uses_gt=false`、`test_gt_read=false`，且新增 `consensus_pose`/`memory_pose`。动作计数保持不变：static `161/219/198/0`、dynamics `3/22/130/0`、LT `67/112/243/0`、dynamic+LT `26/134/381/0`（accept/memory/geometry/reject）；因此 `reject=0` 是规则实际行为，不能被解释为已经验证拒绝策略。
- **A.4 风险—覆盖率结果（GT 仅事后评分）**：有效输出为 `outputs/lilocbench/AVL-004/risk_coverage_v2/summary_all.json`。固定风险阈值 1.0/1.5/2.0/3.0/5.0 的四序列加权 coverage/success-selected 分别为 `15.09%/96.09%`、`27.36%/94.61%`、`35.38%/91.33%`、`43.69%/89.34%`、`57.19%/87.22%`；全量 consensus candidate 为 `100%/70.64%`。按实际 action 输出的 coverage/success-selected 为 static `65.74%/93.95%`、dynamics `16.13%/92.00%`、LT `42.42%/93.85%`、dynamic+LT `29.57%/88.13%`。低风险子集成功率高于全量，支持证据风险可用于 selective prediction；但 Manager 当前主要把中高风险帧升级 geometry，完整 route 仍未回灌，故不能把这些数值称为端到端 Agent 收益。
- **A.4 决策（追加）**：deterministic Manager 已达到“可审计风险分层”门槛，但也暴露出 geometry 调用率过高、reject 未触发和 LT/combo 的低覆盖问题。下一步必须先完成小预算/全序列 route 回灌或明确的 geometry budget 对齐，再比较 `accept/memory/geometry/reject` 与固定强几何；在此之前不进入 LLM Agent。
- **AVL-004 A.5 全序列 geometry route（运行前登记，2026-10-09）**：Claim：对 deterministic Manager 触发的全部 `call_geometry` 帧，先用 mapping-only NetVLAD Top-100，再用 SuperPoint+LightGlue Essential-RANSAC 在 100 个 reference 中无 GT 重排并固定选 4 个；把同一 4-ref 集合分别交给 HLoc-style 2D/3D PnP 与 RoMa 三角化+PnP，能否在公平 reference budget 下恢复 Manager 覆盖率和准确率。Manager trace 固定为 `outputs/lilocbench/AVL-004/manager_dryrun_rerank_v2/`，动作不能重算或按 GT 改写。
- **A.5 预算与工具**：每个 geometry-call query 只允许 100 个 mapping reference 做 LightGlue 匹配，最终 4 refs；HLoc-style 使用三角化点+PnP，RoMa 使用同一 4 refs 的最多 6 对、每对固定 `num_matches=6000`、`max_intersections=1800`、`max_pnp_points=3000`。HLoc 与 RoMa 均保存逐帧 `elapsed_s`、reference scores、selected refs、inliers、reprojection 和 pose availability。使用 8-GPU 分片并行只改变 wall-clock，不改变单帧预算；RoMa 固定 `use_custom_corr=false` 原生 fallback。
- **A.5 预注册判定**：分别把 HLoc-style 或 RoMa pose 回灌 Manager route，与当前 dry-run、AVL-003 memory-3、以及固定 Top-4 RoMa 参考比较；报告四序列 coverage、success `(0.25m,5°)`、median t/r、最长失败段、geometry-call 成功率、误接受率、每帧延迟和总 wall-clock。若任一工具在四序列 geometry-call 池相对当前 consensus/memory fallback 的加权成功率提升 ≥5 个百分点且无 query GT，则进入 deterministic Manager 候选动作；若 HLoc/RoMa 差异小于 2 个百分点，保留“工具范式等价、reference 选择主导”的结论。不得把 geometry-call 帧的 query GT 用于 reference、PnP、阈值或 Memory。
- **A.5 审计与输出**：retrieval、HLoc、RoMa 运行进程均不得加载 query GT；仅 `tools/eval_avl004_route.py`/独立 post-hoc evaluator 读取 GT。保留失败/中断分片，不覆盖已有 A.3/A.4。预期输出根目录 `outputs/lilocbench/AVL-004/full_route_a5/`，协议命令、GPU、环境和日志在完成后追加；标准 HLoc 3D-PnP 仍不声称已运行。
- **A.5 运行审计与无效尝试（2026-10-09，追加）**：四类 geometry-call 触发池共 952 帧（static/dynamics/LT/combo=`198/130/243/381`）。mapping-only Top-100 retrieval 完成于 `outputs/lilocbench/AVL-004/full_route_a5/retrieval/`；HLoc-style 8-GPU 分片完成于 `.../hloc_shards/` 并严格合并为 `.../hloc/*_top100_merged.jsonl`。第一次 RoMa 启动因未设置项目级 `TORCH_HOME`，尝试访问默认缓存并未生成有效结果；日志和空/不完整目录保留在 `outputs/lilocbench/AVL-004/full_route_a5/roma_shards/`，该尝试不用于结论。设置 `TORCH_HOME=/data2/liusy/reloc3r/.cache/torch` 后，第二次 8-GPU 运行完成并写入独立目录 `.../roma_shards_localcache/`，合并结果为 `.../roma/*_top100_merged.jsonl`。所有合并审计均通过：行数严格为 198/130/243/381，`query_gt_read=false`、`selection_uses_gt=false`；GT 只被 `tools/eval_avl004_route.py` 和 `tools/analyse_avl004_recovery.py` 用于事后评分。
- **A.5 全序列 route 结果（2026-10-09，追加）**：统一回灌 `tools/eval_avl004_route.py` 后，四序列 route 的 success `(0.25m,5°)`（static/dynamics/LT/combo）为 HLoc-style `90.83/81.29/86.73/86.32%`，RoMa `92.91/87.10/86.97/87.62%`；加权 1696 帧分别为 `1484/1696=87.50%` 与 `1513/1696=89.21%`，完整 route coverage 分别为 `1689/1696=99.59%` 与 `1696/1696=100%`。相对 AVL-003 memory-3 的加权 `1301/1696=76.71%`，HLoc-style 和 RoMa 分别提高 `10.79` 与 `12.50` 个百分点；静态 route 已接近 memory-3（HLoc `90.83%`、RoMa `92.91%` 对 `91.18%`），动态/长期变化/组合条件的收益主要来自 geometry recovery。
- **A.5 geometry-call 池结果（GT 仅 evaluator）**：同一 952 帧触发池上，memory-3 为 `612/952=64.29%`，HLoc-style 为 `795/952=83.51%`，RoMa 为 `824/952=86.55%`；两种工具均达到相对当前 consensus/memory fallback 至少 5 个百分点的预注册门槛。各条件 geometry success（HLoc/RoMa）为 static `84.85/90.91%`、dynamics `79.23/86.15%`、LT `81.48/81.89%`、combo `85.56/87.40%`。RoMa 相对 HLoc 的 geometry-pool 优势为 `3.05` 个百分点，但全 route 优势为 `1.72` 个百分点，不能把两者称为完全等价；同一 reference 集合上 952 帧成功混淆为 both `764`、HLoc-only `31`、RoMa-only `60`、neither `97`，说明工具有互补性且仍有共同不可恢复帧。
- **A.5 误差、连续失败与 latency**：RoMa route 的最长连续失败段为 static/dynamics/LT/combo=`4/5/5/5` 帧，HLoc-style 为 `5/7/6/6` 帧；RoMa geometry-call pose availability 为 `952/952`，HLoc-style 为 `945/952`。RoMa 单帧 geometry elapsed 中位数/P95（秒）分别为 static `4.88/6.26`、dynamics `4.57/6.81`、LT `4.81/6.39`、combo `4.72/6.09`；HLoc-style 为 `20.20/31.30`、`19.21/30.93`、`19.75/28.02`、`18.76/30.23`。逐帧 elapsed 总和为 RoMa `4506.7 s`、HLoc-style `18921.0 s`；8 路分片输出文件时间跨度约为 RoMa `10.0 min`、HLoc-style `33.0 min`（并行 wall-clock 仅作运行记录，不与单帧 latency 混用）。geometry-call 的 route false-accept（覆盖帧中不满足阈值）仍为 RoMa `9.09/13.85/18.11/12.60%`、HLoc `15.15/17.60/18.18/14.21%`（static/dynamics/LT/combo），因此“有 pose”不能等价于“已验证可靠”。
- **A.5 困难段与 reference 结论**：Dynamic+LT query `416–436` 的 memory-3 仍为 `0/21`；在全序列同一 route 中 HLoc/RoMa 均为 `13/21=61.90%`，与 A.2 的重排结果一致，说明 Top-100→LightGlue re-ranking 能恢复大部分但不是全部困难段。四类工具对照均使用完全相同的 4-ref 集合（952/952 相同）；HLoc 与 RoMa 的 pose 差异中位平移约 `0.013–0.024 m`、中位旋转约 `0.32–0.51°`，但 P95 可达 `1.27–2.14 m`、`17.4–32.2°`，共同失败帧和尾部冲突仍明显。`outputs/lilocbench/AVL-004/full_route_a5/recovery_analysis.json` 保存逐序列、逐工具、agreement、latency、失败段及 GT 审计汇总。
- **A.5 结论与限制（追加）**：预注册的“geometry-call 池相对 memory/consensus fallback 提升至少 5 个百分点”门槛已满足；因此“无 GT reference re-selection→强几何”可作为 deterministic Manager 的候选恢复动作。证据不支持“RoMa/HLoc 单独解决所有失败”：仍有 97/952 共同失败帧，LT geometry success 仅约 82%，combo 416–436 仍有 8/21 失败；当前规则仍 `reject=0`，且 geometry 成本高、未建立审计过的 COLMAP map，HLoc 结果只能称 HLoc-style 2D/3D feasibility。初次 RoMa 缓存错误已作为 invalid artifact 保留，不覆盖；有效结论来自项目缓存重跑。该结果完成 A.5，但不构成 LLM Agent 收益证明。
- **A.5 阶段决策（追加）**：主线不变，停止在真正引入 Agent 之前。下一步只做充分的 deterministic 分析：固定 geometry budget 的 route 对照、reject/验证校准、工具互补与共同失败的逐帧/场景审计，以及 reference 变化和 latency 的公平比较；在这些分析完成并明确剩余不可由固定规则解释的决策价值前，不实现或训练 LLM/多 Agent。

## AVL-004 A.6–A.9：Agent 前四项终止性实验（运行前登记，2026-10-09）

- **共同约束**：四项实验只使用现有 mapping-only 数据、Reloc3r/AVL-003 输出、A.5 无 GT Manager trace 和 A.5 recovery 证据；GT 只能用于独立 post-hoc evaluator。所有阈值、预算和 action 定义先冻结，不用 GT 调参。结果不得写成 LLM Agent 收益。
- **A.6 Always-RoMa / Always-HLoc**：对四序列全部 1696 帧，不读取 Manager action，使用与 A.5 完全相同的 Top-100 mapping retrieval、LightGlue/Essen­tial-RANSAC reranking、4 refs、RoMa/HLoc-style 几何预算。报告全路线 coverage、success、median t/r、longest failure、每帧/总 latency，并与 A.5 Manager route、memory-3 和固定计算预算比较。输出根目录：`outputs/lilocbench/AVL-004/always_geometry/`。
- **A.7 Action Oracle**：对每帧读取已保存的 Reloc3r `consensus3`、`memory3`、A.5 HLoc、A.5 RoMa（以及 Manager route）结果，逐帧计算“任一候选成功”的 upper bound；只用于 post-hoc 上界，不参与任何推理。报告 oracle success、各动作单独贡献、oracle 与 deterministic route 的 gap，以及各条件/失败段分布。输出：`outputs/lilocbench/AVL-004/action_oracle/`。
- **A.8 97 个共同失败帧审计**：固定 A.5 中 HLoc/RoMa 均失败的 97 个 geometry-call query，保存逐帧 reference scores、selected refs、inliers、reprojection、candidate dispersion、appearance/frame-change、GT 空间标签和代表图像/失败类型；GT 空间标签只作为审计标签，不参与选择。按 `reference_visibility/mismatch`、`severe_occlusion_or_dynamic`、`long_term_structure_change`、`geometric_degradation`、`mixed/uncertain` 分类，允许多标签并保留不确定项。输出：`outputs/lilocbench/AVL-004/common_failure_audit/`。
- **A.9 Risk–cost–accuracy**：以 deterministic Manager 的 geometry-call rate / 实际 geometry latency / action budget 为横轴，以全帧 success、coverage 和 selective accuracy 为纵轴；比较当前 Manager、always-RoMa、always-HLoc、memory/consensus、固定 geometry-rate 预算和加入 reject 的 post-hoc selective policy。Reject 只允许使用预先冻结的无 GT 风险/工具证据阈值，不能用 GT 拟合。输出：`outputs/lilocbench/AVL-004/risk_cost_accuracy/`。
- **终止性 Go/No-Go**：若 deterministic Manager 在相同 geometry/latency 预算下接近 A.7 Action Oracle，停止引入 LLM Agent；若存在大量风险数值相近但视觉上下文决定不同动作的样本，且该差异可由可审计视觉证据解释，才进入 Agent 设计。四项完成前不实现或训练 Agent。

## AVL-005 — 复核后确定性、成本、闭环与 held-out 补严（运行前登记，2026-10-10）

- **总 Claim**：在进入任何 Agent 实现前，补齐 RoMa 随机性、统一 action cost、recovery-aware closed loop 和 validation/test 隔离；所有结果继续服务于 failure-aware Agentic Visual Localization 主线，不把补实验写成 Agent 收益。
- **AVL-005-A RoMa multi-seed**：固定 Python/NumPy/Torch/CUDA/OpenCV seed=`0,1,2`，在 A.5 相同 952-frame geometry-call pool、相同 mapping-only Top-100 reranking 和 selected refs 上运行 `run_avl004_roma_recovery.py --seed S`。保留每 seed 原始 JSONL；逐帧 evaluator 事后读取 GT。报告每序列/加权 success 均值、样本标准差、95% bootstrap CI、逐帧成功一致率、HLoc-only/RoMa-only/neither 均值/方差、共同失败集合 Jaccard 和 Action Oracle 均值/方差。任何 seed 访问 query GT、使用未来帧或改变 refs 均为 invalid。
- **AVL-005-B unified cost**：固定同一 query/reference/硬件协议，把 `retrieval`、`Top-100 rerank`、`geometry`、`verification`、`Reloc3r`、`Memory` 分开打点；同时记录串行 elapsed、GPU device、GPU-hours、调用次数、reference budget 和 wall-clock。不得用不同阶段 elapsed 直接比较 RoMa/HLoc，不得从 call-rate 推导 latency。先做 deterministic Manager 与 Always-RoMa 的相同端到端 stage accounting。
- **AVL-005-C closed-loop Manager**：以当前无 GT Manager trace 为决策规则起点；对 `call_geometry` 必须读取当帧 recovery pose/verification 结果，再更新 `previous_prediction`、memory reliability 和 jump state，后续帧决策只能使用已经发生的过去结果。保存逐帧 state-before/state-after、action、pose source、verification、GT-forbidden audit；与冻结离线回灌 A.5 对照。
- **AVL-005-D held-out**：先定义 validation/test 划分和冻结流程。validation 只能用于阈值/规则选择，test 只运行冻结规则；不得用 test GT 选择 threshold/reference/tool。优先在 LILocBench 的时间/条件留出上做内部验证；OpenLORIS/RIO10 只有在相机、map、GT 和工具适配完成并单独登记后才作为跨数据集验证，不得混入 LILocBench 调参。
- **Go/No-Go**：四项均有原始逐帧 artifact、无 GT 在线审计和可复现统计后，才重新判断 Agent Go。若 deterministic closed-loop 在相同端到端成本下已接近 Always-RoMa/oracle，则继续 Hold；只有 held-out 上存在稳定的上下文分流收益且超过 deterministic baseline，才进入 Agent。

### AVL-005 补充执行批次（运行前登记，2026-10-10）

- **状态**：`running`。本批次只补 AVL-005-A/B/C/D 的缺口，不改变主线、不引入 Agent，也不改写先前 artifact。
- **A（全帧多 seed）**：在 1696 个 all-frame query 上按 seed `0/1/2` 重跑 RoMa；每 seed 使用 frozen mapping-only Top-100→LightGlue selected references、相同 RoMa 与 PnP 参数、同一硬件环境。GT 仅由独立 evaluator 读取。报告 all-frame Always-RoMa success 的均值/样本标准差、HLoc/RoMa confusion 与 oracle 对 seed 的敏感性。现有 952-frame manager-call pool 多 seed 结果保留为独立结果，不能替代全帧结果。
- **B（统一成本）**：HLoc-style 全 1696 帧按 `query descriptor/extraction`、Top-100 re-rank、geometry 拆分计时；RoMa 的 matching/triangulation/PnP 已拆分，将以本批次 full-frame fixed-seed runs 汇总。retrieval profiling 已完成但只计 mapping/query descriptor 与 ranking；Reloc3r inference/Memory 的耗时若不能在同一环境从原始模型重测，必须明确标为缺失，不能估算。所有比对报告串行 stage sum、调用次数、设备和 wall-clock，不以分片 wall-clock 取代每帧成本。
- **C（闭环）**：以全帧固定 seed=0、无 GT RoMa cache 重放 recovery-aware Manager，验证每一 geometry 输出在下一帧前更新 prediction/memory，并对新的 trace 作独立 GT-only evaluator。该实验是缓存工具输出的 deterministic online-policy replay；若未重新在线执行 geometry，必须保留该限定。
- **D（held-out）**：保持先前 static+dynamics validation、LT+combo frozen-test 的内部 condition split，同时审计当前工作区是否有 OpenLORIS/RIO10 或其他未参与策略发现的带 map+GT 数据。若无数据，不以下载/不兼容数据替代；记录为跨数据集验证 blocked，内部 split 只称 calibration-shift test，不能声称独立泛化。
- **预注册判定**：若全帧 RoMa seed 波动大于 1 pp 或 oracle/mutual-failure 集合 Jaccard 不稳定，则所有精确互补计数降级为 seed-specific；若 unified stage sum 无法显示 Manager 在相同成功水平下低于 Always-RoMa 的成本，不得主张效率收益；若 internal frozen test 的 budget 明显失配或无跨数据集数据，则 held-out Go 条件未满足。
- **路径**：新增输出仅写入 `outputs/lilocbench/AVL-005/fullframe_multiseed/`、`outputs/lilocbench/AVL-005/unified_cost/`、`outputs/lilocbench/AVL-005/closed_loop_seed0/` 和 `outputs/lilocbench/AVL-005/heldout_audit/`；日志写入同级 `logs/`。所有推理行须保留 `query_gt_read=false`、`selection_uses_gt=false`。

### AVL-005 补充批次范围更正（2026-10-10，运行前追加）

- 用户确认本轮只完成此前尚未完成的两项：**B 统一成本核算**与 **D 真正独立 held-out**。A 的 952-frame 三 seed 结果与 C 的缓存式 closed-loop Manager 已完成，不启动 full-frame multi-seed 或 closed-loop 重跑。本次更正发生在任何新 GPU 任务启动前；上条中 A/C 的计划与 `fullframe_multiseed/`、`closed_loop_seed0/` 路径均未执行、不得视为实验结果。
- **执行范围**：B 将从已完成的 retrieval profile、既有 fixed-seed RoMa stage fields、以及新运行的带 stage timing HLoc-style 产物构建统一表；若 Reloc3r/Memory 原始端到端计时无法复现，则在表中明确标为缺失并不作 Manager-vs-Always-RoMa 的完整成本主张。D 只接受未参与此前 discovery/tuning 的外部带 map+GT 数据；先审计本机可用数据，必要时调查公开的 OpenLORIS/RIO10 获取与协议兼容性，未经明确适配和冻结配置不运行、不把 LILocBench 内部 split 误称独立 held-out。

### AVL-005-B 统一成本结果（2026-10-10，completed）

- **审计**：HLoc-style 与 RoMa 均完成四序列全帧计时，行数严格为 static/dynamics/LT/combo=`578/155/422/541`；所有 recovery 行保留 `query_gt_read=false`、`selection_uses_gt=false`。RoMa 使用固定 seed=0；HLoc-style 使用同一 mapping-only Top-100 retrieval 与 4-ref geometry 参数。结果根目录：`outputs/lilocbench/AVL-005/unified_cost/`，聚合：`summary.json`。
- **全帧阶段总耗时（串行 stage sum）**：NetVLAD mapping+四条 query retrieval=`37.953 s`；HLoc-style query extraction=`89.352 s`、Top-100 rerank=`2182.693 s`、geometry=`182.311 s`，总 HLoc stage=`2454.518 s`；RoMa geometry matching=`3944.379 s`、triangulation=`102.440 s`、PnP=`265.031 s`，总 RoMa geometry=`4312.566 s`。这些是独立 stage sum，不是 8-GPU wall-clock。
- **Manager-vs-Always-RoMa**：Manager trace 的 783/1696 帧调用 geometry（`46.17%`）；调用子集的 query extraction=`42.917 s`、Top-100 rerank=`1152.206 s`、RoMa geometry=`2073.230 s`，candidate processing=`0.164 s`、Memory=`86.363 s`、verification=`0.027 s`。Reloc3r 网络/预处理以每序列固定 8 帧时间抽样并按帧数缩放，估计=`3273.025 s`；该项明确是 sampled estimate，不伪装为逐帧实测。统一估计总成本：Manager=`6665.885 s`，Always-RoMa=`6622.564 s`，Manager 高 `43.321 s`（约 `0.65%`），因此当前规则只降低 geometry 调用率，不带来端到端成本优势。
- **同 seed=0 事后准确率对照**：Always-RoMa=`1552/1696=91.51%`，Manager 使用同一 seed=0 recovery cache 且保留原 verification gate 的闭环 route=`1521/1696=89.68%`，coverage=`1672/1696`；Manager 低 `1.83 pp`。旧 unseeded Manager `89.50%` 与 Always-RoMa `91.21%` 保留为历史单次结果，不与本条 seed=0 数值混用。
- **解释边界**：Manager 的 Reloc3r 时间来自 32 帧总样本（每序列 8 帧），所以绝对总成本是不确定度较高的估计；Manager 与 Always 的 geometry/rerank/retrieval 部分为全帧实测。结论只支持“当前 Manager 没有已证明的端到端成本/精度优势”，不支持任意硬件或其他并行度下的普适速度结论。

### AVL-005-D 状态（2026-10-10，paused）

- 用户要求暂不运行新场景 held-out；RIO10 只完成只读审计，未进入实验。LILocBench 的条件 split 结果仍保留为内部 calibration-shift test，不能声称独立泛化。跨数据集/未见场景验证保持暂停，不影响 B 的成本结果。

### AVL-004 A.6–A.9 结果（2026-10-09，追加；Agent 仍未实现）

- **A.6 Always-HLoc / Always-RoMa**：四序列共 1696 帧均完成。严格 shard 合并审计通过：static/dynamics/LT/combo 分别 578/155/422/541 帧，recovery 行均为 `query_gt_read=false`、`selection_uses_gt=false`；GT 只被独立 evaluator 读取。全量 HLoc-style coverage/success 为 `99.47%（1687/1696）/88.62%（1503/1696）`，RoMa 为 `100%/91.21%`（1547/1696）。逐序列 success（HLoc/RoMa）为 static `90.14/94.64%`、dynamic `82.58/89.03%`、LT `88.15/88.39%`、combo `89.09/90.39%`。全量 median t/r：HLoc `0.0295 m/0.437°`，RoMa `0.0271 m/0.431°`；最长失败段（static/dynamic/LT/combo）HLoc `10/7/7/6` 帧，RoMa `9/4/5/5` 帧。Always rerun 逐帧 elapsed 中位数为 HLoc `12.48 s`、RoMa `2.29 s`；2026-10-10 复核确认 HLoc elapsed 包含 Top-100 重排与 HLoc-style PnP，而 RoMa elapsed 只包含复用 selected refs 后的 RoMa 阶段，二者不能作端到端速度比较。A.6 结果目录：`outputs/lilocbench/AVL-004/always_geometry/{hloc,roma}/`，评估：`always_route_eval/`。
- **A.6 无效尝试**：首次 RoMa 启动因错误拼接环境路径退出（`roma_shards_gpu4_7/`，日志显示 Python 路径不存在）；不覆盖有效重跑。有效重跑位于 `roma_shards_gpu4_7_retry/`，使用项目级 `TORCH_HOME` 和 `/data2/liusy/conda_envs/reloc3r`。
- **A.7 Action Oracle**：读取 consensus-3、memory-3、Always-HLoc、Always-RoMa 的 post-hoc 成功标记，不参与路由。加权成功率为 consensus `70.64%`（1198/1696）、memory `76.71%`（1301/1696）、Always-HLoc `88.62%`（1503/1696）、Always-RoMa `91.21%`（1547/1696）、任一候选 oracle `95.75%`（1624/1696）。按序列 oracle success：static `98.62%`（570/578）、dynamic `90.32%`（140/155）、LT `95.26%`（402/422）、combo `94.64%`（512/541）。A.5 deterministic Manager 的 RoMa route 为 `1513/1696=89.21%`，与 oracle 差 111 帧/6.54 个百分点；Always-RoMa 与 oracle 仍差 77 帧/4.54 个百分点。Oracle 文件：`outputs/lilocbench/AVL-004/action_oracle/summary_always.json`。
- **A.7 工具互补**：在全部 1696 帧的单次 post-hoc 成功混淆中，HLoc/RoMa both `1466`、HLoc-only `37`、RoMa-only `81`、neither `112`；比例分别为 `86.44/2.18/4.78/6.60%`。这不是工具选择的在线证据；2026-10-10 复核又发现 RoMa seed 未固定，因此只能作为探索性互补证据，精确计数暂定。
- **A.8 共同失败审计**：A.5 单次运行 geometry-call 池中的 97 个 HLoc/RoMa 共同失败帧已逐帧保存到 `outputs/lilocbench/AVL-004/common_failure_audit/`，并生成四序列 contact sheet（static/dynamic/LT/combo=`16/18/28/35` 帧）。operational multi-label 计数为：`trajectory_jump 94`、`both_pose_available_but_wrong 90`、`appearance_change_proxy 79`、`candidate_conflict 69`、`turn_motion 75`、`low_texture_or_blur 60`、`low_texture_or_blur_proxy 65`、`reference_mismatch 65`、`large_reference_distance_proxy 54`、`appearance_change 51`。这些是审计信号而非语义因果真值；73/97 帧含 reference mismatch/distance 证据，79/97 含 appearance proxy，65/97 含 blur/low-texture proxy，90/97 两种工具均有 pose 但仍错误。该实现未完成预注册的语义/因果类别真值，且 97 帧集合受 RoMa seed 影响，因此 A.8 状态修正为“proxy 审计完成、语义分类与稳定性未完成”。
- **A.9 Risk–call-rate–accuracy**：有效输出为 `outputs/lilocbench/AVL-004/risk_cost_accuracy/summary.json`，风险阈值沿用预注册 `{0.5,1,1.5,2,3,5}`，无 GT 风险不变。四序列按帧汇总的 risk-gated RoMa `geometry_call_rate / coverage / success_all_frames / selective_accuracy` 为：阈值 0.5 `98.05%/100%/91.21%/91.21%`；1.0 `84.91%/100%/90.92%/90.92%`；1.5 `72.64%/98.88%/89.45%/90.46%`；2.0 `64.62%/96.46%/86.85%/90.04%`；3.0 `56.31%/91.75%/81.90%/89.27%`；5.0 `42.81%/81.96%/72.46%/88.42%`。旧报告把各序列 selective accuracy 按序列总帧数加权，聚合方法错误，旧值立即无效；正确 reject policy 在阈值 0.5/1.0/1.5/2.0/3.0/5.0 的 `coverage / success_all_frames / selective_accuracy` 为 `1.95/1.95/100.00%`、`15.09/14.50/96.09%`、`26.24/24.88/94.83%`、`31.84/30.01/94.26%`、`35.44/33.25/93.84%`、`39.15/36.50/93.22%`。当前 A.5 Manager 的 geometry-call rate 为 `952/1696=56.13%`，RoMa route success `89.21%`，高于简单阈值 3.0 的 `81.90%`；但脚本没有读取实际 latency，故 A.9 尚未完成预注册的 risk–latency–accuracy，只完成 call-rate 曲线。
- **A.9 成本解释（2026-10-10 修正）**：Always-RoMa 的 2.29 s 与 Always-HLoc 的 12.48 s、A.5 的 4.6–4.9 s 与 18.8–20.2 s 计时边界不一致。HLoc 字段包含 Top-100 reranking+HLoc-style PnP，RoMa 字段只包含 selected refs 后的 RoMa 阶段。旧端到端速度比较无效；工具调用次数和单次运行记录保留。
- **四项审计结论（2026-10-10 修正）**：A.6 的单次 accuracy、A.7 的单次 oracle 和 A.9 的 call-rate 曲线已完成；RoMa 稳定性、A.8 语义分类、统一 latency/action budget 与闭环 Manager 未完成。当前冻结 Manager route 在约 56% geometry-call rate 下为 `89.21%`，单次 Always-RoMa 为 `91.21%`、单次 oracle 为 `95.75%`；这些支持继续研究调度价值，但不能作为稳定 Oracle、真实成本或闭环 Agent 结论。当前仍为 **Agent 前暂停（Hold/No-Go for immediate LLM implementation）**，主线不变。

### 2026-10-10 关键实验复核与结论勘误

- 完整复核报告：`docs/key_experiment_reaudit.md`。AVL-002 核心统计、位姿方向和 GT 边界正确；AVL-003 四序列确定性重放与原产物逐字节一致；Manager trace 重放也逐字节一致。
- **RoMa 随机性未控制**：`run_avl004_roma_recovery.py` 未设置 seed，而 RoMa `sample()` 使用 `torch.multinomial`。A.5 与 A.6 在相同 952 帧/相同 refs 上有 34 帧 success label 翻转；A.5 的共同失败 97 帧在第二次运行变为 98 帧，交集 88/并集 107。聚合恢复率可保留为各自单次运行值，但逐帧互补、共同失败集合和 Oracle 精确值降级为暂定，需固定 seed 并多次重跑。
- **延迟口径不公平**：HLoc elapsed 包含 Top-100 reranking+HLoc-style PnP，RoMa elapsed 只计 selected refs 后的 RoMa 阶段。因此此前“RoMa 端到端更快”和按 56.13% call-rate 推导实际计算节省的解释无效；A.9 也只实现 call-rate，没有实现实际 latency/action budget。
- **Manager 非闭环**：`call_geometry` 的 pose 没有回写到后续 `previous_prediction`/memory。A.5 `89.21%` 是冻结 action 的离线回灌结果，不是 recovery-aware closed-loop Manager。
- **A.8 部分完成**：97 帧 proxy/contact-sheet 审计有效，但未完成预注册的语义/因果类别真值；且集合受 RoMa seed 影响。A.6–A.9 应表述为“accuracy/oracle/call-rate 分析完成，稳定性、语义审计和统一 latency 尚待补齐”，不能笼统称全部终止性问题已完成。
- 同一四个 query 序列用于发现 failure、制定阈值与最终评价；没有在线 GT 泄漏，但存在 test-set adaptation。所有结论限当前 LILocBench feasibility，不构成 held-out 泛化证据。

## AVL-006 — 统一成本 break-even 与 GT-free pre-routing feasibility（运行前登记，2026-10-10）

- **主张与范围**：在不引入 LLM/语言 Agent 的前提下，回答两项终止性问题：A）用 AVL-005-B 的统一 stage timing 计算 Manager 的准确率—成本 break-even；B）检验一个只读取廉价前置信号的 deterministic pre-router 能否在运行 Reloc3r/Memory 或 RoMa 之前作出可审计的路线选择。该实验只服务于 failure-aware Agentic Visual Localization 主线，不改变主线。
- **A 协议**：共享成本为 mapping/query retrieval 与廉价上下文提取；Reloc3r 使用 AVL-005-B 每序列 8 帧无 GT timing 的明确抽样估计，RoMa 路径包含 query extraction、Top-100 rerank 与 RoMa matching/triangulation/PnP。以 Always-RoMa seed=0 的 `91.51%`、`6622.564 s` 作为比较锚点，报告在不同 Manager overhead、Reloc3r/RoMa 调用数下的严格成本不等式和达到 `>91.51%` 所需的最低有效 RoMa 路由收益。不得把抽样 Reloc3r 时间写成逐帧实测。
- **B 输入约束**：路由器只能使用 mapping-only NetVLAD 的 top-1 分数、top-1/top-2 margin、top-5 均值、当前图像 Laplacian/清晰度、相邻 query 图像变化，以及已经发生的上一动作/连续帧状态；不能读取当前帧 Reloc3r 候选、dispersion、GT、RoMa/HLoc 输出或未来帧。检索 descriptors 可作为共享廉价输入，但不执行 Top-100 LightGlue rerank 后再路由。
- **B 动作与冻结**：`reloc3r_memory` 路径定义为已保存 AVL-003 memory-3 结果，`roma` 路径定义为固定 seed=0 Always-RoMa 结果；路由规则在读取任何 GT 前冻结。采用预注册阈值网格/固定规则只用于展示成本—成功率 Pareto，不按 test GT 选最优阈值；独立 evaluator 事后合并两条路径的 success、coverage、call-rate、longest failure 和 per-sequence 结果。
- **判定标准**：B 只有在无 GT 路由器在不高于 Always-RoMa 成本下达到超过 `91.51%` 的全帧 success，或形成稳定且可解释的 accuracy–cost Pareto，才构成进入下一阶段 deterministic Manager 的证据；否则结论为当前廉价前置信号不足。A/B 的所有路由 trace、特征、阈值、公式和审计字段写入 `outputs/lilocbench/AVL-006/`，GT 只出现在 evaluator 结果中。

### AVL-006 结果（2026-10-10，completed）

- **A break-even**：以 AVL-005-B 的 Always-RoMa seed=0 成本 `6622.564 s` 为严格预算，四序列共 `1696` 帧。廉价图像特征提取实测 `14.355 s`，共享 mapping/query retrieval `37.953 s`；Reloc3r 每帧成本为每序列 8 个固定时间样本的均值（加权 `1.930 s/frame`），完整 RoMa 路径（query extraction+Top-100 rerank+RoMa）加权 `3.883 s/frame`。计入 AVL-005 的 memory/candidate/verification 状态开销 `86.553 s` 后，严格低于 Always-RoMa 的上限为 `1644/1696=96.93%` RoMa calls；`1645` calls 只能达到预算边界，不能称严格低于。若所有帧都调用 RoMa，任何额外 Manager/context overhead 都没有正预算余量。
- **A 解释**：该 break-even 是统一串行 stage accounting，不是并行 wall-clock；Reloc3r 绝对时间仍是抽样缩放估计。它说明“减少 geometry calls”有足够的理论成本空间，但并不自动产生准确率收益。
- **B pre-routing**：运行 `conservative/balanced/context/retrieval_sensitive/sensitive` 五个预冻结 deterministic 规则。路由只读 mapping-only top-1、margin、top-5、Laplacian、相邻图像变化和上一动作；逐帧 trace 均 `query_gt_read=false`、`selection_uses_gt=false`，未读取当前 Reloc3r pose、dispersion、RoMa/HLoc 输出或未来帧。结果（全 1696 帧，seed=0 RoMa cache）：RoMa call-rate / success 分别为 conservative `3.01%/78.13%`、balanced `22.70%/82.08%`、context `22.88%/83.20%`、retrieval-sensitive `54.83%/88.09%`、sensitive `75.18%/90.04%`。五个策略均未超过 Always-RoMa `91.51%`。
- **B 判定**：当前两动作缓存的 post-hoc union 为 `1606/1696=94.69%`（both/memory-only/RoMa-only/neither=`1247/54/305/90`），因此理想路由仍有超过 `91.51%` 的上界；但五个真实 GT-free pre-router 的最高结果只有 `1527/1696=90.04%`。B 没有形成进入 Agent 的 Go 证据；它证明廉价信号能形成风险—成本排序，却尚不足以识别 memory-only/RoMa-only 的互补帧。
- **输出与审计**：路由 trace、特征计时、GT-only evaluator、break-even 表和四条件汇总分别位于 `outputs/lilocbench/AVL-006/early_router/`、`early_router/context_feature_timing.json`、`early_router_eval.json`、`break_even.json`、`four_condition_summary.json`；实现为 `tools/run_avl006_early_router.py`、`tools/eval_avl006_early_router.py`、`tools/analyse_avl006_break_even.py`、`tools/aggregate_avl006_four_condition.py`。所有产物保留，未覆盖 AVL-005。
- **下一步决策**：继续 Hold/No-Go，不引入 Agent。主线仍保留；若以后推进，应先做有独立场景或新工具输出的验证，不能把本次 post-hoc cache replay 写成在线 Agent 收益。

### AVL-006-B2 补充协议：leave-one-sequence-out 轻量 pre-router（运行前登记，2026-10-10）

- **目的**：避免把五个手工阈值的失败误判为廉价信号没有信息，检验一个小型、可审计的 deterministic classifier 在未见 query sequence 上能否选择 `reloc3r_memory` 或 `roma`。
- **输入与约束**：模型输入固定为 AVL-006-B 已生成的 top-1/margin/top-5、Laplacian、frame-change、上一动作/时间状态；训练和测试的在线路由都不读取当前 Reloc3r pose、candidate dispersion、RoMa/HLoc 输出、GT 或未来帧。GT 只在离线训练标签生成和独立 evaluator 中出现。
- **划分**：leave-one-sequence-out；每次完整留出一个 query sequence，训练只用其余三条序列，阈值/模型结构不按留出序列 GT 调整。比较 logistic/浅层树等轻量模型与固定手工规则，报告每个留出序列、总体 success、RoMa call-rate、成本和 action confusion。
- **标签与边界**：post-hoc action label 只用于估计可学习上限：当仅 RoMa 成功时标 `roma`，当 Memory 成功而 RoMa 失败时标 `reloc3r_memory`；both/neither 采用预注册的低成本 tie-break。该标签不是在线输入，也不把结果写成因果证明。
- **判定**：若 leave-one-sequence-out 的 GT-free 推理在不高于 Always-RoMa 成本下超过 `91.51%`，则说明廉价前置信号具备 deterministic routing 潜力；否则只能保留为弱风险排序证据，继续 Hold Agent。输出写入 `outputs/lilocbench/AVL-006/early_router_cv/`。

### AVL-006-B2 结果（2026-10-10，completed）

- **审计**：四折 leave-one-sequence-out 完成；每个留出序列的预测只读取 cheap feature，trace 均 `query_gt_read=false`、`selection_uses_gt=false`。离线标签只用于训练/事后评分，未进入路由 trace。
- **结果**：`logistic_balanced` 为 `728/1696=42.92%` RoMa calls、`1483/1696=87.44%` success、统一成本估计 `5050.285 s`；`tree_depth4` 为 `775/1696=45.70%` calls、`1468/1696=86.56%` success、成本 `4949.562 s`。两者均低于 Always-RoMa `91.51%`，但均在成本预算内。
- **解释**：轻量模型能从廉价信号学到条件相关风险排序，但留出序列上的 action confusion 仍较大（总体 target-action accuracy logistic `65.39%`、tree `60.26%`）；因此当前 cheap feature 集尚不能可靠区分 RoMa-only 与 memory-only 互补帧。B2 不构成 Agent Go，但比手工阈值更强地支持“有弱可利用信号、尚不足以完成路线选择”的结论。
- **输出**：`outputs/lilocbench/AVL-006/early_router_cv/summary.json`；实现 `tools/eval_avl006_early_router_cv.py`。该实验仍是 LILocBench 条件级 feasibility，不是独立场景泛化。

## AVL-007 — Cheap Context Representation Routing（运行前登记，2026-10-10）

- **Claim**：在不引入 LLM/VLM Agent 的情况下，冻结的 DINOv2 图像表征加 retrieval/blur/frame-change/短期历史，是否能在运行完整 Reloc3r 之前区分 `memory-only` 与 `RoMa-only` 帧，并形成超过 AVL-006 cheap-feature router 的可泛化 pathway routing。
- **数据与划分**：同一 LILocBench mapping database 与四条 query sequence；主评估为 leave-one-sequence-out，留出序列的 embedding、router 和结果独立落盘。该结果仍是条件级 feasibility，不冒充独立场景泛化。
- **冻结表征**：本地缓存 `dinov2_vitl14_pretrain.pth`，DINOv2 ViT-L/14，weights frozen、eval mode、无微调；输入图像只经固定 resize/normalize。保存 mapping/query embedding、模型 SHA256、代码 revision 与推理硬件。
- **输入与动作**：当前 query embedding、上一 query embedding、top-1 mapping embedding 相似度、top-1/top-2/top-5 retrieval 统计、Laplacian、frame-change、有限历史状态；输出只允许 `reloc3r_memory` 或 `roma`。路由器不能读取当前帧 Reloc3r pose、candidate dispersion、RoMa/HLoc 输出、GT 或未来帧。
- **模型与标签边界**：只比较预注册的 logistic regression 与浅层 decision tree；`memory-only`/`RoMa-only` 标签只由独立 evaluator 离线生成，用于训练和事后评分，不能进入在线 trace。both/neither 使用 cheap-path tie-break，并单独报告四类 action confusion。
- **判定标准**：主要看 leave-one-sequence-out 的 memory-only recall、RoMa-only recall、balanced accuracy、action confusion、overall success、RoMa call-rate 和统一成本；若 embedding router 在不高于 Always-RoMa 成本下超过 `91.51%`，则说明 cheap visual context 已有 deterministic routing 潜力；若仍无法区分且人工/可视化审计显示两类存在语义差异，才保留引入 VLM/Agent 的依据。不得只以普通二分类 accuracy 判定。
- **输出**：预期写入 `outputs/lilocbench/AVL-007/`，包括 frozen embeddings、feature manifest、LOSO traces、GT-only evaluator、confusion/Pareto 汇总和日志。失败或无效运行保留，不覆盖 AVL-006。

### AVL-007 结果（2026-10-10，completed）

- **冻结表征审计**：DINOv2 ViT-L/14 本地权重加载成功，SHA256=`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`；mapping 785 帧、query 1696 帧、CLS 维度 1024，权重 eval/frozen。query embedding 提取串行 elapsed `47.665 s`，mapping 一次性 elapsed `22.267 s`；所有提取行 `query_gt_read=false`、`pose_or_reloc3r_read=false`。
- **LOSO 表征路由**：输入包含当前 DINO embedding 的 mapping-only PCA-64、NetVLAD top reference 的 DINO similarity、DINO max/top-5 similarity、前一帧/最近三帧 embedding similarity，以及 AVL-006 cheap signals。训练只用其余三条 sequence，测试 trace 只输出 action 和 cheap features。
- **重点类别结果（359 个 exactly-one-success 帧：54 memory-only、305 RoMa-only）**：DINO-similarity + logistic 的 memory-only/RoMa-only recall=`31.48%/65.90%`、balanced accuracy=`48.69%`；DINO-similarity + depth-4 tree 为 `24.07%/74.10%`、`49.09%`。加入 PCA embedding/ref/difference 的 full representation 后，logistic 为 `50.00%/43.93%`、`46.97%`，tree 为 `38.89%/49.84%`、`44.36%`。随机/Always-RoMa 风格的单类偏置约为 50% balanced accuracy，因此冻结 DINO 没有解决两类互补帧的区分。
- **全帧结果与成本**：DINO-similarity tree 最好但仍为 `1485/1696=87.62%`、`52.00%` RoMa calls；加入 query DINO 提取后统一成本约 `5219.843 s`，低于 Always-RoMa 预算但精度不足。DINO full logistic/tree 分别为 `83.02%/33.02%` 和 `83.73%/34.55%`，没有优于 AVL-006 B2。
- **人工 operational audit**：代表性 contact sheet 显示 memory-only 更多出现可由短期历史维持的重复房间/门/柜体视角，RoMa-only 在 dynamic/combo 中较多出现近距离行人、运动模糊，在 LT/combo 中较多出现纸箱、家具/门/走廊外观变化；但两类高度重叠，也存在清晰无遮挡的 RoMa-only 帧。该观察不是语义 GT 或因果标签。
- **结论与下一步**：冻结视觉表征能提供弱的上下文排序，但不能稳定区分 memory-only 与 RoMa-only；因此 AVL-007 未满足“cheap visual context 已足够”的条件，也未单独构成 VLM/Agent Go。它保留了一个受限的研究依据：若后续确需 Agent，理由应是处理表征无法编码的上下文/因果语义，而不是普通数值分类能力不足。输出见 `outputs/lilocbench/AVL-007/summary.json`、`embeddings/manifest.json`、`audit/semantic_audit.json`，实现见 `tools/extract_avl007_dinov2.py`、`tools/run_avl007_representation_router.py`、`tools/audit_avl007_special_frames.py`。

## AVL-008 — Semantic Context Probe（运行前登记，2026-10-10）

- **状态 / Claim**：`running`。不构建完整 Agent，只检验冻结 Qwen2.5-VL-7B-Instruct 的零样本、多图像语义判断是否提供 AVL-006/007 未捕获的决策信息：A）语义类别是否与 Memory/RoMa 的离线 outcome cohort 有稳定关系；B）在 `memory-only` 与 `RoMa-only` 帧上，预先冻结的语义→路径映射是否优于同一帧的 AVL-007 DINO-similarity tree。
- **模型、输入与时间边界**：模型固定为本地 `/data2/liusy/agent/vlm` 的 Qwen2.5-VL-7B-Instruct，经本地 vLLM OpenAI-compatible endpoint 调用，`temperature=0`、`top_p=1`、`max_tokens=96`。每个请求只给当前 query、mapping-only NetVLAD top-3 reference 及最多前三个**过去** query frame；不得给 GT、pose、Reloc3r/Memory/RoMa 成功失败、候选离散度、未来帧、sequence condition 名或任何 outcome cohort。Top-3 是固定预算，不因结果改变。
- **数据与抽样**：outcome cohort 由既有 `memory3` 与 fixed-seed=0 Always-RoMa 的 `(0.25m,5°)` evaluator 标签在 VLM 运行后仅用于评估：`memory-only`、`RoMa-only`、`both-fail`、`both-success`。用随机种子 `20261010` 对每 cohort 作按四条 sequence 比例的确定性分层抽样，目标每 cohort `min(54, cohort size)` 帧（预期共 216 帧）；样本清单先冻结并记录所有 index/path/hash。cohort 只用于抽样平衡，绝不写入 VLM request/prompt。
- **冻结的输出 schema**：VLM 只能返回 JSON `{label, confidence, evidence}`；`label` 必须为 `temporary_dynamic_occlusion`、`motion_blur_or_weak_observation`、`persistent_scene_change`、`reference_semantic_mismatch`、`normal_or_compatible`、`uncertain` 之一；`confidence` 为 0–1；`evidence` 最多 20 个英文词。解析失败只允许以同一图像和固定“JSON only”提示重试一次；重试率与无效输出单独报告。不得要求或保存详细内部推理。
- **冻结的语义→动作映射**：`temporary_dynamic_occlusion`、`motion_blur_or_weak_observation`、`normal_or_compatible`→`memory`；`persistent_scene_change`、`reference_semantic_mismatch`→`roma_or_reretrieve`（在本二路径离线对照中记为 `roma`）；`uncertain`→`abstain`。该映射在读取 VLM 输出前冻结，不针对结果调整；它不是部署中的完整 action set。
- **指标与预注册判定**：A）四 cohort×六标签的 contingency、Cramér's V、10,000 次 cohort-preserving permutation p 值和 sequence-stratified bootstrap CI；只有 V≥0.30、p<0.05 且 95% CI 下界>0.10 才称“明显语义关系”。B）在 sampled `memory-only`/`RoMa-only` 帧上，把 `uncertain` 计为未命中，报告语义映射的 balanced accuracy、两类 recall、abstain rate，与同一 index 的 AVL-007 `dino_similarity/tree_depth4` LOSO action 作配对比较；只有 VLM 比 DINO 高≥10 pp 且 bootstrap 95% CI 下界>0，才称 VLM 有额外路径预测信息。另报告非 abstain selective 指标但不得用它替代主指标。
- **泄漏 / GT 审计**：VLM request、prompt、response trace 必须显式保存 `query_gt_read=false`、`future_frame_read=false`、`outcome_or_action_read=false`；outcome/cohort 只由独立 evaluator 在预测完成后 join。任何 prompt 含 sequence/condition、GT、pose 或 action outcome，或抽样后人工按 VLM 输出替换样本，均使该运行 `invalid`。
- **输出 / 下一步决策**：输出写入 `outputs/lilocbench/AVL-008/`：frozen sample manifest、redacted requests、raw JSON responses、解析日志、prompt/model manifest、独立 evaluator、DINO paired baseline 和代表性 contact sheets。若 A/B 均通过，只构成进入小规模 VLM Context Manager 设计的证据，仍需 held-out 与统一成本；若任一不通过，维持 Agent No-Go，并将原因区分为“VLM 无额外信息”或“当前观测本身不足”。

### AVL-008 结果（2026-10-10，completed with limits）

- **执行与审计**：使用本地 Qwen2.5-VL-7B-Instruct（`/data2/liusy/agent/vlm`，vLLM 本地 endpoint，GPU 1，temperature=0、top_p=1、seed=0），对四个 outcome cohort 各抽取 54 帧，共 216 个固定请求；每次输入为当前 query、mapping-only Top-3 reference 和最多前三个过去 query frame。VLM request 不含 cohort、GT、pose、Reloc3r/Memory/RoMa outcome 或未来帧；请求和响应均保留 `query_gt_read=false`、`future_frame_read=false`、`outcome_or_action_read=false`。
- **解析情况**：200/216 个响应严格通过 JSON schema，16 个响应在两次尝试中都超过预注册的 evidence 20 词限制，按 invalid 保留，不回填、不重采样；invalid 按 cohort 为 both-fail `3`、both-success `2`、memory-only `5`、RoMa-only `6`。有效响应标签只有 `motion_blur_or_weak_observation` `31`、`normal_or_compatible` `149`、`reference_semantic_mismatch` `16`、`uncertain` `4`；`temporary_dynamic_occlusion` 与 `persistent_scene_change` 未被输出。
- **问题 1：语义与 outcome 的关系**：四 cohort×六标签的 Cramér’s V=`0.2187`，cohort-preserving permutation p=`0.0004`，stratified bootstrap 95% CI=`[0.2007,0.2959]`。统计上不是完全独立，但效应低于预注册的 V≥`0.30` 且 CI 下界>`0.10` 门槛，因此只能称弱关联，不能据此定义“遮挡→Memory、长期变化→RoMa”的可靠规则。
- **问题 2：是否优于 DINO 路由**：在有效的 97 个 `memory-only/RoMa-only` 样本上，冻结语义映射把 uncertain 计为错误时，VLM action balanced accuracy=`47.02%`，memory-only recall=`85.71%`，RoMa-only recall=`8.33%`，abstain rate=`3.09%`；同一帧 AVL-007 DINO-similarity/tree 的 balanced accuracy=`47.64%`。VLM−DINO=`−0.62 pp`，paired stratified bootstrap 95% CI=`[−10.23,9.04] pp`，未达到预注册的 +10 pp 且 CI 下界>0 标准。VLM selective coverage=`96.91%`、selective balanced accuracy=`48.10%`，不能替代主指标。
- **结论**：AVL-008 不支持当前 Qwen zero-shot semantic probe 提供超出 DINO/数值特征的 action routing 信息；尤其无法识别 RoMa-only 帧。该结果支持“当前多图像观测/提示下的语义判断尚不足以进入 Agent”的结论，但不证明所有 VLM、微调或更好的 temporal/reference 表示都无效。完整摘要：`outputs/lilocbench/AVL-008/summary.json`；样本、请求、原始响应和 evaluator 行分别见同目录 `sample_manifest.json`、`requests.jsonl`、`responses.jsonl`、`evaluator_rows.jsonl`。
- **阶段决策**：维持 **Hold / No-Go for immediate Agent implementation**。不构建完整 Agent；若未来重启，只能先登记新的 prompt/input ablation 或独立 held-out VLM probe，并修正 schema 解析率问题，不能把本次结果写成 Agent 负收益或跨模型普遍结论。

## 主线架构更新（2026-10-10，实验策略决策）

- **决策**：后续实验从“逐帧 action routing”转为“连续 RGB 序列上的 stateful localization”。这不是改变研究主线，而是把 failure-aware 决策的基本单位从单帧改为状态转移。
- **冻结流程**：`连续 RGB → Temporal Tracking/Motion → Stateful Localization State → Global Localization/Recovery → Verification → 更新状态`。
- **状态字段**：当前位姿状态、历史可靠帧/Memory、连续失败状态、reference reliability。Global Recovery 的候选动作仍为 Reloc3r、reference reranking、RoMa；不新增 LLM Agent。
- **信息边界**：tracking、状态更新和动作选择只能读当前帧及过去帧；query GT、未来帧和事后 reference 选择只允许出现在独立 evaluator/审计中。验证通过的恢复 pose 才能写回后续状态。
- **下一步实验**：先登记并实现 temporal tracking + stateful deterministic baseline，比较“逐帧 Reloc3r/Memory”与“状态驱动恢复”的连续失败长度、轨迹跳变、覆盖率、成功率和统一成本；完成前不进入完整 Agent。

## AVL-009 — Causal Sequence-based Visual Localization Baseline（运行前登记，2026-10-10）

- **状态 / Claim**：`completed with limits`（协议已在运行前登记）。在四条 LILocBench query 的连续 RGB 顺序上，严格只读当前及过去帧的 temporal edge、当前 global candidate 和可靠历史状态，是否能比独立帧 Reloc3r/Memory 产生更短失败段、更少跳变，尤其跨过短时动态遮挡；本阶段不引入 LLM、VLM、SFT 或 Agent。
- **数据与边界**：mapping=`mapping` 的 785 帧保持 mapping-only；query=`static_0/dynamics_0/lt_changes_0/lt_changes_dynamics_0` 共 1696 帧，按原 manifest 顺序逐帧处理。mapping camera pose 是允许的地图资产；query GT 只在独立 evaluator 读取。前 40% 为 validation、后 60% 为 test；test 阈值冻结后才运行，结果明确称 same-scene temporal hold-out，不称跨场景泛化。
- **Temporal tool**：用固定 Reloc3r-512 对 `(I_{t-1}, I_t)` 生成相邻 edge，按已有 `mapping_pose @ relative_pose` 组合约定解释为 `T_{t-1→t}`，在线传播 `T_world_t=T_world_{t-1} @ T_{t-1→t}`。edge 无效则标 unavailable；不读取 query GT，不用未来帧。方向通过实现审计和独立 evaluator 检查，不用 test GT 选择方向。
- **State**：每序列维护 `last_frame_idx/last_accepted_pose/last_accepted_frame/previous_pose/previous_confidence/failure_streak/recent_reliable_history/last_global_anchor`；只有经过几何一致性验证的预测才能写回，GT 永不写回。memory 是历史状态，不是未来帧检索。
- **Global candidate 与预注册阈值**：复用 AVL-002 的 mapping-only Top-10 相对 pose 和 AVL-003 的 consensus-3；global 内部一致性阈值固定为 translation dispersion `1.5 m`、rotation dispersion `15°`。候选与历史/temporal 的一致性门限固定为 `2.0 m/30°`，单步 jump guard 固定为 `3.0 m/45°`，memory hold 最多连续 `3` 帧；本批次不在 test 上调阈值。
- **确定性动作**：`accept_global`、`accept_temporal`、`use_memory`、`reject`。global 与 temporal 一致时优先 global；global 不稳定时在 jump guard 内优先 temporal；两者不可用时，只有 memory hold 未超过 3 帧才使用 memory，否则 reject。该规则不读取任何 pose GT、RoMa 输出或未来帧。
- **Recovery ablation**：基础序列 baseline 后才运行 reference reranking→RoMa recovery；RoMa 固定 seed，使用统一成本边界，仍不读取 query GT。恢复结果必须通过同一验证门限后才能写回状态。
- **必须比较的方法**：原始 Reloc3r Top-10、consensus-3、causal memory-3、Always-RoMa、temporal-only、global+temporal、global+temporal+memory、global+temporal+memory+RoMa recovery。Always-RoMa 和 RoMa recovery 的缓存结果只作为候选/计时输入，在线 trace 仍需标记 GT-free。
- **指标**：success@0.25m/5°、median translation/rotation error、最大/均值/中位连续失败长度、trajectory jump rate、failure recovery latency、temporal drift length、reject rate、global/RoMa call rate；成本分别记录 temporal Reloc3r、global Reloc3r、Memory、rerank、RoMa、verification、总 serial latency 和 GPU-hours，不能从 call rate 直接推算 latency。
- **输出与审计**：写入 `outputs/lilocbench/AVL-009/{temporal_edges,sequence_baseline,sequence_memory,sequence_recovery,validation,test,failure_analysis,cost}/` 和 `summary.json`。每帧 trace 保存 query index、候选可用性、selected action/source、failure streak 前后、`uses_future=false`、`query_gt_read=false`。任何 GT 错配、未来帧访问、test 阈值调参或协议变更均标记 invalid，不覆盖旧结果。
- **Go / No-Go**：不要求平均 success 必须提升；若时序信息明显缩短失败段、跨越短期遮挡、显示 global/temporal 互补且 recovery 状态回写有价值，则支持继续 Stateful Agent 研究；若时序几乎没有独立收益，则暂停该假设并维持 Agent No-Go。

### AVL-009 结果（2026-10-10，completed with limits）

- **执行与审计**：四序列相邻 edge 共 1692/1692 有效；在线 trace 均为 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。方向审计保存在 `outputs/lilocbench/AVL-009/validation/temporal_direction_audit.json`，正向组合约定优于取逆，但仍有显著累积漂移。
- **Test 总体（1016 帧）**：Top-10=`27.85%`、consensus-3=`72.15%`、memory-3=`76.87%`、temporal-only=`0.00%`、global+temporal=`53.64%`、global+temporal+memory=`53.64%`、global+temporal+memory+RoMa=`53.74%`、Always-RoMa=`92.62%`。最大失败段分别为 `37/20/21/346/270/270/270/6` 帧。
- **场景 test**：memory-3 / global+temporal / recovery / Always-RoMa success 分别为 static `91.04/84.10/84.10/93.64%`，dynamic `56.99/52.69/53.76/93.55%`，LT `85.77/79.45/79.45/92.49%`，dynamic+LT `60.49/1.23/1.23/91.36%`。Dynamic+LT 的 270 帧连续失败来自 temporal drift，不是瞬时 trajectory jump。
- **成本**：temporal edge inference batch 实测总计 `457.43 s`（未单独记录 preprocessing）；global Reloc3r 继承 AVL-005 八帧样本的透明缩放估计 `3273.02 s`；retrieval=`25.60 s`；recovery 缓存 RoMa 使用 1 次、`2.94 s`；memory/verification 使用已登记的 CPU proxy。已知组件合计=`3845.38 s`，reference reranking 未在 AVL-009 replay 中重新计时，不支持速度优势声明。
- **结论**：当前 Reloc3r 相邻帧 edge 不能证明连续时序具有独立定位收益；memory-3 单独仍有价值，但在 temporal 漂移后简单 memory hold 不足。长期变化和 dynamic+LT 仍需可靠 global/强几何恢复。该结论不外推为 DPVO/其他 VO 失败。
- **无效/修正记录**：初版 global 冲突优先级错误保留在 `outputs/lilocbench/AVL-009/invalid_fusion_v1/`；初版 recovery 未覆盖低置信度 memory hold 保留在 `invalid_recovery_v2/`；两者不进入主结果。
- **下一步**：保持 Agent No-Go；若继续 Stateful，先替换真正的 temporal tracker（如 DPVO/几何光流），重新进行同一 causal validation/test，再评估 Agent 是否有必要。

## AVL-010 — Strong Temporal Tracker Baseline（运行前登记，2026-10-10）

- **状态 / Claim**：`completed with limits`。在不引入 Agent 的前提下，用真正的连续单目 temporal tracker（首选 DPVO）替换 AVL-009 的 Reloc3r 相邻帧 edge，检验时序失败究竟来自“时序信息本身没有独立价值”，还是来自 Reloc3r temporal edge 的漂移/模型能力不足。主线仍为 failure-aware continuous localization，不改变研究问题。
- **数据与协议**：完全复用 AVL-009 的四条 LILocBench query（static、dynamic、LT、dynamic+LT）、mapping-only reference database、原 manifest 顺序以及 40% validation / 60% test 的 causal split。阈值、reference 候选、Memory-3、Always-RoMa 和 evaluator 边界冻结，不因 DPVO 结果在 test 上调参。该 test 仍是 same-scene temporal hold-out，不称跨场景泛化。
- **信息边界**：tracker 只读当前与过去 RGB；global anchor 只来自已接受的 mapping-only Reloc3r/RoMa 结果；anchor 以前的 query GT、未来帧和 test evaluator 信息不可见。只有 verification 通过的 anchor 才可更新状态。所有输出 trace 必须保存 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。
- **Temporal tool**：首选 DPVO 官方实现/预训练权重；若环境或权重不可用，必须记录为环境 No-Go，不能用未验证替代物冒充 DPVO。MASt3R-SLAM 只作为后续可选强对照，不与 DPVO 结果混写。记录 tracker 版本、权重 hash、CUDA/环境和每序列处理耗时。
- **单目尺度与 causal anchoring**：DPVO 的局部轨迹不直接与 metric map 比较。状态维护局部 VO pose 和最近已接受 global anchors；至少两个时间有序、经 verification 的 global anchors 才估计 Sim(3) 的尺度/旋转/平移映射，之后在 anchor 间传播。相邻 anchor 平移基线小于 `0.05 m` 时不更新尺度；尺度更新限制在预注册 `[0.1, 10]`，异常则 reject/等待下一 anchor。首个 anchor 或不足两个 anchor 时只报告 temporal-local 状态，不伪造 metric pose。
- **比较方法**：`Memory-3`、`DPVO temporal-only`、`Global+DPVO`、`Global+DPVO+Memory`、`Global+DPVO+Memory+RoMa recovery`、`Always-RoMa`；AVL-009 的 Reloc3r temporal-only 和 global+temporal 作为历史对照，不覆盖旧结果。Global、Memory、RoMa 的 action 规则沿用 AVL-009 冻结规则，不能读取 GT。
- **主要指标**：success@`0.25 m/5°`、median translation/rotation error、最长/均值/中位连续失败段、trajectory jump rate、failure recovery time、global relocalization frequency、RoMa call rate、coverage/reject rate，以及 DPVO 局部轨迹漂移和 anchor 间 scale consistency。成本按相同串行边界记录 tracker、global、Memory、verification、RoMa 和总时间；不能由 call rate 推断加速。
- **预注册判定**：若 `Global+DPVO(+Memory)` 在 dynamic 上明显减少连续失败、跨过短时遮挡，并相对 AVL-009 的 global+temporal 有稳定 success/恢复优势，则支持继续构建无 Agent 的 stateful environment；若 DPVO 在相同 anchor/成本边界下没有独立收益，暂停 sequence-based 假设，回到 map-based failure-aware routing。无论结果如何，不在本实验中实现 LLM/VLM Agent。
- **输出与审计**：写入 `outputs/lilocbench/AVL-010/`，至少包含 tracker 原始轨迹、anchor/Sim(3) 参数、GT-free causal traces、validation/test evaluator、failure analysis、cost ledger、环境/权重 manifest 和最终 `summary.json`；失败、环境不可用或协议错误均保留，不覆盖 AVL-009 产物。

### AVL-010 结果（2026-10-10，completed with limits）

- **正式输入边界**：原始 LILocBench front-RGB 全流约 15 Hz（static/dynamic/LT/combo=`8975/2398/6535/8378` 帧），只在与 AVL-009 相同的 1 Hz query 时间戳上评测；DPVO 每帧在线输出，未调用 terminate/final batch optimization。1 Hz-only DPVO 结果保留为边界对照，不作为主结果。
- **运行与审计**：官方 DPVO commit=`0ac95b656d1fda91c271d2a106460d19ad966fc7`，权重 hash 记录在 `docs/avl010_strong_temporal_tracker_report.md`。DPVO full-rate online pose 全部有效；四个 trace 的 `query_gt_read/uses_future/selection_uses_gt` 均为 false，审计见 `outputs/lilocbench/AVL-010/eval_full_rate_v3/audit/causal_audit.json`。
- **Test 总体（1016 帧）**：Memory-3=`76.87%`、DPVO temporal-only=`41.54%`、Global+DPVO=`40.55%`（coverage=`53.54%`）、Global+DPVO+Memory=`40.55%`（coverage=`88.39%`）、Global+DPVO+Memory+RoMa=`68.31%`、Always-RoMa=`92.62%`；最大失败段分别 `21/324/346/346/76/6` 帧。
- **条件结果**：static=`91.04/49.13/0.00/77.75/93.64%`，dynamic=`56.99/39.78/55.91/55.91/93.55%`，LT=`85.77/84.98/87.75/87.75/92.49%`，dynamic+LT=`60.49/0.00/42.59/46.60/91.36%`（顺序为 Memory-3 / DPVO temporal-only / Global+DPVO / +Memory+RoMa / Always-RoMa）。LT 上 Global+DPVO 相对 Memory-3 仅 `+1.98 pp`，为条件性小收益；dynamic 与 combo 没有独立收益。
- **成本**：DPVO full-rate online tracking 实测 `698.88 s`（约 `26–27 ms/frame`，GPU=`0.194 h`）；继承 global Reloc3r 估计=`3273.02 s`，固定 seed RoMa recovery 选择 `17` 次、缓存时间=`51.09 s`，已知组件=`4022.99 s`。因为边界混合，不作端到端速度声明。
- **实现修正记录**：`stateful_full_rate_v1/v2` 使用的初版 anchor 对齐与固定方法汇总已保留但不进入主结果；最终仅使用 `stateful_full_rate_v3` + `eval_full_rate_v3`。1 Hz-only、旧 evaluator 和失败目录均不覆盖。
- **结论 / Go-No-Go**：DPVO 证明“时序完全无价值”不成立，但不足以证明 sequence-first stateful pipeline 普遍优于 Memory/geometry；主线保持不变，Agent 仍 No-Go。DPVO 仅保留为受限 tracking 信号，下一步回到 deterministic global-anchor/recovery 和 Dynamic+LT 共同失败分析。

## AVL-011 — Dynamic+LT 共同失败诊断与强化 deterministic recovery（运行前登记，2026-10-10）

- **主张与范围**：本实验服务于同一条 failure-aware visual localization 主线，不构建 Agent。第一部分对 `lt_changes_dynamics_0` 逐帧诊断：reference 空间覆盖不足/检索错选、候选冲突、动态/遮挡代理信号、长期结构变化代理、RoMa 有输出但错误、Memory 是否可用。所有带 GT 的类别仅为运行后诊断，不进入在线动作选择。
- **在线协议**：完全复用 AVL-009/010 的 mapping-only database、1 Hz query、40% validation / 60% test causal split。在线 trace 只能使用当前/历史 RGB、Reloc3r 候选证据、已验证 Memory 和当前 geometry 输出；禁止 query GT、未来帧和事后标签。GT 只由独立 evaluator 读取。
- **强化 baseline**：固定 `Reloc3r → reliability → reference/rerank geometry → Memory fallback → RoMa verification → Reject` 顺序。geometry verification 除 inlier/reprojection 外增加 reference-pair support、candidate/memory jump guard；阈值只在 validation 预注册网格中选择，test 冻结。使用已落盘的 full-frame RoMa cache 作为工具输入，明确标注其历史 seed/运行版本限制，不把缓存 replay 写成在线耗时。
- **判定指标**：逐帧 success、coverage、最长连续失败、recovery latency、RoMa call rate、Reject rate；诊断报告必须给出 Memory-only、RoMa-only、both-success、both-fail 及各失败类别的交叉表，并保存代表帧清单/contact sheet。
- **输出**：协议、trace、validation/test evaluator、诊断表和 contact sheet 均追加至 `outputs/lilocbench/AVL-011/`；任何 GT 泄漏、类别无法由现有证据支持或阈值使用 test 标签的结果标记为限制，不覆盖 AVL-002–010。

### AVL-011 结果（2026-10-10，completed with limits）

- **Dynamic+LT 逐帧诊断（541 帧）**：`both_success=326`、`memory_only=14`、`RoMa_only=163`、`both_fail=38`。在 GT 后验的空间覆盖 proxy 下，Top-100 中 `516/541` 帧存在空间上接近的 mapping reference，`25/541` 帧表现为“Top-10 选错但 Top-100 有近邻”，没有出现“Top-100 全部空间上不可用”的帧。因此当前共同失败主要不是地图区域完全缺失，而是 reference 外观不兼容、候选冲突、动态/长期变化和几何误接受。
- **共同失败交叉证据**：38 个 both-fail 帧中，32 帧带有 `reference_mismatch` operational label，37 帧带有 candidate conflict，22 帧带有 low-texture/blur proxy，8 帧带有 occlusion-or-motion proxy，5 帧带有 persistent-scene-change proxy；38 帧均有 RoMa pose 字段但 GT 后验错误。16 帧同时显示“内部一致的 Memory 候选但实际错误”，说明仅靠 candidate dispersion 不能保证 Memory 可靠。以上是后验/代理标签，不是真值语义分类。
- **代表性证据**：`outputs/lilocbench/AVL-011/failure_diagnosis/lt_changes_dynamics_0/` 保存 rows、both-fail/memory-only/RoMa-only/both-success contact sheets，以及按 reference selection、动态观测和长期变化 proxy 的子集图。该审计显示部分帧有行人或门/家具变化，但不能据此声称完成了行人遮挡真值标注。
- **强化 deterministic baseline**：validation 只在 40% causal prefix 上选择规则；test 完全冻结。最终规则为“candidate conflict 直接调用 geometry；geometry 通过 inlier/reprojection/pair-support 基本验证后接受；失败时优先当前可靠 Consensus，再使用历史 Memory；只有无可用回退才 Reject；只有 accepted 输出可更新 Memory”。最终 trace：`outputs/lilocbench/AVL-011/final_v1/`。
- **Test（1016 帧）**：总体 success=`91.24%`、coverage=`100%`、geometry call rate=`67.13%`、最长失败=`4` 帧、平均 recovery latency=`1.47` 帧。四条件 test success 为 static=`91.62%`、dynamic=`92.47%`、LT=`90.12%`、dynamic+LT=`91.36%`。Validation success=`89.12%`，规则没有使用 test GT 选择。
- **边界与解释**：AVL-011 使用已有 full-frame RoMa cache；缓存运行版本/seed 与 AVL-005 不同，故这里只能证明 deterministic action trace 和 verification 逻辑的效果，不能宣称新的在线 RoMa latency 或跨 seed 稳定性。geometry call rate 也不能直接等价为端到端加速。
- **结论 / 下一步**：AVL-011 将主线正式收敛为 **Failure-aware Visual Localization in Dynamic Indoor Environments**。连续序列保留为 Memory、failure streak 和 motion evidence 的输入；主要创新不再定义为 sequence-first localization。下一步优先针对 38 个 both-fail 和 16 个错误 Memory 帧做 reference reranking、跨 reference 一致性与 reject 校准；不引入 Agent。

## AVL-012 — Localization Reliability & Reject（运行前登记，2026-10-10）

- **主张与范围**：继续不引入 Agent，检验 deterministic pipeline 能否把“有 pose”与“可信 pose”分开。实验只服务于 failure-aware visual localization 主线，重点是 reference reliability、pose verification 和 Reject calibration。
- **输入边界**：复用 AVL-011 的 mapping-only reference、Reloc3r candidates、causal Memory、full-frame RoMa evidence 和最终 trace；在线可靠性分数只能使用当前帧与历史状态，禁止 query GT、未来帧、后验 outcome label。GT 仅由独立 evaluator 计算 coverage/accuracy。
- **Reference reliability**：冻结以下证据字段：Top-k candidate translation/rotation dispersion、RoMa pair-support/intersection 数、inlier ratio、median reprojection、reference score/selection 信息，以及 geometry 与 Reloc3r consensus、Memory、previous prediction 的 cross-check。空间 GT 近邻只用于事后诊断，不进入 score。
- **Pose verification**：对 consensus、Memory 和 RoMa 输出分别计算 evidence-only reliability score；RoMa 必须同时记录多 reference support、inlier/reprojection 和 temporal/memory/candidate cross-check。不得把单一 inlier 或低 reprojection 当作充分接受条件。
- **Reject calibration**：在 validation 的 causal prefix 上冻结 score 公式和阈值网格 `{0.25,0.5,0.75,1,1.25,1.5,2,3}`；test 只生成 Accuracy–Coverage / selective accuracy 曲线，不按 test GT 选阈值。报告 accept/recovery/reject 数量、coverage、success-all-frames、selective accuracy、错误接受率、最长失败和恢复延迟。
- **预注册判定**：若 test 上降低 coverage 能显著提高 selective accuracy、降低错误接受，并形成稳定可解释曲线，则说明 deterministic reliability 尚有可用校准空间；若风险分数不能排序错误，记录为 reliability bottleneck，作为未来 Agent 的潜在入口，但本实验不实现 Agent。
- **输出与审计**：追加到 `outputs/lilocbench/AVL-012/`，保存逐帧 evidence、冻结 score manifest、validation/test 曲线、代表性 accepted/rejected/error contact sheets 和 GT-free audit。任何 score 使用 GT/未来信息均使对应结果 invalid，不覆盖 AVL-011。

### AVL-012 结果（2026-10-10，completed with limits）

- **实现与边界**：`tools/run_avl012_reliability.py` 从 AVL-011 的 causal trace、AVL-002 Reloc3r candidate 和 AVL-004 RoMa cache 生成 1696 条 evidence-only 记录；记录 candidate dispersion、appearance/motion、inlier ratio、reprojection、multi-reference pair support，以及 geometry 对 consensus/Memory/previous prediction 的 cross-check。score 分支只读取 mapping GT 作为地图坐标资产，不读取 query GT；所有原 trace 的 `query_gt_read/uses_future/selection_uses_gt` 均为 false。
- **冻结规则**：风险分数为 candidate、temporal/appearance、geometry verification 三类无量纲风险的最大值。validation 只在阈值网格 `{0.25,0.5,0.75,1,1.25,1.5,2,3}` 中选择；按预登记规则（coverage≥50% 时 selective accuracy 最大，平局取更高 coverage）冻结 `threshold=1.5`。test 未参与阈值选择。
- **Test Accuracy–Coverage 曲线（1016 帧）**：阈值 `0.5/0.75/1.0/1.25/1.5/2.0/3.0` 的 coverage 分别为 `5.91/22.64/37.50/48.23/55.22/62.70/74.70%`；selective accuracy 分别为 `96.67/96.52/96.33/96.12/95.72/95.45/94.60%`；错误接受率分别为 `3.33/3.48/3.67/3.88/4.28/4.55/5.40%`。曲线显示风险分数能排序一部分不可靠 pose，但 coverage 提升会稳定牺牲选择性准确率。
- **冻结阈值结果**：`threshold=1.5` 接受 `561/1016` 帧（coverage=`55.22%`），其中正确 `537`、错误 `24`；按动作 accepted 为 `call_geometry=242`（recovery）、`accept_consensus=169`、`use_memory=150`，Reject=`455`。最长“正确输出/否则失败”段为 `36` 帧。相同 AVL-011 route 不拒绝时为 coverage=`100%`、success=`91.24%`；因此 Reject 校准降低错误接受，但代价是大量 abstention，不能声称整体定位 success 提升。
- **条件 test（阈值 1.5）**：static coverage/selective accuracy=`70.81/93.88%`，dynamic=`29.03/100.00%`，LT=`55.73/96.45%`，dynamic+LT=`45.68/97.30%`。dynamic 的低 coverage 表明当前证据分数对困难观测过于保守，不能直接作为最终在线策略。
- **可视化与审计**：代表帧 contact sheets 位于 `outputs/lilocbench/AVL-012/contact_sheets/`；evaluator-only 行位于各 sequence 的 `evaluator_rows.jsonl`，score 输入仍不读取这些文件。实现和输出保留了 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false` 审计字段。
- **结论 / 下一步**：AVL-012 支持“reference/几何/历史交叉证据可以校准 Reject”这一中间结论，但不支持把当前分数直接部署为闭环 Manager：在可接受 coverage 下仍有明显 abstention，且本批次是 AVL-011 缓存回放，未测新的在线端到端成本和 Reject 后状态回灌。继续保持 Agent No-Go；下一步应做真正 online reject→state update 的 causal replay，并在统一成本边界下比较 coverage、错误接受和恢复延迟。

## AVL-013 — Closed-loop Reject & Recovery（运行前登记，2026-10-10）

- **主张与范围**：继续不引入 Agent，检验 AVL-012 的可靠性 Reject 在真正 causal 状态机中是否减少错误传播，并能在后续帧通过 consensus、历史 Memory 或 RoMa recovery 恢复。实验回答最终在线系统的 success、coverage、错误接受、连续 Reject/恢复时间和成本，而不是只报告 accepted-pose accuracy。
- **协议与信息边界**：完全复用 AVL-011 的 mapping-only reference、1 Hz query、40% validation / 60% test 和 RoMa cache；每一帧只访问当前 candidates、当前 geometry evidence 和此前已接受的 state。Reject 不更新 `previous_prediction`、Memory 或可靠性状态；后续帧从未被污染的状态重新计算 selections。query GT 仅由独立 evaluator 读取，不能参与 action、score、state update 或阈值选择。
- **动作状态**：`accept_consensus`、`use_memory`、`call_geometry`/`geometry_recovery`、`reject`。geometry 通过 AVL-011 的 inlier/reprojection/pair-support/运动验证后才可写回；geometry 失败时先尝试当前 consensus，再尝试历史 Memory，否则保持 Reject。只有最终 accepted pose 才更新 state/memory。
- **阈值与比较**：固定使用 AVL-012 validation 冻结的 risk threshold=`1.5`；同时保留 `no_reject` AVL-011 route 作为对照。不得在 test 上调整 threshold。报告 validation/test 的 success、coverage、accepted-pose accuracy、错误接受率、最长连续 Reject/失败段、从 Reject 到下一次可靠 pose 的恢复帧数、state contamination proxy 和各动作调用率。
- **成本边界**：逐帧记录 routing、candidate selection、verification、state update 和 cache lookup 的墙钟时间；同时单独列出 Reloc3r/RoMa 的真实在线 inference 时间。若使用历史 cache replay，必须明确标记为 replay cost，不能冒充端到端在线 latency。统一 ledger 以同一序列、同一帧集合和同一串行边界汇总。
- **预注册判定**：若 Reject 闭环相对 no-reject 对照降低错误传播/最长失败，且在 coverage 不低于 `0.50` 时保持可解释的恢复收益，则支持继续 deterministic stateful pipeline；若 Reject 只降低 coverage 而不改善最终 success、恢复或错误传播，则记录为 reliability calibration bottleneck，仍不引入 Agent。
- **输出与审计**：追加到 `outputs/lilocbench/AVL-013/`，保存 causal trace、state transitions、validation/test evaluator、recovery/reject 分析、统一 cost ledger 和 audit。任何 query GT、未来帧或 test 标签进入 state/action 均使运行 invalid，不覆盖 AVL-011/012。

### AVL-013 补充协议 v2（运行前登记，2026-10-10）

- **问题驱动修正**：严格 v1 若连续 Reject 后仍把最后一个可靠 pose 用作无限期 jump reference，可能把“状态陈旧”误判为当前 geometry 错误，造成 recovery deadlock。v1 trace 保留为 exploratory，不能作为最终策略结论。
- **冻结恢复规则**：当 `reject_streak >= 3` 时进入 `stale_state_recovery`；不删除历史状态，但 risk score 暂停使用陈旧 previous-prediction 的 temporal/cross-check 惩罚，只使用当前 candidate dispersion、当前帧 appearance/frame evidence 和 geometry intrinsic evidence。恢复 pose 通过验证且 risk≤1.5 后，才首次写回 prediction/Memory，并清零 reject streak。
- **因果边界**：该规则只使用过去 reject streak 和当前证据，不读取 GT、未来帧或 test outcome；v1/v2 均必须独立报告，不能事后只保留较优版本。

### AVL-013 结果（2026-10-10，completed with limits）

- **v1 严格闭环**：Reject 后保持最后可靠 pose 的 temporal/cross-check，test coverage=`0.39%`、最长连续 Reject=`991` 帧，出现 recovery deadlock。该版本保留为 exploratory，不能作为最终策略。
- **v2 stale-state recovery**：连续 Reject 达到 3 帧后暂停陈旧 previous-prediction 的 temporal/cross-check 惩罚；只有 risk≤1.5 且 verification 通过的 pose 才写回 state/Memory。validation test 冻结协议下，test（1016 帧）coverage=`54.23%`、最终 success=`51.57%`、accepted-pose accuracy=`95.10%`、错误接受率=`4.90%`、最长连续 Reject=`26`、平均恢复到首个 accepted=`3.94` 帧、平均恢复到首个 correct=`4.27` 帧。接受 `551` 帧（正确 `524`、错误 `27`），拒绝 `465` 帧。
- **错误传播比较**：AVL-011 no-reject test 的错误 state update=`89`、连续错误 state proxy=`30`；AVL-013 v2 分别为 `27` 和 `5`。这支持 Reject 阻断错误写回，但全帧 success 从 `91.24%` 降至 `51.57%`，说明当前阈值/恢复证据过于保守。
- **条件 test（v2）**：static/dynamic/LT/dynamic+LT coverage=`68.79/29.03/54.15/45.99%`，最终 success=`64.45/27.96/52.17/44.14%`，accepted accuracy=`93.70/96.30/96.35/95.97%`。Dynamic 的低 coverage 是当前主要瓶颈。
- **成本与审计**：状态机 replay 实测墙钟=`101.91 s`；历史 RoMa cache 对应 geometry elapsed=`2634.97 s`。本批次未成功重新执行 Reloc3r/RoMa neural inference，`cost_complete_online_inference=false`，不能把 cache elapsed 写成新的端到端 latency。补做的全帧 Reloc3r 在线计时尝试因未设置本地 HF 离线变量而触发 Hub 访问，随后 GPU driver 不可用，未产生有效 pose shard；失败产物保留在 `outputs/lilocbench/AVL-013/online_cost/`，manifest 标为 `invalid_blocked`。v1/v2 trace 均 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。
- **结论 / 下一步**：AVL-013 证明 Reject 能减少错误状态污染，且 stale-state recovery 可避免死锁；但当前 deterministic 闭环以 coverage 和最终 success 为代价，尚不支持进入 Agent。下一步补齐真实在线 Reloc3r/RoMa 计时，并优化 coverage–error–recovery 联合校准；继续 Agent No-Go。

## AVL-014 — 关键实验实现修正与配对重验（运行前登记，2026-10-10）

- **目的**：修复审计发现的 AVL-010 Sim(3) 旋转缩放错误；让 AVL-012 的 geometry evidence 只在 causal route 实际调用 geometry 的帧可用；修正 AVL-013 的跨序列连续段、右删失和状态 trace；并在相同输入缓存上对 AVL-011 做有效 geometry jump guard 配对消融。
- **信息边界**：不重新生成 DPVO/RoMa/Reloc3r 神经输出；所有回放只读取当前帧及过去状态，query GT 只由独立 evaluator 读取。旧结果和旧 evaluator 不覆盖，新增产物使用 `AVL-014/*` 子目录。
- **AVL-010 修正**：Sim(3) 对 SE(3) pose 分开变换旋转和平移，验证输出旋转正交性；使用同一 full-rate DPVO trajectory 重放并重新评测。
- **AVL-012 修正**：geometry evidence 仅在 AVL-011 route 的 `call_geometry` 帧读取；validation 仍按“coverage≥0.5 下 selective accuracy 最大，tie 取 coverage 更高再取阈值更低”冻结阈值；test 不调参。
- **AVL-013 修正**：使用 AVL-014 阈值和同一 selective-geometry risk 规则；状态 trace 保存更新前可靠性/上一接受帧；evaluator 按序列独立计算 run，聚合时报告正确恢复数、右删失数和恢复率。
- **AVL-011 配对消融**：复用 frozen validation/test protocol、mapping/reference/RoMa cache 和所有非 jump 阈值，仅将 geometry jump guard 从 `100m/180°` 改为 `2.5m/60°`；不使用 test GT 选择参数。
- **预注册判定**：若修正后的 DPVO 指标仍无 Dynamic/Dynamic+LT 独立收益，才保留该限定结论；若 selective geometry 重校准后 v2 仍在 coverage≥0.5 下减少错误接受而不造成不可解释死锁，则支持继续 deterministic recovery；jump guard 只报告 paired accuracy/coverage/error propagation 差异，不据单次消融进入 Agent。
- **状态**：`running`。结果追加到本条目，不覆盖 AVL-010–013。

### AVL-014 结果（2026-10-10，completed with limits）

- **AVL-010 Sim(3) 修正**：复用原始 DPVO full-rate trajectory，修正后 DPVO propagated 输出旋转满足正交性（最大行列式偏差约 `1.4e-6`，最大正交误差约 `2.0e-6`）。test（1016 帧）为 DPVO temporal-only=`0.20%`、Global+DPVO=`4.04%`、Global+DPVO+Memory=`4.04%`、Global+DPVO+Memory+RoMa=`65.26%`、Always-RoMa=`92.62%`。旧 AVL-010 派生指标因旋转缩放错误作废；原始 DPVO 推理轨迹保留有效。新产物：`outputs/lilocbench/AVL-014/AVL-010_stateful_sim3fix/`、`AVL-010_eval_sim3fix/`。
- **AVL-012 selective-geometry 重校准**：geometry evidence 仅在 AVL-011 route 的 `call_geometry` 帧读取；validation 仍选择 threshold=`1.5`。test coverage=`55.41%`、selective accuracy=`95.56%`、错误接受率=`4.44%`。该结果仍是缓存回放校准，不是完整 neural online cost。新产物：`outputs/lilocbench/AVL-014/AVL-012_selective/`。
- **AVL-013 v3**：使用同一 threshold 和 selective-geometry risk，修正状态 trace（保存更新前可靠性和上一接受帧）及按序列 recovery evaluator。test coverage=`54.23%`、全帧 success=`51.57%`、accepted-pose accuracy=`95.10%`、最长 Reject=`26`；Reject 段 `119`，恢复 `117`，右删失 `2`；平均恢复到首个 accepted=`3.87` 帧、首个 correct=`4.21` 帧。GT/future audit 全部通过。新产物：`outputs/lilocbench/AVL-014/AVL-013_v3/`。
- **AVL-011 jump-guard paired ablation**：在相同 frozen inputs 下将 geometry jump guard 设为 `2.5m/60°`。test coverage=`98.82%`、success=`87.50%`、最长失败=`14`、geometry call=`67.13%`，相对旧宽松 guard 的 `100%/91.24%/4` 下降。说明旧 91.24% 是 cached deterministic route 结果，不能表述为严格运动验证后的可靠性提升。新产物：`outputs/lilocbench/AVL-014/AVL-011_jumpguard/`。
- **统一判定**：AVL-010 修正后的时序结论可引用为“当前 DPVO/anchor 实现没有独立定位收益”；AVL-012/013 仍是 coverage–error calibration bottleneck；AVL-011 的强几何路线需保留宽松 guard 与严格 guard 两个版本。Agent 继续 No-Go。
- **成本决策**：完整 Reloc3r/RoMa online timing 暂缓。当前 deterministic route 的 Sim(3)、selective evidence 和 jump-guard 刚完成修正，尚未形成稳定的最终策略；现在测量会把成本绑定到仍在变化的路线。待下一版 recovery policy 冻结后再做统一串行成本测量。
- **状态**：`completed with limits`。

## AVL-015 — Reliability–Recovery Frontier（运行前登记，2026-10-10）

- **状态 / 范围**：`completed with limits`。全程保持 Agent Hold，未训练或运行 LLM/VLM Agent。本实验固定底层环境为 **Map Localization + Experience Memory + Geometry + Reliability**，回答了动作互补、可靠性判断和历史关键帧恢复三个问题，但 dynamic re-trust 仍是 evidence replay、Historical Memory 当前为负结果。
- **Q1：动作互补空间**：只把最终产生 pose 的分支计为 pose action：`Reloc3r consensus`、`Historical-keyframe relocalization`、`rerank + Reloc3r`、`rerank + RoMa`，必要时加入 `HLoc-style pose`。`rerank` 单独是中间工具动作，`Reject` 是 abstention，二者不定义“定位成功集合”。
- **Pose Oracle**：在独立 evaluator 中用 query GT 计算 `exists a: P_a correct`，报告每个 pose action 的独立成功率、两两互补矩阵、only-action 集合和 union 上界。GT 结果只用于离线 oracle/evaluator，不进入在线候选、验证或决策。
- **Cost-aware Oracle**：在统一串行成本边界下，求达到指定 full-frame success 时的理论最低工具成本，并绘制 `Full Success – Geometry Calls – End-to-end Cost` Pareto frontier。重点比较最佳固定策略与 oracle 是否存在“更高成功率”或“相同成功率、更低成本”的有意义改进；不预设必须提高 3–5 pp。
- **Q2：source-specific verification**：Consensus 分支使用 candidate dispersion、reference agreement、retrieval similarity；Historical Memory 使用 keyframe similarity、relative-pose geometry、keyframe reliability 与 age；RoMa/HLoc-style 使用 multi-reference agreement、inlier/reprojection 与 cross-tool agreement。不同来源分别校准，禁止复用一个无来源区分的统一 reliability score。
- **Dynamic re-trust**：旧状态约束随最近可靠 anchor 的 age、reject streak 和 motion uncertainty 动态衰减。刚获得可靠定位时使用强 consistency constraint；长期 Lost 后 stale pose 不得否决具有高质量独立证据的 fresh global localization。固定 `2.5m/60°` jump guard 仅保留为 AVL-014 对照，不作为本实验默认规则。
- **Frontier 指标**：不预设单一 `coverage>=80%` 门槛。完整报告 `Coverage – Selective Accuracy – Full-frame Success – False Acceptance – Recovery Time – Cost` frontier；Reject 帧在 full-frame success 中按失败计，并单独报告 abstention/右删失。
- **Q3：Historical Keyframe Relocalization**：Memory 从历史 pose fallback 升级为经过验证的临时地图节点。只写入可靠历史 keyframe 的 `RGB + descriptor + global pose + reliability + timestamp/age`；当前定位失败时检索历史关键帧，估计当前图像到历史帧的相对位姿，并按 `T^w_t = T^w_k T^k_t` 组合为当前 global pose，之后接受独立 source-specific verification。禁止直接复制历史 pose。
- **因果与泄漏审计**：在线模块只能访问当前帧、过去帧、mapping assets 和过去已验证状态；禁止 query GT、未来帧和事后标签。所有 trace 保存 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。GT 仅由分离 evaluator 计算 pose 正确性、oracle 和 frontier。
- **数据与冻结协议**：第一阶段复用 LILocBench mapping-only database、原四条 query 顺序和既有 40% validation / 60% same-scene temporal test；所有阈值、模型和 action cost 在 validation 冻结。该 test 仍不称独立泛化证据；进入最终 Agent 主张前必须补充未参与探索的新场景或跨数据集 held-out。
- **Agent Go 判定**：只有当冻结 test 上 Cost-aware Action Oracle 相比最佳固定策略存在有意义的 Pareto 改进、存在可观测的 action-dependent context、source-specific verifier 能控制错误接受/状态污染，且后续任务特定 Agent 在相同输入/动作/预算下超过规则与轻量 router，才进入 Agent。若 Always-RoMa 持续支配，或新增 Historical Memory 不形成稳定 only-action 集合，则保持 Agent No-Go。
- **计划产物**：`outputs/lilocbench/AVL-015/` 下分别保存 `pose_oracle/`、`cost_oracle/`、`source_verification/`、`historical_keyframe_memory/`、`frontier/`、`audit/` 和最终 `summary.json`。实验运行前不得用 test GT 调整协议；失败、无效与实现修正全部追加保留。

### AVL-015 Q2/Q3 实现补充协议（2026-10-10，运行前登记）

- **Q2 source-specific verifier**：从已有 causal AVL-012/013 evidence-only trace 中按 `consensus`、`memory`、`geometry` 分源计算风险，不使用 query GT/未来帧。阈值网格仍为 `{0.25,0.5,0.75,1,1.25,1.5,2,3}`；本次探索性校准对每个来源在 validation 选择 selective accuracy 最大且 coverage≥0.20 的阈值，test 冻结。该 20% 是分源校准的最低可观测覆盖约束，不是最终系统 coverage 目标。anchor age、reject streak 和 motion uncertainty 在本批 evidence 中尚未完整记录，dynamic re-trust 只作为下一版状态扩展，不能把本结果称为完整 verifier 闭环。
- **Q3 Historical-keyframe Relocalization**：在线回放使用 AVL-013 v3 中 `accepted=true && state_updated=true` 的过去 query 帧作为 keyframe，只保存其 RGB path、DINO descriptor、global pose、reliability/trace 状态和 age；当前帧从过去 keyframe 中按当前 DINO descriptor 取候选，要求 keyframe index 小于当前 index，默认 `num_refs=4`、`min_gap=5`。RoMa 仅在当前帧与历史 keyframe 之间匹配；历史 keyframe 对三角化 3D tracks，PnP 估计 `T^w_t=T^w_kT^k_t`，输出独立 evidence（inliers、reprojection、pair support），不直接复制历史 pose。没有至少两帧历史 keyframe 时输出 unavailable。该模块先产生 candidate，最终接受必须由后续 source-specific verifier 决定。
- **Q3 causal audit**：每帧保存 `max_historical_index < query_index`、`query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`；DINO embedding 只作为当前图像与已产生历史 descriptor 的检索表示。GT 只能由分离 evaluator 读取，用于报告 Memory-only/Map-or-RoMa-only/共同失败及 Accuracy–Coverage–Cost。

### AVL-015 Q1 结果（2026-10-11 重算，completed with limits）

- **输入与审计**：复用 AVL-004 `action_oracle/summary_always.json` 的 frozen pose outcomes，并加入 Q3 独立 causal Historical-keyframe trace；未驱动任何在线路由。`consensus`、`memory`、`HLoc-style`、`rerank+RoMa`、`Historical-keyframe` 均作为最终产生 pose 的 action，`rerank` 单独不计 action，Reject 不计成功。成本使用 AVL-005/006 已登记 timing boundary，Historical 使用其 trace 的实测 elapsed；GT 只由本 evaluator 读取，不能作为在线 action 输入。
- **全 1696 帧独立成功率**：`consensus=1198/1696=70.64%`、`memory=1301/1696=76.71%`、`HLoc-style=1503/1696=88.62%`、`rerank+RoMa=1547/1696=91.21%`、`Historical=698/1696=41.16%`；五者 Pose Oracle union=`1628/1696=95.99%`。冻结 temporal test 对应为 `72.15/76.87/88.98/92.22/44.78%`，union=`985/1016=96.95%`。Historical 相对全部其他动作只贡献 `4` 个全量、`2` 个 test 独占成功帧，未形成稳定独立恢复能力。
- **两两互补**：Memory/RoMa 为 `both=1245`、`memory-only=56`、`RoMa-only=302`、`neither=93`；HLoc/RoMa 为 `both=1466`、`HLoc-only=37`、`RoMa-only=81`、`neither=112`。原四动作严格 only-action 计数为 `consensus=4`、`memory=4`、`HLoc=17`、`RoMa=37`；加入 Historical 后其相对全部其他动作的 only-action 仅 `4`（test=`2`）。这证明原工具之间有动作互补，但 Historical 没有带来稳定新增集合，且精确计数仍受 seed/route 实现限制。
- **冻结 temporal test（1016 帧）**：`consensus=72.15%`、`memory=76.87%`、`HLoc=88.98%`、`RoMa=92.22%`、`Historical=44.78%`、Pose Oracle=`985/1016=96.95%`。Validation 为 `68.38/76.47/88.09/89.71/35.74%`（同顺序）。该 split 是 same-scene calibration-shift test，不是独立泛化证据。
- **Cost-aware oracle（GT 后验下界，成本边界修正后重算）**：只对 pose-producing action 计费，shared retrieval 只计一次；Reloc3r 为 AVL-006 估计，Memory 加 AVL-005 memory proxy，HLoc 包含其 rerank+geometry，RoMa 包含 top-100 query extraction+rerank+RoMa，Historical 使用 Q3 trace 的实测 cost。全量固定动作成本为 Consensus=`3311.0s`、Memory=`3397.3s`、HLoc=`2492.5s`、RoMa=`6622.6s`、Historical=`3714.7s`。理论 oracle 在达到 RoMa action pool 的 `91.21%` 成功率时最低约 `2461.5s`，达到五动作 union `95.99%` 时约 `2599.0s`；这是知道每帧 GT outcome 的不可部署下界，不能写成在线加速。
- **Accuracy–Geometry Calls–Latency frontier**：固定动作点已写入 `summary.json`；全量非支配点为 Memory=`76.71%/0/3397.3s`、HLoc=`88.62%/1696/2492.5s`、RoMa=`91.21%/1696/6622.6s`（格式为 success/geometry calls/latency）。Historical=`41.16%/1696/3714.7s` 被 HLoc 严格支配；Consensus 被 Memory 支配。该 frontier 是固定动作比较，不是可部署 Manager frontier。
- **Q1 结论**：原始 action pool 有理论互补和 cost Pareto 空间，但新增 Historical action 的独占集合极小，不能作为 Memory 新能力或 Agent Go 证据。继续保留 Q2/Q3 结果用于界定 failure-aware deterministic system；不进入 LLM Agent。
- **产物**：`tools/run_avl015_pose_oracle.py`；`outputs/lilocbench/AVL-015/pose_oracle/summary.json`。该脚本是 evaluator-only，输出中的 oracle、互补和 cost curve 均标记为 post-hoc upper bound。

### AVL-015 Q2 结果（2026-10-10，completed with limits）

- **分源校准结果**：在已有 AVL-012/013 causal trace 中，按来源分别冻结 validation 阈值后，test 的 source-conditional 结果为：Consensus threshold=`0.75`，coverage=`84.97%`、selective accuracy=`94.56%`、false acceptance=`5.44%`；Memory threshold=`1.0`，coverage=`50.58%`、selective accuracy=`93.10%`、false acceptance=`6.90%`；Geometry threshold=`0.5`，coverage=`34.58%`、selective accuracy=`99.14%`、false acceptance=`0.86%`。
- **解释边界**：这些 coverage 是“已被旧 causal route 选择到该来源的帧”中的 coverage，不是全帧 coverage；geometry 的高准确率以大量拒绝为代价。anchor age、reject streak 和 motion uncertainty 尚未进入 score，因此不能称为 dynamic re-trust 完成，也不能替代最终 online verifier。
- **结论**：不同 pose 来源确实需要不同 reliability score；统一 risk score 不是唯一合理设计。但 Q2 目前只证明 calibration signal，尚未解决 Dynamic/Dynamic+LT 的高 coverage recovery bottleneck。
- **产物**：`tools/eval_avl015_source_verifiers.py`；`outputs/lilocbench/AVL-015/source_verification/summary.json`。该结果保留为 `completed with limits`。

### AVL-015 Q2 dynamic re-trust 首轮结果（2026-10-10，completed with limits）

> 以下为当时的原始实验记录；其中“previous-state disagreement 已按 trust 衰减”的实现解释已被 2026-10-12 复核推翻。数值仅代表该冻结实现的回放结果。

- **实现**：在 AVL-013 v3 的已有 causal candidate trace 上回放 source-specific threshold；当 `reject_streak>=3` 时移除 stale temporal/previous-state veto，但保留当前 intrinsic geometry 与 cross-source evidence。该脚本不重新生成 route 或 pose，GT 只在独立 evaluator 中用于评分。
- **结果**：对全部 1696 个 evidence frame 进行 replay，并以新 replay 的 `reject_streak`、最近 accepted anchor 的 `anchor_age` 和 evidence 中的 `temporal_risk` 计算 `state_trust=exp(-anchor_age/20-reject_streak/3)`；旧状态与 geometry 的 previous-pose disagreement 按 trust 衰减。全量 coverage=`46.29%`、accepted accuracy=`97.07%`、full-frame success=`44.93%`；test coverage=`45.87%`、accepted accuracy=`96.57%`、full-frame success=`44.29%`，false acceptance=`3.43%`，平均恢复=`3.92` 帧，最长 Reject=`33`，test stale frames=`326`、accepted stale=`69`。trace 审计为 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`。
- **限制与结论（已降级）**：这是全帧 evidence replay，不重新生成 Reloc3r/geometry pose 或在线 route；其 coverage/success 只能描述冻结实现，不能据此判断 dynamic re-trust 是否改善全帧表现。下一步必须先拆分风险项，再重新生成端到端 causal route。
- **产物**：`tools/eval_avl015_dynamic_retrust.py`；`outputs/lilocbench/AVL-015/dynamic_retrust/evaluation.json`。

### AVL-015 dynamic re-trust 实现复核（2026-10-12，结论降级）

- **发现**：上游 `tools/run_avl012_reliability.py` 的 `geometry_risk` 已包含 `geometry_vs_previous_m/deg`；下游 `tools/eval_avl015_dynamic_retrust.py` 在保留未衰减 `geometry_risk` 的同时，又对重复的 previous-disagreement 项乘 `state_trust`。未衰减的复合项覆盖了预期衰减。
- **只读验证**：对现有 `1108` 条 geometry-source evidence，将 `(anchor_age,reject_streak)` 从 `(0,0)` 改为极大值，最终 `relaxed_score` 改变数为 `0/1108`；且 `1108/1108` 条的 `geometry_risk` 均已不小于单独 previous-disagreement 风险。
- **判定**：原 replay 的 coverage/accuracy 数字仍描述该实现的冻结回放，但“旧状态 disagreement 已按 trust 衰减”这一实现表述不成立；不得用该结果否定 dynamic re-trust。后续需保存不可约风险项，只衰减 state-dependent evidence，并重新生成完整 causal route。
- **状态**：AVL-015 Q2 dynamic re-trust 从有限负证据进一步降级为 **implementation-limited / must revalidate**。

## 2026-10-12 Agent 主线与实现决策

### Phase A 实现落点（2026-10-12，代码 smoke-tested）

- `tools/agentic_localization_env.py`：提供同帧多步 evidence acquisition、`reset/step`、验证后状态推进和严格 Memory 写入门控。
- `tools/lilocbench_agent_backend.py`：reference action 触发当前 query 与所选 mapping references 的真实 Reloc3r pair inference；返回 `executed=true/cache_hit=false` 回执，未加载 query GT。
- `tools/run_agentic_localization_env.py`：第一版主动 reference batch baseline，用于比较 fixed/reference-policy，而不是 Agent 论文 claim。
- `tools/cpu_orb_agent_backend.py`：无 GPU 时的真实图像 smoke backend；每次 reference action 重新执行 ORB matching、triangulation、PnP 和 verifier，仅用于环境验收，不作为论文 baseline。
- `tools/run_agentic_reference_switch_smoke.py`：固定一帧连续选择两组不同 reference，确认新候选覆盖旧 verification；产物位于 `outputs/lilocbench/AVL-agent-cpu-smoke/reference_switch_static_0/`。
- `tools/test_agentic_localization_env.py`：验证不同 reference 确实进入 backend、execution receipt 的 query/reference IDs 必须匹配、未验证候选不能写入 Memory、cache-hit 回执被拒绝、evaluator 字段不能越过在线边界、state trust 随 anchor age/reject streak 衰减、future memory 被拒绝、新 reference 会清除旧 verification，并检查 `audit_trace()`；当前 9/9 通过。
- `tools/run_avl012_reliability.py`：新增 intrinsic / cross-source / previous-state 风险分解；已在 `outputs/lilocbench/AVL-012_componentized/` 重跑，dynamic re-trust 的 fresh/stale score 在 `303/1108` 条 geometry-source evidence 上发生变化。

该实现仍有两个未完成的部署级验证：在 LILocBench GPU 数据上运行全序列并核对实际 matching/pose elapsed；将 componentized risk 接入真实 route regeneration。未完成前不把 Phase A 写成 Agent 收益。

在线 Reloc3r backend 在 CUDA 不可用时显式报错，不允许静默回退到 AVL-002/015 的冻结候选；当前机器的 `torch.cuda.is_available()` 为 `False`，因此真实 Reloc3r neural smoke run 仍未执行。CPU ORB backend 已在 `static_0` 前 3 帧真实图像上完成 9-transition smoke，`environment_audit.passed=true`；同一帧两组不同 reference 的重算也已通过。

- **Agent 定义**：`Observation/State -> choose missing evidence -> tool call -> new observation -> update belief -> continue/Accept/Reject/Defer`。研究对象是连续定位状态和证据获取，不是一次选择 Reloc3r/Memory/RoMa。
- **实现问题 1（固定候选池）**：AVL-015 Oracle 只覆盖既有 pose 输出；它不能约束通过替代 reference 或 held-out verification 生成的新候选。后续 action 必须改变 evidence set。
- **实现问题 2（冻结回放）**：Historical Memory 读取旧 route 的 accepted trace，dynamic re-trust 不重新生成 pose/route；策略改变不会因果地改变后续 Memory 和 observation。训练前必须建立 `reset/step` 环境。
- **实现问题 3（re-trust）**：旧实现的状态信任衰减被复合 geometry risk 覆盖；风险项现已拆分并通过 `303/1108` fresh/stale score change 复核，仍需真实交互 route 重验。
- **基本系统**：MapIndex/ReferenceGroups、PoseTools、EvidenceStore、source-aware pose-conditioned Verifier、ExperienceMemory、CausalStateMachine、CostLedger 与可替换 Policy。
- **第一项主动取证实验**：固定 Reloc3r/RoMa/verifier，只开放“选择下一组 reference 或 held-out verification evidence”，在相同 matching/tool budget 下比较 fixed/random/geometry-greedy/有限深度 oracle；在它证明新证据价值后再生成 Agentic trajectory。
- **训练路线**：多场景动作展开与 cost-aware expert search -> 结构化 Agentic SFT -> policy-induced state 数据聚合 -> 必要时 RL。现有缓存只作初始状态/部分反事实，不能直接视为完整交互轨迹。
- **IROS 证据**：需加入 standard HLoc 或等价强基线、scene-disjoint/cross-dataset 长期变化测试、实机 lost-and-recovery、统一延迟/成本、false acceptance、Memory pollution、右删失恢复统计及关键消融。
- **主张边界**：可以开始受控 Agent prototype；只有 learned policy 在相同工具/verifier/budget 下超过 fixed-order、one-shot router、geometry-greedy 和轻量 sequential policy，才主张 Agent/VLM 独立收益。

### AVL-015 Q3 结果（2026-10-11，completed with negative result）

- **实现与因果审计**：四类序列共 1696 帧，Historical-keyframe route 只从当前帧之前 `accepted=true && state_updated=true` 的历史帧建立临时地图；当前图像与历史图像做 RoMa 匹配，历史全局 pose 三角化 3D tracks，再用 PnP 组合 `T_t^w=T_k^wT_t^k`。trace 均满足 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`、`historical_future_access=false`。
- **原始候选结果**：全量 pose availability=`1685/1696=99.35%`、success=`698/1696=41.16%`；test availability=`100%`、success=`455/1016=44.78%`。按序列 test success 为 static=`45.09%`、dynamic=`24.73%`、LT=`54.15%`、dynamic+LT=`42.90%`；实测总 elapsed=`3676.76s`（另加 shared retrieval 后 Q1 成本为 `3714.71s`）。
- **互补性**：相对 Consensus/Memory/HLoc/RoMa 的 raw Historical-only 为 all=`4`、test=`2`；test 中 `Historical ∩ RoMa` 为 `442`、RoMa-only=`495`、Historical-only=`13`，说明 pairwise difference 不等于相对整个 action pool 的独立恢复。
- **Memory 专属 verifier**：validation 冻结全局 threshold=`1.0`，使用历史相似度、reprojection、inliers、pair support、历史 keyframe risk 与 age；test coverage=`21.85%`、selective accuracy=`63.96%`、false acceptance=`36.04%`、full-frame success=`13.98%`，且 Historical-only test=`0`。因此 verifier 不能把当前候选变成可靠恢复。
- **结论**：本 Q3 实现没有证明 Experience Memory 产生稳定的新定位能力；当前更像高成本、低可靠的历史候选生成器。该负结果不否定未来更好的历史关键帧选择或局部地图构建，但不支持现在引入 Agent。实现/结果：`tools/run_avl015_historical_memory.py`、`tools/eval_avl015_historical_memory.py`、`tools/eval_avl015_historical_verifier.py`、`outputs/lilocbench/AVL-015/historical_keyframe_memory/`。

### AVL-016 — Active Evidence Pilot（2026-10-12，completed with negative/limited result）

- **目的**：在同一 causal environment、同一 evidence budget 和同一 verifier 下，先验证“策略主动选择下一组 reference”是否相对于固定顺序产生独立收益；该实验是进入 Agentic SFT 前的门槛，不预设 VLM 有效。
- **协议**：四个序列的 calibration-prefix 只用于 light policy 的离线标签；严格复验测试使用后续 304 帧。每帧最多两组 reference，每组最多 3 个 references；fixed、random、rule、light 共用 historical-memory action、held-out verification、Accept/Reject/Defer 和 action budget。reference、verification、memory 三类工具调用均计入总 budget。oracle 使用测试 GT，仅作为 finite-depth counterfactual upper bound。
- **连续帧 Memory 边界**：测试段内 accepted pose、历史关键帧和 reject state 会因果传给下一帧；历史 action 从历史帧 RGB、map references 建立局部 3D，再对当前帧重新 PnP，不复制旧 pose。为避免 calibration/test 泄漏，测试段从空 online Memory 冷启动。因此本条目不是 warm-start Memory ablation，也不是 learned memory agent；后续须同时报告 cold-start 与合法历史状态 warm-start。
- **独立验证边界**：`verify_candidate` 从候选未使用的 retrieval references 中重新匹配，并按候选 pose 计算极线一致性；不复用候选自身的 inlier/reprojection 作为唯一验证证据。
- **结果**：fixed=`81/304=26.6%` full-frame success、`81/87=93.1%` accepted accuracy；random=`69/304=22.7%`、`69/74=93.2%`；rule=`77/304=25.3%`、`77/83=92.8%`；light=`81/304=26.6%`、`81/87=93.1%`。平均总工具调用为 fixed/random/rule/light=`3.168/3.336/3.211/3.158`；light 与 fixed 并列，仅略低成本。历史 fixed RoMa cache 使用不同工具与 cost boundary，不能作为同成本 baseline。
- **审计**：四个在线 policy 的 `query_gt_read=false`、`uses_future=false`、`selection_uses_gt=false`，environment audit 全部通过。light 的 `gt_used_for_labels=true` 仅表示 calibration-prefix 离线训练，`online_selection_uses_gt=false`；oracle 的 GT 使用显式标注为不可部署。
- **判定**：当前 CPU-ORB action pool、强几何候选和 held-out verifier 尚未形成可观测 action-dependent context，主动策略没有超过 fixed；light 与 fixed 完全并列。该负结果不能外推为主动定位无效，但足以维持 Agent/VLM No-Go；先完成真实 GPU Reloc3r/RoMa、统一 online cost 和 scene-disjoint 复验，再决定是否生成 tool-expanded trajectories。
- **硬件边界**：本机 `torch.cuda.is_available()=false`、device count=`0`，`nvidia-smi` 无法连接 NVIDIA driver；在线 Reloc3r backend 显式失败，未使用历史 cache 冒充强几何在线调用。因此 AVL-016 的负结果限定为 CPU action pool，不能外推为 GPU Reloc3r/RoMa 的最终结论。
- **产物**：`tools/eval_active_evidence_policies.py`、`outputs/lilocbench/active_evidence_pilot128_heldout_v2/summary.json`、`docs/active_evidence_pilot_report.md`。

### Stage 3B Pareto Gate v1（2026-10-12，development gate implemented）

- **冻结目标**：success 相对 feedback-rule 非劣界 `-2 pp`；FAR `<=10%`；mean measured tool cost `<1.0 s/frame` 且低于 feedback-rule。三项必须同时通过，并要求四序列至少 80 个逐帧配对样本。
- **provenance 门槛**：每个 trace 必须有对应 completed summary 且 causal audit passed；旧 `lt_changes_dynamics_0` feedback trace 因原进程退出码为 1、summary 缺少该 policy audit，被 Gate 正确排除。随后在预热 cache 上重跑四序列 feedback-rule，80 帧全部完成并审计通过。
- **development 结果**：MLP=`41.25% success / 8.33% FAR / 0.675s`，feedback-rule=`48.75% / 7.14% / 2.533s`。MLP 通过安全和成本目标，success 差 `-7.5 pp` 未通过非劣界；overall fail，Pareto 关系为 non-dominated tradeoff。
- **配对不确定性**：10,000 次 bootstrap 的 learned-minus-rule success 差 95% CI=`[-13.75,-2.50] pp`，cost 差 CI=`[-2.818,-1.023]s`。本结果是 development diagnostic，不是 final held-out claim。
- **产物**：`configs/stage3b_pareto_gate_v1.json`、`tools/evaluate_stage3b_pareto_gate.py`、`tools/test_stage3b_pareto_gate.py`、`outputs/lilocbench/stage3b_v2_protocol_fixed_eval_r1/pareto_gate_development.json`。

### Stage 3B corrected targeted DAgger r2（2026-10-12，completed / Gate fail）

- **数据**：40--59帧四序列重新执行 live-cost 两步 Oracle，80帧产生208个 on-policy states、893个 root action nodes；全部 prior cost 与在线 state cost 一致。on-policy targets 为 Accept-optimal=32、Geometry-optimal=11、Memory-optimal=30、high-regret=14、premature-Accept=2。
- **训练修正**：DAgger states 强制进入 train，原始 Oracle 32--39帧作为不加权 validation；targeted weights 只作用于带 DAgger provenance 的状态，最大权重4，不做乘法叠加。
- **结果**：成本约束内最佳仍是进入本轮前 MLP=`41.25% success / 8.33% FAR / 0.675s`。r2 policy variants 为37.5--40% success；全 value head 达45%且FAR=0，但成本2.352s、Geometry=55%；warm-start 为32.5%/7.14%/0.991s。没有模型达到46.75%/10%/1s三目标。
- **判定（随后经 r1 审计修正）**：不替换当前 checkpoint，不继续在60--79 development帧调权重。原先将低成本 policy 的不足归为 terminal under-accept 只是待验假设；r1 on-policy Q audit 已在保留 checkpoint 上否定该假设。下一步先做 acquisition value/observation/target diagnosis。
- **产物**：`outputs/lilocbench/stage3b_v2_corrected_dagger_r2/round_summary.json`、`outputs/lilocbench/stage3b_v2_corrected_targeted_*`、`configs/stage3b_targeted_dagger_v1.json`。

### Stage 3B on-policy Q-calibration audit r1（2026-10-13，completed / diagnostic）

- **范围**：当前保留的低成本 checkpoint 在 corrected r2 的 208 个 policy-induced states
  上只读重算预测 Q、policy logits 与同状态 live-cost Oracle 值；不运行工具、不改训练集或 checkpoint。
- **terminal 结论**：79 个 verifier-passed states 中，Accept Q 的 signed bias 是 `+0.087`
  （MAE=`0.215`）；32 个 Oracle-Accept-optimal state 没有一个被 policy 或 Q-head 漏 Accept，
  两个头各有2次 over-accept。故“当前 MLP 因 `Qhat(Accept)` 系统偏低而 under-accept”被本批
  on-policy evidence 否定。
- **acquisition 结论**：Geometry/Localize/Memory 的 signed Q bias 为
  `+0.049/+0.054/-0.020`；Geometry 相对廉价替代的 margin 相关性为`0.216`、符号准确率
  `77.5%`，模型的正 Geometry margin 占`21.25%`，Oracle 仅`8.75%`。候选来自
  Reloc3r/Memory 时 margin sign accuracy 仅`47.1%/61.9%`，无 candidate root 为`96.3%`。
- **head 分歧**：policy/Q-head 在`84/208=40.4%` states 分歧；Q-head 的平均 Oracle
  regret 不低于 policy（`0.0880 vs 0.0878`），但平均相对 Oracle cost 更高
  （`0.468s vs 0.065s`）。这与全 value-head 的高 geometry/高成本结果一致。
- **判定**：终端接受不是下一步主假设；下一步是 cost-sensitive acquisition advantage/
  success-gain 分解和 candidate/Memory recovery strata 的对比性数据。60--79 不再用于调参。
- **产物**：`tools/analyze_stage3b_q_calibration.py`、`tools/test_stage3b_q_calibration.py`、
  `outputs/lilocbench/stage3b_v2_q_calibration_audit_r1/q_calibration_audit.json`。

### Stage 3C recovery/cost controller r1--r2（2026-10-13，completed / Gate fail）

- **目的**：检验“分别学习 action 的 correct-recovery probability 与新增成本，再由
  显式 controller 排序”是否能解决 Stage 3B 的 recovery-state Geometry 选择不足。
- **训练**：由 corrected live-cost Oracle / targeted DAgger 构建 1,116 grouped states、
  4,650 acquisition labels；MLP 的 temporal-tail validation 为 recovery Brier=0.1275、
  cost MAE=0.229s、Geometry margin sign=89.38%。XGB classifier/regressor 与
  XGBRanker 仅作离线 diagnostic（显式 controller/XGBRanker Geometry margin sign
  为 76.25%/95.63%），不进入 deployed checkpoint。
- **r1 作废原因**：controller 错将 Memory `verified` candidate 作为可 Accept 的终止
  证据，造成 20 次 direct Memory Accept 且全错。该安全接口已修复为 Memory 只能
  提供 context，必须重新取得 Reloc3r 或 RoMa evidence；不能引用 r1 指标。
- **r2 真实 GPU 四序列结果**：80 个共同帧、causal audit 全部通过，
  success=42.5%、FAR=8.11%、mean cost=1.165s、Geometry frame rate=1.25%。
  feedback-rule 为48.75%/7.14%/2.533s/53.75%。Pareto Gate fail：success 差
  -6.25pp（CI [-12.5,-1.25]pp）且 cost 超过1s，只有 FAR 与相对 rule 的成本优势通过。
- **判定**：不能以高离线 ranking accuracy 主张 learned acquisition 已经成功；当前
  bottleneck 是 policy-induced verified-candidate recovery state 的在线排序/终止耦合。
  保留 Stage 3B checkpoint，不进入 Qwen/RL；下一项是新 split 的 recovery-state
  calibration 与一个真正部署的 ranking-controller baseline。
- **产物**：`configs/stage3c_recovery_policy_v1.json`、
  `tools/{build_stage3c_recovery_dataset,train_stage3c_recovery_policy,stage3c_recovery_policy,train_stage3c_xgboost_diagnostic}.py`、
  `outputs/lilocbench/stage3c_recovery_eval_r2/pareto_gate.json`。

### Stage 3C.5 fresh-split XGBRanker diagnosis（2026-10-13，completed / diagnostic negative）

- **协议**：冻结只见过 0--59 帧的 Stage 3C XGB classifier/regressor/ranker；在四序列
  80--99 帧部署 XGBRanker acquisition controller。rank score 仅用于 acquisition 排序，
  calibrated `P-C` 与 verifier 继续负责 terminal safety。每个实际 policy decision 后
  fork 同一 live state，并以相同两步、live-cost Oracle 标注机会；在线 policy 不可见 GT/
  Oracle。
- **对照结果**：XGBRanker=52.5% success / 6.67% FAR / 0.719s / 0% Geometry；
  feedback-rule=57.5% / 9.80% / 2.272s / 41.25% Geometry。四序列80帧 causal audit
  全通过。Pareto Gate fail：success -5pp（CI [-10,-1.25]pp），FAR 与成本通过。
- **Geometry opportunity audit**：Oracle Geometry-optimal=5（其中 recovery=4），
  policy Geometry calls=0，recall=0%，因此 precision 未定义；missed opportunity 平均
  utility regret=0.750。160 个 Geometry-vs-cheap state 中，55.6% 为预测 `ΔP>0` 但
  被 `lambda*ΔC` 压制；XGBRanker 只在1个、explicit controller 只在13个 state 偏好
  Geometry。
- **判定**：反驳“仅 MLP/joint-training 造成 Geometry 漏调”的单点解释；当前 evidence
  state + data coverage 未在 fresh policy-induced recovery states 上泛化。此结果不足以声称
  information-theoretic observability failure，故不进入 Qwen/VLM；先补高价值几何可观察
  特征，并在全新 split 重复同一 recall audit。
- **产物**：`configs/stage3c5_xgbranker_fresh_split_v1.json`、
  `tools/stage3c_xgboost_policy.py`、`tools/analyze_stage3c_geometry_opportunities.py`、
  `outputs/lilocbench/stage3c5_xgbranker_fresh_eval_r1/{geometry_opportunity_audit,pareto_gate_fresh_split}.json`。

### Stage 3D Recovery Observability v1（2026-10-13，implemented / diagnostic negative）

- **问题定义**：验证 Geometry 调用前的可观测证据是否足以预测其边际恢复价值；冻结
  XGBRanker 和 terminal verifier，不以更大模型掩盖 observation 缺口。
- **实现**：`tools/stage3d_features.py` 提供 F0/F1/F2/F3（维度20/30/36/43）的
  pre-Geometry feature ladder，并对 RoMa/PnP/inlier/reprojection/epipolar 字段 fail
  closed。`tools/lilocbench_agent_backend.py` 的 `--stage3d-preaction-features` 在
  cheap Reloc3r action 内计算 ORB/Hamming correspondence spatial statistics 和
  appearance-change cue；计算耗时纳入该 action receipt/cost。`tools/collect_stage3d_recovery_opportunities.py`
  从 live-cost Oracle 选择 751 个 targeted recovery/boundary states。
- **fresh split 诊断**（四序列80帧，80--99；旧 ladder 记录中的可选字段大多缺失）：
  F0=`51.25% / 4.65% / 0.945s / 7.5%`，F1=`50.00% / 4.76% / 0.677s / 7.5%`，
  F2=`52.50% / 8.70% / 0.602s / 2.5%`，F3=`51.25% / 6.82% / 0.806s / 5.0%`
  （依次为 success/FAR/cost/Geometry rate）；feedback-rule 为 `57.50% / 9.80% /
  2.272s / 41.25%`。四个版本的 Geometry opportunity recall 均为 `0%`（分别
  0/8、0/4、0/6、0/5），Gate 未通过。
- **解释边界**：本轮证明当前 feature+coverage+controller 组合仍漏掉 fresh
  policy-induced Geometry opportunities，但不能据此声称 structured observation
  已达到信息论上限；此前训练轨迹没有真实 ORB optional cues。下一轮必须在新 split
  开启 pre-action extractor，重训 F1/F3，再做相同 opportunity audit。
- **No-Go**：在 appearance/change cue 尚未证明增加 opportunity recall 前，不进入
  Qwen/Qwen-VL 或 RL；最终指标仍为 opportunity recall/precision/regret、Success--Cost--FAR
  Pareto，而不是单独 offline ranking accuracy。

### Stage 3D Recovery Observability v2（2026-10-13，completed / Gate pending baseline）

- **协议**：Stage 3C.6 component utility controller；F0/F1/F2/F3 为24/34/40/47维；
  四序列100--119帧、每层80帧、真实GPU工具、两步live-cost Oracle。可选特征均在
  Geometry调用前获得并计入成本，post-Geometry字段fail closed。
- **闭环结果**：F0=`58.75% success / 2.08% FAR / 1.553s / 41.25% Geometry`；
  F1=`56.25% / 0% / 1.459s / 43.75%`；F2=`57.50% / 0% / 1.298s / 41.25%`；
  F3=`52.50% / 2.33% / 1.125s / 35.00%`。
- **Opportunity audit**：F0/F1/F2/F3 的 Geometry recall 分别为
  `9/17=52.94%`、`8/15=53.33%`、`8/15=53.33%`、`5/13=38.46%`；precision 分别为
  `27.27%/22.86%/24.24%/17.86%`。acquisition admission recall 全部高于96.8%。
- **判定**：v1 的0% recall不是结构化 observation ceiling；targeted coverage 与协议
  修正有效。但 F1--F3 未优于 F0，F3 appearance/change proxy 明显负迁移。瓶颈已从
  terminal admission 收敛为 Geometry-vs-cheap 的排序与校准。
- **Gate边界**：100--119尚无同协议 feedback-rule，故不能正式宣称 learned policy
  超过规则策略。先补同split paired baseline，再在F0/F2上做显式 pairwise Geometry
  advantage calibration；暂不进入Qwen/Qwen-VL/RL。
- **产物**：`outputs/lilocbench/stage3d_v2_fresh_collect_r1/aggregate_summary.json`、
  `tools/summarize_stage3d_ladder.py`。

### Stage 3D v2.1 same-split feedback-rule baseline（2026-10-14，completed / Gate fail）

- **协议**：在 Stage 3D 100--119 四序列/80帧 split 上，使用同一 Reloc3r backend、
  verifier、Memory、两步 live-cost Oracle、2次 acquisition、30s budget 和 pre-action
  feature cost accounting 跑 feedback-rule。
- **baseline**：feedback-rule=`61.25% success / 2.00% FAR / 1.597s / 52.50% Geometry`。
  所有80帧 causal 字段检查通过。
- **paired结果**：F0 相对 rule=`-2.50pp success / +0.08pp FAR / -0.045s cost`，success
  95% CI=`[-6.25,0]pp`；F2=`-3.75pp / -2.00pp / -0.299s`，success CI=`[-8.75,0]pp`，
  cost CI=`[-0.523,-0.072]s`。
- **判定**：F0/F2 均未过三目标 Gate：F0 只差0.5pp非劣界但成本优势不稳定；F2 安全/成本
  更好但 success 低于非劣界，且二者 cost 都高于预注册1s上限。故只能报告
  non-dominated success--cost trade-off，不能报告 learned > feedback-rule。
- **下一步**：冻结F0/F2，使用 shared-state audit 后专门训练 Geometry-vs-best-cheap
  advantage pairwise/ranking calibration；停止扩充F3，不进入Qwen/VLM/RL。
- **产物**：`outputs/lilocbench/stage3d_v2_feedback_rule_fresh_baseline_r1/aggregate_summary.json`、
  `outputs/lilocbench/stage3d_v2_fresh_collect_r1/{f0,f2}/pareto_gate_feedback_rule_same_split.json`、
  `tools/evaluate_stage3d_pareto_gate.py`。

### Stage 3D F0/F2 baseline freeze（2026-10-14）

- **冻结对象**：F0 checkpoint（最高success）和F2 checkpoint（零FAR/较低成本），以及
  同 split feedback-rule、verifier、utility、feature-ladder config 和全部结果摘要。
- **约束**：后续 Geometry-advantage、shared-state audit 和新策略不得覆盖、微调或基于
  100--119结果选择 F0/F2 超参数；它们只作为固定可复现对照。新的最终 split 必须重新
  运行 feedback-rule。
- **完整性**：19个配置、checkpoint与结果产物的 SHA-256 记录在
  `outputs/lilocbench/stage3d_v2_frozen_baselines_r1/freeze_manifest.json`，由
  `tools/verify_stage3d_frozen_baselines.py` 验证。

### Stage 3E recovery/cost pairwise ranking（2026-10-14，completed / fresh diagnostic mixed）

- **实现**：由1,106个 recovery-targeted states（含114个 policy-induced DAgger train-only
  states）训练双头 MLP，预测 `P_recover(s,a)` 与执行前 `C_hat(s,a)`；以 regret-scaled
  pairwise Oracle-value loss 监督 Geometry/Localize/Memory 的排序。terminal control、verifier
  与F2 pre-action observation 保持冻结。在线 controller 唯一使用 `P_recover-lambda*C_hat`，
  Geometry margin 只是在 validation 上校准的单一 score 阈值，不再使用 classifier与回归头
  的AND gate。
- **协议修正**：最初发现 Stage 3E 为15帧、其对照为20帧，立即作废该比较；runner现接收
  `MAX_FRAMES`，聚合器加入 `--require-aligned`，会验证每种 policy 的序列名与逐序列帧数。
  最终评测统一四序列140--154共同前缀，共60帧；`dynamics_0`后续无帧，因此不将60帧伪装为
  预注册的80帧 Gate。
- **结果**：Stage3E=`60.00% success / 0% FAR / 3.562s / 60.00% Geometry`；冻结
  F2=`63.33% / 0% / 4.961s / 50.00%`；feedback-rule=`61.67% / 5.13% / 5.267s / 48.33%`。
  相对F2：success `-3.33pp`（CI `[-8.33,0]pp`）、cost `-1.40s`（CI `[-1.99,-0.82]s`）；
  相对 rule：success `-1.67pp`（CI `[-8.33,+3.33]pp`）、FAR `-5.13pp`、cost `-1.70s`
  （CI `[-2.42,-0.98]s`）。全部 online causal audit 通过。
- **Geometry audit**：Stage3E opportunity recall=`5/7=71.4%`、precision=`5/36=13.9%`、
  acquisition regret=0.0606；F2=`6/8=75.0%`、`6/30=20.0%`、0.0934；rule=`7/10=70.0%`、
  `7/29=24.1%`、0.1085。Stage3E 的核心失败不是漏掉 Geometry，而是过度放行低价值
  Geometry，导致更低局部 ranking regret 未转换为更多终端正确接受。
- **判定**：不替换 F2；不报告 Gate pass（共同帧数小于80，且绝对成本>1s）。本轮证明
  recovery-targeted ranking 具有更好的 Geometry opportunity recall/regret 和安全--成本
  trade-off，但尚未证明端到端 learned policy 优于 F2/rule。下一步固定 F2 feature schema，
  只学习 Geometry 相对 best-cheap 的 expected terminal-success advantage，并用 precision /
  expected-gain-constrained threshold 校准；不继续加特征，不进入 Qwen、Qwen-VL或RL。
- **产物**：`outputs/lilocbench/stage3e_recovery_cost_ranking_r1/fresh_140_154/aggregate_summary_aligned.json`、
  `pareto_vs_frozen_f2.json`、`pareto_vs_feedback_rule.json`，以及
  `tools/{build_stage3e_recovery_ranking_dataset,train_stage3e_recovery_ranking,stage3e_recovery_ranking_policy}.py`。

### Stage 3F frozen F2 stability validation（2026-10-14，completed / partial positive）

- **冻结与范围**：F2 checkpoint、F2 feature schema、verifier、utility、budget和seed全部冻结；
  仅复验100--119、120--139、140--154三个不重叠 causal block，未用任何结果改变策略。
- **success/FAR**：F2-minus-rule success依次为`-3.75pp/-1.25pp/+1.67pp`，方向不稳定；
  pooled 220帧为F2=`54.09% success / 0.83% FAR`，rule=`55.45% / 3.17% FAR`，success差
  `-1.36pp`（CI `[-3.64,+0.45]pp`）。故否定“F2已稳定全面优于 feedback-rule”。
- **去偏成本复验**：新增同 GPU 同 backend 的 AB/BA crossover runner；首轮发现 policy
  position 可显著改变 receipt time，故仅用同 `(sequence, block)` 的反向双运行均值。去偏后
  F2-minus-rule cost为`-0.725s/-1.662s/-0.098s`，pooled=`-0.828s/frame`；效率优势方向稳定，
  但一对AB/BA不能作为最终论文 latency estimate。
- **判定**：F2 保留为 safety--cost Pareto baseline，不替换 feedback-rule，也不以最新60帧
  宣称 success dominance。当时预注册的后续为 paired F2-vs-rule disagreement audit；
  该审计现已由 Stage 3G 完成，且未支持继续训练 Geometry advantage。
- **产物**：`tools/run_stage3f_f2_crossover.sh`、`tools/aggregate_stage3f_f2_stability.py`、
  `outputs/lilocbench/stage3f_f2_stability_r1/f2_stability_summary_order_debiased.json`。

### Stage 3G paired disagreement + terminal alignment（2026-10-14，completed / Stage 3 closed）

- 220共同帧中 `both-success/both-failure/F2-only/rule-only=118/97/1/4`；rule-only 仅1帧
  是 F2 漏掉 Oracle Geometry opportunity，1帧是错过 Oracle Accept，其余是非 Geometry
  acquisition 分歧。
- legacy feedback-rule 为 verifier 后自动 Accept，F2 为显式 Accept/Defer；旧比较存在
  terminal-interface confound。
- terminal-aligned rule 固定 acquisition、使用冻结 F2 terminal controller。F2=
  `54.09% success / 0.83% FAR`，legacy rule=`55.45% / 3.17%`，terminal-aligned rule=
  `53.64% / 3.28%`；legacy rule 对 F2 的正确帧差由 `+3` 变为 `-1`。
- 判定：Stage 3 收口，不继续同 split 的 Geometry/controller 调参，不进入 Qwen/VLM/RL；
  转向 failure-triggered、scene-disjoint 的 active-evidence realizable-value validation。
- 产物：`tools/audit_stage3f_f2_disagreements.py`、`tools/close_stage3.py`、
  `outputs/lilocbench/stage3f_terminal_aligned_feedback_r1/`。
