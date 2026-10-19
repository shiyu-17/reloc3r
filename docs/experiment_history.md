# 已退役实验的简要结论

此页合并历史文档的决策信息；原始 JSON/轨迹仍在 `outputs/`，退役的 199 个独立工具及文档文件保存在[压缩包](../outputs/project_cleanup_20261019/retired_sources.tar.gz)和[逐文件 SHA256 清单](../outputs/project_cleanup_20261019/retired_source_manifest.json)。压缩包**不是**旧 v2 的完整可执行环境：共享后端和运行入口已回归 v1，旧轨迹中的源码指纹只用于核对当时记录。这些历史实验不构成当前 LILocBench v1 运行依赖。

| 实验 | 可靠结论 | 限制／决策 |
| --- | --- | --- |
| LILocBench 步骤二 | 28 个 failure episode 中，规则与有限深度 Oracle 均恢复 28/28；Oracle 相对规则的执行成本下降约 63.8%；F2 访问状态约 22.3% 存在不同取证动作的终局差异。 | 只有一个物理场景，成功率饱和；Oracle 知道动作结果，不能直接当作 Agent 能学到的收益。[Gate](../outputs/lilocbench/active_evidence_v2_audit_20261015/stage2_mlp_v2_gate_20261018.json)／[动作归因](../outputs/lilocbench/active_evidence_v2_audit_20261015/action_attribution_report.json) |
| 旧执行前特征诊断 | 同站点、留一序列的结构化特征对部分动作价值有预测信号。 | 错误接受样本太少，且旧标签基于旧动作与参考图协议，不可用于新在线训练。[诊断](../outputs/lilocbench/stage3_readiness_diagnostic_20261018/diagnostic.json) |
| RIO10 多场景与 PnP | 完成过数据适配、参考检索、Reloc3r、几何与 PnP 探索。 | scene02 仍有错误接受及正确候选被拒；PnP 增加成本且未证明稳定收益，已退出主线。scene06 曾用于少量适配烟测，不能称为完全未看的盲测。原始记录仍在 `outputs/rio10/` 与 `outputs/lilocbench/stage3_active_evidence_*`。 |
| LILocBench 在线 DINO v2 对照 | 四段完整 episode、20 帧中，v1/v2 正确接受为 10/5，错误接受均为 0；v2 在有用参考图的保留和分组方面退步，验证器也常拒绝正确候选。 | 同一物理场景的小样本；v1 的离线检索成本未计。在线 DINO 不提升为训练基线。[机器可读报告](../outputs/lilocbench/unified_retrieval_paired_20261019/paired_review_report.json) |

**当前可复现主线**只有[项目状态](codex_project_state.md)列出的 LILocBench v1。对历史结论有疑问时，先查看相应原始输出，再从压缩包提取退役源码；不要把旧文档中的多套门槛叠加成当前训练许可。
