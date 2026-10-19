# 视觉定位 Agent：当前项目状态

更新：2026-10-19。当前唯一运行主线是 **LILocBench v1 基线**。RIO10、在线 DINO 检索、PnP 试验、旧阶段的多种策略与训练脚本已从日常 `tools/` 移出；原始输出不改动，退役源码见 [`retired_sources.tar.gz`](../outputs/project_cleanup_20261019/retired_sources.tar.gz) 与[文件清单](../outputs/project_cleanup_20261019/retired_source_manifest.json)。

当前流程：当前 RGB 与历史定位状态 → 使用**预计算的 NetVLAD 参考图排名** → 依次调用 Reloc3r 和必要时的 RoMa 强几何 → 对候选单独验证 → 接受或延迟本帧。环境提供因果 `reset/step`、动作预算、工具耗时与轨迹审计。查询 GT 只由事后评分器读取。Memory 目前只保存状态，尚不能用历史图像重新估计当前位姿；当前策略是固定顺序基线，**尚非训练完成的 Agent**。

运行入口是 [`run_lilocbench_unified_episode.py`](../tools/run_lilocbench_unified_episode.py)，评分入口是 [`evaluate_lilocbench_unified_episode.py`](../tools/evaluate_lilocbench_unified_episode.py)。基线仍依赖 `outputs/lilocbench/AVL-002/<sequence>/retrieval.npz`；这些排名在运行前生成，检索的生产耗时**没有计入**逐帧在线成本。因此现有 v1 结果只能用作该协议内的性能对照，不能声称完整端到端时延。

近期成对复核在同一物理场景的四段完整 episode、共 20 帧上，v1 正确接受 10 帧、错误接受 0；实验性在线 DINO v2 正确接受 5 帧、错误接受 0。下降既有参考组拆散/漏图，也有正确候选被验证器拒绝。v1 继续作为性能基线；v2 不用于训练。详细数字和逐帧记录在 [`paired_review_report.json`](../outputs/lilocbench/unified_retrieval_paired_20261019/paired_review_report.json)。20 帧不足以建立跨场景结论。

当前研究决策：先保持这条基线可运行、可评分。若以后需要让 Agent 自主执行“查地图”，应优先把已用的 NetVLAD 查询改成当帧执行并计费，在**保留原有参考排序/分组效果**的前提下复测；不再为了接口一致强制 LILocBench 使用 DINO。只有候选覆盖、验证可靠性以及执行前动作收益在隔离场景上成立，才重新启动学习策略训练。路线见[精简计划](visual_localization_agent_roadmap.md)，历史结果见[实验摘要](experiment_history.md)。
