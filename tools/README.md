# Active tools: LILocBench v1 only

- Prepare data: `download_lilocbench.py` / `.sh`, `prepare_lilocbench.py`.
- Build frozen NetVLAD reference rankings: `eval_lilocbench_reloc3r.py` (`retrieve` stage).
- Run and score episodes: `run_lilocbench_unified_episode.py`, then `evaluate_lilocbench_unified_episode.py`.
- Runtime: `agentic_localization_env.py`, `active_evidence_protocol.py`, `unified_localization_policy.py`, `lilocbench_agent_backend.py`, `lilocbench_geometry.py`.
- Regression checks: `test_agentic_localization_env.py`, `test_unified_localization_policy.py`.
- `itt_compat_stub.c` only resolves an optional PyTorch/MKL profiler-symbol mismatch in this conda environment.

The retrieval file `outputs/lilocbench/AVL-002/<sequence>/retrieval.npz` is precomputed; its production time is not charged to the episode. Historical scripts and documents were retired to [`outputs/project_cleanup_20261019/`](../outputs/project_cleanup_20261019/). See [project status](../docs/codex_project_state.md) for the current scope.

Run the existing v1 rankings and score a short episode from the repository root:

```bash
conda activate reloc3r
export LD_PRELOAD="$PWD/outputs/runtime_compat/libitt_compat_stub.so"
python -m tools.run_lilocbench_unified_episode --sequence dynamics_0 --start 100 --count 7 --geometry-on-failure --output outputs/lilocbench/v1_run/dynamics_100_106.json
python -m tools.evaluate_lilocbench_unified_episode --trace outputs/lilocbench/v1_run/dynamics_100_106.json --output outputs/lilocbench/v1_run/dynamics_100_106_score.json
```

If the ranking file must be rebuilt, run `python -m tools.eval_lilocbench_reloc3r --stage retrieve --data-root data/lilocbench/prepared_1hz --query-sequence dynamics_0 --output-dir outputs/lilocbench/AVL-002/dynamics_0 --device cuda:0` first.
