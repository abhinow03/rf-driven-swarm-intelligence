# scripts/

## Production (the two commands in the top-level README)

- `generate_data.py` — generates the synthetic STGT dataset.
- `train_model.py` — trains STGT on it.

## V5 research / diagnostic tools

Everything else in this directory is a diagnostic or measurement script from the V5 STGT
retraining program (`docs/V5_LOG.md`, `docs/CEILING.md`). Each one is a distinct, real
measurement (not a duplicate/abandoned rewrite of another) — most have a docstring citing
the exact V5 step that introduced them and what question they answer. Kept in `scripts/`
rather than moved into a nested `experiments/` directory because most of them share
sys.path-relative logic (`Path(__file__).resolve().parent.parent`) and import each other by
bare module name (e.g. `phase0_robust_threshold_sweep.py` imports directly from
`phase0_decompose_failures.py`) — moving them would require rewriting that logic across every
file for a purely cosmetic reorganization, which was judged not worth the risk of introducing
a bug during this repository's cleanup. If you're looking for "the current, authoritative
version" of anything shared across these scripts, it's `src/swarm_intent/eval_trajectories.py`
— all of `phase0_ceiling.py`, `phase0_guard_audit.py`, `phase0_decompose_failures.py`,
`phase0_chainlength_breakdown.py`, `phase0_chain2_observability.py`,
`phase0_full_guard_audit.py`, `phase0_rules_coverage_report.py`, and
`llm_finetuning/measure_coverage.py` import the same `sample_chain`/
`build_long_sequence_labeled` implementation from there — not independent copies.

Key ones, in the order the V5 program used them (full detail in `docs/V5_LOG.md`):

| script | what it measures |
|---|---|
| `verify_upstream_physics.py` | asserts the upstream generator physics fix (dispersed/converging geometry split, acceleration) is present in `src/swarm_intent/` |
| `phase0_ceiling.py` | pair-level / window-level accuracy ceiling on a fixed seed=999 population |
| `phase0_threat_ceiling.py` | the same, re-scored against RULES threat/intent/action instead of exact-pair match |
| `phase0_guard_audit.py` / `phase0_full_guard_audit.py` | audits every `stgt_bridge`/`coverage` guard condition against what it actually tests vs. claims to test |
| `phase0_decompose_failures.py` | per-trajectory failure decomposition: bad windows vs. guard-blocked vs. structurally unresolvable |
| `phase0_chainlength_breakdown.py` | stratifies the ceiling by chain length (chain-1 steady state vs. chain-2 single transition vs. chain-3+) |
| `phase0_chain2_observability.py` | whether a chain-2 trajectory's destination/source formation is ever actually observable by any sliding window |
| `phase0_mining_split_sweep.py` / `phase0_robust_threshold_sweep.py` | dev-split threshold tuning for the `robust=True` reduction mode, per the dev/mining discipline in `docs/methodology.md` |
| `phase0_chain2_blend_distributions.py` / `_v2.py` | Monte Carlo comparison of train-time vs. eval-time blend-timing distributions |
| `phase0_rules_coverage_report.py` | chain-3+ pattern-space analysis (does it collapse to a small enumerable set?) |
| `phase0_endtoend_projection.py` | projects end-to-end threat accuracy from the measured bucket A/B/C split |
| `check_delta_v_geometry.py`, `measure_reg_distribution.py`, `recover_reg_stats.py`, `export_real_reg_percentiles.py`, `report_synth_context_recalibration.py` | regression-label (velocity/approach-rate/stability) distribution diagnostics, referenced in `AUDIT.md` sec V/W |
| `bench_memory.py` | GPU memory benchmark (AUDIT.md sec T) |
