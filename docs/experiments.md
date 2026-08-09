# Experiments: final, historical, abandoned, obsolete

A quick index so a reader doesn't have to infer status from filenames. "Final/active" means
currently the shipped default or the current best-known state; "historical" means superseded
but scientifically load-bearing (a later result depends on understanding it); "abandoned"
means built, tested, and deliberately not used, for a documented reason; "obsolete" means
safe to ignore, kept only for provenance.

## Final / active

- **STGT checkpoint selection rule**: judge on pair-level/threat-level ceiling, never test
  accuracy alone (established after the strategy-6 experiment below made this mistake once).
- **`stgt_bridge.py` default reduction** (`robust=False`): unanimity-based, not majority-vote.
- **Three fixed generator bugs, all shipped**: the upstream dispersed/converging geometry +
  acceleration fix (commit `27adc23`); the destination dwell-time fix and source-side
  `LEAD_IN_RANGE` symmetrization (`src/swarm_intent/eval_trajectories.py`, V5 steps 25-26).
- **Three fixed guard bugs, all shipped**: `dispersed_converging_ambiguity` (top-2
  competitiveness check added), `oov_name` (genuine-OOV vs. valid-`transitioning`-class
  distinction), `dominant_history_contradiction` (fixed to test the actual claim; proven
  unreachable post-fix, kept as a defensive check).
- **`pipeline_v2.py`'s three-layer routing**, with the Layer-1 decision-field
  overwrite-and-log guarantee.

## Historical (superseded, but load-bearing for later results)

- **AUDIT.md sec AE-AF**: measuring real RULES coverage (1.8% of real observations) and the
  bucket A/B/C coverage design — the coverage measurement itself is superseded by later guard
  fixes (bucket A is now much larger), but the *design* it produced (`coverage.py`) is still
  exactly what's shipped.
- **`docs/CEILING.md`'s pooled (non-stratified) ceiling numbers** (steps 1-9): individually
  correct at the time, but pooling chain-1 and chain-2 accuracy hid a large effect later
  unpooled at step 22. Left in place, not deleted, per the project's "supersede via banner,
  never rewrite history" convention.

## Abandoned (built, tested, deliberately not shipped)

- **`stgt_bridge.py`'s `robust=True` majority-vote reduction mode.** Fully implemented,
  10 dedicated regression tests, evaluated end-to-end on a 5-system/24,650-case-run battery.
  Rejected because it recovers more `(from,to)` pairs at much lower precision — most
  recoveries still trip the (separately real) ambiguity guard, so the net effect on real
  pipeline output was to convert potential correct LLM answers into guaranteed silence, not
  to recover genuinely correct pairs. The code and its tests remain in the repository and
  are exercised by CI-equivalent unit tests; `DEFAULT_ROBUST_THRESHOLD` is set but the flag
  defaults to off.
- **Strategy 6 (steadier LR schedule) STGT checkpoint.** Achieved 99.6% in-distribution test
  accuracy — its own literal objective — while roughly *halving* the real pair/threat
  ceiling (a generalization/overfitting tradeoff to the training distribution's narrower
  spread/noise ranges). Reverted the same session; backup checkpoint kept
  (`best_model_strategy6_backup.pt`, not committed per `.gitignore`) for anyone who wants to
  reproduce the comparison.
- **A retired dev-split threshold-tuning conclusion.** An early "robust reduction" threshold
  sweep, tuned on a dev split that was (unknowingly, at the time) reused from an earlier
  session, concluded a lower threshold dominated the shipped default. A fresh, disjoint,
  single-use split reversed that conclusion. The original sweep's result is kept in
  `AUDIT.md`/`docs/V5_LOG.md` exactly as originally reported, with the reversal recorded
  alongside it as a process-discipline finding, not silently corrected.

## Obsolete (safe to ignore, kept for provenance only)

- **`docs/archive/matrix_early_rules_draft.py`** — an early, unreferenced duplicate of the
  RULES dict from the very first migration commit. Superseded by
  `llm_finetuning/build_sft_dataset.py`'s `RULES`, which is the canonical version everywhere
  else in the codebase.
- **`docs/archive/PROJECT_HANDOFF.md`, `docs/archive/handoff_audit_report.md`** — an early
  project-state snapshot and its verification report, both predating the coverage-routing and
  V5 work described above. Left for provenance; do not treat as reflecting current state.
