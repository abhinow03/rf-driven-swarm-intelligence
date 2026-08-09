# Evaluation

Full methodology (independent ground truth, independent judge, dev/mining split discipline)
is in `docs/methodology.md`. This page summarizes what is currently measured and what the
numbers mean; raw output backing every table below is in `evaluation/*.json`.

## Metrics used, and why

- **STGT window-level accuracy** — per-window classification accuracy against the true label
  at that timestep. Reported, but explicitly **not** the headline metric — a model can score
  well here while still failing the metric that matters (see "the ceiling gap" below).
- **Pair-level accuracy** — did the full pipeline recover the correct `(from, to)` formation
  pair for a whole trajectory? This is the metric the V5 program's internal 70% floor is
  stated in terms of.
- **Threat-level / intent / action accuracy ("threat ceiling")** — did the recovered pair map
  to the correct RULES outcome? Because RULES maps 49 pairs onto only 4 threat levels, this is
  a more generous, and arguably more decision-relevant, metric than exact-pair accuracy — a
  wrong pair can still produce the right threat level. Both are reported; `docs/V5_LOG.md`
  step 6 argues the program's stated floor should be restated in threat-ceiling terms, since
  pair-level and threat-level ceilings have diverged and will keep diverging as pair-specific
  fixes land.
- **Macro-F1** (not plain accuracy) for STGT's in-distribution classification report — the
  `transitioning` class is imbalanced; plain accuracy would hide that.
- **Objective intent/threat accuracy** (not judge score) for LLM evaluation — see
  `docs/methodology.md`'s note on the self-grading failure this replaced.

## The ceiling gap (why "the model is 93.5% accurate" is not the headline number)

STGT's own in-distribution test accuracy (`train_model.py`'s reported `test_acc`) has been
above 93% since early in the V5 program, and reached 99.6% at one point. Neither number is
close to the real pipeline ceiling. Two separate, now-diagnosed reasons, in order of
discovery:

1. **In-distribution test data is not representative of realistic long evaluation
   trajectories.** `generate_dataset()`'s training regime and the long, multi-hop trajectories
   used for realistic evaluation have different blend-timing distributions — as of the latest
   V5 entry, these distributions have **zero overlap** (0/20000 Monte Carlo draws land inside
   any training regime's realized timing box), confirmed unchanged across three independent
   generator-formula revisions.
2. **Evaluation-harness/bridge bugs**, found by auditing every guard condition in
   `stgt_bridge.py`/`coverage.py` against what it *claimed* to test versus what it *actually*
   tested. The single largest one (a dispersed/converging ambiguity guard that fired on raw
   probability closeness with no competitiveness check) alone moved the end-to-end threat
   ceiling from 13.0% to 52.3%/58.7% — with no retraining at all.

## Current numbers (docs/CEILING.md is the authoritative, continuously-updated source)

Stratified by chain length (pooling these hides a large effect — see `docs/methodology.md`):

| | chain-1 (steady state) | chain-2 (single real transition) |
|---|---|---|
| pair accuracy | ~85-88% | 65.8% (from an 18.7% original baseline) |
| threat accuracy | ~88% | 76.3% |

Chain-3+ (multi-hop) trajectories are evaluated on a different axis (bucket-A
false-positive rate — how often a multi-hop trajectory is *wrongly* resolved as if it were a
simple 2-step pair): currently 1.8%, down from a 9.0% original baseline, after a fix that also
resolved a temporary regression (statistically confirmed via a two-proportion z-test,
p=4.3e-07 for the net improvement).

## Layer-firing rates (how often each pipeline_v2 layer actually handles a case)

Measured on real STGT output, not templated text cases (`llm_finetuning/measure_coverage.py`,
`docs/development-history.md` Phase 2). As of the pre-V5-guard-fix baseline, reported for
transparency alongside the fact that the guard fixes above have since materially changed
bucket A's size — see `docs/CEILING.md`'s "Current state" banner for the latest bucket
A/B/C split, and treat any single historical layer-firing number in `AUDIT.md` as dated to
the checkpoint/guard-code state at the time it was measured, not as the current figure.

## What "robust=True" evaluation established (a negative result, kept)

A full 5-system, `n_runs=20`, 24,650-case-run comparison
(`llm_finetuning/eval_real_stgt_output_robust.py`) tested the majority-vote robust reduction
mode against three stated success criteria (Layer-1 firing >40%, over-abstention <25%,
escalation error <=20.5%). It passed only the escalation-error criterion, and only because it
answered almost nothing (90.9% over-abstention). The full comparison table, and the
reasoning for not shipping `robust=True` as the default, is in `docs/development-history.md`
and `AUDIT.md` sec AG/AH.
