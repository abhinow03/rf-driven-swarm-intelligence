# V5 program: strategy history

Living document. One entry per strategy attempted, in order, kept up to date every time a new
strategy starts. Full blow-by-blow detail lives in `docs/V5_LOG.md`; per-attempt ceiling
numbers live in `docs/CEILING.md`; the two diagnosed structural gaps live in
`docs/GAP_DIAGNOSIS.md`. This file is the fast-to-read summary of the whole arc: what was
tried, why, what happened, and what's being tried now. Current machine-readable state is
`docs/V5_STATE.json`.

**The number being tracked throughout: pair-level accuracy** on 1000 fresh, realistic,
long, multi-hop trajectories (`scripts/phase0_ceiling.py --n 1000`, seed=999) — the fraction
of trajectories where `stgt_bridge`'s reduced `(a, b)` pair matches the generator's own known
formation chain. This is Phase 0's "ceiling": the maximum end-to-end tactical accuracy any
downstream LLM layer could ever achieve, since if the wrong pair is recovered, `RULES` is
keyed wrong and no LLM can fix it. Plan's stated floor: **70%.** Nowhere close yet.

| # | strategy | pair-level ceiling before → after | verdict |
|---|---|---|---|
| 0 | (baseline) unfixed generator physics | n/a — HALTED before measuring | dispersed/converging shared geometry, no acceleration; both Phase 0 preconditions failed |
| 1 | pull upstream's geometry+acceleration fix (commit `9158b081`), retrain | n/a → **3.3%** (17/509) | fix verified correct and faithfully ported; ceiling catastrophically low, triggered HALT GATE 1 |
| 2 | fix `generate_dataset()`'s transition labeling (dominant-formation, 3-regime) | 3.3% → **4.7%** (24/509) | correctly diagnosed and implemented; helped `v_shape` a lot (1.3%→53.2%) but 3 other classes regressed; still HALT GATE 1 |
| 3 | scale data 3.3x (9k→30k seqs) + epochs 1.9x (80→150) | 4.7% → **6.7%** (34/509) | real, monotonic improvement but insufficient rate of return; `v_shape` regressed again (53.2%→31.7%) while others recovered — capacity trading, not convergence |
| 4 | centroid-relative node features (not absolute position) in `build_graph`, retrain | 6.7% → **6.1%** (31/509); robust=True gives 6.5% | window-level accuracy transformed (36.9%→72.7%, the session's biggest single jump, per-class now even 52-85%) but pair-level ceiling stayed flat. Root cause found: uniform 20-28% false-positive `"transitioning"` rate across all 7 steady formations — no longer a classification problem, a `"transitioning"`-specific calibration/bias problem |
| 5 | **(current)** target the transitioning false-positive rate specifically | in progress | see below |

## Strategy 5 (current): target the transitioning false-positive rate

**Starting point:** strategy 4 left window-level classification good (72.7% overall) but with
every steady formation getting misread as `"transitioning"` 20-28% of the time — uniformly
across classes, not concentrated in one or two. With 15-30 windows per realistic long
trajectory, this makes at least one spurious ambiguous window per trajectory near-certain,
which trips the guard logic in `stgt_bridge`/`coverage.py` regardless of how accurate
classification is otherwise.

**Before touching anything, diagnosing WHERE the false positives concentrate** (per this
project's standing discipline: diagnose before fixing) — specifically, whether they land on
windows that are genuinely, fully unambiguous (drawn from a chain of length 1, no blend
anywhere in the whole trajectory) or whether they cluster near real blend boundaries (where a
sliding 50-step window can legitimately contain a meaningful fraction of blend content even
if the window's majority-vote label reads as the pure formation). These have different fixes:
a true calibration bug on fully-unambiguous windows points at training data/loss changes; a
concentration near blend boundaries would instead point at evaluation-window selection or the
reduction algorithm's tolerance, not a training bug at all.

*(This section is updated as strategy 5 progresses — see `docs/V5_LOG.md` for the live
detail and `docs/CEILING.md` for the resulting numbers once retrained.)*
