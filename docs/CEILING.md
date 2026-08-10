# Phase 0: the ceiling — measured, and it is far below the plan's floor

> ## 📌 STRATIFICATION POLICY (2026-08-09, discipline-catch turn)
>
> **Pooled ceiling numbers are BANNED from this point forward. Every measurement in this
> file from here on must report per-chain-length stratum (1 / 2 / 3+), never a single
> number averaged across all pair-eligible trajectories.** Chain-1 (steady state) and
> chain-2 (single transition) have had wildly different ceilings for this program's entire
> history — chain-2 was near-zero (6.0%) while chain-1 sat at 87%+, and the "509
> pair-eligible" pooled number this file reported throughout quietly averaged the two
> together, hiding exactly the kind of differential problem (and differential improvement)
> that mattered most. **Every historical entry below this notice, dated before
> 2026-08-09, reports POOLED numbers as originally measured and is left unedited for the
> historical record — do not cite them as current.** The stratified section immediately
> below supersedes every pooled figure in this file.

## Current state, 2026-08-10 — supersedes the chain-2 numbers in the 2026-08-09 section below

The section below (`## Current state (stratified), 2026-08-09`) is itself now stale on
chain-2 specifically: steps 24-26 (`V5_LOG.md`) landed a destination dwell-time fix and a
source-side symmetrization AFTER that section was written. Current chain-2 numbers:
**pair accuracy 65.8%** (robust=False; 66.7% robust=True), **threat accuracy 76.3%**
(77.2% robust=True) — see `V5_LOG.md` step 26 for the full before/after chain. Chain-1 and
chain-3+ rows in the section below are still current (neither generator fix touches them
materially).

**Critical caveat, established 2026-08-10 (`V5_LOG.md` steps 27-29, `HISTORY.md`'s
2026-08-10 decision entry) — read before citing 65.8% as "the model generalizes":** steps
24-26 fixed *observability* in the EVALUATION harness only (`eval_trajectories.py`) —
`generate_dataset()`, the actual STGT training-data path, is unchanged since strategy 5.
The two generators' blend-timing distributions have **0.0% overlap** (Monte Carlo, n=20000,
`V5_LOG.md` step 28), and the refined failure taxonomy shows 100% of remaining chain-2
failures are still boundary/blend-timing-concentrated. **65.8% means "the model does well
once evaluation stopped handing it windows shaped unlike anything it was trained on" — it
is not evidence STGT generalizes to the corrected, realistic blend-timing shape.** Decision
recorded (not executed): port the corrected distribution into `generate_dataset()` and
retrain from scratch — see `HISTORY.md`'s 2026-08-10 decision entry.

**Update, same day, part 3**: the port was designed (3 toggleable flags, `V5_LOG.md` steps
31-34) and pre-scaling-checked (`V5_LOG.md` steps 35-38) before any full-scale attempt. The
decisive small-scale test found the corrected-format model **collapses to near-random
accuracy** (13.2% test_acc, 0% chain-2 pair/threat accuracy) — **NO-GO**, full-scale
generation and retraining are not authorized on the current port design. The 65.8%/76.3%
numbers above remain the current, real figures for the *existing* frozen checkpoint; they are
not superseded by a retrain, because no retrain happened. See `HISTORY.md`'s 2026-08-10 part-3
decision entry for the full finding and the leading root-cause hypothesis (unbounded
acceleration over the port's longer hop lengths).

**Update, same day, part 4**: the acceleration hypothesis above was tested directly and
rejected — capping it produced an identical collapse. The real mechanism is
`split_and_normalize`'s global position-normalization scalar (1.71x inflated by high-drift
outlier windows); a robust (percentile-trimmed) version, with acceleration left fully
uncapped, recovered the small-scale corrected model from 13.2%/0.0%/0.0% to
**87.7%/51.8%/60.5%** (test_acc/pair/threat) — close to, but not clearly exceeding, baseline
in a single noisy run. Verdict remains **NO-GO, narrowly** pending a multi-seed confirmation.
See `HISTORY.md`'s 2026-08-10 part-4 decision entry.

## Current state (stratified), 2026-08-09 — supersedes every pooled number below (chain-2 numbers here are further superseded above)

`scripts/phase0_chainlength_breakdown.py --n 1000`, seed=999 (the standing ceiling
battery), measured AFTER all three of this session's guard/trim fixes (`oov_name`,
`dominant_history_contradiction`, the `robust=True` trim step — see `V5_LOG.md`'s
2026-08-09 entries for each).

| chain_length | n | % of population | pair accuracy (robust=False) | pair accuracy (robust=True) | threat accuracy (robust=False) | threat accuracy (robust=True) |
|---|---|---|---|---|---|---|
| **1 (steady state)** | 258 | 25.8% | 87.6% | 88.4% | 88.0% | 88.8% |
| **2 (single transition)** | 251 | 25.1% | **18.7%** | 18.7% | 31.9% | 39.8% |
| **3+ (no RULES key exists)** | 491 | 49.1% | n/a — see false-positive rate below | n/a | n/a | n/a |

**Chain-3+ bucket-A false-positive rate** (any bucket-A resolution here is wrong by
construction — no 2-tuple ground truth can exist for a 3+-hop chain): **44/491 (9.0%)**,
both reduction modes. This is UP from 1.8% measured earlier in the program — a real,
disclosed side effect of this session's guard fixes: the guards being corrected (`oov_name`,
`dominant_history_contradiction`) used to accidentally catch some chain-3+ false positives
even though they were testing the wrong condition; now correctly scoped to their claimed
purpose, they no longer incidentally block a chain-3+ trajectory whose classifier errors
happen to make it LOOK like a clean ≤2-length reduction. Not hidden: a pooled number would
never have surfaced this trade at all.

**What the pooled number was hiding, made visible by stratification**: chain-2's pair
accuracy moved from 6.0% (before today's fixes) to 18.7% (after) — a 3x improvement — while
chain-1 barely moved (86.8%→87.6%). The POOLED "509 pair-eligible" number this file reported
throughout today's fixes (47.0%→53.6%) compressed a 3x within-stratum improvement and a
near-zero one into a single figure that looks like a modest, uniform gain. It wasn't uniform.
Chain-2's remaining 81.3% failure rate is now dominated (95%, per the stage trace re-run
this session) by the windowing-artefact mechanism (`docs/HISTORY.md`, `UPSTREAM_ISSUES.md`
issue #3) — genuinely not fixable by more bridge-logic changes, unlike the guard bugs that
were.

Full data: `evaluation/phase0_chainlength_breakdown.json`, `evaluation/phase0_chain2_trace.txt`.

---

**Headline: pair-level accuracy on realistic long trajectories is 3.3% (17/509).** Per the
V5 plan's own pre-stated decision rule ("If < 70%: the bottleneck is upstream, not the LLM,
and the plan changes"), this is a HALT GATE 1 trigger, not a borderline call.

This is measured on the RETRAINED checkpoint (`swarm_data/best_model.pt`, trained
2026-08-08 01:50, on the now-fixed geometry/acceleration physics — see `V5_LOG.md`), not
the pre-fix model. `scripts/verify_upstream_physics.py` confirms both fixes are present in
the source this checkpoint was trained against.

## What was measured (`scripts/phase0_ceiling.py`, seed=999, n=1000)

| metric | value |
|---|---|
| window-level overall accuracy | 22.3% (n=8599 windows) |
| pair-level accuracy (the ceiling) | **3.3%** (17/509 eligible chains) |
| pair-eligible bucket distribution | A: 100, B: 174, C: 235 (out of 509) |

Per-class window accuracy is wildly uneven and dominated by over-prediction of
`"transitioning"`:

| true class | n | accuracy |
|---|---|---|
| v_shape | 1134 | 1.3% |
| encirclement | 1023 | 0.2% |
| column | 1130 | 15.8% |
| diamond | 981 | 20.6% |
| dispersed | 1001 | 25.6% |
| converging | 1071 | 0.1% |
| shield | 988 | 24.0% |
| transitioning | 1271 | 80.4% |

Full confusion matrix in `evaluation/phase0_ceiling.json`. `"transitioning"` absorbs 600-750
of ~1000 predictions for nearly every true steady-state class.

## Before trusting this number: ruled out an evaluation-script bug

This is the first time in this project's history per-window accuracy has been scored against
true per-timestep labels (every prior coverage measurement only scored trajectory-level
*bucket* outcomes, never raw window-vs-ground-truth). A new, unvalidated script producing a
catastrophic number is exactly the situation to distrust first. Checked directly:

- **Window/label alignment**: `sliding_window_inference` indexes windows at
  `start = w_idx * stride`, `end = start + window_size` (`stgt/inference.py:79-80`).
  `phase0_ceiling.py`'s label lookup uses the identical formula. Not a misalignment.
- **Early-vs-late window position within a trajectory** (60 fresh trajectories, seed=999,
  independent of the main run): accuracy on the first 2 windows of each trajectory is
  **36.3%**; on the last 2 windows, **32.7%**. Essentially flat, not a monotonic collapse.
  **This rules out simple position-drift-into-OOD-normalization as the sole or primary
  mechanism** — if raw absolute-position drift accumulating across a long, rigid-translated,
  now-accelerating trajectory were the main driver, early windows (minimal drift) should
  score dramatically better than late ones. They don't.

## What actually separates "good" from "bad" — a three-way gap, not one bug

| condition | overall accuracy | inference path |
|---|---|---|
| `train_model.py`'s own reported test accuracy | 93.5% | `train.py`'s `evaluate()` (`SwarmDataset`/`collate_fn`, batched) |
| Training-distribution-matched data (`generate_swarm_sequence`, n_timesteps=50, single steady formation — the EXACT regime `generate_dataset()` trains on) | **69.5%** (n=105) | `sliding_window_inference` / `predict_v2` (the SAME path every real evaluation in this project uses) |
| Realistic long, multi-hop trajectories (`build_long_sequence`'s regime: n_timesteps 50-100 per hop, randomized `blend_start`/`blend_end`, rigid-translated concatenation) | **22.3%** window / **3.3%** pair | `sliding_window_inference` / `predict_v2` |

Two separate, stacked problems, not one:

1. **A 93.5% → 69.5% gap that shows up even on matched-regime data**, purely from switching
   inference path (`train.py`'s internal `evaluate()` vs. the `sliding_window_inference`/
   `predict_v2` path every other script in this project uses to talk to a trained STGT).
   Per-class breakdown on this matched-regime check is itself alarming: `dispersed`/
   `converging`/`shield` score 100%, `column`/`diamond` ~93%, but **`v_shape` and
   `encirclement` are 0/15 (0%) each** — two formations with fixed, non-randomized geometry
   templates, unrelated to anything this session's physics fix touched. This needs its own
   root-cause pass (a bug in `predict_v2`/`sliding_window_inference`'s window construction or
   normalization vs. `SwarmDataset`'s, or a genuine training regression on those two classes)
   before anything downstream can be trusted. Not diagnosed further here — flagging, not
   fixing, per the standing instruction not to improvise around a broken assumption.
2. **A further, much larger 69.5% → 22.3%/3.3% gap specific to realistic long trajectories.**
   Ruled out simple position drift (see above). The leading remaining hypothesis, not yet
   confirmed: `build_long_sequence()`'s hop regime (`llm_finetuning/measure_coverage.py`,
   reused here) samples `blend_start`/`blend_end` as random fractions of a variable
   `seg_len` (50-100 timesteps) — but `generate_dataset()` (STGT's actual training data
   generator) calls `generate_transition_sequence` at its **hardcoded defaults**
   (`blend_start=20, blend_end=30`, fixed `n_timesteps=50`) for every single transitioning
   training example. STGT has never seen a transition blend timed or shaped any other way
   than exactly 20-30 out of 50. A long trajectory's hop, blended over an arbitrary,
   randomized fraction of a 50-100 step segment, is a meaningfully different-shaped input
   than anything in its training set — independent of absolute position magnitude. Not
   confirmed with a targeted ablation here (would require another round of testing); flagged
   as the most likely next thing to check.

## The ceiling, stated per the plan's own terms

**Pair-level accuracy (3.3%) is the ceiling on end-to-end tactical accuracy for the
realistic-trajectory evaluation protocol every other section of this project's AUDIT.md
uses (sec AE step 2 onward).** If this number is right, no downstream LLM layer, no matter
how well built, can exceed it on real input — RULES would be keyed against the wrong pair in
96.7% of resolvable-in-principle cases. Per the plan's own pre-registered decision rule, this
is squarely in the "< 70%: the bottleneck is upstream, not the LLM, and the plan changes"
band, not the "revise target" band and nowhere near the "proceed at 80%" band.

**This is a HALT GATE 1 report, not a diagnosis-complete report.** Two open, unconfirmed
questions block any responsible next step: (1) is the 93.5%→69.5% inference-path gap (and
the `v_shape`/`encirclement` 0% collapse specifically) a bug, and (2) is the further
69.5%→22.3%/3.3% gap really the training-regime/blend-timing mismatch hypothesized above, or
something else. Neither should be fixed unilaterally — both call for a scoped diagnostic
pass with an explicit plan, the same discipline this project applied to every prior
brittleness investigation (AUDIT.md secs AF/AG).

## Update 2026-08-07: gap 2 fix applied and retrained — ceiling barely moves, still HALT GATE 1

Per `docs/GAP_DIAGNOSIS.md`'s confirmed mechanism, `generate_dataset()` was fixed (3-regime
blend-timing labeling — dominant endpoint formation instead of always `"transitioning"`),
dataset regenerated, STGT retrained (80 full epochs this time, no early stop, best epoch 75,
`test_acc=0.9333`, 13m07s). `scripts/verify_upstream_physics.py` still passes; full suite
still 134/134.

Re-ran `scripts/phase0_ceiling.py --n 1000` (same seed=999, same protocol) against the new
checkpoint (`evaluation/phase0_ceiling_v2.json`):

| metric | before (broken transition labeling) | after (gap-2 fix) |
|---|---|---|
| window-level accuracy | 22.3% | 27.7% |
| **pair-level accuracy (the ceiling)** | **3.3% (17/509)** | **4.7% (24/509)** |

Per-class, the picture is mixed, not a clean win:

| true class | before | after |
|---|---|---|
| v_shape | 1.3% | **53.2%** (large improvement — consistent with gap 1's hypothesis that more varied/plentiful training exposure helps) |
| encirclement | 0.2% | 14.3% (still very poor) |
| column | 15.8% | 35.1% (improved) |
| diamond | 20.6% | **10.2%** (got worse) |
| dispersed | 25.6% | 45.3% (improved) |
| converging | 0.1% | 7.7% (still very poor) |
| shield | 24.0% | **15.4%** (got worse) |
| transitioning | 80.4% | **35.5%** (dropped sharply — expected, since far fewer training examples are now labeled `"transitioning"`, but its own recall also fell, not just its share) |

**Verdict: the gap-2 fix was correctly diagnosed and correctly implemented (verified via a
direct before/after regime test in `GAP_DIAGNOSIS.md` before ever touching training data), and
it measurably helped — `v_shape` in particular went from a confident, systematic 0% failure to
53.2%. But pair-level accuracy moved only 3.3%→4.7%, nowhere near the plan's 70% floor.**
Three formations (`diamond`, `shield`, and `transitioning`'s own recall) got WORSE, not just
unchanged — plausibly because boosting endpoint-formation representation via transition
examples came at some cost to how well-represented/well-separated other classes are in the
same fixed training budget (still only 9000 sequences, 80 epochs). Gap 1's core failure mode
(confident misclassification, not just the `v_shape`/`encirclement` instance of it) is still
clearly present for `converging` (7.7%) and `encirclement` (14.3%).

**This remains an unambiguous HALT GATE 1 trigger — 4.7% is not a borderline call any more
than 3.3% was.** Nothing further has been attempted; reporting per the halt protocol rather
than iterating on fixes without checking back in first.

## Update 2026-08-08: data size (3.3x) + epochs (80→150) — real but insufficient improvement

Per instruction, scaled both data (`--per-formation 3000 --transitions 9000`, 30000 total
sequences vs. 9000 before) and epochs (`--epochs 150`, early-stopped at 42, best epoch 30,
`test_acc=0.9771` — up from 0.9333). `verify_upstream_physics.py` and the full suite (134/134)
still pass.

`scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v3.json`):

| metric | 9k data / 80 epochs | 30k data / 150 epochs |
|---|---|---|
| window-level accuracy | 27.7% | 36.9% |
| **pair-level accuracy (the ceiling)** | **4.7% (24/509)** | **6.7% (34/509)** |

Real, monotonic improvement — 3.3%→4.7%→6.7% across the two fixes so far — but at this rate
of return (roughly +1.5-2 points per ~3x compute increase), reaching even a 30-40% ceiling
would require multiple further doublings of an already-substantial (30k sequences, 150
epochs) budget, let alone the plan's 70% floor. Per-class movement is not uniform or
monotonic either — `v_shape` actually fell back from 53.2% to 31.7% this round while
`diamond`/`shield`/`converging` recovered and improved past their original baselines —
consistent with a model still trading capacity between classes within a fixed architecture,
not converging toward a ceiling anywhere near usable.

**Conclusion: "increase data and epochs" is producing real gains but not remotely enough,
and the trend does not support it closing the gap on its own.** Per instruction, moving to
the next strategy (centroid-relative node features) rather than continuing to scale the same
lever further.

## Update 2026-08-08: centroid-relative node features — window accuracy transformed, pair-level ceiling barely moves, root cause identified

Per instruction's pre-authorized contingency, implemented the second strategy: `build_graph`
now uses centroid-relative offsets (`positions - positions.mean(dim=0)`) instead of absolute
positions as GAT node features, in both copies (`graph.py` and `stgt/model.py`, kept in sync).
Retrained from scratch on the same 30k dataset, 150 epochs (early-stopped at 29, best epoch
17, `test_acc=0.9873` — the highest yet, and convergence was dramatically faster: 98-99%
train accuracy by epoch 16). `verify_upstream_physics.py` and the full suite (134/134) pass.

`scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v4.json`):

| metric | 30k/150ep, absolute position | 30k/150ep, centroid-relative |
|---|---|---|
| window-level accuracy | 36.9% | **72.7%** |
| per-class range | 14.9%-72.0% (wide, uneven) | 52.3%-85.4% (much more even) |
| **pair-level accuracy (the ceiling)** | **6.7% (34/509)** | **6.1% (31/509)** |

**Window-level classification was transformed — this is the largest single improvement of
the whole session, and per-class accuracy is now reasonably even across all 8 classes
(52-85%, vs. previous runs' 0-80% spread with confident systematic failures on specific
classes).** But pair-level accuracy did not move — if anything it's marginally lower.

**Checked whether the existing (already-built, sec AG) `robust=True` majority-vote reduction
unlocks the gap between window- and pair-level accuracy that unanimity-based reduction can't
handle** (this was cheap to check — same predictions, no new GPU inference, added to
`phase0_ceiling.py` directly): `robust=True` gives 6.5% (33/509) vs. `robust=False`'s 6.1%
(31/509) — barely different. Bucket C shrinks substantially (188→100 — robust reduction does
structurally resolve many more sequences into a candidate pair), but almost all of that shifts
into bucket B (271→356), not bucket A (50→53) — the exact "recovers a pair, guard eats it
anyway" pattern sec AG already documented, now reproduced on a much better-classified model.

**Root cause identified: the model still over-predicts `"transitioning"` pervasively, at a
strikingly uniform 20-28% false-positive rate across every one of the 7 steady formations**
(`v_shape` 24.3%, `encirclement` 27.5%, `column` 20.5%, `diamond` 21.3%, `dispersed` 25.4%,
`converging` 25.1%, `shield` 27.1% — computed directly from the confusion matrix). With 15-30
windows per long, realistic trajectory, a ~20-27%-per-window false "transitioning" rate makes
it near-certain that ANY given trajectory contains at least one spurious ambiguous window,
which is enough to trip the `oov_name`/ambiguity-style guards regardless of how accurate
classification is everywhere else. **This is no longer a classification-quality problem —
window accuracy is now good. It is a residual, uniform-across-classes calibration/bias
problem specific to the "transitioning" class**, plausibly still connected to gap 2's
partially-addressed training regime (the fixed labeling now teaches 3 blend-timing regimes,
but the model may still be keying "transitioning" off a broader noise/drift signature than
genuine mid-blend geometry, especially given the higher-variance positions the acceleration
fix introduces even within nominally-steady segments).

**Both pre-authorized strategies have now been executed. Neither closed the gap.** Data+epochs
gave real but insufficient improvement (3.3%→4.7%→6.7%). Centroid-relative features
transformed window-level accuracy but left pair-level accuracy flat (6.7%→6.1%/6.5%), for a
specific, now-diagnosed reason (pervasive transitioning false-positives, not confusion
between real formations). **This remains an unambiguous HALT GATE 1 trigger.** Not
attempting a third strategy without checking in first.

## Update 2026-08-08: targeted fix for the transitioning false-positive rate — pair-level ceiling roughly doubles

Per instruction, targeted the false-positive rate specifically rather than guessing. Before
touching anything, diagnosed WHERE it concentrates (40 fresh chain-length-1 trajectories vs.
40 chain-length-2 trajectories, seed=4242, disjoint from everything else):

| window population | false-positive "transitioning" rate |
|---|---|
| fully unambiguous (chain length 1, zero blend anywhere) | **1.8%** |
| within 15 timesteps of a real blend boundary | **53.2%** |
| more than 15 timesteps from any blend boundary | **0.0%** |

**Not a broad calibration bug — the model is well-behaved on genuinely unambiguous input.**
The false positives concentrate almost entirely near real blend boundaries, where strategy
5's own gap-2 regime fix still let a wide grey zone into training: regime 1 ("transitioning")
examples could have as little as 28% genuine blend content with up to 64% residual pure
formation; regimes 0/2's "pure" examples could carry real blend content near their window
edges. Tightened the regime bounds (`src/swarm_intent/data.py`, commit `5baaceb`): regime 1
now requires the blend region to dominate the window (45-62% of `n_timesteps`, verified
numerically at generation time — mean 52%, was 28-44%); regimes 0/2 push blend to the very
edge with a short duration (74-90% pure content, was 66-86%).

Regenerated (30k sequences) and retrained (150 epochs, early-stopped at 22, best epoch 10,
`test_acc=0.8631` — notably lower and the validation curve was visibly more volatile than
prior runs, val_loss oscillating between 0.6 and 4.1). `verify_upstream_physics.py` and the
full suite (134/134) still pass.

`scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v5.json`):

| metric | before (strategy 4) | after (strategy 5) |
|---|---|---|
| window-level accuracy | 72.7% | 70.4% (roughly flat) |
| **pair-level accuracy, robust=False** | **6.1% (31/509)** | **12.2% (62/509)** |
| pair-level accuracy, robust=True | 6.5% (33/509) | 12.8% (65/509) |
| transitioning FP rate (mean across 7 classes) | ~24% | ~16% (diamond 21.3%→4.0%, shield 27.1%→8.5% improved most; others improved modestly) |

**Pair-level accuracy roughly doubled — the largest single relative jump from any one fix in
this program so far.** Not uniform, though: `encirclement`'s raw window accuracy regressed
sharply (62.2%→41.3%), the same capacity-trading pattern seen in strategy 3. The volatile
training curve (early stop at epoch 22 of 150, lower aggregate test accuracy than strategy 4)
suggests this checkpoint may not be fully converged — untested whether a steadier training
run (lower LR, more patience) would do better on the same fixed data.

**Still far below the 70% floor. This remains an unambiguous HALT GATE 1 trigger** — 12.2% is
real, meaningful progress, not a plateau, but not remotely close to "proceed" or even
"revise target" territory yet.

## Update 2026-08-08: the REAL ceiling — RULES-aware threat/intent/action accuracy, not just exact-pair

Exact-pair accuracy is a pessimistic lower bound: `RULES` (`llm_finetuning/build_sft_dataset.py`)
maps 49 `(from, to)` pairs onto only 4 threat levels, so a wrong recovered pair can still land
on the correct `threat_level`. Re-scored the SAME 509 pair-eligible records from
`evaluation/phase0_ceiling_v5.json` (strategy 5, no retraining/resampling) against `RULES` for
`threat_level`/`likely_intent`/`recommended_action` (`scripts/phase0_threat_ceiling.py`,
`evaluation/phase0_threat_ceiling_v5.json`):

| metric | robust=False | robust=True |
|---|---|---|
| (a) exact pair accuracy | 12.2% (62/509) | 12.8% (65/509) |
| **(b) THREAT ceiling** | **13.0%** | **13.6%** |
| (c) intent ceiling | 12.2% | 12.8% |
| action ceiling | 13.0% | 13.6% |
| n with no recovered pair at all (bucket B/C) | 431/509 (84.7%) | 428/509 (84.1%) |

4x4 threat confusion matrix (robust=False; robust=True nearly identical, `low` row 37→40):

| true \\ pred | low | medium | high | critical | no_recovery |
|---|---|---|---|---|---|
| low | 37 | 0 | 1 | 0 | 135 |
| medium | 0 | 0 | 1 | 0 | 182 |
| high | 10 | 0 | 29 | 0 | 104 |
| critical | 0 | 0 | 0 | 0 | 10 |

**Answering the question plainly: the 70% floor is NOT met on the threat ceiling either.**
13.0%/13.6% barely moves off exact-pair accuracy (12.2%/12.8%). The reason it barely moves is
itself the important finding: **`RULES`-mapping tolerance was never the bottleneck.** Of the
78 (robust=False) / 81 (robust=True) trajectories that DO reach a resolvable bucket-A pair,
conditional threat accuracy is 84.6%/85.2% — genuinely good. The bottleneck is that **84.7% of
trajectories never reach a resolvable pair in the first place** (bucket B, guard-blocked —
mostly the dispersed/converging ambiguity guard and oov-name/dominant-history-contradiction
guards per `stgt_bridge.py` — or bucket C, multi-hop/unresolvable). No `RULES` value is even
looked up for those. `critical` (10 true cases) has zero recoveries under either variant — the
rarest, highest-stakes class is also the one with the least evidence to judge.

This reframes the ceiling question: the classifier itself, when it commits to a clean pair, is
mostly right. The gate is `stgt_bridge`'s bucket-A resolution rate (15.3%/15.9% of eligible
trajectories), not the granularity of `RULES` or exact-pair matching. Step 2 (decomposing
pair-recovery failures) investigates whether that resolution-rate bottleneck sits in the
classifier's per-window accuracy or in the reduction/guard logic on top of it.

## Update 2026-08-08: pre-strategy-6 step 2 — decomposing pair-recovery failures

Re-ran inference (no retraining) with the identical seed=999 sampling regime as
`phase0_ceiling.py`, reproducing the same 509 pair-eligible trajectories index-for-index
(confirmed: 62 successes, 447 failures, exactly matching `phase0_ceiling_v5.json`). For each
failure, recorded per-window correctness, WHERE bad windows fall, whether the correct pair
was recoverable from a filtered prediction list containing only the classifier's correct
windows, and the exact `stgt_bridge` guard reason(s) that blocked bucket A.
(`scripts/phase0_decompose_failures.py`, `evaluation/phase0_decompose_failures.json`.)

**Finding 1 — chain-length-2 (an actual formation transition) NEVER succeeds:**

| chain length | n | n correct | accuracy |
|---|---|---|---|
| 1 (steady state) | 258 | 62 | 24.0% |
| 2 (single transition) | 251 | 0 | **0.0%** |

Every one of the 62 successes in strategy 5's whole ceiling measurement is a steady-state
trajectory. Not one of the 251 genuine single-hop transitions — arguably the most tactically
interesting case this whole system exists to detect — was ever correctly recovered.

**Finding 2 — the reduction logic is NOT the bottleneck; a specific unconditional guard is.**
47.2% of failures (211/447) had **zero misclassified windows** — the classifier's per-window
top-1 prediction was perfect for the entire trajectory, and it *still* failed to recover the
pair. Filtering each failure down to only its correctly-classified windows and re-running the
exact same reduction logic recovered the correct pair in just **4/447 (0.9%)** of cases — so
this is not "a few noisy windows tripping unanimity," the signal is clean and the pipeline
still rejects it. Tracing WHY, across all 447 failures:

| guard / bucket reason | count | fraction of failures |
|---|---|---|
| **`dispersed_converging_ambiguity`** | **272** | **60.9%** |
| bucket C (oscillation/multi-hop/terminal-unknown from classifier noise) | 154 | 34.5% |
| `oov_name` (a window read as "transitioning") | 42 | 9.4% |
| `dominant_history_contradiction` (tie) | 18 | 4.0% |
| bucket A but wrong pair (edge case, see below) | 16 | 3.6% |
| `low_confidence` | 11 | 2.5% |
| (a trajectory can trip more than one guard, so this does not sum to 447) |

`dispersed_converging_ambiguity` — `stgt_bridge.py`'s guard that fires when a window's
classifier probability for `dispersed` vs. `converging` is within 0.15 of each other — is by
far the single largest cause, present in **61% of all failures**. It is explicitly
**unconditional** per `stgt_bridge.py`'s own docstring (fires even after a correct robust
recovery) and it fires on windows of EVERY formation, not just dispersed/converging ones (this
run: diamond 81, v_shape 80, shield 62, column 50, encirclement 50, converging 22, dispersed
19 — roughly population-proportional, i.e. it is a generic classifier-calibration artifact,
not something specific to those two formations' trajectories). This exact defect was already
flagged, unfixed, in `AUDIT.md` sec AG ("this guard alone still blocks 92/96 of otherwise-
robustly-recovered cases") — this run is the first time it's been isolated and quantified as
THE dominant cause against the current (strategy 5) checkpoint specifically.

Bad windows, when present, concentrate in the interior of the trajectory (58.4%) over
leading/trailing edges (20.7%/20.9% each) — consistent with mid-trajectory blend-adjacent
confusion, not edge-of-window artifacts.

**Implication for strategy 6:** a steadier training run (the user's next-authorized step) can
plausibly raise per-window accuracy further and narrow some classifier probability margins,
but the dominant blocker — an unconditional, fixed-threshold (0.15) guard that fires
independent of whether the top-1 prediction was even correct — is bridge/reduction-code
logic, not a training-data or convergence problem. Retraining alone is unlikely to move the
pair-level ceiling by much unless this guard's threshold or unconditional behavior is also
revisited; that revisit is not authorized in this instruction and is not attempted here.
Proceeding to strategy 6 (retrain) exactly as instructed, reporting the result honestly
against this expectation.

## Update 2026-08-08: strategy 6 — steadier training schedule fixes the under-convergence

**Dataset class distribution checked first** (per instruction, to rule out "encirclement just
has fewer examples now" before retraining): strategy 5's 30k dataset is essentially balanced —
`v_shape` 3910, `encirclement` 3877, `column` 3865, `diamond` 3862, `dispersed` 3821,
`converging` 3867, `shield` 3832, `transitioning` 2966 (transitioning modestly lower by
design, not a data-starvation artifact for `encirclement` specifically). Ruled out.

**Change** (`src/swarm_intent/config.py`, `train.py`, `scripts/train_model.py`): replaced
`OneCycleLR` (peak lr=3e-4, decays to ~0 via `final_div_factor`) with linear warmup
(`warmup_pct=0.1` of total steps) + cosine decay to a **nonzero floor**
(`lr_min_frac=0.05` × peak), and retrained at a **lower peak LR (1e-4, was 3e-4)** with
**patience raised to 35 (was 12)**. Same strategy-5 dataset, no regeneration. 134/134 tests
still pass after the config/scheduler change.

**Full train/val curve** (`/tmp/v5_strategy6_train.log`, committed as
`evaluation/phase0_strategy6_train_log.txt`): smooth, mostly monotonic improvement through
epoch ~24 (val_loss 2.12→0.08, val_acc 0.16→0.99), then val_loss oscillates in later epochs
(occasionally spiking to 0.7-1.5) similar in shape to strategy 5's volatility — but now
around a much lower floor, with the best checkpoint found much later:

| | strategy 5 (OneCycleLR, lr=3e-4, patience=12) | strategy 6 (warmup+cosine-floor, lr=1e-4, patience=35) |
|---|---|---|
| best epoch | 10 | **51** |
| early-stopped at | 22/150 | **86/150** |
| best val_loss | not recorded precisely, test_acc-inferred poor | **0.0505** |
| test_acc | 0.8631 | **0.9958** |

**Per-class test accuracy** (`evaluation/phase0_strategy6_classification_report.json`; note:
found `evaluate_ml_model` in `llm/evaluate.py` double-normalizes already-normalized
`X_test.npy` — a pre-existing bug, not touched here, bypassed by evaluating directly):

| class | precision | recall | f1 |
|---|---|---|---|
| v_shape | 0.987 | 1.000 | 0.993 |
| **encirclement** | 1.000 | **0.986** | 0.993 |
| column | 1.000 | 1.000 | 1.000 |
| diamond | 0.990 | 1.000 | 0.995 |
| dispersed | 0.995 | 0.988 | 0.991 |
| converging | 0.997 | 1.000 | 0.998 |
| shield | 1.000 | 1.000 | 1.000 |
| transitioning | 1.000 | 0.991 | 0.995 |

**`encirclement` fully recovered** (0.986 recall, up from strategy 5's window-level 41.3% on
the ceiling test — though that's a different, harder eval; the in-distribution regression is
gone). Every class is now ≥98.6% on both precision and recall — essentially saturated on the
in-distribution test split. This confirms the under-convergence hypothesis: strategy 5's
aggressive LR schedule and low patience stopped training right as it was still finding a
better optimum.

**Caveat, flagged in advance (step 2's finding still applies):** this is in-distribution test
accuracy, not the ceiling metric. Step 2 showed the pair-level ceiling's dominant blocker
(`dispersed_converging_ambiguity`, 60.9% of failures) is a `stgt_bridge.py` guard that fires
independent of classification correctness — a better-converged classifier can narrow
probability margins and help at the margin, but is not expected to resolve that guard by
itself. Phase 0 step 4 (next) re-measures the actual pair-level and threat-level ceilings on
this checkpoint to see how much, if any, of this in-distribution gain transfers.

## Update 2026-08-08: step 4 — strategy 6 REGRESSES the ceiling despite fixing convergence

`scripts/phase0_ceiling.py --n 1000` and `scripts/phase0_threat_ceiling.py` re-run on the
strategy-6 checkpoint (`evaluation/phase0_ceiling_v6.json`,
`evaluation/phase0_threat_ceiling_v6.json`), same seed=999 protocol.

**The headline result is a regression, not an improvement — reported plainly, not softened:**

| metric | strategy 5 | strategy 6 | change |
|---|---|---|---|
| in-distribution test_acc | 86.3% | **99.6%** | fixed, as intended |
| window-level accuracy (ceiling test) | 70.4% | 69.1% | ~flat |
| **pair-level accuracy, robust=False** | **12.2% (62/509)** | **4.9% (25/509)** | **roughly HALVED** |
| pair-level accuracy, robust=True | 12.8% (65/509) | 5.3% (27/509) | roughly halved |
| **threat ceiling, robust=False** | **13.0%** | **6.3%** | **roughly HALVED** |
| threat ceiling, robust=True | 13.6% | 6.9% | roughly halved |
| bucket A size (n reaching a resolvable pair) | 78 | 77 | ~flat |
| **conditional accuracy WITHIN bucket A** | **62/78 = 79.5%** | **25/77 = 32.5%** | **collapsed** |

Per-class window accuracy shows the mechanism — the same capacity-trading pattern seen all
program, but now net-negative:

| class | strategy 5 | strategy 6 |
|---|---|---|
| v_shape | 78.0% | 77.4% |
| **encirclement** | 41.3% | **61.8%** (recovered) |
| column | 68.8% | 76.2% |
| **diamond** | 94.0% | **70.0%** (regressed) |
| dispersed | 71.3% | 71.4% |
| **converging** | 66.7% | **37.4%** (regressed hard) |
| **shield** | 90.3% | **66.5%** (regressed) |
| **transitioning** | 57.3% | **87.5%** (big gain) |

**Diagnosis:** strategy 6 converged much more sharply to the training distribution
(`generate_dataset()`'s spread/noise ranges are narrower — U(0.7,1.5)/U(0.3,0.8) — than the
ceiling test's realistic long-trajectory sampling — U(0.6,1.8)/U(0.15,1.4)). Fixing the
under-convergence let the model fit that narrower distribution far more tightly (near-100%
in-distribution accuracy), which helped `encirclement`/`transitioning` generalize better but
made `diamond`/`shield`/`converging` generalize WORSE to the wider real-world variation — a
classic overfitting/generalization tradeoff, not a bug. The net effect on bucket A's own
conditional accuracy (79.5%→32.5%) shows this isn't just "more guard-blocking" — even when
the pipeline DOES commit to an answer, it is now right less than half as often.

**Strategy 6 achieved its own literal objective (fix the training curve, raise test_acc) while
making the metric that actually matters significantly worse.** `swarm_data/best_model.pt` is
currently the strategy-6 checkpoint; `swarm_data/best_model_strategy5_backup.pt` holds the
better-performing strategy-5 checkpoint if reverting is wanted. **HALT GATE 1 is unchanged —
still far below the 70% floor — and this turn's exploration ends with the ceiling lower than
where strategy 5 left it.** Per instruction, stopping here to report both trajectories.

## Update 2026-08-09: revert + step 1 — the ambiguity guard IS broken, confirmed directly

Reverted `swarm_data/best_model.pt` to the strategy-5 checkpoint (SHA-256 verified). Standing
rule recorded: checkpoint selection must be judged on ceiling, never test accuracy alone.

**Step 1: audited `dispersed_converging_ambiguity` directly against real predictions.** The
guard (`stgt_bridge.py:114-120`):

```python
def _is_ambiguous_dispersed_converging(class_probabilities: dict) -> bool:
    d, c = class_probabilities.get("dispersed"), class_probabilities.get("converging")
    return abs(d - c) < DISPERSED_CONVERGING_AMBIGUITY_MARGIN  # margin = 0.15
```

This checks only whether the two RAW probabilities happen to be close to each other in
absolute terms — never whether either is actually competitive for the window's top
prediction. Ran `scripts/phase0_guard_audit.py` (same seed=999 population, strategy-5
checkpoint, inference only): across 1469 windows, the guard fires on **75.8%** of them. Of
those firings:

| condition | count | fraction of firings |
|---|---|---|
| BOTH dispersed and converging in top-2 (genuinely competing) | 19 | 1.7% |
| ONE of the two in top-2 | 358 | 32.2% |
| **NEITHER in top-2 (spurious)** | **736** | **66.1%** |

**Confirmed: the guard is not testing what it claims to test.** Example firings: a window
predicted `shield` at 98.97% confidence, with `dispersed=0.0012`/`converging=0.0005` —
`|0.0012-0.0005| = 0.0007 < 0.15`, so the guard fires, on a window that is not remotely
ambiguous about anything. With 8 softmax classes, when one class dominates, the remaining
~7 split a small residual probability mass, and ANY two of them (not just dispersed/
converging) will very often land within 0.15 of each other purely because both are near
zero — the guard was written as if 0.15 were a meaningful gap regardless of scale, but at
these magnitudes it's nearly always satisfied by chance. This fully explains step 2's
finding (60.9% of pair-recovery failures, firing across every formation class uniformly) —
it was never actually testing dispersed/converging contention.

## Update 2026-08-09: step 2 — the guard fix, isolated and full re-measurement

**Fix (`stgt_bridge.py`'s `_is_ambiguous_dispersed_converging`):** now requires dispersed and
converging to be the window's TOP-2 predicted classes (genuinely competing for the top spot),
in addition to the original `abs(d-c) < 0.15` closeness check. One added condition, no other
change to the function, no retraining, no checkpoint change (still strategy 5).

**Isolated re-measurement (`scripts/phase0_guard_audit.py --n 1000`, same seed=999 population,
same checkpoint as step 1's audit — the only variable changed is the guard code):**

| condition | before (step 1) | after (step 2) |
|---|---|---|
| guard fires on | 1113/1469 windows (75.8%) | **19/1469 windows (1.3%)** |
| BOTH in top-2 (genuinely competing) | 19 (1.7% of firings) | **19 (100.0% of firings)** |
| ONE in top-2 | 358 (32.2%) | **0 (0.0%)** |
| NEITHER in top-2 (spurious) | 736 (66.1%) | **0 (0.0%)** |

**Every one of the 19 remaining firings is now genuine dispersed/converging contention.
Zero spurious firings, zero one-sided firings — the fix is complete and clean, not a partial
tightening.** The original 19 "genuinely competing" firings from step 1 survive unchanged
(the top-2 condition is additive, never removes a firing that was already both-in-top-2 AND
close), so this is exactly the expected before/after: everything spurious removed, nothing
genuine lost.

**Full ceiling re-measurement (`phase0_ceiling.py --n 1000` → `evaluation/
phase0_ceiling_v5_guardfix.json`, `phase0_threat_ceiling.py` → `evaluation/
phase0_threat_ceiling_v5_guardfix.json`, same seed=999/checkpoint/protocol as every prior
ceiling measurement — the guard fix is the only variable changed):**

| metric | strategy 5, before guard fix | strategy 5, after guard fix | change |
|---|---|---|---|
| bucket A size, robust=False (of 509 eligible) | 78 | **294 (57.8%)** | **+216** |
| bucket A size, robust=True | 79 | **396 (77.8%)** | **+317** |
| pair-level accuracy, robust=False | 12.2% (62/509) | **47.0% (239/509)** | **+34.8pt** |
| pair-level accuracy, robust=True | 12.8% (65/509) | **48.5% (247/509)** | **+35.7pt** |
| precision WITHIN bucket A, robust=False | 79.5% (62/78) | **81.3% (239/294)** | +1.8pt |
| **precision WITHIN bucket A, robust=True** | **~20% (sec AG, capped by this defect)** | **62.4% (247/396)** | **+~42pt** |
| threat ceiling, robust=False | 13.0% | **52.3%** | **+39.3pt** |
| threat ceiling, robust=True | 13.6% | **58.7%** | **+45.1pt** |

**This is by far the single largest gain of the entire Phase 0 program, and it came from a
one-condition bug fix, not from retraining.** `robust=True`'s recovery precision — the exact
number sec AG measured at 17-25% and used to justify "robust reduction should not ship" — is
now 62.4%, because the dominant source of low-precision robust recoveries was this same
mis-specified guard corrupting the class_probabilities signal the majority-vote reduction
also depends on. Sec AG's verdict was correct given the code it was measured against; it does
not hold against this fixed guard, and should not be treated as a permanent verdict on
`robust=True` — a follow-up re-evaluation against this fix is the natural next step if a full
pipeline_v2 re-run is wanted.

**HALT GATE 1, stated plainly: still not cleared.** Threat ceiling (52.3-58.7%) remains below
the 70% floor — this is the biggest single jump in the program (from ~57-58 points below floor
to ~11-18 points below floor) but is not itself sufficient to clear the gate. The gap remaining
is far smaller and now looks structurally different: no longer "the guard is nonsense," but
genuine STGT window-level accuracy (70.4% overall, with `encirclement` at 41.3% and
`transitioning` at 57.3% dragging the average down) and the underlying pair-reduction task's
own remaining difficulty.

## Update 2026-08-09: step 3 — chain-length-2 is still broken, and it's NOT a second guard bug

Before the guard fix, `scripts/phase0_decompose_failures.py` found chain-length-2 (single real
transition) accuracy near zero while chain-length-1 (steady state) succeeded. Re-measured with
the fixed guard (`scripts/phase0_chainlength_breakdown.py --n 1000`, same seed=999 population,
strategy-5 checkpoint):

| chain_length | n | pair_acc (robust=False) | pair_acc (robust=True) | threat_acc (F) | threat_acc (T) |
|---|---|---|---|---|---|
| 1 (steady state) | 258 | **86.8%** | 88.4% | 86.8% | 88.8% |
| 2 (single transition) | 251 | **6.0%** | 7.6% | 16.7% | 27.9% |
| 3+ (no RULES key) | 491 | n/a | n/a | n/a | n/a |

**Confirmed: chain-length-2 is still almost entirely broken (6.0% vs 86.8% for steady state) —
this is real, not fixed by the guard fix, exactly the user's hypothesis.** Chain 3+'s bucket-A
false-positive rate (any A-resolution here is wrong by construction — no 2-tuple ground truth
exists) is low and roughly stable regardless of reduction mode: 1.8% (robust=False), 1.8%
(robust=True) — chain 3+ isn't where the problem lives.

**But it is NOT a second bug of the same class as the dispersed_converging guard.** Traced 20
failing chain-length-2 trajectories stage by stage (window classifications → guard →
temporal transition derivation → reduction → bucket; full trace:
`evaluation/phase0_chain2_trace.txt`):

| diagnosis | n | % of 20 |
|---|---|---|
| all_windows_transitioning | 7 | 35.0% |
| structural_reduction_wrong_pair | 5 | 25.0% |
| trailing_transitioning_run | 4 | 20.0% |
| blocked_by_oov_name_guard | 3 | 15.0% |
| spurious_third_formation_from_misclassification | 1 | 5.0% |

**60% of traced failures (all_windows_transitioning + trailing_transitioning_run + most of
structural_reduction_wrong_pair) trace back to the SAME windowing-artefact mechanism identified
in the earlier engagement (AUDIT.md sec AF step 4): chain-length-2 trajectories are a SINGLE
hop, 50-100 timesteps, often only 1-2 sliding windows total.** Two concrete examples from the
trace:
- Trajectory 34 (`dispersed→encirclement`, 2 windows): BOTH windows' true labels are
  `dispersed`/`encirclement` respectively, but STGT reads `transitioning` on both (98% and 98%
  confidence) — with only 1-2 windows and no redundancy to average over, a single hard
  misclassification collapses the whole trajectory to `all_unknown`.
- Trajectory 18 (`shield→v_shape`, 2 windows): both windows' TRUE label is `shield` — the
  transition to `v_shape` never actually completes within the observed span at all. STGT
  correctly, confidently reads `shield` on both windows (99% conf) and the bridge correctly,
  confidently reduces to `(shield, shield)` — WRONG relative to the generator's chain, but not
  because anything in the classifier or bridge is broken. The ground truth itself isn't
  observable in the data as generated.

**Only 15% (blocked_by_oov_name_guard) is genuine bridge-logic brittleness of the kind step 2
already fixed once.** Example: trajectory 11 (`v_shape→column`, 5 windows) structurally
reduces to the EXACT correct pair (`v_shape`, `column`) — `rules_key` matches ground truth
exactly — but 2 interior windows read `transitioning` (a real, if brief, blend read), tripping
`oov_name` and routing to bucket B instead of A. This IS fixable without retraining (it's
exactly what `robust=True`'s trim/majority-vote logic is for), and IS covered by step 4's guard
audit below.

**5% is genuine STGT misclassification** (trajectory 31: `dispersed`→`converging`→`diamond`
predicted where truth is `converging`→`diamond`, from a single bad window at the very start) —
not a bridge-logic issue, a classifier-accuracy one.

**Verdict, stated plainly: chain-length-2's brokenness is NOT primarily a fixable bridge bug.**
It is dominated (60%) by a structural property of the sampling regime — segment lengths
(50-100 timesteps) close to or barely above `window_size` (50, architecturally fixed for this
checkpoint) mean many single-transition trajectories provide only 1-2 observation windows, too
few for the destination formation to reliably resolve OR be observed at all. No bridge-logic
change (guard fix, robust reduction, anything short of retraining with longer segments or a
larger `max_seq_len`) can fix the dominant share of this. The 15% oov_name-guard share is real
and actionable now — see step 4's guard audit.

## Update 2026-08-09: step 4 — auditing every other guard/rule the same way

`scripts/phase0_full_guard_audit.py --n 1000`, same seed=999 population as every prior
measurement. For each boolean guard: fire rate over the 509 pair-eligible trajectories,
and — among trajectories where the guard is the SOLE reason `bucket != A`
(`guard_reasons == [that guard]`) — what fraction of those "sole firings" are spurious, i.e.
the structural `rules_key` (computed BEFORE guard checks, available regardless of bucket)
already equals the ground-truth pair, so the guard blocked what would otherwise have been the
correct answer:

| guard | fire rate (of 509) | n sole-firing | sole-firing spurious rate | failures attributable |
|---|---|---|---|---|
| **oov_name** | 42 (8.3%) | 29 | **69.0% (20/29)** | **20** |
| **dominant_history_contradiction** | 18 (3.5%) | 4 | **100.0% (4/4)** | **4** |
| low_confidence | 11 (2.2%) | 4 | 25.0% (1/4) | 1 |
| dispersed_converging_ambiguity (post-fix) | 11 (2.2%) | 4 | 50.0% (2/4) | 2 |

**Two more guards of the same defective class as the original dispersed_converging bug,
found by the same methodology:**

- **`dominant_history_contradiction`: 100% spurious when it's the sole blocker (n=4, small but
  unambiguous).** This guard fires when the two known formations' PREDICTED window counts are
  exactly tied — but a raw count tie says nothing about whether the underlying (a,b) pair is
  actually ambiguous; a clean, obviously-correct 2/2 window split ties just as easily as a
  genuinely uncertain one. Every sole-firing case in this sample blocked an answer that was
  already right. Same defect shape as the original bug: testing a condition that doesn't
  actually correlate with untrustworthiness.
- **`oov_name`: fires on 8.3% of trajectories, 69.0% spurious when sole, blocking 20 correct
  answers outright — the highest-VOLUME actionable defect found.** Window-level check: of the
  68 total unknown windows across all oov_name-firing trajectories, 57.4% are
  `spurious_misclassification` (STGT wrongly read `"transitioning"` on a window whose true
  label was a real, settled formation) vs. 42.6% `genuine_transitioning` (real blend
  ambiguity). The guard is reacting to real classifier noise more than half the time, but its
  response (blanket-block on ANY unknown window at all, zero tolerance, no threshold) massively
  overreacts relative to how much signal the OTHER windows in the same trajectory usually
  still carry.
- `low_confidence` and the post-fix `dispersed_converging_ambiguity` are comparatively minor
  and mostly-justified (25% and 50% spurious respectively, both on n=4 — small volume either
  way, not a priority).

**Two non-guard reduction mechanisms, audited the same way:**

- **`key_windows` cap (`DEFAULT_MAX_KEY_WINDOWS=10`): no bug found.** 373/1000 trajectories
  exceed the cap (almost entirely chain 3+, longer sequences) and get capped, but **0/373**
  capped selections are missing a true endpoint formation from the narrative shown downstream —
  the priority-based selection (first/last/unknown/low-confidence/ambiguous, sec V's original
  design) is working as intended.
- **Leading/trailing transitioning-run trim (`robust=True`'s `_robust_reduce`, sec AG): a real,
  significant precision cost, found by the same window-level ground-truth check.** Fires
  (strips >=1 window) on 101/509 (19.8%) pair-eligible trajectories, 168 windows trimmed total,
  and **105/168 (62.5%) of trimmed windows have a true label that was NOT `"transitioning"`** —
  i.e. the trim step is discarding genuine signal (a window the model simply misclassified as
  transitioning) more often than it's discarding real noise. This directly explains why
  `robust=True`'s within-bucket-A precision plateaus at 62.4% even after the dispersed_converging
  fix (sec AG/step 2): the majority-vote algorithm's own trim step is itself contaminated by
  the same "assume any unknown-run is genuine ambiguity" fallacy as `oov_name`, just applied
  before the vote instead of as a guard after it.

**Verdict: the dispersed_converging bug was not an isolated defect — it's an instance of a
recurring pattern (treat a raw/local signal as evidence of genuine ambiguity with no
competitiveness or ground-truth-correlated threshold) that also appears in
`dominant_history_contradiction`, `oov_name`, and the `robust=True` trim step, in that rough
order of severity. `key_windows` capping is the one mechanism audited that is NOT part of this
pattern.** Audit only, per this turn's scope — no code changes made; these are documented
findings for a future fix turn, not implemented here.

## Update 2026-08-09: step 5 — robust=True threshold re-swept post-guard-fix

`DEFAULT_ROBUST_THRESHOLD=0.7` was tuned once (sec AG, the earlier engagement), before this
session's guard fix and full guard audit — both changed the signal the threshold operates on.
Re-swept against the CURRENT pipeline (`scripts/phase0_robust_threshold_sweep.py`), tuned on a
DEV split ONLY (seed=1, disjoint from the seed=999 held-out population every other Phase 0
measurement uses), then confirmed on held-out seed=999:

| threshold | dev coverage | dev precision | held-out coverage | held-out precision |
|---|---|---|---|---|
| 0.45-0.50 | 84.8% | 60.4% | 83.5% | 60.5% |
| **0.55-0.65** | **81.1%** | **63.1% (best)** | — | — |
| 0.70-1.00 (current shipped default) | 77.9% | 62.6% | 77.8% | 62.4% |

**Dev-vs-held-out precision gap at the tested operating points: 0.1pt — essentially perfect
generalization, the threshold is not overfit to the dev split** (this session's own version
of sec AG step 4's standing check).

**The precision curve across the ENTIRE sweep is flat (60.4-63.1%, a 2.7-point range across
every threshold from 0.45 to 1.00).** This is itself a finding, not just a null result:
consistent with step 4's guard audit, the threshold barely moves precision because the
DOMINANT source of contamination — the leading/trailing trim step discarding genuine signal
62.5% of the time — happens BEFORE the majority vote and isn't gated by this parameter at
all. Raising or lowering the vote threshold cannot fix a problem the vote never gets a chance
to see.

**Recommended operating point: 0.55-0.65 (e.g. 0.6), NOT the current shipped 0.7.** It
Pareto-dominates the current default — more coverage (81.1% vs 77.9%, +3.2pt) at equal-or-
better precision (63.1% vs 62.6%, +0.5pt) — a low-risk, unambiguous improvement over what's
shipped today. Going lower (0.45-0.50) buys further coverage (84.8%, +6.9pt over current) but
at a real precision cost (60.4% vs 62.6%, -2.2pt) — a legitimate choice if coverage is valued
more than contamination, but not a free win the way 0.6 is.

**Tradeoff stated explicitly, at the recommended 0.6:** moving from 0.7 to 0.6 lets ~3.2% more
of the 509 pair-eligible population reach Layer 1 (roughly 16 more trajectories out of 509),
with the fraction of WRONG answers among everything reaching Layer 1 essentially unchanged
(36.9% wrong at 0.7 → 36.9% wrong at 0.6, both ≈37% contamination). At 0.45, coverage rises
further (~35 more trajectories vs 0.7) but wrong-key contamination among Layer-1 answers rises
to ~39.5%. **None of these operating points get contamination meaningfully below ~37% without
also fixing the trim step itself** — the threshold is a minor lever on top of a much bigger,
already-identified problem. Audit/recommendation only, per scope — `DEFAULT_ROBUST_THRESHOLD`
left at 0.7 in code; changing it is a decision for a future turn, not made here.

## Update 2026-08-09: step 6 — HALT GATE 1 re-examined, end-to-end threat accuracy projected

The 70% floor was set when pair-level (12.2%) and threat-level (13.0%) ceilings were nearly
identical, so a single number implicitly gated both. They have since diverged sharply — 47.0%
pair vs 52.3% threat (robust=False) — because RULES maps 49 (from,to) pairs onto only 4
threat levels: a wrong recovered pair frequently still lands on the correct threat. This
divergence will only widen as pair-level brittleness gets fixed (this session's guard fixes
help pair-level directly; RULES' many-to-one structure means threat-level doesn't move 1:1
with it). `scripts/phase0_endtoend_projection.py` computes what the CURRENT pipeline actually
projects to, end-to-end (pure post-processing of already-saved ceiling data, no new inference):

**Identity used**: `phase0_threat_ceiling.py`'s reported "threat ceiling" (52.3%/58.7%) is
ALREADY `P(bucket A) × threat_accuracy_within_bucket_A` — bucket B/C score 0 in that metric
(no LLM layer runs in these ceiling scripts). The only missing term for a true end-to-end
number is Layer 3's real contribution on bucket C; bucket B (guard) contributes 0 either way,
by design.

| | robust=False (shipped default) | robust=True @ 0.7 |
|---|---|---|
| bucket A / B / C | 57.8% / 12.0% / 30.3% | 77.8% / 10.6% / 11.6% |
| **threat accuracy WITHIN bucket A** | **90.5% (266/294)** | 75.5% (299/396) |
| current measured "threat ceiling" | 52.3% | 58.7% |
| end-to-end (Layer 3 @ 20%, conservative) | 58.3% | 61.1% |
| **end-to-end (Layer 3 @ 30.9%, sec AF's measured v3b-fix)** | **61.6%** | **62.3%** |
| end-to-end (Layer 3 @ 40%, optimistic) | 64.4% | 63.4% |

Threat-accuracy-within-bucket-A (90.5% robust=False) is dramatically higher than pair-accuracy-
within-bucket-A (81.3%, sec AG/step 2) — direct confirmation of the many-to-one RULES mapping
effect. Layer 3's contribution is a DISCLOSED ESTIMATE, not measured this turn: reused from
the earlier engagement's real-STGT-output eval (AUDIT.md sec AF, `evaluation/
eval_real_stgt_output.json`, v3b-fix 77/249 = 30.9%) — a different checkpoint/bridge state
(pre-guard-fix) and not bucket-conditioned (that eval ran v3b-fix on every case, not just
bucket-C-shaped ones), so reported as a central estimate with an explicit conservative/
optimistic band rather than a single number treated as precise.

**Central projection: end-to-end threat accuracy is ~61.6-62.3%, essentially the same under
either reduction mode** — `robust=True` reaches more of the population via the cheap,
deterministic Layer 1 pathway (77.8% vs 57.8% coverage) at lower per-case quality within it
(75.5% vs 90.5%), and those two effects roughly cancel in the end-to-end number. `robust=True`
DOES have a materially narrower sensitivity band (61.1-63.4%, 2.3pt) than `robust=False`
(58.3-64.4%, 6.1pt), because it depends on the uncertain Layer-3 estimate for a much smaller
share of the population (11.6% vs 30.3%) — a real, if secondary, argument for shipping
`robust=True` beyond precision alone.

**HALT GATE 1 verdict, stated plainly: even under the most favorable end-to-end framing tested
(optimistic Layer 3, robust=False, 64.4%), the projection does not clear a 70% floor.** The
gap is real, not an artefact of scoring bucket B/C as zero. But the RIGHT QUESTION per this
turn's instruction is whether 70% stated in PAIR-LEVEL terms is still the right gate, and the
answer is no — it is measuring the wrong quantity for what the gate is actually meant to
protect (end-to-end tactical correctness), and pair-level and threat-level will keep
diverging as bridge fixes land. **Recommendation: restate HALT GATE 1 in end-to-end threat-
accuracy terms going forward, not pair-level terms.** Whether the numeric floor stays 70% or
is revised is a policy decision this projection informs but does not make — the projected
central estimate (~62%) is ~8 points short of 70% either way, so the gate does not
mechanically clear regardless of which of the two metrics it's restated against; what changes
is which number future turns should actually be trying to move.

## Update 2026-08-09: chain-2 observability — is 18.7% a generator ceiling or an STGT problem?

`UPSTREAM_ISSUES.md` issue #3 hypothesized chain-2's failure is dominated by the destination
formation never becoming observable at all. This directly measures it, per-trajectory, against
the same seed=999/n=1000 population every table above uses
(`scripts/phase0_chain2_observability.py`; full derivation in `docs/V5_LOG.md`'s 2026-08-09
step-24 entry). **Observability criterion (reused, not invented): a window's true label is
`Counter(true_labels[window]).most_common(1)`; destination B is "observable" if >=1 window's
true-label majority is B.**

| chain-2 observability | n | % of 251 | pair_acc (robust=False) | pair_acc (robust=True) |
|---|---|---|---|---|
| OBS_CLEAR (B majority in >=2 windows) | 28 | 11.2% | 57.1% | 57.1% |
| OBS_PARTIAL (B majority in exactly 1 window) | 96 | 38.2% | 19.8% | 19.8% |
| OBS_NONE (B never a window majority) | 127 | 50.6% | 9.4% | 9.4% |

**Destination B is never observable in 50.6% of chain-2 trajectories — confirms issue #3's
hypothesis as the single largest cause, and puts a hard 49.4% ceiling on chain-2 pair accuracy
under the current generator, regardless of STGT quality.** But observability is not the whole
story: **OBS_CLEAR (the best-case group, redundant unambiguous destination signal) still only
reaches 57.1%** — a real, separate STGT-recognition-quality gap remains even when the generator
does its part.

A manually-reviewed 20-case failure trace (balanced across all three observability groups,
correcting the script's programmatic first-pass categorization where it conflated source- vs.
destination-formation misclassification) found, among traced failures: **40% destination not
observable (generator), 55% genuine STGT misclassification, 5% thin-plurality ground-truth
labeling ambiguity, 0% bridge/reduction-logic defects.** The STGT-misclassification share
traces to two recurring, largely-already-documented mechanisms: near-blend-boundary
`"transitioning"` over-prediction (the same pattern strategy 5 above measured at a 53.2% false-
positive rate), and confident (non-near-tie) `dispersed`/`converging` SOURCE misclassification
away from any blend region — a failure mode the `dispersed_converging_ambiguity` guard cannot
catch, since it only fires on a close top-2 tie.

A companion Monte Carlo comparison (`scripts/phase0_chain2_blend_distributions.py`, no GPU)
found **0% overlap** between eval's per-hop blend-timing shape (start 28-50% into the segment)
and any of `generate_dataset()`'s three training blend regimes (which only ever teach blends
starting at 74-88%, 12-28%, or 2-14% into a fixed 50-timestep window) — a plausible mechanistic
explanation for the near-blend-boundary misclassification specifically.

**Verdict: not single-cause.** ~50.6% generator-observability ceiling, ~32-33% independent STGT
recognition-quality gap (extrapolated from the 20-case trace onto the full observable-failure
population), ~3% labeling-ambiguity residual, 0% bridge-logic. A minimal, no-retrain,
eval-harness-only generator fix (widen `build_long_sequence`'s segment length / decouple
destination dwell time from segment length so `D = seg_len - blend_end >= 30` timesteps
reliably) is specified but **not implemented this turn** — see `docs/V5_LOG.md` step 24 for
the full spec and the disclosed caveat that this fix alone would raise the ceiling toward
~85-90% at best (extrapolating OBS_CLEAR's 57.1%, not 100%), not close the whole gap, because
the STGT-recognition-quality share is real and independent. Full data:
`evaluation/phase0_chain2_observability.json`, `evaluation/phase0_chain2_observability_trace.txt`.

## Update 2026-08-09: the dwell-time generator fix, implemented — chain-2 more than doubles, no retrain

Step 24's spec (above) implemented, in `scripts/phase0_decompose_failures.py`'s
`build_long_sequence_labeled` only (the 4 other verbatim duplicates of this sampling logic
left untouched, disclosed follow-up — see `docs/V5_LOG.md` step 25). Same frozen strategy-5
checkpoint throughout, zero retraining.

**Fix:** `dwell` (post-blend settled-B timesteps) is now sampled DIRECTLY, `~Uniform{40,60}`,
guaranteeing `D >= 40` against a derived minimum requirement of `D >= 35` (26 for a window
majority + 9 for stride=10 grid slack — precisely re-derived this turn, see V5_LOG). `seg_len`
is now DERIVED (`lead_in + blend_duration + dwell`) instead of sampled first with dwell left
as an unconstrained byproduct.

| observability | before | after |
|---|---|---|
| Observable | 49.4% | **100.0%** |
| OBS_CLEAR | 11.2% (n=28) | **99.1% (n=231)** |
| OBS_PARTIAL | 38.2% (n=96) | 0.9% (n=2) |
| OBS_NONE | 50.6% (n=127) | **0.0% (n=0)** |

| chain_length | n | pair_acc (F) | threat_acc (F) |
|---|---|---|---|
| 1 (steady state) | 252 | 84.1% (flat within noise, unrelated branch untouched) | 85.7% |
| **2 (single transition)** | 233 | **39.9% (was 18.7% — more than doubled)** | **72.1% (was 31.9%)** |
| 3+ (no RULES key) | 515 | n/a | n/a |

Chain-3+ bucket-A false-positive rate ticked up 9.0%→12.6% — modest, disclosed, not alarming
(more windows per chain-3+ trajectory now, slightly more surface for spurious ≤2-length-
looking noise).

**Two disclosed caveats, neither hidden:**

1. **A new, self-inflicted, ~15%-of-failures SOURCE-observability gap.** `LEAD_IN_RANGE=(15,35)`
  was chosen to match the old formula's realized scale, not derived with the same `D>=35`
  rigor as `MIN_DWELL_RANGE` — roughly the bottom half of that range is too short for the
  SOURCE formation to ever win a window majority, the exact same failure mode mirrored onto
  the wrong end. Trivial next fix: widen `LEAD_IN_RANGE` the same way.
2. **Train/eval blend-timing overlap is STILL 0.0%** (`scripts/phase0_chain2_blend_distributions_v2.py`)
  — the mismatch shape shifted (eval's blend start now reaches into training regime 1's start
  range) but duration still never matches regime 1's blend-dominates requirement, so the two
  conditions never jointly hold. Flagged as the likely driver of the now-dominant (80% of a
  freshly manually-traced 20-case sample, up from 55%) genuine STGT-misclassification failure
  mode — explicitly NOT fixed this turn.

**Decision: A — observability fix successful, remaining problem primarily STGT recognition**,
with the two caveats above as the concrete next steps (a two-line symmetrization fix, then a
targeted STGT experiment on the blend-timing mismatch specifically, not a generic retrain).
Full detail: `docs/V5_LOG.md`'s 2026-08-09 step-25 entry.

## Update 2026-08-09: consolidation + source symmetrization — chain-2 more than triples total

Both of step 25's caveats resolved this turn (`docs/V5_LOG.md` step 26 for full detail). (1)
The 5 duplicate copies of the eval-trajectory sampling logic consolidated into
`src/swarm_intent/eval_trajectories.py`, imported by all 4 LIVE eval scripts (one, a frozen
historical-reproduction script, deliberately excluded, documented). Confirmed **none of the 5
ever fed real STGT training data** — that's `generate_dataset()`, entirely separate code — so
consolidation is about eval-harness reproducibility, not training-data provenance. (2)
`LEAD_IN_RANGE` symmetrized from `(15,35)` (realized-scale guess) to `(30,50)` (derived the
same way as `MIN_DWELL_RANGE`: source needs `lead_in>=25` to hold an outright window-0
majority, no stride-slack term on this side since window 0 always starts at `t=0`).

| | dest-only fix | **both fixes** |
|---|---|---|
| source OBS_NONE | 16.3% | **0.0%** |
| destination OBS_NONE | 0.0% | 0.0% |
| **chain-2 pair_acc** | 39.9% | **65.8%** (was 18.7% at original baseline — 3.5x) |
| **chain-2 threat_acc** | 72.1% | **76.3%** |
| chain-3+ false-positive rate | 12.6% | **1.8%** (below the 9.0% original baseline; z=-6.68, p=2.4e-11 vs. dest-only, mechanistically explained — same shared per-hop sampling loop affects every hop of every chain length, not just chain-2's single hop) |

**Refined failure taxonomy (20 fresh failures, manually traced): 100% boundary/blend-timing-
concentrated at the trajectory level, 0% clean (non-boundary) misses.** Blend-overlap
Monte Carlo re-run a third time: **still 0.0%**, unchanged across three independent formula
revisions.

**Decision: B — blend-timing distribution mismatch is the dominant remaining issue and
should be fixed BEFORE any STGT training/capacity change.** Both observability fixes worked
cleanly and the chain-3+ false-positive concern is resolved (not a live bug). But the
taxonomy gives no support for a capacity-limited-STGT story (0% clean misses) and the
blend-overlap gap has now persisted, unmoved, through three formula iterations — the next
experiment should target that mismatch directly (a 4th training regime, or a widened regime
1, matching eval's actual realized blend shape), not a generic retrain. Not started this
session, per instruction. Full detail: `docs/V5_LOG.md`'s 2026-08-09 step-26 entry.
