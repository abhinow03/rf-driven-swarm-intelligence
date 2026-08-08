# Phase 0: the ceiling — measured, and it is far below the plan's floor

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
