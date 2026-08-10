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
| 5 | target the transitioning false-positive rate, diagnosis-driven regime retightening | 6.1% → **12.2%** (62/509); robust=True 6.5%→12.8% | diagnosed first (1.8% FP on unambiguous windows, 53.2% near a real blend boundary — not a broad calibration bug, a regime-boundary grey zone); tightened `generate_dataset()`'s regime bounds accordingly; roughly DOUBLED the ceiling, the largest single jump so far; `encirclement` regressed (capacity trading again); still far below 70% |
| 6 | steadier LR schedule (warmup+cosine-floor, lower peak lr, higher patience), retrain | 12.2% → **4.9%** (25/509); robust=True 12.8%→5.3% | fixed in-distribution under-convergence (test_acc 86.3%→99.6%) but REGRESSED the ceiling — overfit the narrower training distribution at the cost of generalization; diamond/shield/converging regressed hard even though encirclement/transitioning improved |

## Strategy 5: target the transitioning false-positive rate — done, ceiling doubled

**Starting point:** strategy 4 left window-level classification good (72.7% overall) but with
every steady formation getting misread as `"transitioning"` 20-28% of the time.

**Diagnosed before touching anything** (40 chain-length-1 + 40 chain-length-2 fresh
trajectories, seed=4242): false-positive rate is **1.8%** on fully unambiguous windows (zero
blend anywhere in the whole trajectory) but **53.2%** within 15 timesteps of a real blend
boundary, **0.0%** far from one. This ruled out a broad model-calibration bug — the false
positives trace specifically to strategy 2's (gap-2) regime boundaries still admitting a wide
grey zone: `"transitioning"`-labeled examples could carry as little as 28% genuine blend
content (up to 64% residual pure formation), and "pure"-labeled examples could carry real
blend content near their edges.

**Fix:** tightened `generate_dataset()`'s three regime bounds (`src/swarm_intent/data.py`,
commit `5baaceb`) so regime 1 ("transitioning") requires the blend region to dominate the
window (45-62% of `n_timesteps`, verified numerically before regenerating), and regimes 0/2
("pure") push blend to the very window edge with a short duration (74-90% pure content).
Regenerated (30k sequences) and retrained (150 epochs, early-stopped at 22, best epoch 10,
`test_acc=0.8631` — notably lower and the validation curve visibly more volatile than strategy
4's run, an open, unexplored question rather than a resolved one).

**Result:** pair-level accuracy (the ceiling) **roughly doubled: 6.1%→12.2%** (`robust=True`:
6.5%→12.8%) — the largest single relative jump from any fix in this program so far.
Transitioning false-positive rate dropped meaningfully for some classes (`diamond`
21.3%→4.0%, `shield` 27.1%→8.5%) and modestly for others. Not uniform: `encirclement`'s raw
window accuracy regressed sharply (62.2%→41.3%), the same capacity-trading pattern seen in
strategy 3. **Still far below the 70% floor — remains HALT GATE 1 — but this is real,
diagnosis-driven progress, not a plateau.** Full detail: `docs/CEILING.md`'s 2026-08-08
"targeted fix" section, `docs/V5_LOG.md`'s matching entry.

## Pre-strategy-6 measurement: is exact-pair accuracy even the right ceiling?

Before choosing strategy 6, re-scored strategy 5's existing 509 pair-eligible trajectories
(no retraining) against `RULES` for `threat_level`/`likely_intent`/`recommended_action`
instead of exact-pair match, since `RULES` maps 49 pairs onto only 4 threat levels — a wrong
pair can still land on the right threat. Result: **13.0%/13.6%, barely above exact-pair
(12.2%/12.8%)**. Full 4x4 confusion matrix and reasoning: `docs/CEILING.md`'s 2026-08-08
"the REAL ceiling" update.

**The 70% floor is not met on the threat ceiling either — same HALT GATE 1 verdict.** But the
finding is still load-bearing: conditional on the classifier actually reaching a resolvable
`(a,b)` pair (bucket A), threat accuracy is ~85% — good. The real bottleneck is that 84.7% of
trajectories **never reach a resolvable pair at all** (bucket B guard-blocked or bucket C
unresolvable). `RULES`-mapping tolerance was never the constraint; bucket-A resolution rate
(15.3%) is. This reframes what strategy 6 and step 2's decomposition should target.

## Pre-strategy-6 measurement 2: decomposing pair-recovery failures

Re-ran inference (no retraining) reproducing all 509 pair-eligible trajectories from
strategy 5's exact measurement, and traced WHY each of the 447 failures happened. Two
findings sharper than anything in the program so far:

1. **Chain-length-2 (a real formation transition) has 0.0% accuracy (0/251) — never once
   succeeded.** Every one of the 62 successes is a steady-state (chain-length-1) trajectory.
2. **The reduction/guard logic, not classifier accuracy, is the dominant bottleneck.** 47.2%
   of failures had zero misclassified windows and still failed. Filtering to only
   correctly-classified windows and re-reducing recovered the pair in just 0.9% of failures.
   Tracing the actual guard: `stgt_bridge.py`'s **`dispersed_converging_ambiguity` guard
   accounts for 60.9% of all failures**, unconditional, firing across every formation class
   (not just dispersed/converging) — a known, previously-documented (AUDIT.md sec AG) but
   never-before-quantified-against-this-checkpoint defect.

Full detail: `docs/CEILING.md`'s 2026-08-08 "step 2" update. Flagged, not acted on: since the
dominant blocker is bridge/reduction-code logic (fixed 0.15 threshold, fires regardless of
correctness), strategy 6's retrain is unlikely to move the pair ceiling much by itself.
Revisiting that guard is out of scope for this instruction; proceeding to strategy 6 as
given and reporting the result honestly against this expectation.

## Strategy 6: steadier training schedule — under-convergence fixed, ceiling re-measurement pending

Checked the strategy-5 dataset's class distribution first (balanced across all 8 classes,
ruled out "encirclement has fewer examples"). Replaced `OneCycleLR` (decays to ~0) with
linear warmup + cosine decay to a nonzero floor, retrained on the SAME data with peak lr
lowered (3e-4→1e-4) and patience raised (12→35).

**Result: best epoch 10→51, early-stop 22→86/150, test_acc 0.8631→0.9958.** Every class is
now ≥98.6% precision/recall, including `encirclement` (recall 0.986, fully recovered from
strategy 5's regression). Confirms the under-convergence hypothesis was correct.

**Step 4 re-measurement: the gain did NOT transfer — it made the ceiling WORSE.**

| metric | strategy 5 | strategy 6 |
|---|---|---|
| pair-level accuracy (robust=False) | 12.2% (62/509) | **4.9% (25/509)** |
| threat ceiling (robust=False) | 13.0% | **6.3%** |
| conditional accuracy within bucket A | 79.5% (62/78) | **32.5% (25/77)** |

Strategy 6 converged much more tightly to `generate_dataset()`'s training distribution
(near-100% in-distribution accuracy) than strategy 5 did — but that distribution's spread/
noise ranges are narrower than the ceiling test's realistic long-trajectory sampling. The
sharper fit generalized BETTER for `encirclement`/`transitioning` and WORSE for `diamond`
(94.0%→70.0%), `shield` (90.3%→66.5%), and `converging` (66.7%→37.4%) — a genuine overfitting/
generalization tradeoff from fixing the under-convergence, not a new bug. Even when the
pipeline DOES reach a resolvable pair (bucket A), it is now right less than half as often as
before (79.5%→32.5%).

**Strategy 6 met its own literal objective (fix the training curve) while making the actual
ceiling significantly worse.** `swarm_data/best_model.pt` is currently the WORSE strategy-6
checkpoint; `swarm_data/best_model_strategy5_backup.pt` holds the better strategy-5 checkpoint
if reverting is wanted. Also found (not fixed, flagged): `evaluate_ml_model` double-normalizes
already-normalized test data, a pre-existing bug. Full detail: `docs/CEILING.md`'s 2026-08-08
"strategy 6" and "step 4" updates.

**HALT GATE 1 remains unambiguously triggered — further below the 70% floor than strategy 5
left it.** Stopping here per instruction. Not proceeding to any further strategy without
explicit direction.

## 2026-08-09: revert to strategy-5 checkpoint

Per instruction, restored `best_model_strategy5_backup.pt` as `swarm_data/best_model.pt`
(verified by SHA-256: pre-revert `best_model.pt` = `9090093c...` matched the strategy-6
backup; post-revert `best_model.pt` = `18fc201d...` matches `best_model_strategy5_backup.pt`
exactly; loaded checkpoint confirms `epoch=10, val_loss=0.610` — strategy 5's own best epoch).
The strategy-6 checkpoint is preserved at `swarm_data/best_model_strategy6_backup.pt` in case
it's ever wanted again, but is no longer `best_model.pt`.

**Standing rule going forward: STGT checkpoint selection must be judged on pair-level/
threat-level ceiling (`phase0_ceiling.py`/`phase0_threat_ceiling.py`), never on in-distribution
test accuracy alone.** Strategy 6 hit 99.6% test accuracy — the best of the whole program — and
roughly halved the ceiling (12.2%→4.9%) doing it. Test accuracy measures fit to
`generate_dataset()`'s own distribution; the ceiling measures what the system can actually do
on realistic long trajectories. They diverged sharply here and will again.

## 2026-08-09: step 1 — the ambiguity guard bug confirmed directly, not just inferred

Read `stgt_bridge.py`'s `_is_ambiguous_dispersed_converging` (lines 114-120): it checks only
`abs(dispersed_p - converging_p) < 0.15` on raw probabilities, with no check on whether either
class is actually competitive. Audited it directly against real predictions
(`scripts/phase0_guard_audit.py`, strategy-5 checkpoint, no training): **fires on 75.8% of all
windows**, and of those firings, **66.1% have neither dispersed nor converging in the window's
top-2 predicted classes** — e.g. a window predicted `shield` at 98.97% confidence still trips
it, because `dispersed=0.0012` and `converging=0.0005` are "close" in absolute terms despite
both being irrelevant noise. Only 1.7% of firings are genuine both-in-top-2 contention. This
is the exact, now-confirmed mechanism behind step 2's finding — the guard was never testing
what its name claims.

## 2026-08-09: step 2 — fixed the guard, the single biggest gain of the whole program

`_is_ambiguous_dispersed_converging` now also requires dispersed/converging to be the
window's top-2 predicted classes, not just close in raw probability. One condition added,
no retraining, still the strategy-5 checkpoint.

Isolated re-audit: guard fire rate 75.8%→**1.3%** (1113→19 of 1469 windows), and the 19
remaining firings are now **100% genuine** (0% spurious, down from 66.1%).

Full ceiling re-measurement, same seed=999/checkpoint/protocol as every prior measurement:

| metric | before | after |
|---|---|---|
| pair-level accuracy (robust=False) | 12.2% | **47.0%** |
| pair-level accuracy (robust=True) | 12.8% | **48.5%** |
| threat ceiling (robust=False) | 13.0% | **52.3%** |
| threat ceiling (robust=True) | 13.6% | **58.7%** |
| `robust=True` precision within bucket A | ~20% (sec AG) | **62.4%** |

Biggest jump of the program, from a one-line bug fix. Also retroactively explains sec AG's
"robust reduction should not ship" verdict: that verdict was measured against the BROKEN
guard corrupting the same class_probabilities signal robust reduction's majority vote reads
— it doesn't hold against this fix and shouldn't be treated as permanent.

**HALT GATE 1 still not cleared** (52-59% vs a 70% floor) but the gap shrank from ~57-58
points to ~11-18 points. Full detail: `docs/CEILING.md`'s 2026-08-09 step 2 update.

## 2026-08-09: step 3 — chain-length-2 confirmed still broken, but NOT a second guard bug

Re-measured pair/threat ceiling by chain length with the fixed guard: chain-1 (steady state)
86.8%, **chain-2 (single transition) still 6.0%**. Confirmed real, not fixed by the guard fix.

Traced 20 failing chain-2 trajectories stage by stage. 60% (all_windows_transitioning 35% +
trailing_transitioning_run 20% + most of structural_reduction_wrong_pair 25%) is the SAME
windowing-artefact mechanism from the earlier engagement — chain-2 trajectories are a single
50-100-timestep hop, often only 1-2 sliding windows, too few for the destination formation to
reliably resolve or even be observed. Only 15% is genuine bridge-logic brittleness
(`oov_name` blocking an already-correct structural reduction — fixable, see step 4). 5% is
plain classifier misclassification.

**Not a second bug of the dispersed_converging class — no bridge-logic fix (short of
retraining with longer segments or a larger `max_seq_len`) touches the dominant 60%.** Full
detail: `docs/CEILING.md`'s 2026-08-09 step 3 update.

## 2026-08-09: step 4 — audited every other guard/rule the same way; found two more of the same class

`scripts/phase0_full_guard_audit.py --n 1000`. Same methodology as the dispersed_converging
audit: fire rate, and (among trajectories where a guard is the SOLE blocker) how often it
blocks an already-correct structural answer.

| guard | fire rate | sole-firing spurious rate | failures attributable |
|---|---|---|---|
| **oov_name** | 8.3% | **69.0% (20/29)** | **20** |
| **dominant_history_contradiction** | 3.5% | **100.0% (4/4)** | **4** |
| low_confidence | 2.2% | 25.0% (1/4) | 1 |
| dispersed_converging_ambiguity (post-fix) | 2.2% | 50.0% (2/4) | 2 |

**Two more guards of the same defective class, found the same way.**
`dominant_history_contradiction` (ties in raw predicted window counts) blocks a correct answer
100% of the time it's the sole blocker (n=4, small but unambiguous). `oov_name` is the
highest-volume actionable defect: fires 8.3% of the time, 69% spurious when sole, 20 correct
answers blocked outright — and window-level checking shows 57.4% of the windows triggering it
are themselves classifier misclassifications, not genuine ambiguity, which the guard then
overreacts to with a zero-tolerance blanket block.

Also audited `robust=True`'s leading/trailing trim step: **62.5% of trimmed windows (105/168)
discard genuine signal, not noise** — this is WHY `robust=True` precision plateaus at 62.4%
even after the guard fix; the trim step has the same defect pattern baked into its own logic.
`key_windows` capping, by contrast: no bug found (0/373 capped selections miss a true
endpoint).

Audit only this turn, no code changes. Full detail: `docs/CEILING.md`'s 2026-08-09 step 4
update.

## 2026-08-09: step 5 — re-swept the robust=True threshold post-guard-fix

`DEFAULT_ROBUST_THRESHOLD=0.7` was tuned before this session's guard fix and guard audit, both
of which changed the underlying signal. Re-swept on a dev split (seed=1) only, confirmed on
held-out (seed=999): dev/held-out precision gap is 0.1pt — not overfit.

**The precision curve is flat across the WHOLE sweep (60.4-63.1%)** — consistent with step 4's
finding that the trim step (not the vote threshold) is the dominant contamination source, and
happens before the vote even runs.

**Recommended: 0.6, not the current 0.7 — it Pareto-dominates** (coverage 81.1% vs 77.9%,
precision 63.1% vs 62.6%, both better). Going lower (0.45) buys more coverage (84.8%) at a
real precision cost (60.4%). None of the tested thresholds get wrong-key contamination
meaningfully below ~37% without fixing the trim step itself. Recommendation only —
`DEFAULT_ROBUST_THRESHOLD` left at 0.7 in code. Full detail: `docs/CEILING.md`'s 2026-08-09
step 5 update.

## 2026-08-09: step 6 — HALT GATE 1 re-examined: end-to-end threat accuracy projected at ~62%

The 70% floor was set when pair-level and threat-level ceilings were nearly identical (12.2%
vs 13.0%); they've since diverged to 47.0% vs 52.3% (RULES maps 49 pairs onto only 4 threat
levels, so wrong pairs often still land on the right threat). Computed the projection the
user asked for: `end_to_end = current_threat_ceiling + P(bucket C) * Layer_3_accuracy`.

| | robust=False | robust=True @ 0.7 |
|---|---|---|
| threat accuracy within bucket A | **90.5%** | 75.5% |
| current threat ceiling | 52.3% | 58.7% |
| **end-to-end projection (central)** | **61.6%** | **62.3%** |

Layer 3's contribution is a disclosed estimate (v3b-fix's 30.9% from the earlier engagement's
real-STGT-output eval, pre-guard-fix checkpoint, not bucket-conditioned) with a
conservative(20%)/optimistic(40%) sensitivity band: 58.3-64.4% (robust=False), 61.1-63.4%
(robust=True, narrower since it depends less on Layer 3).

**Verdict: even the most optimistic projection tested (64.4%) does not clear 70%.** But the
actual instruction was to settle whether 70% stated in PAIR-LEVEL terms is still the right
gate — it isn't; it measures the wrong quantity now that the two ceilings have diverged.
**Recommendation: restate HALT GATE 1 in end-to-end threat-accuracy terms.** The numeric
floor itself is a policy call this projection informs but doesn't set — the central estimate
(~62%) is ~8 points short either way. Full detail: `docs/CEILING.md`'s 2026-08-09 step 6
update.

## Diagnostic checkpoint, 2026-08-09: is chain-2's 18.7% a generator ceiling or an STGT problem?

Not a new strategy attempt — a scoped measurement answering `UPSTREAM_ISSUES.md` issue #3
directly, no generator/model/LLM code changed. Defined observability (a window's true-label
majority is B — reused from existing scoring code, not invented) and stratified all 251
chain-2 trajectories in the standard population by it:

| observability | n | % | pair accuracy |
|---|---|---|---|
| OBS_CLEAR (B majority in >=2 windows) | 28 | 11.2% | 57.1% |
| OBS_PARTIAL (1 window) | 96 | 38.2% | 19.8% |
| OBS_NONE (never) | 127 | 50.6% | 9.4% |

**Not single-cause.** ~50.6% of chain-2 trajectories never make the destination formation
observable at all (confirms issue #3, generator ceiling). But even the best-observed group
still only reaches 57.1% — a manually-reviewed 20-case trace attributes the residual to real
STGT misclassification (55% of traced failures: near-blend-boundary `"transitioning"`
over-prediction and confident `dispersed`/`converging` source confusion the existing ambiguity
guard can't catch), not bridge/reduction logic (0%). A minimal, no-retrain, eval-harness-only
generator fix was specified (decouple destination dwell time from segment length) but not
implemented this turn, per instruction — diagnosis and recommendation only. Full detail:
`docs/V5_LOG.md`'s 2026-08-09 step-24 entry, `docs/CEILING.md`'s matching update.

## Experiment, 2026-08-09: the dwell-time generator fix, implemented — chain-2 more than doubles

Implemented the checkpoint's dwell fix specified above, in `scripts/phase0_decompose_failures.py`'s
`build_long_sequence_labeled` only (dwell now sampled directly, `~Uniform{40,60}`, guaranteeing
the derived `D>=35` minimum; `seg_len` now derived from `lead_in+blend_duration+dwell` instead
of sampled first). Same frozen checkpoint, zero retraining.

| | before | after |
|---|---|---|
| chain-2 OBS_NONE | 50.6% | **0.0%** |
| chain-2 pair accuracy | 18.7% | **39.9%** |
| chain-2 threat accuracy | 31.9% | **72.1%** |

**Two disclosed caveats, not hidden:** (1) a new, self-inflicted ~15%-of-failures SOURCE-
observability gap — `LEAD_IN_RANGE` wasn't derived with the same rigor as the dwell fix, a
trivial next fix; (2) train/eval blend-timing overlap is STILL 0.0% after the fix, flagged as
the likely driver of the now-dominant (80%, up from 55%) genuine STGT-misclassification
failure mode. **Decision: A — observability fix successful, remaining problem primarily STGT
recognition**, with those two caveats as the concrete next steps. Full detail:
`docs/V5_LOG.md`'s 2026-08-09 step-25 entry, `docs/CEILING.md`'s matching update.

## Experiment, 2026-08-09: consolidation + source symmetrization — chain-2 more than triples total

Resolved both of step 25's caveats. Consolidated the 5 duplicate copies of the eval-trajectory
sampling logic into `src/swarm_intent/eval_trajectories.py` (4 live scripts now import it; one
frozen historical-reproduction script deliberately excluded, documented) — confirmed none of
the 5 ever fed real STGT training data. Symmetrized `LEAD_IN_RANGE` from `(15,35)` to `(30,50)`,
derived the same way as the destination fix's `MIN_DWELL_RANGE`.

| | dest-only fix | both fixes | original baseline |
|---|---|---|---|
| source OBS_NONE | 16.3% | **0.0%** | — |
| chain-2 pair accuracy | 39.9% | **65.8%** | 18.7% (3.5x total) |
| chain-2 threat accuracy | 72.1% | **76.3%** | 31.9% |
| chain-3+ false-positive rate | 12.6% | **1.8%** | 9.0% (net improvement, p=4.3e-07) |

A refined 20-case failure taxonomy found the remaining chain-2 failures 100%
boundary/blend-timing-concentrated (0% clean, non-boundary misses) — the still-unresolved
0.0% train/eval blend-overlap (unchanged across three independent formula revisions) is the
likely cause. **Decision: B — the blend-timing distribution mismatch is the dominant
remaining issue and should be fixed BEFORE any STGT training/capacity change**, not decision
A as the previous entry projected. Not started this session, per instruction. Full detail:
`docs/V5_LOG.md`'s 2026-08-09 step-26 entry, `docs/CEILING.md`'s matching update.

## Decision, 2026-08-10: does STGT need to be retrained on a corrected distribution?

**This is deliberately NOT "Strategy 7."** Strategies 1-6 above all tuned architecture,
training schedule, or capacity against a training distribution that was held fixed (aside
from strategy 2/5's own regime-margin adjustments). This decision is about whether that
training distribution — `generate_dataset()`'s 3-regime blend timing, last touched by
strategy 5 and untouched since — is itself the bottleneck. That is a categorically different
question: fixing the distribution invalidates strategy-to-strategy comparisons made above
unless read correctly. Strategies 1-6 answer "given this (possibly flawed) distribution, what
architecture/schedule/capacity choice does best on it." This decision asks "is the
distribution the thing that was flawed all along." No retraining happened this session —
this entry establishes facts only, per explicit instruction.

**Fact 1 — the two generators have been diverging silently.** `eval_trajectories.py` (steps
24-26's dwell-time and source-symmetrization fixes) builds long EVALUATION trajectories only.
`generate_dataset()`, the actual training-data path, has not changed since strategy 5. Every
Phase 0 ceiling number reported above (strategies 1 through 6, and steps 24-26) measures the
frozen strategy-5 checkpoint against an increasingly-accurate but entirely separate evaluation
harness — never against a training set built the corrected way. Full parameter-by-parameter
diff: `docs/V5_LOG.md` step 27.

**Fact 2 — the two distributions do not overlap.** Monte Carlo comparison (n=20000/side,
`docs/V5_LOG.md` step 28): eval's realized `(start_frac, duration_frac)` region has **0.0%
overlap** with any of `generate_dataset()`'s 3 current regime boxes. If `generate_dataset()`
adopted eval's current timing distribution, effectively 100% of transitioning examples would
land outside every regime STGT has ever been trained on. This is not a partial-tuning gap —
it is a full distributional replacement, and a structural one: eval's segment is 1.6-2.6x
longer than train's fixed 50-step window, so a genuine port means restructuring how
`generate_dataset()` builds a transitioning example, not substituting three constants.

**Fact 3 — the step-26 gains (18.7%→65.8%) answer a different question than "does the model
generalize to the corrected distribution."** Steps 24-26 fixed *observability* — ensuring a
chain-2 trajectory's source and destination formations each dominate at least one sliding
window, via wide lead-in/dwell margins (`lead_in>=30`, `dwell>=40`). That construction
deliberately produces windows heavily weighted toward PURE, low-blend-content material — it
does not test whether STGT correctly classifies a genuinely blend-dominant window. The
refined failure taxonomy from step 26 itself already shows this directly: 100% of the 20
freshly-traced chain-2 failures under both fixes are boundary/blend-timing-concentrated, and
the train/eval blend-overlap remained exactly 0.0% even after the observability fix landed.
**These are not the same claim.** "65.8% pair accuracy" means "the model does well once the
eval harness stopped handing it windows dominated by blend content it was never trained on" —
it is evidence the observability bug was real and worth fixing, not evidence the model
generalizes to the actual corrected (realistic) blend-timing shape. Observability and
distribution-mismatch are orthogonal problems; fixing one doesn't touch the other.

**Decision gate:**
- **A** (port the fix, retrain) is justified if fact 2 shows substantial divergence AND fact 3
  shows the current checkpoint's gains are eval-artifact-driven rather than genuine
  generalization to the corrected distribution.
- **B** (leave `generate_dataset()` as-is) is justified if the frozen checkpoint already
  generalizes well to corrected eval trajectories — meaning the training distribution,
  though different, wasn't actually the bottleneck.
- **C** (insufficient evidence) if neither is cleanly established.

**Both of A's conditions hold: fact 2 shows 0.0% overlap (full divergence, not partial), and
fact 3 shows the step-26 gains are explained by the observability fix alone, with the
blend-boundary failure mode both predating and surviving it unchanged. Verdict: A — port the
corrected timing distribution into `generate_dataset()` and retrain STGT from scratch,
recorded as a decision, not executed this session** (no training this session, per explicit
instruction). Doing so means every strategy-1-through-6 comparison above must be read as
"tuning against a flawed distribution," not as a closed chapter — a genuinely new baseline is
needed once the distribution itself is fixed, and strategy-style architecture/capacity tuning
should resume only after that baseline exists. Full detail: `docs/V5_LOG.md` steps 27-30,
`docs/CEILING.md`'s matching update.

## Decision, 2026-08-10 (part 2): designing the port as three separable changes

**Also deliberately NOT "Strategy 7," and a distinct decision from the one directly above —
that decision established WHETHER to port; this one designs HOW.** The parameter diff (step
27) surfaced two structural differences beyond blend timing: per-example vs per-timestep
labeling granularity, and fixed-50 vs derived-80-132 example length. Porting blend timing
alone, without resolving these, risks a dataset wrong in a new way. Three independently
toggleable flags were designed and implemented on `generate_dataset()`
(`corrected_blend_timing`, `windowed_examples`, `content_majority_labeling`, all default
`False`, verified bit-identical to the pre-change generator with all off) — no training this
session either.

**Labeling rule** (full derivation: `V5_LOG.md` step 31): a window is a confident pure
formation at `>=70%` of its own content (reuses eval's own `MIN_DWELL_RANGE`/`LEAD_IN_RANGE`
derivation, so "eval trusts this as observed" and "training labels this pure" agree by
construction); `"transitioning"` requires blend content to be the plurality of the window
AND at least `20%` of it (mirroring `0.70` exactly was checked and rejected — `BLEND_
DURATION_RANGE`'s own max is 50% of a window, so `0.70` would make the class unreachable);
windows meeting neither bar are **excluded, not mislabeled**.

**Windowing/architecture constraint** (`V5_LOG.md` step 32): checked `model.py`/`config.py`
directly — `PositionalEncoding` slices its buffer (`T<=50` works, `T>50` crashes), and
`max_seq_len=50` is baked into every existing checkpoint plus the entire eval harness'
`window_size=50` convention. Increasing it is a full architecture change, out of scope.
Decision: keep training examples at exactly 50 timesteps; slide `WINDOW_SIZE=50, STRIDE=10`
across a long corrected-timing hop instead — the identical grid `sliding_window_inference`
already uses at eval time.

**Label-sanity results, four diagnostic datasets, n_transition=300 each** (full table:
`V5_LOG.md` step 34):

| run | kept | excluded | label disagrees with own content |
|---|---|---|---|
| baseline | 300 | 0 | 1/300 (0.3%) |
| blend-timing only | 300 | 0 | 0/300 (0.0%) |
| windowing only | 1800 | 0 | **73/1800 (4.1%)** |
| labeling only | 297 | 3 (1.0%) | 0/297 (0.0%) |
| **all three combined** | 886 | **928 (51.2%)** | **0/886 (0.0%)** |

Two findings disclosed, not smoothed over: **windowing alone reintroduces mislabeling
(4.1%)** — direct evidence it cannot ship without content-majority labeling paired with it —
and **the combined port excludes 51.2% of candidate windows**, a genuine yield cost (not a
correctness problem) that means full-scale generation needs roughly 2x the target
`n_transition` to compensate.

**Readiness verdict: labels pass sanity checks for the proposed port (all three flags
combined) — 0% disagreement, both thresholds empirically confirmed to hold exactly as
derived. Ready to proceed to full-scale dataset generation and STGT retraining, planning for
the 51.2% yield loss (≈2x oversampling) and the mandatory windowing+labeling pairing.
NOT STARTED this session** — no full-scale generation, no retraining, per explicit
instruction. Full detail: `docs/V5_LOG.md` steps 31-34.

## Decision, 2026-08-10 (part 3): pre-scaling checks — NO-GO, a real problem found before spending the full-scale budget

**Also its own decision, not "Strategy 7," and distinct from parts 1 and 2 above** — those
established WHETHER to port and HOW to design it; this one is the pre-registered gate that
was supposed to say go/no-go on actually doing it. Two checks were run first (exclusion bias,
seed stability), then the decisive test itself (small-scale train comparison), before any
full-scale commitment.

**Step 1 — exclusion bias**: no severe systematic bias found. Content-space concentration
sits exactly in the expected near-miss ambiguity zone; formation-level exclusion is
near-uniform (std 1.8 points); the one pair-level outlier (`encirclement`->`shield`, 66.7%
vs. 51.1% mean) is disclosed for monitoring, not severe enough to block on its own. **This
gate condition passed.** Full detail: `docs/V5_LOG.md` step 35.

**Step 2 — seed stability**: keep rate is very stable across 5 seeds (std 0.8 points, range
48.6%-50.7%). The ~2x compensation estimate holds; revised slightly to 2.06x using the
worst-case seed. Full detail: `docs/V5_LOG.md` step 36.

**Step 3 — the decisive test**: matched-size small-scale datasets, two STGT models from
identical initial weights, differing only in training data format. **The corrected-format
model collapsed to near-random accuracy (13.2% test_acc, ~chance for 8 classes) and scored
0% on both chain-2 pair and threat accuracy — against baseline's 92.2% test_acc, 60.5% pair
accuracy, 64.9% threat accuracy on the identical eval population.** A real bug in the test's
own oversampling arithmetic was found and fixed mid-step (disclosed above and in
`V5_LOG.md` step 37) — the collapse is not that bug; it was fixed before this result, and
dataset sizes were confirmed matched (3033 vs 3000 total examples) when the collapse was
measured. **This gate condition failed, decisively.** Full detail: `docs/V5_LOG.md` step 37.

**Root cause, quantified, not just asserted**: `generate_transition_sequence`'s acceleration
term is unbounded and was exercised almost exclusively at `n_timesteps<=50` throughout this
project's history. The port's hops run 80-132 timesteps — measured per-timestep centroid
displacement growth of 16x across a 132-step hop vs. 4.5x across 50 steps. Every
`eval_trajectories.py`-based ceiling measurement this whole program has relied on already
calls the same function at the same long lengths, so this mechanism isn't new — but this port
is the first thing to TRAIN directly on windows spanning that whole growing-velocity range,
where regression targets for a window near hop-start vs. hop-end differ by an order of
magnitude under one shared normalization. Offered as the leading hypothesis; not confirmed by
an isolation experiment this session.

**Decision gate, as stated in advance: proceed to full-scale generation + retrain only if (a)
no severe exclusion bias AND (b) the small-scale corrected model beats baseline. (a) passed;
(b) failed decisively. Verdict: NO-GO.** Full-scale generation and retraining are **not**
authorized on the current port design. Compute budget for the (now moot) full-scale attempt
was ~1.5-2 hours — not spent, precisely because this small-scale gate existed to catch this
first (`docs/V5_LOG.md` step 38).

**What needs fixing before any further attempt**: the acceleration/velocity-growth mechanism
identified above, most likely by capping or decaying acceleration's effect so it doesn't
compound unboundedly over hop lengths several times longer than the regime it was designed
for — then re-running step 37's exact small-scale comparison (not skipping straight back to
full scale) before reconsidering this gate.

## Decision, 2026-08-10 (part 4): mechanism isolated — normalization, not acceleration; gate remains NO-GO, narrowly

**Also its own decision, distinct from parts 1-3 and from Strategy 7.** Part 3 named
acceleration/velocity-growth as the leading hypothesis and recommended capping it. This part
tested that directly, found it insufficient, isolated the real mechanism, and re-measured.

**Acceleration capping tested and rejected as the explanation.** `ACCEL_SPEED_CAP=33.0` was
implemented (derived from `generate_transition_sequence`'s own 50-step design envelope,
provably inert for every existing ≤50-step call site), measurably reduced long-hop growth
(16x→11x), and **produced an identical collapse** (13.2%/0.0%/0.0%, matching the uncapped
result to 4 decimal places). That exact match was itself the signal that acceleration wasn't
the operative mechanism.

**Second contributor confirmed sufficient on its own.** `split_and_normalize`'s global
`train_mean`/`train_std` scalar is 1.71x larger for the corrected dataset (143.2 vs 83.7),
inflated by a minority of high-drift windows drawn from deep inside long hops — a robust
(percentile-trimmed) version of the same scalar, with acceleration left **completely
uncapped**, resolved the collapse on its own: test_acc 13.2%→**87.7%**, chain-2 pair/threat
accuracy 0.0%/0.0%→**51.8%/60.5%**. Steps 2 and 3 of the isolation protocol (split
regression-target vs. graph-edge-threshold distortion; re-run with the identified fix) were
addressed without separate runs — moot once the collapse was directly resolved, and already
satisfied by the same run, respectively. `ACCEL_SPEED_CAP` has been **reverted**
(`src/swarm_intent/data.py`, verified bit-identical to the pre-cap code) — the record
corrected so it doesn't overstate what that change accomplished. Full detail:
`docs/V5_LOG.md` steps 39-41.

**Decision gate, reported literally, not rounded up**: "proceed only if the fixed version
beats baseline" — in this single run it does not (51.8%<57.0% pair, 60.5%<63.2% threat),
though the gap is now small (previously 60+ points, now single digits) and baseline's own
figures swung by up to 10 points on test_acc and ~5 on threat_acc across nominally-identical
reruns of this exact config across steps 37/39/41 — real training-run noise at this scale,
not a data difference. **A single run does not carry enough statistical power for a confident
go/no-go call in either direction. Verdict: NO-GO, narrowly** — reported as the literal
result the data supports, not stretched into a GO on the strength of "close." Full-scale
generation and retraining remain **not authorized**.

**What would resolve this cleanly**: repeat this exact small-scale comparison across several
seeds (data seed and/or model-init seed) to separate genuine signal from the observed
run-to-run noise, before either authorizing full-scale generation or concluding the port
still underperforms. Not done this session — no full-scale generation, per instruction.
