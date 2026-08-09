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
