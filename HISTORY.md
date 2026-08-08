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
