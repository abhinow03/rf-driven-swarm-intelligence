# Scoped diagnostic pass on the two gaps `docs/CEILING.md` flagged, not fixed

Per instruction: diagnose only, nothing patched. All checks below are read-only against the
retrained checkpoint and the current (fixed-physics) generator; no model, dataset, or code
changed as part of this pass.

## Gap 1: 93.5% (`train.py`'s own eval) → 69.5% (matched-regime, `sliding_window_inference` path) → 0% for `v_shape`/`encirclement` specifically

**Ruled out, cleanly, with evidence:**
- **Duplicate model/graph implementation** (`swarm_intent/model.py`+`graph.py` vs.
  `swarm_intent/stgt/model.py`, a parallel, undocumented copy — see note below). Re-ran the
  identical matched-regime check through the "official" `swarm_intent.model.STGTModel` +
  `swarm_intent.graph.sequence_to_graphs` path: **identical result** (73/105, 69.5%, same
  per-class breakdown, `v_shape`/`encirclement` still 0/15 each). Not the cause.
- **Normalization round-trip inconsistency.** Took 5 actual `v_shape` examples straight out of
  `X_test.npy` (already-normalized, on disk), scored them two ways — (a) exactly as `train.py`
  does (normalized array straight into `sequence_to_graphs`), (b) un-normalized back to raw
  and re-normalized inside `sliding_window_inference`. **Both paths agree on all 5, both
  correctly predict `v_shape`.** The pipeline is internally consistent; not a normalization bug.
- **Small-sample luck.** Re-ran at n=100 per class, at two independent seeds (111, 222):
  **0.0% for `v_shape` at both**, uniformly, including instances with modest position spread
  (`pos_std` as low as 32, comparable to typical `X_test.npy` examples). Not sampling noise.

**What's actually happening — confirmed:** the model doesn't produce garbage/uncertain output
on fresh `v_shape` instances — it confidently (97.1% softmax probability) predicts
**"diamond."** 30/30 fresh `v_shape` examples (seed=111) were predicted as `diamond`, 100% of
the time. This is a confident, systematic, wrong answer, not noise or OOD collapse.

**Leading, evidence-consistent explanation (not proven down to a single line):** `v_shape` and
`encirclement` are two of the five formations with a completely deterministic offset template
(no `rng` draws at all inside `get_formation_offsets` beyond the `spread` scale factor) — the
*only* thing that varies example-to-example is centroid drift (angle/speed/**now
acceleration**) and additive noise. The model's node features are **absolute** (normalized)
`(x, y, z)` positions, not offsets relative to the swarm centroid (`build_graph`,
`graph.py`/`stgt/model.py`, both identical on this point) — so the network has to implicitly
learn to discount whatever the centroid is doing. With only 67 epochs / ~11 minutes of
training on a 9000-example dataset, and acceleration now making per-example position drift
far more variable than before this session's physics fix (raw single-example ranges observed
up to ±400+ units, vs. a training-set-global scalar std of 68), the two *least* geometrically
distinctive, purely-template-driven formations look like the most likely candidates to have
been memorized via incidental positional correlations in this specific 67-epoch run rather
than learned as a general, centroid-invariant shape — `X_test.npy`'s own held-out `v_shape`
rows (drawn from the exact same generation run the model trained on) score 100%, while
independently regenerated `v_shape` instances (never part of that specific run) collapse to
0%. That pattern — perfect on in-run held-out data, collapsed on genuinely novel data — is the
signature of a generalization failure tied to this specific training run, not a code bug in
the inference path (which has now been checked three independent ways and found consistent).
**Not confirmed to a single root cause; the acceleration fix is the most likely aggravating
factor given the magnitude of the position-variance increase it introduces, but this is
inference from correlated evidence, not a proven causal chain.**

**Also found, unrelated to the physics fix, worth flagging on its own:** `src/swarm_intent/stgt/`
is an entire second, independently-maintained copy of the model/graph/config code
(`stgt/model.py`, `stgt/config.py`) that CLAUDE.md's documented architecture does not mention.
It reintroduces exactly the hidden-global-state pattern the project's own migration was meant
to eliminate — `stgt/model.py`'s `forward()` reads a module-level `device` global from
`stgt/config.py` instead of inferring device from `next(self.parameters()).device` like
`swarm_intent/model.py` does. In this single-GPU environment it happens not to matter (both
resolve to the same `cuda` device), but it is a real, disclosable duplication-and-drift risk:
every eval/inference script in this project (`sliding_window_inference`, `predict_v2`, and
therefore every coverage/eval script since sec AE) imports from `stgt/`, while `train.py`
imports from the "main" package — two copies of the same logic, one used for training, a
different one used for every downstream measurement, kept in sync only by someone remembering
to edit both.

## Gap 2: 69.5% (matched-regime) → 22.3%/3.3% (realistic long trajectories)

**Ruled out:** simple position-drift-into-OOD-normalization as the sole mechanism (already
shown in `docs/CEILING.md` — early vs. late window accuracy in a long trajectory is flat,
36.3% vs. 32.7%).

**Confirmed mechanism, directly tested:** `generate_dataset()` (STGT's actual training-data
builder) labels an ENTIRE `generate_transition_sequence(a, b, n_timesteps=50, blend_start=20,
blend_end=30)` output as the single class `"transitioning"` — full stop, regardless of the
fact that timesteps 0-20 are genuinely pure `a` and timesteps 30-49 are genuinely pure `b`.
**The model has never once been trained on a window where only PART of it is transitioning
and the rest should be labeled as the steady endpoint formation** — every "pure A"/"pure B"
training label comes exclusively from `generate_swarm_sequence` (a full 50-timestep window of
one formation, no blend at all, ever). `build_long_sequence`'s real-world regime (used by
`measure_coverage.py` and `phase0_ceiling.py` alike) produces exactly this "partially blended"
situation constantly — a 50-100 step hop with the blend occupying only a modest fraction of it,
so a sliding window taken well before or after the blend region should legitimately read as
the pure endpoint formation, something the training data structurally never teaches.

Directly tested this by checking whether a transition's first window predicts the correct pure
`A` formation, under two regimes, 20 trials each, 4 formation pairs:

| pair | matched regime (n_timesteps=50, blend 20-30 — `generate_dataset()`'s own format) | randomized regime (`build_long_sequence`'s format, longer/randomized blend) |
|---|---|---|
| dispersed→shield | 0/20 (0.0%) | 8/20 (40.0%) |
| shield→dispersed | 0/20 (0.0%) | 11/20 (55.0%) |
| column→diamond | 0/20 (0.0%) | 17/20 (85.0%) |
| dispersed→column | 0/20 (0.0%) | 10/20 (50.0%) |

The "matched regime" scoring 0% is the training data's OWN definition working as intended,
not a failure — a 50-timestep window with `blend_start=20`/`blend_end=30` is *exactly* what
`generate_dataset()` labels `"transitioning"`, so the model correctly says
`"transitioning"` there, not the endpoint formation (this was this diagnostic's own framing
error, not a model error — flagged plainly). The randomized/long regime scoring meaningfully
*better* (40-85%, still short of the matched-regime single-formation ceiling of ~70-100%) is
consistent with the mechanism: a longer segment with the blend confined to a narrower
fraction gives the model a window that looks closer to a genuine steady-state training
example, and it does correspondingly better — but still far short of ideal, and still exposed
to gap 1's `v_shape`/`encirclement`-specific failure for any window touching those formations.

**Gap 2 is not fully independent of gap 1** — `phase0_ceiling.json`'s aggregate confusion
matrix shows every formation, not just `v_shape`/`encirclement`, degrading sharply on long
trajectories relative to its matched-regime score (`column` 93.3%→15.8%, `diamond`
93.3%→20.6%, `dispersed` 100%→25.6%, `shield` 100%→24.0%) — this is squarely explained by the
blend-timing/window-position training gap just confirmed, layered on top of, not replacing,
gap 1's separate `v_shape`/`encirclement` collapse.

## Summary, stated plainly

Two real, evidence-backed, DISTINCT mechanisms, neither a simple bug:
1. **A generalization failure specific to `v_shape`/`encirclement`** (deterministic-template,
   feature-poor formations), plausibly aggravated by the newly-added acceleration term's much
   larger position-variance, on an absolute-position (not centroid-relative) node-feature
   representation, trained for only 67 epochs / 11 minutes.
2. **A structural train/eval mismatch in how "transitioning" is taught**: `generate_dataset()`
   only ever labels a FULL, fixed-size (50-step), fixed-position (blend at 20-30) window as
   `"transitioning"`, and never teaches the model that a window mostly-but-not-entirely before
   or after a blend region should read as the pure endpoint formation — exactly the situation
   every long, realistic, sliding-window-evaluated trajectory constantly produces.

Neither has been fixed here. Both point toward the training DATA REGIME (how `generate_dataset()`
labels transition examples, and possibly how much data/epochs are used) rather than the physics
fix itself being wrong — the physics fix (`docs/UPSTREAM_ISSUES.md` #1, ported this session) is
verified correct and is not implicated as a bug in either mechanism; it is, at most, an
aggravating factor for gap 1 via increased position variance.
