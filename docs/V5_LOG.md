# V5 production retraining program — run log

## 2026-08-07 — Phase 0, Step 1: HALTED before any compute

Per Phase 0 s1's own instruction ("Verify the two upstream fixes are actually in the
checkpoint's source... If either is unfixed, HALT and tell me"), checked both preconditions
by reading current source directly — no trajectory generation, no STGT load, no GPU time
spent, since both checks are answerable from the files on disk as they exist right now:

- **(a) dispersed/converging distinct base geometry — FAIL.** `src/swarm_intent/formations.py`
  lines 68-73: both formation types hit the identical `rng.uniform(...)` branch. This is the
  exact defect `docs/UPSTREAM_ISSUES.md` (written earlier this session) requests a fix for —
  that request has not been actioned by anyone; it is an open ask, not a landed fix.
- **(b) velocity varies within a sequence (acceleration) — FAIL.** `src/swarm_intent/data.py`,
  `generate_swarm_sequence()`: velocity is sampled once before the timestep loop and held
  constant for the entire trajectory (`centroid = centroid + velocity * dt`, same `velocity`
  every step). No acceleration term exists anywhere in the file. This fix was never requested
  in any prior AUDIT.md section or in `docs/UPSTREAM_ISSUES.md` — it is new scope.
- **Checkpoint freshness check:** `swarm_data/best_model.pt` (Aug 3) postdates
  `formations.py`/`data.py` (Jul 11) in raw mtime, but `git log` shows those two files have
  never been touched since the initial handoff commit — there is no version of the generator
  anywhere in this repo's history that contains either fix. The checkpoint was necessarily
  trained on the current, unfixed physics.

**Both of Phase 0's own stated preconditions fail. Per the plan's global rule ("If any step
fails or produces a result that contradicts a stated assumption, STOP... Do not improvise
around a broken assumption"), execution stops here.** No trajectories were generated, no
model was loaded, no Groq calls were made, no tmux job was launched. Reported back to the
user rather than proceeding into Phase 0 steps 2-4 (which would produce ceiling numbers whose
only valid interpretation is "measured against pre-fix physics," not the number the plan
needs) or improvising a fix to either defect unilaterally.

State written to `docs/V5_STATE.json`: `phase=0, step=1, status="halted_precondition_failure"`.

## 2026-08-07 — Phase 0, Step 1: upstream verification — VERDICT A, still halted

Per the user's follow-up instruction, verified the teammate's claimed push to
`https://github.com/pizz-beep/capstone` before doing anything else, rather than trusting the
claim or patching locally.

**1. Fetched upstream, all branches.** Cloned fresh into a scratch dir. **Only one branch
exists (`main`), no tags.** 11 commits total. Last 5, newest first:

| hash | date | message |
|---|---|---|
| `9158b081` | 2026-08-03 18:25:07 +0530 | added acceleration and split dispersed and converging logic |
| `b139dcee` | 2026-08-01 20:33:38 +0530 | docs: add README for LLM handoff |
| `106eca9d` | 2026-08-01 20:21:11 +0530 | updated gitignore |
| `6069c760` | 2026-08-01 20:18:17 +0530 | pushed model and updated gitignore |
| `deb24f75` | 2026-08-01 20:01:18 +0530 | fixed bugs |

**2. Located the generator code in both places.** Upstream's file list matches what the user
described exactly (`config.py`, `data_gen.py`, `dataset.py`, `inference.py`, `model.py`,
`train.py`, `visualize*.py`, `capstoner (2).py`) — no `formations.py`/`data.py`. Locally,
`src/swarm_intent/formations.py` and `data.py` were introduced wholesale in this repo's very
first commit (`2653c96`, "Initial commit... handoff snapshot", 2026-07-30) with no earlier
history — per `MIGRATION_GUIDE.md`, they are the migrated/restructured form of the same
logical functions (`get_formation_offsets`, `generate_swarm_sequence`), not a literal copy of
upstream's file layout. Confirmed common lineage directly: local `formations.py`'s pre-fix
numbers (`low=[-20,-20,-10], high=[20,20,10]`, shared `dispersed`/`converging` branch) are
byte-identical to upstream's commit `9158b081`'s PARENT — i.e. local was captured before
upstream's Aug 3 fix landed, from the same starting point.

**3. Checked both fixes.** Diffed `9158b081^` → `9158b081` directly (`data_gen.py` only):

```diff
-    elif formation_type == "dispersed" or formation_type == "converging":
-        offsets = rng.uniform(low=[-20,-20,-10], high=[20,20,10], size=(6,3))
+    elif formation_type == "dispersed":
+        offsets = rng.uniform(low=[-30,-30,-15], high=[30,30,15], size=(6,3))
+    elif formation_type == "converging":
+        angles = np.linspace(0, 2*np.pi, 6, endpoint=False); radius = 25.0
+        offsets = [[radius*cos(a), radius*sin(a), rng.uniform(-5,5)] for a in angles]
...
+    accel_mag = rng.uniform(-1.0, 1.0)
+    acceleration = np.array([accel_mag*cos(angle), accel_mag*sin(angle), rng.uniform(-0.1,0.1)])
     for t in range(n_timesteps):
+        velocity = velocity + acceleration * dt
         centroid = centroid + velocity * dt
```

| version | geometry fixed | acceleration fixed | last commit date |
|---|---|---|---|
| upstream `main` HEAD (`9158b081`) | **YES** | **YES** | 2026-08-03 18:25:07 +0530 |
| upstream `main`, parent commit | NO | NO | 2026-08-03 17:xx (pre-fix) |
| local `src/swarm_intent/` (unchanged) | NO | NO | 2026-07-11 (never edited since initial commit) |

**4. Checkpoint provenance.** `swarm_data/best_model.pt` keys: `epoch, model_state_dict,
optimizer_state_dict, val_loss, val_acc, cfg, reg_mean, reg_std, train_mean, train_std`.
**No git hash, dataset hash, or source-file identifier of any kind.** Only hyperparameters and
normalization stats. Cannot verify which generator version any checkpoint was trained against
by inspecting the file alone — confirmed as its own finding. Its mtime (Aug 3, 13:08) is
5h17m BEFORE upstream's fix commit (18:25) regardless, so it necessarily predates the fix.
Upstream's own repo carries no checkpoint either (`swarm_data/`, `*.pt`, `*.pth` all
gitignored there too) — the "pushed model" commit message does not correspond to an actual
checkpoint in git history.

**5. VERDICT: A.** Both fixes are present upstream, commit `9158b081e812a4f4a757840c677ea43d3cfb476c`,
on `main` (the only branch — no B/non-main-branch scenario applies). Per the user's own
protocol for verdict A: report and STOP, wait for confirmation a retrained checkpoint exists
before resuming Phase 0. **No retrained checkpoint exists anywhere yet — not locally, not
upstream.** Did not patch the local generator (explicitly instructed not to) and did not pull
upstream's fix into local `formations.py`/`data.py` on my own initiative — that is the "pull
and have her retrain" step the user reserved for themselves.

**6. Provenance guard shipped.** `scripts/verify_upstream_physics.py`: asserts both fixes are
present in whatever `src/swarm_intent/` currently contains (static regex check for the exact
shared-branch anti-pattern + a behavioural check per fix), fails loudly with the specific
evidence, non-zero exit code. Run against the CURRENT (still-unfixed) local repo, it correctly
fails on both checks — proving the gate works before it's ever relied on. 7 unit tests in
`tests/test_verify_upstream_physics.py`, all passing. **Deferred:** the user also asked to
extend `scripts/rebuild_derived.py`'s hash to cover the generator source files — that script
does not exist yet (it is a Phase 3 deliverable, and Phase 3 has not been reached). Noted as a
carried-forward requirement for whenever Phase 3 actually builds it.

State written to `docs/V5_STATE.json`: `phase=0, step=1, status="halted_awaiting_retrained_checkpoint"`.

## 2026-08-07 — Phase 0, Step 1: fix pulled locally

User authorized: "Pull the fix and retrain, then resume Phase 0." Ported upstream commit
`9158b081` into `src/swarm_intent/formations.py` and `data.py` (hand-migrated logic, not a
git merge — file layouts differ from upstream's `data_gen.py`):

- `formations.py`: `dispersed`/`converging` split into separate branches. `dispersed` keeps
  a (widened, `low=[-30,-30,-15]`/`high=[30,30,15]`) uniform scatter; `converging` is now a
  fixed-radius ring (`radius=25`, 6 angles via `np.linspace`, small per-drone z-jitter via
  `rng.uniform(-5,5)`).
- `data.py`: both `generate_swarm_sequence` and `generate_transition_sequence` (upstream
  unifies these into one function; kept split here, so the acceleration term went into both)
  now sample `accel_mag = rng.uniform(-1.0, 1.0)`, build an acceleration vector colinear
  with the initial heading, and apply `velocity = velocity + acceleration * dt` inside the
  per-timestep loop.
- Kept this repo's threaded-seeded-`rng` discipline throughout — did NOT port upstream's
  unseeded `np.random.default_rng(seed=None)` calls inside the geometry branches.
- **Explicitly not ported:** upstream's `generate_transition_sequence` also applies a
  converging-specific offset shrink DURING an active blend (when `formation_a`/`b ==
  "converging"` inside the alpha-blend loop). This predates commit `9158b081` — it was
  already in upstream before this fix and was never captured in our original migration
  either. It's a separate, pre-existing gap, not part of the verified diff being pulled here.
  Flagging for awareness: real coverage-measurement trajectories (`measure_coverage.py`'s
  `build_long_sequence`) are built entirely from `generate_transition_sequence`, so
  `converging`'s temporal shrink cue currently applies to steady-state sequences
  (`generate_swarm_sequence`) but not to transitions — the static ring-vs-scatter geometry
  difference is the only discriminating signal during an actual A→B hop right now. Worth a
  deliberate decision later, not silently fixed here.

`scripts/verify_upstream_physics.py` now passes both checks (previously correctly failed).
`tests/test_verify_upstream_physics.py`'s integration tests flipped from expect-failure to
expect-success. Full suite: 134/134 pass, no other test depended on the old shared-branch
values. Committed as `27adc23`.

Next: regenerate the dataset and retrain STGT on the fixed physics (Phase 0 still hasn't run
any of its own steps 2-4 yet — this was all precondition work).

## 2026-08-07 — Phase 0, Step 1: dataset regenerated, STGT retrained

Backed up the pre-fix `swarm_data/` (checkpoint + splits) to
`swarm_data_prefix_backup_20260807/` before overwriting — referenced throughout AUDIT.md
secs AE-AH, kept rather than discarded.

Ran in tmux session `v5` (`tail -f /tmp/v5_generate_data.log` then
`tail -f /tmp/v5_train_model.log`):

- `PYTHONPATH=src python scripts/generate_data.py --per-formation 1000 --transitions 2000`
  — 9000 sequences across 8 classes, 2.23s. (Note: neither `scripts/generate_data.py` nor
  `scripts/train_model.py` work without `PYTHONPATH=src` or `pip install -e .` — the package
  isn't installed in this venv; used `PYTHONPATH=src` rather than installing, to avoid an
  unrequested environment change.)
- `PYTHONPATH=src python scripts/train_model.py --classes 8 --epochs 80` — early stopped at
  epoch 67 (patience=12), best checkpoint saved from epoch 55, `test_loss=0.2017,
  test_acc=0.9348`. Wall time 11m13s on the RTX 4090.

`scripts/verify_upstream_physics.py` re-run against the retrained checkpoint's source (the
guard checks the CURRENT repo source, which is what the checkpoint was trained against since
retraining happened after the port) — both checks pass. Full suite 134/134.

Moving to Phase 0 steps 2-4: generate 1,000 fresh held-out trajectories (seed=999, disjoint
from seed=0 used in every real-output eval this session, seed=1 used for sec AG's dev split,
and seed=42 used for the train/val/test split itself), measure per-class accuracy, the 8x8
confusion matrix, and pair-level accuracy (the ceiling).

## 2026-08-07 — Phase 0, Steps 2-4: HALT GATE 1 — ceiling measured at 3.3%

`scripts/phase0_ceiling.py --n 1000` (seed=999, tmux `v5`, `tail -f /tmp/v5_phase0_ceiling.log`,
95s wall time). Full breakdown in `docs/CEILING.md`. Headline:

- Window-level overall accuracy: **22.3%** (n=8599 windows). Per-class collapse is severe and
  uneven: `v_shape` 1.3%, `encirclement` 0.2%, `converging` 0.1%, vs. `transitioning` 80.4% —
  the model is overwhelmingly defaulting to "transitioning" on real formations.
- **Pair-level accuracy (the ceiling): 3.3% (17/509).**

Before trusting a brand-new, never-before-run script's catastrophic output, checked for a bug
in the script itself first:
- Window/label alignment verified against `sliding_window_inference`'s own indexing
  (`stgt/inference.py:79-80`) — correct, not a bug.
- Early-vs-late window position within a trajectory (60 fresh sequences): 36.3% early vs.
  32.7% late — essentially flat. **This rules out simple position-drift-into-OOD-normalization**
  as the primary mechanism (a monotonic drift story would predict much better early-window
  accuracy).
- Training-distribution-matched check (exact `generate_dataset()` regime — n_timesteps=50,
  single steady formation, same `sliding_window_inference`/`predict_v2` path): **69.5%**
  overall (n=105) — better than the long-trajectory number, but well below `train_model.py`'s
  own reported `test_acc=0.9348`, and with `v_shape`/`encirclement` specifically collapsing to
  **0/15 each**. This gap (93.5% → 69.5%) is unrelated to anything this session's physics fix
  touched and is itself an open finding, not diagnosed further here.
- Leading unconfirmed hypothesis for the further 69.5% → 22.3%/3.3% drop:
  `build_long_sequence()`'s hop regime (randomized `blend_start`/`blend_end` as fractions of a
  variable 50-100 step segment) is shaped nothing like `generate_dataset()`'s training
  regime, which calls `generate_transition_sequence` at its hardcoded defaults
  (`blend_start=20, blend_end=30`, fixed `n_timesteps=50`) for every transitioning training
  example. STGT has never seen a blend timed/shaped any other way. Not confirmed with a
  targeted ablation — flagged as the most likely next thing to check, not fixed here.

**Per the plan's own pre-stated rule ("If < 70%: the bottleneck is upstream, not the LLM, and
the plan changes"), this is an unambiguous HALT GATE 1 trigger** — not a borderline call, and
not something to improvise a fix for. Two separate, stacked, unconfirmed mechanisms are on the
table (an inference-path gap with a `v_shape`/`encirclement`-specific collapse, and a
long-trajectory blend-timing/regime mismatch); neither has a root cause established yet.
Reporting to the user now, per HALT GATE 1's protocol.

State written to `docs/V5_STATE.json`: `phase=0, step=4, status="HALT_GATE_1..."`.

## 2026-08-07 — Phase 0: scoped diagnostic pass on both gaps (no fix applied)

Per instruction, diagnosed both gaps `docs/CEILING.md` flagged, without changing any code,
data, or model. Full writeup in `docs/GAP_DIAGNOSIS.md`.

**Gap 1 (93.5%→69.5%→0% for `v_shape`/`encirclement`).** Ruled out three candidate causes
with direct evidence: duplicate model/graph implementation
(`swarm_intent.model`/`graph.py` vs. the parallel, undocumented `swarm_intent.stgt.model`
copy) — identical result via both, not the cause; normalization round-trip inconsistency —
verified consistent on real `X_test.npy` examples through both code paths; small-sample luck
— 0% reproduced at n=100 across two independent seeds. **What's confirmed:** the model
confidently (97.1%) predicts `"diamond"` for fresh `v_shape` instances, 30/30 — a real
generalization failure, not noise. Leading (unproven) explanation: `v_shape`/`encirclement`
are deterministic-template formations with no random offset component; the model's node
features are absolute, not centroid-relative, positions; the newly-added acceleration term
makes per-example position variance far larger than before; only 67 epochs/11 minutes of
training. `X_test.npy`'s own held-out rows (same generation run) score 100% while genuinely
novel instances collapse to 0% — the signature of overfitting to this specific run, not a
code bug (checked three ways). **Side finding, unrelated to this fix:**
`src/swarm_intent/stgt/` is an undocumented second copy of the model/graph/config code that
every eval script in this project imports from, while `train.py` uses the "main" package —
and it reintroduces the exact hidden-global-state pattern (`device` as a module-level global)
this project's own migration was meant to eliminate. Harmless today (single GPU), but a real
drift risk going forward.

**Gap 2 (69.5%→22.3%/3.3% on long trajectories).** Confirmed mechanism, directly tested:
`generate_dataset()` labels an ENTIRE 50-step transition sequence (`blend_start=20`,
`blend_end=30`) as `"transitioning"`, full stop — the model has never been trained on a
window where only part of it is blend and the rest should read as the pure endpoint
formation. Tested directly: 4 formation pairs, 20 trials each. Matched-regime (exactly
`generate_dataset()`'s own format) scores 0% on "does the first window predict the pure
endpoint formation" — correctly, since that IS what "transitioning" training data looks like
(this was the diagnostic's own framing error, flagged as such, not a model error). The
randomized/long regime (`build_long_sequence`'s actual format) scores meaningfully better
(40-85%) — closer to a genuine steady-state window — but still well short of the ~70-100%
matched-single-formation ceiling. Not independent of gap 1: `phase0_ceiling.json`'s aggregate
matrix shows every formation, not just `v_shape`/`encirclement`, degrading sharply on long
trajectories relative to its matched-regime score.

**Nothing fixed. Both mechanisms point at the training DATA REGIME
(`generate_dataset()`'s blend-labeling convention, possibly epoch/data budget), not at the
physics fix itself being wrong** — the fix is verified correct (`scripts/verify_upstream_physics.py`)
and is, at most, an aggravating factor for gap 1 via increased position variance, not the root
cause of either gap. Reporting back per the "diagnose only" instruction; awaiting direction on
whether/how to fix.

State written to `docs/V5_STATE.json`: `phase=0, step=4`, still HALT_GATE_1, diagnosis
complete, nothing fixed.

## 2026-08-08 — gap-2 fix implemented, retrained, ceiling barely moves (3.3%→4.7%)

User authorized: "Fix option 1 and retrain." Implemented the 3-regime dominant-formation
labeling in `generate_dataset()` (commit `8e08e02`) exactly as diagnosed in
`docs/GAP_DIAGNOSIS.md`, ported from upstream's own `generate_transition_dataset` regime
design, adapted to threaded seeded rng and proportional (not hardcoded-50) bounds. Sanity
checked label distribution before trusting it (34% transitioning, matches regime 1's ~1/3
probability; no NaNs). Full suite 134/134.

Backed up `swarm_data/` again (`swarm_data_prefix_backup2_20260807_gap2/`), regenerated
(9000 sequences, 2.18s), retrained (`PYTHONPATH=src python scripts/train_model.py --classes 8
--epochs 80`, tmux `v5`) — this run went the FULL 80 epochs with no early stop (best epoch
75, `test_acc=0.9333`, 13m07s). `verify_upstream_physics.py` still passes against the new
checkpoint's source.

Re-ran `scripts/phase0_ceiling.py --n 1000` (identical seed=999 protocol,
`evaluation/phase0_ceiling_v2.json`): window-level accuracy 22.3%→27.7%, **pair-level
accuracy (the ceiling) 3.3%→4.7% (17/509→24/509)**. Mixed per-class picture: `v_shape` jumped
from a confident 0% failure to 53.2% (large win, consistent with the diagnosis), but
`diamond` (20.6%→10.2%) and `shield` (24.0%→15.4%) got WORSE, and `transitioning`'s own
recall dropped sharply (80.4%→35.5%, more than its shrunken population share alone would
predict). `encirclement` (14.3%) and `converging` (7.7%) remain very poor — gap 1's
generalization-failure signature is still clearly present for those two.

**Full writeup appended to `docs/CEILING.md`. This remains an unambiguous HALT GATE 1
trigger** — 4.7% is not meaningfully closer to a "proceed" or even "revise target" decision
than 3.3% was. The fix was correctly diagnosed and correctly implemented (verified with a
direct before/after regime test before ever touching training data), and it did measurably
help where it targeted — but it did not come close to resolving the ceiling problem on its
own, and introduced new regressions in classes it wasn't targeting. Nothing further attempted
without checking back in, per the halt protocol.

State written to `docs/V5_STATE.json`: `phase=0, step=4`, still HALT_GATE_1.

## 2026-08-08 — data+epochs scaled (3.3x/1.9x): real improvement, still nowhere near the floor

User authorized: "Increase training epochs and data size, then retrain, IF RESULTS STILL
DONT IMPROVE TRY ANOTHER STRATERGY." Scaled `--per-formation 1000->3000`,
`--transitions 2000->9000` (30000 total sequences vs. 9000), `--epochs 80->150`
(early-stopped at 42, best epoch 30, `test_acc=0.9771`, 24m45s, tmux `v5`). Backed up the
prior (9k/80-epoch) `swarm_data/` to `swarm_data_backup3_gap2fix_9k/` first — skipped a full
backup of the LARGER dataset given disk was at 100%/3.7GB free at the time (checked before
proceeding; the 30k dataset only added ~140MB, headroom remained adequate throughout).
`verify_upstream_physics.py` and the full suite (134/134) still pass.

Re-ran `scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v3.json`):
window-level 27.7%→36.9%, **pair-level (the ceiling) 4.7%→6.7% (24/509→34/509)**.

Real, monotonic improvement across both fixes so far (3.3%→4.7%→6.7%), but the rate of return
(~+1.5-2 points per ~3x compute increase) does not support closing the gap to the 70% floor
without many further doublings of an already-substantial budget. Per-class movement isn't
even monotonic — `v_shape` fell back from 53.2% to 31.7% this round while
`diamond`/`shield`/`converging` recovered past their original baselines — reads as capacity
trading within a fixed architecture, not convergence toward a usable ceiling.

**Judged this as "results still don't improve [enough to matter]" per the user's own stated
conditional, and moved to the pre-authorized next strategy: centroid-relative node
features.** Full comparison appended to `docs/CEILING.md`.

## 2026-08-08 — strategy 2 (centroid-relative features): window accuracy transformed, ceiling still flat, root cause identified

Per the pre-authorized contingency (data+epochs alone was judged insufficient — see above),
implemented centroid-relative node features: `build_graph`'s `data.x` is now
`positions - positions.mean(dim=0)` instead of absolute positions, in both `graph.py` and the
duplicate `stgt/model.py` copy (kept in sync — see the earlier-flagged drift risk). Committed
`f9df8a3`. Retrained from scratch on the SAME 30k dataset (positions unchanged, only the
graph-construction interpretation changes, so no need to regenerate data), 150 epochs,
early-stopped at 29 (best epoch 17), `test_acc=0.9873` — highest yet, and convergence was
dramatically faster (98-99% train accuracy by epoch 16). `verify_upstream_physics.py` and the
full suite (134/134) pass.

`scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v4.json`): **window-level
accuracy 36.9%→72.7%, the largest single jump of the session, with per-class accuracy now
much more even (52-85%, vs. previous runs' wide 0-80% spread with confident systematic
failures on specific classes). But pair-level accuracy barely moved: 6.7%→6.1%.**

Checked cheaply (same predictions, no new GPU inference — added a `robust=True` comparison
directly into `phase0_ceiling.py`) whether sec AG's already-built majority-vote reduction
unlocks the gap between window- and pair-level accuracy: `robust=True` gives 6.5% (33/509) vs.
`robust=False`'s 6.1% (31/509) — barely different. Bucket C shrinks substantially (188→100)
but nearly all of that shifts into bucket B (271→356), not bucket A (50→53) — the exact
"recovers a structural pair, guard eats it anyway" pattern sec AG documented, reproduced here
on a MUCH better-classified model.

**Root cause identified, computed directly from the confusion matrix: the model over-predicts
`"transitioning"` at a strikingly uniform 20-28% false-positive rate across all 7 steady
formations** (`v_shape` 24.3%, `encirclement` 27.5%, `column` 20.5%, `diamond` 21.3%,
`dispersed` 25.4%, `converging` 25.1%, `shield` 27.1%). With 15-30 windows per realistic long
trajectory, this makes at least one spurious ambiguous window near-certain per trajectory,
which is enough to trip the guard logic regardless of how accurate classification is
everywhere else. **This is no longer a classification-quality problem — it's a residual,
class-uniform calibration/bias problem specific to the `"transitioning"` class.**

**Both pre-authorized strategies have now been executed. Neither closed the gap to anywhere
near the 70% floor, though the second one transformed the underlying classification quality
and produced a much more specific, actionable diagnosis of what's actually left.** Full
writeup appended to `docs/CEILING.md`. Remains an unambiguous HALT GATE 1 trigger — not
attempting a third strategy without checking in first, since the instruction authorized one
pivot ("try another strategy"), not open-ended iteration.

State written to `docs/V5_STATE.json`: `phase=0, step=4`, still HALT_GATE_1.

## 2026-08-08 — strategy 5: targeted fix for the transitioning false-positive rate, ceiling roughly doubles

User authorized: "Target the transitioning false-positive rate specifically, then retrain,"
and asked for `HISTORY.md` first (strategy-level running summary, created and committed
`0f7f404` before touching anything else).

Before writing any code, diagnosed where the false positives actually concentrate (40
chain-length-1 + 40 chain-length-2 fresh trajectories, seed=4242): **1.8% false-positive
`"transitioning"` rate on fully unambiguous windows (zero blend anywhere in the whole
trajectory), 53.2% within 15 timesteps of a real blend boundary, 0.0% far from one.** This
refuted the "model has learned an overly broad transitioning detector" story from strategy
4's writeup — the model is well-calibrated on clean data. The false positives trace to
strategy 2's (gap-2 fix) regime boundaries still leaving a wide grey zone: `"transitioning"`-
labeled training examples could have as little as 28% genuine blend content (up to 64%
residual pure formation), and "pure"-labeled examples could carry real blend content near
their edges.

Tightened `generate_dataset()`'s regime bounds accordingly (commit `5baaceb`): regime 1 now
requires the blend region to dominate the window (45-62% of `n_timesteps`, verified
numerically before regenerating — mean 52%, was 28-44%); regimes 0/2 push blend to the very
edge with a short duration (74-90% pure content, was 66-86%). Regenerated (30k sequences,
5.84s) and retrained (150 epochs, early-stopped at 22, best epoch 10, `test_acc=0.8631` —
notably lower and the val curve visibly more volatile, oscillating 0.6-4.1, than strategy 4's
run). `verify_upstream_physics.py` and the full suite (134/134) still pass.

`scripts/phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v5.json`): window-level
accuracy roughly flat (72.7%→70.4%), but **pair-level accuracy (the ceiling) roughly
DOUBLED: 6.1%→12.2% (robust=True: 6.5%→12.8%)** — the largest single relative jump from any
one fix in this program so far. Transitioning false-positive rate dropped meaningfully for
some classes (`diamond` 21.3%→4.0%, `shield` 27.1%→8.5%) and modestly for others. Not a clean
win throughout, though: `encirclement`'s raw window accuracy regressed sharply (62.2%→41.3%),
the same capacity-trading pattern seen in strategy 3.

**Still far below the 70% floor — remains an unambiguous HALT GATE 1 trigger — but this is
real, meaningful, diagnosis-driven progress, not a plateau.** The volatile training curve
(early stop at epoch 22 of 150, lower aggregate `test_acc` than strategy 4) is untested as a
further lever — a steadier training run on the same, now-improved data might do better,
but that's a new question, not explored here.

`HISTORY.md` updated with this strategy's entry. State written to `docs/V5_STATE.json`:
`phase=0, step=4`, still HALT_GATE_1.

**Note on the message's final instruction.** The message opened with "Do NOT patch the
generator yourself under any circumstances" and closed with "If the bugs still exist, just
clone the repo and fix the bugs yourself and continue working" — these directly contradict
each other. Moot here since the verdict is A, not C (the bugs do NOT still exist upstream), so
neither branch of that contradiction was triggered. Flagged to the user for awareness in case
it was a drafting slip, since a future turn could land on verdict C and the contradiction
would then be load-bearing.

## 2026-08-08: pre-strategy-6 step 1 — is exact-pair accuracy even the right ceiling?

User instruction, before choosing strategy 6: compute the RULES-aware threat/intent/action
ceiling on the EXISTING strategy-5 measurement (no retraining, no new sampling). Built
`scripts/phase0_threat_ceiling.py`, which re-scores `evaluation/phase0_ceiling_v5.json`'s
509 pair-eligible `pair_records` (already computed) against `RULES`
(`llm_finetuning/build_sft_dataset.py`) instead of requiring an exact pair match — since
`RULES` maps 49 `(from, to)` pairs onto only 4 threat levels, a wrong recovered pair can still
land on the correct `threat_level`.

Result (`evaluation/phase0_threat_ceiling_v5.json`):

| metric | robust=False | robust=True |
|---|---|---|
| exact pair accuracy | 12.2% (62/509) | 12.8% (65/509) |
| **threat ceiling** | **13.0%** | **13.6%** |
| intent ceiling | 12.2% | 12.8% |
| action ceiling | 13.0% | 13.6% |
| no recovered pair at all (bucket B/C) | 431/509 (84.7%) | 428/509 (84.1%) |

**Plain answer: the 70% floor is not met on the threat ceiling either.** It barely moved off
exact-pair accuracy. Why it barely moved is the actual finding: of the 78/81 trajectories that
DO reach a resolvable bucket-A pair, conditional threat accuracy is ~85% — genuinely good.
`RULES`-mapping tolerance was never the bottleneck. The bottleneck is that 84.7% of
trajectories never reach a resolvable pair in the first place (bucket B guard-blocked, or
bucket C multi-hop/unresolvable) — no `RULES` lookup even happens for those. The `critical`
threat class (10 true cases) has zero recoveries under either variant.

This reframes the target for step 2/strategy 6: the classifier, when it commits to a clean
pair, is mostly right. The gate is `stgt_bridge`'s bucket-A resolution rate (15.3%/15.9%),
not `RULES` granularity or exact-pair strictness. Full detail and 4x4 confusion matrix:
`docs/CEILING.md`'s 2026-08-08 "the REAL ceiling" update. `docs/V5_STATE.json` updated
(`phase=0, step=5`). `HISTORY.md` updated with a pre-strategy-6 measurement section. Proceeding
to step 2 (decompose pair-recovery failures) per instruction — no retraining yet.

## 2026-08-08: pre-strategy-6 step 2 — decomposing pair-recovery failures

Re-ran inference (still no retraining) with the identical seed=999 regime, reproducing all
509 pair-eligible trajectories index-for-index (62 correct / 447 failed, exact match to
`phase0_ceiling_v5.json` — good determinism sanity check). For each failure, recorded
per-window correctness/position and the exact `stgt_bridge` guard reason that blocked bucket
A (`scripts/phase0_decompose_failures.py`, `evaluation/phase0_decompose_failures.json`).

**Two findings, both sharper than anything surfaced so far this program:**

1. **Chain-length-2 (an actual formation transition) has NEVER once succeeded**: 62/258
   (24.0%) of steady-state trajectories recover correctly, but **0/251 (0.0%)** of genuine
   single-hop transitions do. Every success in the whole ceiling measurement is a steady
   state. The system currently cannot detect a real transition at all.

2. **The reduction/guard logic, not classifier accuracy, is the dominant bottleneck.** 211/447
   failures (47.2%) had ZERO misclassified windows — perfect per-window classification, still
   failed to recover the pair. Filtering to correct-only windows and re-reducing recovered the
   pair in only 4/447 (0.9%) of cases, ruling out "a few bad windows tripping unanimity."
   Tracing the actual `stgt_bridge` guard: **`dispersed_converging_ambiguity` accounts for
   272/447 (60.9%) of ALL failures** — by far the largest single cause — and it fires on
   windows of every formation (not just dispersed/converging), confirming it's a generic,
   unconditional classifier-calibration guard, not something specific to those two classes'
   trajectories. This exact defect was already documented, unfixed, in `AUDIT.md` sec AG; this
   is the first time it's been isolated as the dominant cause against the strategy-5
   checkpoint specifically. bucket C (oscillation/multi-hop noise) is the next largest at
   34.5%.

Full tables and guard-reason breakdown: `docs/CEILING.md`'s 2026-08-08 "step 2" update.

**Implication flagged, not acted on:** since the dominant blocker is a fixed-threshold guard
in `stgt_bridge.py` that fires independent of whether classification was even correct,
strategy 6's retrain (steadier schedule) is unlikely to move the pair-level ceiling by much on
its own. Not revisiting the guard itself — not authorized this turn. Proceeding to strategy 6
exactly as instructed and reporting the result honestly against this expectation.
`docs/V5_STATE.json` updated (`phase=0, step=6`).

## 2026-08-08: strategy 6 — steadier training schedule, under-convergence fixed

Checked dataset class distribution FIRST (per instruction): strategy 5's 30k dataset is
essentially balanced across all 8 classes (base formations 3821-3910 each, transitioning
2966) — ruled out "encirclement has fewer examples" before touching training code.

Changed the LR schedule (`src/swarm_intent/config.py` + `train.py` + `scripts/train_model.py`,
new `cfg.warmup_pct`/`cfg.lr_min_frac` fields, CLI `--lr`/`--patience`/`--warmup-pct`/
`--lr-min-frac` overrides): `OneCycleLR` (decays to ~0) replaced with linear warmup + cosine
decay to a nonzero floor (5% of peak). Retrained on the SAME strategy-5 data with peak lr
lowered 3e-4→1e-4 and patience raised 12→35. 134/134 tests pass after the change. Backed up
`swarm_data/best_model.pt` → `best_model_strategy5_backup.pt` before overwriting.

Training ran ~86/150 epochs (~43 min on the RTX 4090) before early stopping (best epoch 51,
`val_loss=0.0505`), vs. strategy 5's early stop at 22/150 (best epoch 10). `test_acc` jumped
0.8631→**0.9958**. Per-class test precision/recall are now all ≥98.6%, including
`encirclement` (recall 0.986, up from being the class that regressed in strategy 5). This
confirms strategy 5's checkpoint genuinely was under-converged, not just noisy.

While computing the per-class report, found (not fixed, out of scope): `evaluate_ml_model`
(`src/swarm_intent/llm/evaluate.py`) re-normalizes `X_test.npy`, which is ALREADY normalized
before being saved to disk (`data.py`'s `split_and_normalize`/`save_splits`) — a
double-normalization bug that collapses predictions to a single class when that function is
used directly. Worked around by evaluating without the extra normalization step; flagging for
a future turn since it's not this instruction's scope.

Full curve: `evaluation/phase0_strategy6_train_log.txt`. Per-class report:
`evaluation/phase0_strategy6_classification_report.json`. Full detail and before/after table:
`docs/CEILING.md`'s 2026-08-08 "strategy 6" update.

**This is in-distribution test accuracy, not the ceiling.** Step 2's finding still stands:
the dominant pair-recovery blocker is a `stgt_bridge.py` guard independent of classification
correctness. Proceeding to step 4: re-measure both the pair-level and threat-level ceilings
on this checkpoint to see what, if anything, transfers.

## 2026-08-08: step 4 — re-measured ceilings on the strategy-6 checkpoint: a REGRESSION

Ran `phase0_ceiling.py --n 1000` (`evaluation/phase0_ceiling_v6.json`) and
`phase0_threat_ceiling.py` (`evaluation/phase0_threat_ceiling_v6.json`) on the strategy-6
checkpoint, same seed=999 protocol as every prior measurement.

**Result, reported plainly: strategy 6 made the ceiling WORSE, not better.**

| metric | strategy 5 | strategy 6 |
|---|---|---|
| pair-level accuracy (robust=False) | 12.2% (62/509) | **4.9% (25/509)** |
| pair-level accuracy (robust=True) | 12.8% (65/509) | **5.3% (27/509)** |
| threat ceiling (robust=False) | 13.0% | **6.3%** |
| threat ceiling (robust=True) | 13.6% | **6.9%** |

Window-level accuracy stayed roughly flat (70.4%→69.1%) but the per-class breakdown shows a
sharp capacity trade: `encirclement` recovered (41.3%→61.8%) and `transitioning` improved a
lot (57.3%→87.5%), but `diamond` (94.0%→70.0%), `shield` (90.3%→66.5%), and especially
`converging` (66.7%→37.4%) all regressed. The clearest signal: conditional accuracy WITHIN
bucket A (trajectories that DID reach a resolvable pair) collapsed from 79.5% (62/78) to
32.5% (25/77) — strategy 6 is now wrong more often than right even when it commits to an
answer.

**Diagnosis:** strategy 6 converged much more tightly to `generate_dataset()`'s training
distribution (near-100% in-distribution test_acc) than strategy 5 did, but that distribution's
spread/noise ranges are narrower than the ceiling test's realistic long-trajectory sampling.
The sharper fit generalized better for 2 classes and markedly worse for 3 others — a genuine
overfitting/generalization tradeoff caused by fixing the under-convergence, not a new bug.

`swarm_data/best_model.pt` is now the strategy-6 (worse-ceiling) checkpoint;
`swarm_data/best_model_strategy5_backup.pt` holds strategy 5's better-performing checkpoint if
reverting is wanted. Full tables: `docs/CEILING.md`'s 2026-08-08 "step 4" update.

**HALT GATE 1 remains unambiguously triggered — further below the 70% floor than strategy 5
left it.** Per instruction, stopping here. `docs/V5_STATE.json` updated (`phase=0, step=8`).
Not proceeding to any further strategy without explicit direction.

## 2026-08-09: step 0 — revert checkpoint to strategy 5, investigate the guard bug directly

User's diagnosis: step 2's `dispersed_converging_ambiguity` finding (60.9% of failures,
47% of failures with zero misclassified windows) points to a bug in OUR bridge code, not an
STGT limitation. No training authorized this turn.

**Step 0: reverted the checkpoint.** `swarm_data/best_model.pt` (strategy 6, worse ceiling)
backed up to `best_model_strategy6_backup.pt`; `best_model_strategy5_backup.pt` copied over
`best_model.pt`. Verified by SHA-256: post-revert `best_model.pt` hash
(`18fc201d5a419ff2fb0cfb66a60810af77a9ed52969f2996f57c952d1306a01b`) matches
`best_model_strategy5_backup.pt` exactly; loaded checkpoint confirms `epoch=10,
val_loss=0.610` (strategy 5's own recorded best epoch). `swarm_data/best_model.pt` is once
again the strategy-5 (better-ceiling) checkpoint.

**Standing rule recorded in `HISTORY.md`:** checkpoint selection must be judged on ceiling
(pair-level/threat-level), never test accuracy alone — strategy 6 proved a checkpoint can hit
99.6% test accuracy while roughly halving the metric that matters.

**Step 1: confirmed the guard bug directly.** Read `stgt_bridge.py:114-120`
(`_is_ambiguous_dispersed_converging`) — it checks only `abs(d - c) < 0.15` on the raw
`dispersed`/`converging` probabilities, with no check on whether either is actually
competitive (e.g. in the window's top-2). Wrote `scripts/phase0_guard_audit.py` to measure
this directly (inference only, strategy-5 checkpoint, same seed=999 population): across 1469
windows the guard fires on 75.8% of them, and of those firings, **66.1% have NEITHER
dispersed nor converging in the top-2 predicted classes** — a window predicted `shield` at
98.97% confidence with `dispersed=0.0012`/`converging=0.0005` still trips it, since
`|0.0012-0.0005|=0.0007 < 0.15`. Only 1.7% of firings are genuinely both-in-top-2 contention.
This is the exact mechanism behind step 2's finding (60.9% of pair-recovery failures,
uniform across every formation class): with 8 softmax classes, when one class dominates, the
remaining ~7 split a small residual probability mass and any two of them land within 0.15 of
each other by chance almost always — the guard was never testing dispersed/converging
contention specifically. Full detail: `docs/CEILING.md`'s 2026-08-09 update. Proceeding to
step 2: fix the guard (require both classes in top-2, difference under threshold), re-measure
isolated from any other change.

## 2026-08-09: step 2 — fixed the guard, re-measured isolated and full

**Fix.** `stgt_bridge.py`'s `_is_ambiguous_dispersed_converging` now additionally requires
dispersed and converging to be the window's top-2 predicted classes (sorted by probability),
on top of the original `abs(d-c) < 0.15` closeness check. Both conditions must hold. No other
line changed, no retraining, `swarm_data/best_model.pt` still the strategy-5 checkpoint
reverted to in step 0. Full 134/134 test suite still passes after the change (no existing
test asserted the OLD, broken firing behavior as a requirement).

**Isolated re-measurement.** Re-ran `scripts/phase0_guard_audit.py --n 1000`, identical
seed=999 population and checkpoint as step 1's audit — the fixed guard function is the only
variable. Fire rate collapsed from 75.8% (1113/1469 windows) to **1.3% (19/1469)**, and of
those 19 remaining firings, **100% are genuine both-in-top-2 contention (0% spurious, 0%
one-sided)** — a complete fix, not a partial tightening (the 19 originally-genuine firings
from step 1 all survive, since the new top-2 condition is additive and never removes an
already-both-in-top-2-AND-close firing).

**Full re-measurement.** Re-ran `phase0_ceiling.py --n 1000` (`evaluation/
phase0_ceiling_v5_guardfix.json`) and `phase0_threat_ceiling.py`
(`evaluation/phase0_threat_ceiling_v5_guardfix.json`), same seed=999/checkpoint/protocol as
every ceiling measurement all program.

| metric | before guard fix | after guard fix |
|---|---|---|
| bucket A (robust=False) | 78/509 (15.3%) | **294/509 (57.8%)** |
| bucket A (robust=True) | 79/509 (15.5%) | **396/509 (77.8%)** |
| pair-level accuracy (robust=False) | 12.2% | **47.0%** |
| pair-level accuracy (robust=True) | 12.8% | **48.5%** |
| precision within bucket A (robust=False) | 79.5% | 81.3% |
| precision within bucket A (robust=True) | ~20% (sec AG) | **62.4%** |
| threat ceiling (robust=False) | 13.0% | **52.3%** |
| threat ceiling (robust=True) | 13.6% | **58.7%** |

**This is the single biggest gain of the whole Phase 0 program, and it came from fixing one
mis-specified boolean condition, not from retraining anything.** It also retroactively
reframes sec AG's "robust reduction should not ship" verdict: that verdict was measured with
the BROKEN guard corrupting the exact `class_probabilities` signal robust reduction's
majority-vote step also reads — `robust=True` precision within bucket A was capped at ~20%
by the same guard bug, not by a flaw in the majority-vote algorithm itself. Under the fixed
guard, `robust=True` precision is 62.4%, close to `robust=False`'s 81.3% and clearly usable.
Sec AG's verdict should not be treated as permanent; a fresh pipeline_v2 comparison against
this fix is the natural next check if that's wanted.

**HALT GATE 1: still not cleared, stated plainly.** 52.3-58.7% threat ceiling remains below
the 70% floor. This is the largest single jump in the program (gap shrank from ~57-58 points
below floor to ~11-18 points below floor) but the gate itself does not open. Per-class window
accuracy (`docs/CEILING.md`) shows the remaining gap is now genuine STGT classification
difficulty — `encirclement` (41.3%) and `transitioning` (57.3%) window accuracy are the
biggest drags — not a bridge-code artefact. `docs/V5_STATE.json` updated (`phase=0, step=11`).
Stopping here per the "re-measure isolated" instruction's scope; not proceeding to any further
strategy without explicit direction.

## 2026-08-09: step 3 — user's new instruction, chain-length-2 audit before more training

User's new instruction: audit the bridge for more bugs of the SAME CLASS as the
dispersed_converging guard before any more STGT training, and settle whether the 70% HALT
GATE 1 floor is still the right gate now that pair-level and threat ceilings have diverged
(47.0% vs 58.7%, previously nearly identical at 12.2%/13.0%). Four steps: (1) chain-length
breakdown + trace, (2) audit every other guard/rule, (3) robust-reduction threshold sweep
tuned on a dev split, (4) end-to-end threat accuracy projection to re-examine the gate. No
training this session.

**Step 1: chain-length breakdown.** `scripts/phase0_chainlength_breakdown.py --n 1000`, same
seed=999 population. Confirmed the user's hypothesis is correct on the numbers: chain-1
(steady state) pair accuracy is 86.8%, **chain-2 (single real transition) is still 6.0%** —
essentially unmoved by the guard fix. Chain-3+ (no possible RULES key) has a low, stable
bucket-A false-positive rate (~1.8%, both reduction modes) — not where the remaining problem
concentrates.

**But tracing 20 failing chain-2 trajectories stage by stage (window classifications → guard
→ temporal derivation → reduction → bucket; full trace `evaluation/phase0_chain2_trace.txt`)
shows this is NOT a second guard bug of the dispersed_converging class:**

| diagnosis | n/20 |
|---|---|
| all_windows_transitioning | 35.0% |
| structural_reduction_wrong_pair | 25.0% |
| trailing_transitioning_run | 20.0% |
| blocked_by_oov_name_guard | 15.0% |
| spurious_third_formation_from_misclassification | 5.0% |

60% (the first three rows) trace to the SAME windowing-artefact mechanism the earlier
engagement already diagnosed (AUDIT.md sec AF step 4): chain-2 trajectories are a single
50-100-timestep hop, frequently only 1-2 sliding windows — genuinely too few observations for
the destination formation to reliably resolve, or in several traced cases (e.g. trajectory
18, `shield→v_shape`) to even be REACHED within the generated sequence at all — the model
reads the FIRST formation correctly and confidently on every available window because that's
all that's actually there to see. No bridge-logic change fixes this; it needs longer
segments in the sampling regime or a larger `max_seq_len` (retraining, out of scope this
session).

Only 15% (`blocked_by_oov_name_guard`) is real, actionable bridge-logic brittleness — cases
where the structural reduction ALREADY lands on the exact correct pair (`rules_key ==
gt_pair`) but the strict `oov_name` guard (fires on ANY unknown window, no threshold) routes
it to bucket B anyway. This is exactly what `robust=True`'s majority-vote/trim logic exists
to fix, and is folded into step 2's guard audit below rather than treated as a separate
finding. 5% is plain classifier misclassification (a bad window, not a bridge issue).

**Verdict, stated plainly: chain-length-2's brokenness is real but is NOT primarily fixable
by more bridge-logic changes.** Proceeding to step 2: audit every other guard/rule in
`stgt_bridge.py` the same way the dispersed_converging guard was audited.

**Step 2: full guard/rule audit.** `scripts/phase0_full_guard_audit.py --n 1000`, same
seed=999 population. For each boolean guard, fire rate over the 509 pair-eligible
trajectories, and — restricted to trajectories where the guard is the SOLE reason
`bucket != A` — what fraction of those sole-firings are spurious (the structural `rules_key`,
computed before guard checks and available regardless of bucket, already equals ground
truth):

| guard | fire rate | sole-firing spurious rate | failures attributable |
|---|---|---|---|
| oov_name | 8.3% (42/509) | **69.0% (20/29)** | 20 |
| dominant_history_contradiction | 3.5% (18/509) | **100.0% (4/4)** | 4 |
| low_confidence | 2.2% (11/509) | 25.0% (1/4) | 1 |
| dispersed_converging_ambiguity (post-fix) | 2.2% (11/509) | 50.0% (2/4) | 2 |

**Two more guards of the exact same defective class as the original dispersed_converging bug.**
`dominant_history_contradiction` fires on a raw predicted-window-count tie, which says nothing
about genuine ambiguity — a clean 2/2 split ties as easily as a real coin-flip — and blocks a
correct answer 100% of the time it's the sole blocker (n=4). `oov_name` is the highest-volume
actionable defect found this session: 8.3% fire rate, 69.0% spurious-when-sole, 20 correct
answers blocked. Window-level ground-truth check on the 68 unknown windows behind those
firings: 57.4% are `spurious_misclassification` (STGT wrongly read `"transitioning"` on a
window whose true label was a real, settled formation), 42.6% `genuine_transitioning`. The
guard reacts to real classifier noise more than half the time but overreacts to it with a
zero-tolerance blanket block regardless of how much signal the trajectory's OTHER windows
still carry.

Also audited two non-guard mechanisms. `key_windows` capping (`DEFAULT_MAX_KEY_WINDOWS=10`):
**no bug** — 373/1000 trajectories get capped, 0/373 lose a true endpoint formation from the
narrative. `robust=True`'s leading/trailing transitioning-run trim (sec AG): **a real,
significant cost** — fires on 19.8% of pair-eligible trajectories, and 62.5% (105/168) of
trimmed windows have a true label that was NOT `"transitioning"`, i.e. the trim discards
genuine signal more often than real noise. This directly explains why `robust=True` precision
plateaus at 62.4% even after the guard fix (step 2 of the prior session) — the majority-vote
algorithm's own trim step carries the same "assume any unknown-run means genuine ambiguity"
defect, just applied before the vote instead of as a guard after it.

**Audit only this step, per scope — no code changes.** Full detail: `docs/CEILING.md`'s
2026-08-09 step 4 update. Proceeding to step 3: sweep the `robust=True` majority-vote
threshold, tuned on a dev split, to find its actual coverage/precision operating point.

**Step 3: robust-reduction threshold sweep.** `scripts/phase0_robust_threshold_sweep.py`.
`DEFAULT_ROBUST_THRESHOLD=0.7` was tuned once, before this session's guard fix and guard
audit changed the underlying signal the threshold operates on — re-swept against the CURRENT
pipeline, tuned on a dev split ONLY (seed=1, disjoint from the seed=999 held-out population),
confirmed on held-out afterward:

| threshold | dev coverage | dev precision | held-out coverage | held-out precision |
|---|---|---|---|---|
| 0.45-0.50 | 84.8% | 60.4% | 83.5% | 60.5% |
| 0.55-0.65 | 81.1% | **63.1% (best)** | — | — |
| 0.70-1.00 (current default) | 77.9% | 62.6% | 77.8% | 62.4% |

Dev-vs-held-out precision gap at the tested points: 0.1pt — not overfit, essentially perfect
generalization. **The precision curve is flat across the entire sweep (60.4-63.1%, a 2.7-point
range from threshold 0.45 to 1.00)** — this itself confirms step 2's finding: the trim step
(not the vote threshold) dominates contamination, and trimming happens BEFORE the vote, so no
amount of threshold tuning can fix what the vote never gets a chance to see.

**Recommended operating point: 0.55-0.65 (e.g. 0.6) — it Pareto-dominates the current shipped
0.7** (coverage 81.1% vs 77.9%, precision 63.1% vs 62.6%, both strictly better; a free
improvement, not a tradeoff). Going lower (0.45) buys more coverage (84.8%, +6.9pt over
current) at a real precision cost (60.4%, -2.2pt) — legitimate if coverage is valued over
contamination, but not free the way 0.6 is. **Tradeoff stated explicitly: none of the tested
operating points get wrong-key contamination meaningfully below ~37% of everything reaching
Layer 1** — the threshold is a minor lever on top of the much larger, already-identified trim-
step problem. Recommendation only, per scope: `DEFAULT_ROBUST_THRESHOLD` left at 0.7 in code.

Proceeding to step 4: re-examine HALT GATE 1 and project end-to-end threat accuracy given how
far pair-level and threat ceilings have now diverged.

**Step 4: end-to-end threat accuracy projection, and the HALT GATE 1 re-examination.**
`scripts/phase0_endtoend_projection.py`. Key identity: `phase0_threat_ceiling.py`'s reported
"threat ceiling" (52.3%/58.7%) is ALREADY `P(bucket A) × threat_accuracy_within_bucket_A` —
bucket B/C score 0 there, no LLM layer runs in the ceiling scripts. The only missing term for
a genuine end-to-end number is Layer 3's real contribution on bucket C.

| | robust=False (shipped) | robust=True @ 0.7 |
|---|---|---|
| bucket A / B / C | 57.8% / 12.0% / 30.3% | 77.8% / 10.6% / 11.6% |
| threat accuracy WITHIN bucket A | **90.5%** (vs 81.3% pair-accuracy-within-A — RULES' many-to-one mapping, confirmed directly) | 75.5% |
| current threat ceiling | 52.3% | 58.7% |
| end-to-end @ Layer3=20%/30.9%/40% | 58.3% / **61.6%** / 64.4% | 61.1% / **62.3%** / 63.4% |

Layer 3's 30.9% is a disclosed estimate (v3b-fix, sec AF's real-STGT-output eval, different
checkpoint/bridge state, not bucket-conditioned) — reported with a sensitivity band, not
treated as precise. Central projection: **~61.6-62.3% end-to-end, essentially the same under
either reduction mode** (robust=True trades bucket-A quality for bucket-A coverage and the two
roughly cancel), but robust=True's band is narrower (2.3pt vs 6.1pt) since it depends on the
uncertain Layer-3 number for a much smaller share of the population.

**HALT GATE 1 verdict: even the most optimistic tested projection (64.4%) does not clear 70%.**
But the actual question this step was asked to settle — is 70% stated in PAIR-LEVEL terms
still the right gate — has a clear answer: no. It measures the wrong quantity now that pair-
level and threat-level have diverged, and will keep diverging as pair-level-specific fixes
land without threat-level moving 1:1 (RULES' structure guarantees this). **Recommendation:
restate the gate in end-to-end threat-accuracy terms.** The numeric floor itself (keep 70%,
or set something else) is a policy decision this projection informs but does not make — the
central estimate (~62%) falls short of 70% under either metric, so nothing here argues the
gate should simply be declared cleared by relabeling it.

`docs/V5_STATE.json` updated (`phase=0, step=15`). Stopping here per this turn's explicit
"STOP after step 4" instruction — no further strategy or code change without direction.

## 2026-08-09: step 0 of the next turn — discipline catch, not a result

**Verified first, not assumed:** `grep DEFAULT_ROBUST_THRESHOLD src/swarm_intent/stgt_bridge.py`
confirms it is still `0.7` in code. The step-5 recommendation (0.6) was never applied —
nothing to revert in the shipped default itself.

**The process concern is real regardless, and is recorded as a discipline catch, not
walked back as a wrong number.** Step 5's sweep selected the threshold using ONLY the dev
split (seed=1) and used the seed=999 ceiling battery strictly for a post-hoc confirmation
check — that specific selection step did not read eval-set metrics. But seed=1 had ALREADY
been used once before, in the earlier engagement (sec AG), to make a threshold-tuning
decision — reusing the same nominally-"held-out" split for a SECOND independent round of
threshold selection is exactly the kind of repeated-reuse that erodes a dev split's validity
over time, even when any single round's selection step is clean in isolation. Treating
seed=1 as an evergreen, reusable "the dev split" rather than retiring it after its first use
is the actual discipline lapse — not a specific number being wrong, a norm being loose.
**Standing rule going forward: a dev/mining split is used for tuning ONCE, then retired; a
fresh, disjoint seed is cut for each new tuning question.** Proceeding to step 1: cut a
proper mining split and re-run the threshold sweep there, never touching seed=1 or seed=999
for selection again.

**Step 1: fresh mining split, re-run.** `scripts/phase0_mining_split_sweep.py`,
`MINING_SEED=2024` — disjoint from training (42), the standing ceiling battery (999), and
the now-retired dev split (1). Single-use: spent after this sweep, not to be reused for
future tuning decisions either.

| threshold | mining coverage | mining precision |
|---|---|---|
| 0.45-0.50 | 83.3% | 60.0% |
| 0.55-0.65 | 73.4% | 62.2% |
| **0.70-1.00 (current shipped default)** | 70.2% | **63.3% (best in sweep)** |

**The result reverses the prior (now-retired) sweep's conclusion. On this fresh split, the
current shipped default (0.7) has the HIGHEST precision of the entire sweep, not the lowest.**
The selection rule (lowest threshold within 3pt of max precision) picks 0.55 — but 0.55 has
BOTH lower precision (62.2% vs 63.3%) AND, once confirmed on held-out, does not beat 0.7 either
(held-out: 0.55 → 78.4% coverage / 61.9% precision vs 0.7 → 77.8% coverage / 62.4% precision —
0.55 trades precision for coverage, not a free win). **`dominates_current_default: False`,
computed and checked directly, not eyeballed.**

**Per the explicit instruction ("if 0.6 still dominates, we ship it legitimately"), the
converse applies: it does not dominate on a properly single-use split, so `DEFAULT_ROBUST_
THRESHOLD` stays at 0.7. Not changed.** This is exactly the outcome the discipline catch in
step 0 exists to protect against: the earlier "0.6 is a free improvement" conclusion was an
artefact of testing against a dev split reused past its first legitimate use, not a robust
property of the pipeline. Good process caught a real difference in outcome, not just a
theoretical risk.

## 2026-08-09: step 2a — fix `oov_name` (isolated commit)

**Current triggering condition** (`coverage.py`, `classify_observation`, before this fix):
```python
if summary["n_unknown_windows"] > 0 and not robust_recovered:
    guard_reasons.append("oov_name")
```
**What it CLAIMS to test** (per its own name): a genuinely out-of-vocabulary formation
name appeared — a real data-integrity concern, the kind of signal that should make a
downstream consumer distrust the read. **What it ACTUALLY tests**: `n_unknown_windows`
(`stgt_bridge.py`) counts BOTH a genuine OOV name AND the classifier's own valid
`"transitioning"` class under the same `UNKNOWN_FORMATION` sentinel — `_validate_formation`
maps anything outside `BASE_FORMATIONS` (which includes `"transitioning"`, a real, expected,
trained class) to the identical sentinel. The guard fires on either indiscriminately. Given
the model has exactly 8 output classes (7 base + `"transitioning"`), a genuinely OOV string
is structurally impossible from THIS classifier at all — every one of the 8.3% firings
measured in the full guard audit was, by construction, a `"transitioning"` read, not a data-
integrity issue.

**Fix**: `stgt_bridge.py` now computes `n_genuinely_oov_windows` separately —
`formation_type not in BASE_FORMATIONS and formation_type != TRANSITION_CLASS` — and the
guard in `coverage.py` now checks that field instead of the broader `n_unknown_windows`
(which keeps its existing, broader meaning for narrative-text purposes, unchanged).
`n_unknown_windows` is unaffected; existing tests asserting its value pass unchanged.

**Regression tests** (`tests/test_stgt_bridge.py`, `tests/test_coverage.py`): confirmed
`n_genuinely_oov_windows == 0` for a `"transitioning"` blip (was miscounted as 1 before this
fix) and `== 1` for a real unrecognized name; confirmed a `"transitioning"`-blip trajectory
that structurally reduces cleanly now reaches bucket A (was incorrectly guarded to bucket B);
confirmed a genuinely-OOV-named blip is still correctly guarded to bucket B, unchanged. 137/137
tests pass.

**Re-measured pair + threat ceiling, same seed=999 population, only this fix changed:**

| | robust=False (shipped) | robust=True @ 0.7 |
|---|---|---|
| bucket A | 294 → **323** | 396 → **420** |
| pair-level accuracy | 47.0% → **50.9%** | 48.5% → **52.1%** |
| threat ceiling | 52.3% → **57.2%** | 58.7% → **63.1%** |

Proceeding to step 2b: fix `dominant_history_contradiction`, isolated commit, same protocol.

## 2026-08-09: step 2b — fix `dominant_history_contradiction` (isolated commit)

**Current triggering condition** (`coverage.py`, before this fix):
```python
counts = [p["formation_type"] for p in predictions if p["formation_type"] in BASE_FORMATIONS]
if counts.count(key[0]) == counts.count(key[1]):
    guard_reasons.append("dominant_history_contradiction")
```
**What it CLAIMS to test** (per its name): whether `summary["dominant_formation"]` actually
CONTRADICTS the derived `(from, to)` key — a genuine reason to distrust which formation is
"dominant". **What it ACTUALLY tests**: a raw window-COUNT tie between `key[0]` and `key[1]`.
`key` is derived from TEMPORAL ORDER (the first/last distinct formation in
`formation_history`), never from window counts — a clean, obviously-correct 2/2 split ties
exactly as easily as a genuinely uncertain one. The two quantities are unrelated; the guard
was never testing what its name claims. Audited: 100% spurious when this was the sole
blocker (n=4).

**Fix**: test the actual claim — does `summary["dominant_formation"]` differ from BOTH
members of `key`? **Proof this is now correctly unreachable, not silently broken**: by the
point this guard runs, `known_history` already has ≤2 distinct real formations (the
`len(known_history)>=3` check earlier already routes 3+ to bucket C), and
`dominant_formation` is the mode over the SAME `valid_formations` set `known_history` is
built from — so `dominant_formation ∈ set(known_history) == {key[0], key[1]}` always. This
condition can never fire via `classify_observation`'s own code path. Kept as a defensive
check (matches the existing `no_rules_key` pattern) rather than deleted, in case a future
change to the upstream invariants makes it reachable again — not hidden, documented in the
code with the proof sketch above.

**Regression test**: the exact previously-spurious shape (a 2/2 window-count tie) now
resolves to bucket A with the guard absent, replacing the old assertion that it should
guard. A genuine dominant/key contradiction cannot be hand-built through the public
function (proven above), so — unlike step 2a's OOV case — there is no "still fires
correctly" counterpart test to add; its absence is the correct, audited outcome, documented
in the test's own docstring so a future reader doesn't mistake it for a gap. 137/137 tests
pass.

**Re-measured pair + threat ceiling, same seed=999 population, only this fix changed
(cumulative with step 2a):**

| | robust=False (shipped) | robust=True @ 0.7 |
|---|---|---|
| bucket A | 323 → **339** | 420 → **431** |
| pair-level accuracy | 50.9% → **53.6%** | 52.1% → **54.0%** |
| threat ceiling | 57.2% → **60.3%** | 63.1% → **65.2%** |

`robust=True`'s threat ceiling (65.2%) is now close to the 70% floor — still short, but the
gap has closed substantially across steps 2a+2b combined (52.3% → 65.2%, +12.9pt, from two
guard fixes and zero retraining). Proceeding to step 3: fix the `robust=True` trim step,
the last of the three defects the full guard audit found.

## 2026-08-09: step 3 — fix the `robust=True` trim step (isolated commit)

**Current behaviour**: `_robust_reduce`'s leading/trailing strip loop removed EVERY
`UNKNOWN_FORMATION` edge window unconditionally, trusting each individual "transitioning"
classification at face value. Audited: 62.5% (105/168) of trimmed windows had a true label
that was NOT `"transitioning"` — the classifier simply misclassified a real, settled window,
and the trim step compounded that error by discarding it as if it were genuine blend
geometry rather than noise.

**Fix**: only strip a `"transitioning"`-classified edge window when the model's OWN
`formation_confidence` for it is `>= TRIM_CONFIDENCE_THRESHOLD` (0.6 — REUSED from the
existing `low_confidence` guard's bar elsewhere in this codebase, not a freshly-tuned value,
so this introduces no new parameter needing its own mining-split tuning question). A
genuine out-of-vocabulary name (never `"transitioning"` itself) is still stripped regardless
of confidence — it was never trustworthy for voting purposes either way. An untrusted
low-confidence window is left in place as `UNKNOWN_FORMATION`, where `modal()`'s existing
logic already excludes it from voting and correctly counts it against that half's plurality
fraction — the safe fallback (a case that no longer resolves), never a confidently wrong one.

**Regression tests** (`tests/test_stgt_bridge.py`, new `TestRobustReductionTrim`): a
confident trailing `"transitioning"` window is still stripped (unchanged behaviour);
a low-confidence one is now left unstripped; a genuine OOV edge window is still stripped
regardless of confidence. 140/140 tests pass.

**Re-measured, same seed=999 population.** `robust=False` is untouched by this fix (it never
calls `_robust_reduce`) — its numbers are unchanged from step 2b (53.6% pair / 60.3% threat),
a useful consistency check. `robust=True`:

| | before this fix (2a+2b only) | after this fix |
|---|---|---|
| bucket A | 431 | **418** (-13) |
| pair-level accuracy (of 509) | 54.0% | 54.0% (unchanged count, smaller base) |
| threat ceiling | 65.2% | 64.6% (-0.6pt) |
| **precision WITHIN bucket A** | **63.8% (275/431)** | **65.8% (275/418)** |

**The 13 trajectories the fix removes from bucket A were ALL wrong before (275 correct stays
exactly 275) — the fix precisely targets and excludes over-trusted wrong recoveries without
losing a single correct one.** The headline "62.4%" this whole exercise started from (before
ANY of this session's three fixes) is now **65.8%** — a real, if modest, improvement in the
metric that actually matters for safety (how often a "guaranteed" Layer-1 answer is right).
The overall threat-ceiling number moves slightly the OTHER way (65.2%→64.6%) because this
ceiling script scores bucket C as "no recovery = wrong" — it doesn't credit Layer 3 for
picking up the newly-excluded cases, so a correct trade (fewer confident wrong answers, more
honest abstentions) reads as a flat-to-slightly-down number in a metric that was never
designed to reward that trade. This is expected and does not indicate a regression; it's the
`current_ceiling` term from sec AH's end-to-end identity losing a little while the
`precision × P(bucket C)` term (Layer 3's opportunity) gains — a fuller accounting needs a
fresh end-to-end projection, not attempted this turn (out of the 4 steps this turn scoped).

**Session-cumulative result across all three fixes (2a + 2b + 3), robust=True:**

| | session start | after all 3 fixes |
|---|---|---|
| bucket A | 396 | **418** |
| threat ceiling | 58.7% | **64.6%** |
| precision within bucket A | 62.4% | **65.8%** |

Proceeding to step 4: report on RULES coverage for chain-3+ (report only, no RULES
extension).

## 2026-08-09: step 4 — RULES coverage for chain-3+ (report only, no action taken)

`scripts/phase0_rules_coverage_report.py`, CPU-only (replays the exact seed=999 rng
consumption of the standing ceiling battery, including a full `build_long_sequence_labeled`
call to stay in sync, but never loads or runs STGT — nothing here depends on model output).
**Explicitly report-only, per instruction: no change to RULES or any other file that affects
behaviour.**

**(a) Exact chain-length distribution, n=1000, seed=999 (identical population to the
standing ceiling battery):**

| chain_length | n | % | theoretical (uniform 1-4) |
|---|---|---|---|
| 1 | 258 | 25.8% | 25.0% |
| 2 | 251 | 25.1% | 25.0% |
| 3 | 234 | 23.4% | 25.0% |
| 4 | 257 | 25.7% | 25.0% |
| **3+ (combined)** | **491** | **49.1%** | 50.0% |

**(b) Generator parameter, not emergent.** `sample_chain()`'s `num_formations =
rng.integers(1, 5)` is a hard-coded bound (1..5, numpy default `endpoint=False` → uniform
over {1,2,3,4}) — an explicit design choice in the generator, not a property that falls out
of anything else. Every subsequent formation in the chain is drawn uniformly from
`BASE_FORMATIONS` minus only the immediately-preceding member (no consecutive repeats
forced; non-consecutive repeats — e.g. A→B→A — ARE legal chains, this is exactly what
produces the `oscillation` bucket-C subtype elsewhere in this project). The measured
distribution (25.8/25.1/23.4/25.7%) matches the theoretical 25% each closely — sampling
noise, not a hidden skew.

**(c) Chain-3+ patterns do NOT collapse to a small set.** 491 chain-3+ trajectories produced
**385 distinct patterns** — 60.7% of them appear exactly once, the single most common pattern
repeats only 7 times, and the theoretical pattern space (7 base formations, no consecutive
repeats: 7×6×6=252 length-3 + 7×6×6×6=1512 length-4 = **1764 possible chains**) is large
enough that 491 draws only sample a small fraction of it. **This directly bears on how big a
future RULES-extension effort (HALT GATE 2) would actually be**: literally enumerating
observed chain-3+ patterns one-by-one is not a small, bounded task the way extending RULES
for a handful of recurring shapes would be — the data doesn't support a "just add these 10
common patterns" shortcut. A viable extension would more likely need a COMPOSITIONAL rule
(e.g. reduce a chain to its structurally-meaningful (first, last) endpoints, or to a small
number of derived features) rather than literal pattern enumeration — a design question for
whoever owns HALT GATE 2's sign-off, not resolved or attempted here.

**No RULES change made. No code behaviour changed.** Full data:
`evaluation/phase0_rules_coverage_report.json`. Proceeding to step 5: rewrite
`docs/CEILING.md` stratified by chain length, pooled numbers banned from here on.

## 2026-08-09: step 5 — CEILING.md rewritten stratified; a fresh re-run made the case for itself

Re-ran `scripts/phase0_chainlength_breakdown.py --n 1000` (same seed=999 population) AFTER
all three of today's fixes to get a fully-current stratified picture:

| chain_length | n | pair acc (F) | pair acc (T) | threat acc (F) | threat acc (T) |
|---|---|---|---|---|---|
| 1 | 258 | 87.6% | 88.4% | 88.0% | 88.8% |
| **2** | 251 | **18.7%** | 18.7% | 31.9% | 39.8% |
| 3+ | 491 | n/a | n/a | n/a | n/a |

**Chain-2 pair accuracy moved 6.0%→18.7% (3x) across today's fixes; chain-1 barely moved
(86.8%→87.6%).** The pooled "509 pair-eligible" number this whole session's earlier commits
reported (47.0%→53.6%) compressed a 3x improvement and a near-zero one into one figure that
reads as a uniform, modest gain. It wasn't. This is precisely the failure mode step 5's
instruction exists to stop: a pooled number hides which stratum actually improved and by
how much.

**Also surfaced, only visible once stratified: chain-3+'s bucket-A false-positive rate rose
from 1.8% to 9.0% (44/491) as a side effect of the guard fixes.** The guards being corrected
(`oov_name`, `dominant_history_contradiction`) used to accidentally catch some chain-3+
false positives despite testing the wrong condition; correctly scoped now, they no longer
incidentally block a chain-3+ trajectory that happens to reduce to a clean-looking ≤2-length
history due to classifier error. A real, disclosed trade a pooled number would never have
shown at all.

**`docs/CEILING.md` restructured**: a policy banner + a new "Current state (stratified)"
section added at the top (supersedes every pooled figure in the file), historical dated
entries below it left UNEDITED for the record (matches this project's standing convention
of correcting via a superseding notice, never rewriting history in place). Pooled numbers
are banned in every CEILING.md entry from this point forward.

Proceeding to step 6: add issue #3 to `UPSTREAM_ISSUES.md` (chain-2's destination
formation frequently never reached within the generated sequence).

## 2026-08-09: step 6 — UPSTREAM_ISSUES.md issue #3

Added issue #3: chain-length-2's destination formation is frequently never observed by any
sliding window at all (distinct from issue #2's "trailing window is transitional" — here NO
window's majority is ever the destination formation), a property of `sample_chain()`'s
segment-length distribution relative to `window_size`, not a bridge-code or model defect.

Post today's three guard/trim fixes, chain-2 pair accuracy is 18.7% (up 3x from 6.0%, but
still the worst stratum by far vs chain-1's 87.6%). Re-traced 20 fresh chain-2 failures:
**95% (19/20) trace to this generation-regime mechanism**, 5% to plain misclassification,
**0% to any remaining bridge-logic defect** — confirms the guard/trim fixes fully resolved
their share; what's left is a data-generation ceiling, not a code bug. Cited two concrete
traced examples (`shield→v_shape` where `v_shape` never appears as any window's majority
label; `diamond→column` where `column` never appears at all).

Requested change: widen `sample_chain()`'s per-hop segment-length distribution so a settled
destination tail reliably exceeds `window_size` — a generator-parameter change, not a
retrain (distinct from issue #2, which does need one) — flagged as the fastest of the three
upstream issues to act on if prioritization is being decided.

`docs/V5_STATE.json` updated (`phase=0, step=23`). **STOP per instruction — all 6 steps of
this turn complete.**

## 2026-08-09: step 24 — chain-2 observability diagnostic: is it the generator, STGT, or the bridge?

Issue #3 (above) asserted chain-2's remaining failure is a generator-observability ceiling, but
had not yet directly measured, per-trajectory, whether the destination formation ever becomes
observable and how accuracy actually varies with observability. This turn does that measurement
directly, no code changes to the generator, bridge, or model (`scripts/phase0_chain2_observability.py`,
`scripts/phase0_chain2_blend_distributions.py`, same seed=999/n=1000 population, same
strategy-5 checkpoint, epoch=10/val_acc=0.868, sha256 `18fc201d...`, as every other Phase 0
measurement in this file).

**Observability criterion (not invented fresh — reuses the exact per-window true-label-majority
computation `phase0_chainlength_breakdown.py`'s trace and `phase0_decompose_failures.py`'s
window scoring already use):** a window's true label is `Counter(true_labels[start:end]).most_common(1)`.
Destination formation B is OBSERVABLE if at least one window's true-label majority is B — i.e.
there exists at least one point in the eval protocol's own window grid where reading B would be
scored correct. If no window's majority is ever B, no classifier can produce a correct B read,
because the protocol never presents B as a window's dominant content.

**Step 1 — observability:**

```
Total chain-2 trajectories: 251
Trajectories where B is observable (>=1 window true-majority-B): 124 (49.4%, 95% CI [43.3%, 55.5%])
Trajectories where B is never observable: 127 (50.6%)
Breakdown: OBS_CLEAR (>=2 majority-B windows)=28, OBS_PARTIAL (=1)=96, OBS_NONE (0)=127
```

**Step 2 — accuracy by observability group** (robust=False / robust=True, same as every other
chain-length table in this file):

| group | n | pair_acc (F) | pair_acc (T) | threat_acc (F) | threat_acc (T) |
|---|---|---|---|---|---|
| OBS_CLEAR | 28 | 57.1% | 57.1% | 64.3% | 64.3% |
| OBS_PARTIAL | 96 | 19.8% | 19.8% | 33.3% | 37.5% |
| OBS_NONE | 127 | 9.4% | 9.4% | 23.6% | 36.2% |

Observable (pooled CLEAR+PARTIAL, n=124) vs unobservable (n=127): **28.2% vs 9.4%, a +18.8pt
gap.** Observability clearly matters — a strong, monotonic gradient (57.1% → 19.8% → 9.4%) — but
it is not the whole story: **OBS_CLEAR still fails 42.9% of the time even with a redundant,
unambiguous destination signal**, and OBS_NONE's non-zero 9.4% is the classifier getting lucky
(misreading some other window as B by chance, landing on the right pair despite B genuinely
never appearing).

**Step 3 — 20-case trace, manually reviewed** (not just the programmatic first pass; the
programmatic heuristic in the script mislabeled several cases as "bridge/reduction issue" when
manual review of the full window table showed the true mechanism was a SOURCE-formation (not
destination) misclassification — corrected below, full trace in
`evaluation/phase0_chain2_observability_trace.txt`):

| root cause (manually confirmed) | n | % of 20 |
|---|---|---|
| 1. Destination not observable | 8 | 40% |
| 3. STGT misclassification | 11 | 55% |
| 2. Transition labeling issue (thin plurality, e.g. 38/36/26 split) | 1 | 5% |
| 4. Bridge/reduction issue | 0 | 0% |
| 5. Evaluation/ground-truth issue | 0 | 0% |

**Zero genuine bridge-logic defects found in this trace** — consistent with the guard/trim
fixes from earlier this session having already cleared that category out. Of the 11
misclassification cases, two distinct mechanisms recur, both already partially documented
elsewhere in this file: (a) the model over-predicts `"transitioning"` on a window where the
true blend content is a MINORITY (e.g. trajectory 154: `frac_transitioning=0.26` but predicted
`transitioning` at 98% confidence) — the same near-blend-boundary false-positive pattern
strategy 5 measured at 53.2% FP rate; (b) confident (non-near-tie) `dispersed`/`converging`
SOURCE misclassification away from any blend region (e.g. trajectory 40: true `dispersed`,
predicted `converging` at 96% confidence, nowhere near a formation change) — a form of the
dispersed/converging confusion the `dispersed_converging_ambiguity` guard cannot catch, because
that guard only fires on a close top-2 tie, not a confident wrong single-class call. Both are
real, independent STGT recognition-quality problems, not generator or bridge issues.

**Step 4 — train vs eval blend-timing distributions** (`scripts/phase0_chain2_blend_distributions.py`,
pure Monte Carlo, no GPU, 20000 draws each — mirrors the exact formulas in `generate_dataset()`
and `build_long_sequence()`):

| | start_frac | duration_frac |
|---|---|---|
| train regime 0 (labeled pure A) | [0.740, 0.880] | 0.100 (fixed) |
| train regime 1 (labeled "transitioning") | [0.120, 0.280] | [0.440, 0.600] |
| train regime 2 (labeled pure B) | [0.020, 0.140] | [0.080, 0.100] |
| **eval per-hop (chain-2+ trajectories)** | **[0.283, 0.495]** | **[0.051, 0.458]** |

**Fraction of eval blends whose (start_frac, duration_frac) falls inside ANY training regime's
realized box: 0/20000 (0.0%).** Eval's blend start timing (28-50% into the segment) sits in a
gap between train regime 2's end (14%) and train regime 1's start (12-28%, barely touching the
very edge) and train regime 0's start (74%+) — a start-timing zone `generate_dataset()` never
produces a labeled example for, at any duration. This is a genuine, quantified train/eval
distribution mismatch, and a plausible mechanistic explanation for WHY the classifier
misbehaves specifically near mid-segment blends (mechanism (a) above): it has literally never
seen a training example shaped like eval's typical blend.

**Step 5 — bottleneck, quantified, not forced into one category:**

- **~50.6% of chain-2 failures are capped by generator observability (Case 1)** — no
  bridge/model fix, however good, can raise pair accuracy on these without a generator change.
  This alone puts a hard ceiling of 49.4% on chain-2 pair accuracy under the CURRENT generator.
- **Of the 124 observable trajectories, 89 (71.8%) still fail.** Extrapolating the manually-
  reviewed 20-case trace's within-observable split (11 STGT-misclassification : 1 labeling-
  ambiguity : 0 bridge, among the 12 non-Case-1 traced cases) onto the full 89: roughly
  **~32-33% of the total 251-trajectory population fails due to genuine STGT recognition
  quality (Case 2)**, and a small residual (~3%) to thin-plurality ground-truth labeling
  ambiguity in the observability criterion itself (arguably a mild eval-protocol sharpness
  issue, Case 4-adjacent, not a generator or model defect).
- **0% Case 3 (bridge/reduction)** in this trace — this session's earlier guard/trim fixes
  appear to have fully cleared that category for chain-2.

**Not single-cause.** Roughly half the ceiling is a generator-observability problem, roughly a
third is an independent STGT recognition-quality problem (concentrated in two specific,
partially-already-documented mechanisms), and a small residual is eval-protocol labeling
sharpness. Fixing the generator alone would raise chain-2's theoretical ceiling from 49.4% but
NOT close the gap to anywhere near 100%, because ~43% of even the best-observed group
(OBS_CLEAR) still fails on STGT recognition grounds alone.

**Minimal generator-correction spec (Case 1 only, NOT implemented this turn — spec only, per
instruction):** to guarantee at least one window with a true B-majority, the post-blend
"settled destination dwell" `D = seg_len - blend_end` must satisfy `D >= window_size/2 + margin`
(≈26 timesteps for `window_size=50`, plus a stride-granularity/noise safety margin, say
`D_min ≈ 30`). Under the CURRENT distribution (`seg_len ~ U{50,100}`, `blend_end_frac ~
U(0.55,0.75)`), worst case (`seg_len=50, blend_end_frac=0.75`) gives `D=12.5` — always short of
30. Even best case (`seg_len=100, blend_end_frac=0.55`) gives `D=45`, comfortably enough, but
the distribution's lower tail dominates the failures. A minimal, targeted fix: either (a) raise
the segment-length floor (e.g. `seg_len ~ U{90,140}`) so even worst-case `blend_end_frac`
leaves `D>=30`, or (b) decouple dwell from segment length directly — sample `blend_end` as
`seg_len - D` with `D ~ U(30,50)` fixed, and derive `blend_start` backward from a
separately-sampled blend duration, so dwell time is guaranteed regardless of `seg_len`. Option
(b) is more surgical (doesn't inflate average trajectory length) and is the one requested,
informally, by `UPSTREAM_ISSUES.md` issue #3's "make the minimum segment length a function of
`window_size` plus a fixed settled-tail requirement." Not implemented — this is `build_long_sequence`'s
EVAL-harness sampling only (`llm_finetuning/measure_coverage.py` / `phase0_decompose_failures.py`'s
copy), not `generate_dataset()`'s training-data generator, so no retrain would even be required
to apply it — but per this turn's explicit instruction, spec only, no code change.

**Decision gate: A + B jointly, A first.** Fixing generator observability (Case 1, ~50.6%,
cheap, no-retrain, eval-harness-only parameter change) is the correct first move — it is the
single largest, cheapest, most mechanically clear-cut share of the problem, and the
minimal-fix spec above requires no STGT retrain at all. But it is **not sufficient on its
own**: the ~32-33% STGT-recognition-quality share (Case 2, concentrated in near-blend-boundary
`"transitioning"` over-prediction and confident-wrong `dispersed`/`converging` source
misclassification away from blends) is a real, independent, already-partially-diagnosed
problem (both mechanisms trace to phenomena `docs/CEILING.md` strategy 5 and the
`dispersed_converging_ambiguity` guard-audit already found) that a generator fix will not
touch. **Recommended next experiment, not run this turn:** re-run this same observability
audit AFTER the generator fix (spec above) lands, to (1) confirm the Case-1 share drops as
predicted and (2) get a cleaner, larger OBS_CLEAR population to re-measure the Case-2 STGT
recognition-quality share in isolation, before deciding whether that needs its own targeted
fix (e.g. re-examining the `"transitioning"` decision boundary specifically in the
train/eval-blend-mismatch zone quantified in step 4).

Full data: `evaluation/phase0_chain2_observability.json`,
`evaluation/phase0_chain2_observability_trace.txt`. `docs/V5_STATE.json` updated
(`phase=0, step=24`). **STOP per instruction — diagnosis and recommendation only, no
generator/model/LLM code changed this turn.**

## 2026-08-09: step 25 — minimal chain-2 dwell-time generator fix, implemented and re-measured

Step 24's diagnosis is unchanged and left intact above. This turn implements ONLY the
generator fix it specified, no STGT/bridge/LLM changes, same frozen checkpoint
(strategy 5, epoch=10, val_acc=0.868, sha256 `18fc201d...`) as every measurement in this file.

**Mechanism (re-derived precisely before touching code):** a window's true label is a
majority vote over 50 timesteps, so destination B needs >=26 B-labeled timesteps in some
window to ever win. But `sliding_window_inference`'s stride=10 grid can leave up to 9
trailing timesteps outside EVERY window (confirmed directly: `seg_len=59` gives a last window
`[0,50)` that misses timesteps 50-58 entirely). So the real requirement is post-blend dwell
`D = seg_len - blend_end >= 26 + 9 = 35`, not just 26. The OLD formula sampled `seg_len ~
U{50,100}` and `blend_end` as a FRACTION of `seg_len` (`~U(0.55,0.75)`) independently, so `D`
was an unconstrained byproduct ranging from a worst case of 12.5 to a best case of 45 -- the
lower tail of that joint distribution is exactly what produced step 24's ~50% `OBS_NONE`.

**Fix, implemented ONLY in `scripts/phase0_decompose_failures.py`'s
`build_long_sequence_labeled`** (the copy this observability audit actually exercises via
`phase0_chainlength_breakdown.py`/`phase0_chain2_observability.py`; the 4 other verbatim
duplicates of this sampling logic --
`llm_finetuning/measure_coverage.py`, `scripts/phase0_ceiling.py`,
`scripts/phase0_guard_audit.py`, `llm_finetuning/analyze_bucket_c_windowing.py` -- were
deliberately left untouched so HALT GATE 1's pooled ceiling and sec AE/AF/AG's pipeline_v2
coverage numbers stay reproducible against the OLD formula; propagating this fix to them is a
disclosed follow-up decision, not made here):

```
OLD: seg_len ~ Uniform{50,100}                         (sampled first)
     blend_start = seg_len * Uniform(0.3, 0.5)           (fraction of seg_len)
     blend_end   = seg_len * Uniform(0.55, 0.75)          (fraction of seg_len)
     -> dwell D = seg_len - blend_end is an UNCONSTRAINED BYPRODUCT (range 12.5-45)

NEW: lead_in         ~ Uniform{15,35}   (timesteps of settled formation_a)
     blend_duration  ~ Uniform{10,25}   (timesteps the blend spans -- transition physics
                                          untouched, generate_transition_sequence's cosine
                                          ramp is unchanged, only WHERE it's placed changed)
     dwell           ~ Uniform{40,60}   (timesteps of settled formation_b -- THE FIX,
                                          directly guarantees D >= 40 >= the derived 35 min)
     blend_start = lead_in; blend_end = lead_in + blend_duration
     seg_len = blend_end + dwell        (DERIVED, no longer independently sampled)
```

Reason: dwell was the actual controlling quantity all along; sampling it directly and
deriving `seg_len` from it (instead of the reverse) is the minimal change that fixes the
mechanism rather than the symptom. `lead_in`/`blend_duration` ranges were chosen to match the
OLD formula's REALIZED scale (old `blend_start` realized ~15-50 timesteps, old blend
duration realized mean ~18.8 timesteps --
`scripts/phase0_chain2_blend_distributions.py`) so source-formation duration and transition
speed are preserved, not just observability. Expected effect: `seg_len` mean rises modestly
(~75 -> ~92.5 timesteps, +23%) as a DISCLOSED SIDE EFFECT of guaranteeing dwell, not a
blanket "make trajectories longer" choice.

**Step 4 — numerically verified before running STGT** (`sample_chain`/
`build_long_sequence_labeled` only, no model, seed=999, n=1000):

| | before | after |
|---|---|---|
| chain-2 trajectories | 251 | 233 (population composition shifts slightly -- same seed, but the rng draw pattern per hop changed from 3 uniform/integer calls to 3 integer calls, so which later indices land on chain-2 vs chain-1/3+ shifts; disclosed, not a bug) |
| seg_len (n_timesteps) | mean ~75, range 50-100 | mean 91.4, range 70-114 |
| blend duration | mean ~18.8 | mean 16.1, range 9-23 |
| n_windows per trajectory | — | mean 4.7, range 3-7 |
| OBS_CLEAR | 28 (11.2%) | **231 (99.1%)** |
| OBS_PARTIAL | 96 (38.2%) | 2 (0.9%) |
| OBS_NONE | 127 (50.6%) | **0 (0.0%)** |

Target ("OBS_NONE should become a small minority rather than ~50%") not just met but
exceeded -- OBS_NONE is now exactly 0%. Proceeded to full STGT re-evaluation.

**Step 5 — observability audit re-run** (`scripts/phase0_chain2_observability.py --n 1000`,
same seed=999/checkpoint, GPU inference, ~9s):

| group | n (before -> after) | pair_acc before | pair_acc after | threat_acc before | threat_acc after |
|---|---|---|---|---|---|
| OBS_CLEAR | 28 -> 231 | 57.1% | 39.8% | 64.3% | 71.9% |
| OBS_PARTIAL | 96 -> 2 | 19.8% | 50.0% (n=2, not meaningful) | 33.3% | 100.0% (n=2) |
| OBS_NONE | 127 -> 0 | 9.4% | n/a (n=0) | 23.6% | n/a |

```
Observable:   49.4% -> 100.0%
OBS_CLEAR:    11.2% (n=28)  -> 99.1% (n=231)
OBS_PARTIAL:  38.2% (n=96)  -> 0.9%  (n=2)
OBS_NONE:     50.6% (n=127) -> 0.0%  (n=0)
```

**A counterintuitive but explainable result: OBS_CLEAR's own pair accuracy went DOWN
(57.1%->39.8%) even though observability went up.** This is a population-composition effect,
not a regression: the OLD OBS_CLEAR (n=28) was a rare, favorable subset -- trajectories that
happened to get a long segment with an early blend under the old unconstrained formula, i.e.
systematically easier cases. The NEW OBS_CLEAR (n=231) is essentially the WHOLE chain-2
population, including every formerly-OBS_NONE trajectory that is generically no easier in
other respects (noise, spread, formation pair difficulty). Comparing group-conditional
accuracy across a fix that changes the group's membership that drastically is not
apples-to-apples; the POOLED number is the one that matters and it clearly improved.

**Step 6 — chain-1/2/3+ breakdown** (`scripts/phase0_chainlength_breakdown.py --n 1000`,
same seed/checkpoint, no retraining):

| chain_length | n | pair_acc (F) | pair_acc (T) | threat_acc (F) | threat_acc (T) |
|---|---|---|---|---|---|
| 1 (steady state) | 252 | 84.1% | 84.1% | 85.7% | 85.7% |
| **2 (single transition)** | 233 | **39.9%** | 39.9% | **72.1%** | 72.1% |
| 3+ (no RULES key) | 515 | n/a | n/a | n/a | n/a |

**Chain-2 pair accuracy 18.7% -> 39.9% (+21.2pt, more than doubled) with the SAME frozen
checkpoint, zero retraining.** Chain-1 is flat within noise (87.6%->84.1%, small population
shift, not a regression -- chain-1's own generation branch is untouched). Chain-3+ bucket-A
false-positive rate ticked up 9.0%->12.6% (65/515) -- a modest, disclosed side effect
(chain-3+ trajectories now average more windows too, per-window noise has slightly more
surface area to spuriously look like a clean <=2-length reduction), not alarming but not
ignored either.

**Step 7 — train/eval blend-distribution overlap re-checked**
(`scripts/phase0_chain2_blend_distributions_v2.py`, same 3 training regimes, Monte Carlo,
no GPU):

| | start_frac | duration_frac |
|---|---|---|
| train regime 0 | [0.740, 0.880] | 0.100 |
| train regime 1 | [0.120, 0.280] | [0.440, 0.600] |
| train regime 2 | [0.020, 0.140] | [0.080, 0.100] |
| eval v1 (before fix) | [0.283, 0.495] | [0.051, 0.458] |
| **eval v2 (after fix)** | **[0.153, 0.405]** | **[0.097, 0.304]** |

**Overlap is STILL 0.0% (0/20000)** -- unchanged from before the fix. The mismatch shape
changed, though: eval v2's blend START timing (15-40%) now genuinely reaches INTO train
regime 1's start range (12-28%) for a large share of draws, but regime 1 additionally
requires `duration_frac` in [0.44,0.60] (blend must DOMINATE the window) and eval v2's
duration (mean 0.186, max 0.304) never gets remotely close -- so the two conditions still
never jointly hold. **Flagged explicitly, not silently fixed, per instruction: this remains
a real, unresolved train/eval distribution gap** and is a plausible contributing mechanism to
step 8's dominant STGT-misclassification finding below.

**Step 8 — 20 new failures, manually traced and classified**
(`evaluation/phase0_chain2_observability_trace.txt`, freshly regenerated; the script's
programmatic first-pass categorizer over-used "bridge/reduction issue" again -- 19/20 -- for
the same reason as step 24's trace: it only checks B-window correctness, not A. Every one of
the 20 was manually re-derived from the printed window table.):

| root cause (manually confirmed) | n | % of 20 | vs. step 24 (before fix) |
|---|---|---|---|
| 1. Destination not observable | 0 | 0% | 40% |
| **1b. Source (A) not observable -- NEW, self-inflicted** | **3** | **15%** | 0% (mechanism didn't exist before) |
| 2. Transition labeling issue (near-exact-tie window) | 1 | 5% | 5% |
| **3. STGT misclassification** | **16** | **80%** | 55% |
| 4. Bridge/reduction issue | 0 | 0% | 0% |
| 5. Evaluation/ground-truth issue | 0 | 0% | 0% |

**The fix fully removed its intended target (destination-not-observable: 40%->0%) and
genuinely improved chain-2 pair accuracy (18.7%->39.9%). But it introduced a SYMMETRIC,
self-inflicted problem it wasn't designed to prevent: `LEAD_IN_RANGE=(15,35)` was chosen to
match the OLD formula's realized SCALE, not derived with the same `D>=35` rigor as
`MIN_DWELL_RANGE` -- roughly half its range (15-25) is below the ~25-26 threshold a source
window needs to win a plurality against blend+dwell content in window 0.** Traced examples:
trajectory 3 (`shield->converging`, `lead_in=18`): window 0's true majority is ALREADY
`converging` (21 B-timesteps vs 19 A-timesteps in the first 50), so `shield` never becomes
any window's majority anywhere in the trajectory -- the exact same failure mode step 24 found
for destinations, now hitting the source. This is NOT one of the 6 categories the user
specified (all of which are about the DESTINATION or downstream stages); reported as its own
explicit finding rather than force-fit into "destination not observable" or "other."

**Genuine STGT misclassification is now overwhelmingly the dominant failure mode (80%, up
from 55%)** -- with far more redundant windows per trajectory (mean 4.7 vs fewer before), the
observability confound is mostly gone and what's left is visibly the SAME two mechanisms step
24 already named: near-blend-boundary `"transitioning"` over-prediction on windows where the
true content is a real (if sometimes thin) plurality of a settled formation (e.g. trajectory
76: `v_shape`, `v_shape`, `diamond` -- THREE consecutive confident (83-99%) `"transitioning"`
misreads even though each of those windows has a clean plurality of a real formation), and
confident (non-near-tie) `dispersed`/`converging` confusion away from any blend (trajectory
116: true `dispersed`, predicted `converging` at 88% confidence, nowhere near a formation
change -- the same mechanism step 24 found in trajectory 40/57).

Full data: `evaluation/phase0_chain2_observability.json` (overwritten with the v2/after
result), `evaluation/phase0_chain2_observability_before.json` (v1/before, preserved for
comparison), `evaluation/phase0_chain2_observability_trace.txt` (v2 trace),
`evaluation/phase0_chain2_observability_trace_before.txt` (v1 trace, preserved),
`evaluation/phase0_chainlength_breakdown.json` (overwritten, v2),
`evaluation/phase0_chainlength_breakdown_before.json` (v1, preserved).

**Decision gate: A -- observability fix successful; remaining problem primarily STGT
recognition, with two disclosed caveats.** The stated success criterion (destination
reliably observable) is met outright: `OBS_NONE` 50.6%->0.0%, chain-2 pair accuracy more than
doubled (18.7%->39.9%) with zero retraining. The dominant remaining failure mode (80% of
traced cases) is genuine STGT misclassification, not any form of non-observability of the
original target. Two caveats, both disclosed rather than hidden: (1) a small (~15%),
easily-fixable, self-inflicted SOURCE-observability gap remains because `LEAD_IN_RANGE`
wasn't derived with the same `D>=35` rigor as `MIN_DWELL_RANGE` -- the immediate, trivial
next step, not requiring a new experiment; (2) the train/eval blend-timing overlap is STILL
0.0% and is a plausible contributing mechanism to the 80% STGT-misclassification share, so
the "next STGT experiment" (step 8's recommendation) should target that specific mismatch
(training examples whose blend timing matches eval's actual mid-segment, moderate-duration
shape) rather than a generic retrain.

`docs/V5_STATE.json` updated (`phase=0, step=25`). **STOP per instruction -- this experiment
only, no further generator/STGT/bridge/LLM changes this turn.**

## 2026-08-09: step 26 — consolidate the duplicates, symmetrize the source side, re-measure everything

Step 25 shipped the destination-dwell fix in ONE of 5 duplicate copies of this sampling logic
and disclosed the other 4 as out of scope. This turn: (1) consolidates all 5 into a single
canonical module, (2) symmetrizes the fix onto the source side per Claude's own step-25
recommendation, (3) re-measures chain-2 with both fixes, (4) refines the failure taxonomy,
(5) investigates the chain-3+ false-positive movement, (6) re-checks blend overlap. No
STGT retraining, no LLM/RULES/architecture changes. Same frozen strategy-5 checkpoint
throughout.

### Step 1: consolidation table

| # | file:line (pre-consolidation) | (a) real training | (b) eval/audit | (c) neither |
|---|---|---|---|---|
| 1 | `llm_finetuning/measure_coverage.py:93` (`build_long_sequence`) | no | **yes** — sec AE/AF/AG pipeline_v2 coverage measurement | |
| 2 | `scripts/phase0_ceiling.py:64` (`build_long_sequence_labeled`) | no | **yes** — HALT GATE 1's pooled-ceiling headline number | |
| 3 | `scripts/phase0_guard_audit.py:60` (`build_long_sequence`) | no | **yes** — guard-defect audits (sec V5-p0 steps 1-4) | |
| 4 | `scripts/phase0_decompose_failures.py:94` (`build_long_sequence_labeled`) | no | **yes** — this entire chain-2 observability audit line (steps 24-26) | |
| 5 | `llm_finetuning/analyze_bucket_c_windowing.py:66` (`build_long_sequence_instrumented`) | no | **yes, but FROZEN** — exists solely to reproduce sec AF step 4's seed=0 population bit-for-bit; deliberately excluded | |

**Which one is canonical for a real retrain: NONE of them.** Confirmed by direct inspection
(`src/swarm_intent/data.py`): STGT's actual training data comes from `generate_dataset()`,
which calls `generate_transition_sequence` directly with its own fixed `n_timesteps=50`,
3-regime blend timing — entirely separate code that has never shared a line with any of
these 5 copies. The premise that one of them "feeds a real retrain" doesn't hold; what these
5 copies actually share is EVAL-harness reproducibility risk, which is what this
consolidation fixes.

**Action taken**: created `src/swarm_intent/eval_trajectories.py` as the single canonical
implementation (`sample_chain`/`ground_truth_pair`/`build_long_sequence_labeled`, with the
full derivation of both fixes in its module docstring). Copies 1-4 now import from it; copy 5
(`analyze_bucket_c_windowing.py`) is explicitly left untouched with the reason documented
inline (its whole purpose is bit-for-bit historical reproduction). Landed as its own commit
(`3591051`), verified byte-identical to the pre-consolidation numbers before any behavior
change (chain2=233, OBS_CLEAR=231, OBS_PARTIAL=2, OBS_NONE=0 — matches step 25 exactly), so
consolidation itself introduced zero drift. **This closes the exact failure mode that already
happened once** (the dwell fix silently landing in only 1 of 5 copies) — a future change now
has to touch one file, and any future divergence would be an explicit code change, not a
silent gap.

### Step 2: source-side symmetrization

**Old value**: `LEAD_IN_RANGE=(15,35)` — chosen (step 25) to match the pre-fix formula's
REALIZED scale, not derived from an observability requirement.

**Derivation (mirrors the destination side exactly)**: a window's true label is a majority
vote over 50 timesteps, so source formation A needs `lead_in + 1 >= 26` to hold an outright
majority of window 0. Unlike the destination side, there is **no stride-granularity slack
term** here — window 0 always starts at exactly `t=0` on the `sliding_window_inference` grid
(`range(0, T-window_size+1, stride)` always includes `start=0`), so there's no equivalent of
the 9-timestep trailing loss the destination side has. The minimum is simply `lead_in >= 25`.

**New value**: `LEAD_IN_RANGE=(30,50)` — mirrors `MIN_DWELL_RANGE`'s margin shape
(threshold+5 to threshold+25) above this minimum.

**Verified, population-scale (seed=999, n=1000, no GPU — pure sampling):**

| | SOURCE (A) observability | DESTINATION (B) observability |
|---|---|---|
| OBS_CLEAR, before (dest-only fix) | 33.9% (n=79) | 11.2% (n=28, pre-step-25 baseline) → 99.1% (n=231, post-step-25) |
| OBS_PARTIAL, before | 49.8% (n=116) | — |
| **OBS_NONE, before** | **16.3% (n=38)** | 0.0% (already fixed by step 25) |
| OBS_CLEAR, after (both fixes) | 99.1% (n=217) | 99.5% (n=218) |
| OBS_PARTIAL, after | 0.9% (n=2) | 0.5% (n=1) |
| **OBS_NONE, after** | **0.0% (n=0)** | **0.0% (n=0)** |

**The new "source not observable" category from step 25's trace (15%) is confirmed to drop to
0.0% at population scale** — matches the trace-level finding almost exactly (16.3% population
vs. 15% in the 20-case sample), directly validating that trace as representative, not a fluke.

### Step 3: chain-2 re-measurement, three-way comparison

`scripts/phase0_chain2_observability.py --n 1000` + `scripts/phase0_chainlength_breakdown.py
--n 1000`, same seed=999/checkpoint as always:

| | baseline (pre-step-24) | dest-only fix (step 25) | **both fixes (step 26)** |
|---|---|---|---|
| chain-2 n | 251 | 233 | 219 |
| chain-2 OBS_NONE (dest) | 50.6% | 0.0% | 0.0% |
| chain-2 OBS_NONE (source) | — (not measured) | 16.3% | **0.0%** |
| **chain-2 pair_acc** | **18.7%** | **39.9%** | **65.8%** |
| **chain-2 threat_acc** | **31.9%** | **72.1%** | **76.3%** |
| chain-2 pair_acc, robust=True | 18.7% | 39.9% | 66.7% |
| chain-2 threat_acc, robust=True | 39.8% | 72.1% | 77.2% |
| chain-1 pair_acc (unrelated branch, sanity check) | 87.6% | 84.1% | 85.5% |

**The incremental contribution of source-symmetrization, isolated: +25.9pt pair accuracy
(39.9%→65.8%) and +4.2pt threat accuracy (72.1%→76.3%), on top of destination-fix's own
+21.2pt/+40.2pt gains — roughly as large a jump as the original destination fix, from a
change that only touched one tuple of constants.** Total improvement from original baseline:
pair accuracy +47.1pt (18.7%→65.8%), more than 3.5x. Chain-1 stayed flat within noise across
both fixes (population-composition shifts only, its own generation branch untouched by
either).

### Step 4: refined failure taxonomy — boundary/blend-timing vs. clean miss

20 fresh chain-2 failures under both fixes, manually traced
(`evaluation/phase0_chain2_observability_trace.txt`; the script's programmatic first-pass
categorizer again over-used "bridge/reduction issue" — 14/20 — for the same reason as before,
manually overridden). For every misclassified window, computed `frac_transitioning` (the
window's own fraction of true blend-region content) as the discriminator: a window is
**boundary/blend-timing** if `frac_transitioning > 0.10` (meaningfully touches the blend
region), **clean miss** if `frac_transitioning <= 0.10` (>=90% pure, settled content) yet
still misclassified.

**Trajectory-level result: 20/20 (100%) have their failure driven by at least one
boundary-type (frac_transitioning>0.10) misclassified window; 0/20 fail SOLELY on clean-miss
windows.** But a secondary, disclosed signal: **3/20 (15%) trajectories ALSO contain at least
one genuinely clean (frac_transitioning<=0.10) misclassified window** alongside their
boundary-type ones — e.g. trajectory 276 (`encirclement->dispersed`): window 0 is 92% pure
`encirclement` (only 8% blend content, nowhere near the transition) yet STGT confidently
(99%) predicts `v_shape`; trajectory 286 (`converging->encirclement`): window 0 is 92% pure
`converging` (8% blend content) predicted `dispersed` at 70% confidence. These 3 clean misses
don't individually determine their trajectory's outcome (each of those 3 trajectories ALSO
has other, boundary-type wrong windows), so they don't flip the trajectory-level bucket, but
they are real, disclosed evidence of some genuine non-boundary recognition error alongside
the dominant boundary mechanism — not hidden. 2/20 additionally trip the (already-flagged,
not-touched-this-session) `dispersed_converging_ambiguity` guard, itself on boundary-adjacent
window content.

**This taxonomy split points clearly toward decision B, not A**: the OVERWHELMING majority
of remaining chain-2 failures (100% at the trajectory level, and the large majority of
individual wrong windows) are boundary/blend-timing-concentrated, consistent with the
still-unresolved 0% train/eval blend-overlap finding (step 6, below) being the actual
mechanism, not a generic STGT capacity limit. Caveat, stated honestly: a few recurring
formation-CONFUSION pairs appear disproportionately near boundaries in this small sample
(`column`/`diamond`, `v_shape`/`shield`, `encirclement`/`v_shape`) — it's possible some of
this is a genuine pairwise visual-similarity confound that happens to concentrate near
boundaries rather than being purely blend-timing-caused; the taxonomy split is a strong
first-order signal, not a perfectly clean causal isolation.

### Step 5: chain-3+ false-positive rate — investigated, resolved, not a bug

| stage | n | chain-3+ bucket-A false-positive rate |
|---|---|---|
| baseline (pre-step-24) | 491 | 9.0% |
| dest-only fix (step 25) | 515 | 12.6% (the flagged "regression") |
| **both fixes (step 26)** | 506 | **1.8%** |

**Mechanism, confirmed by direct code inspection**: `build_long_sequence_labeled`'s per-hop
loop (`for i in range(len(chain) - 1)`) applies the SAME `lead_in`/`blend_duration`/`dwell`
sampling to EVERY hop of EVERY chain, regardless of total chain length — there is no
chain-length-specific branch. A chain-3+ trajectory has 2+ hops, each one individually
subject to exactly the same observability mechanics as chain-2's single hop. Under the
dest-only fix, INTERIOR hops' own source-side (their own "A") observability was just as
poor as chain-2's was before step 26 — an intermediate formation could fail to ever become a
window's true majority, causing the classifier to never correctly read it and
`known_history` to collapse to <=2 distinct formations by OMISSION, incidentally routing a
genuine 3+-hop trajectory into a spurious bucket-A resolution. Symmetrizing lead-in fixes
this for every hop, not just chain-2's, which is exactly what step 26 measures.

**Statistical basis** (two-proportion z-test, normal approximation):

| comparison | z | two-sided p |
|---|---|---|
| baseline (9.0%, n=491) → dest-only fix (12.6%, n=515) | 1.87 | 0.062 (borderline, not strongly significant on its own) |
| dest-only fix (12.6%, n=515) → both fixes (1.8%, n=506) | -6.68 | **2.4e-11 (highly significant)** |
| baseline (9.0%, n=491) → both fixes (1.8%, n=506) | -5.05 | **4.3e-07 (highly significant)** |

**Conclusion: EXPECTED, not a bug, and not noise.** The dest-only-fix "regression" itself was
only borderline significant in isolation (p=0.062) — plausibly a real but modest effect of
the disclosed asymmetric-fix mechanism, not a code defect. What is unambiguous is the
RESOLUTION: symmetrizing drops the rate to 1.8%, a highly significant improvement (p=2.4e-11)
that also comes in significantly BELOW the original baseline (p=4.3e-07) — i.e. this isn't
merely "undoing" the earlier regression, it's a genuine net improvement, mechanistically
explained by the shared per-hop code path, not a coincidence.

### Step 6: blend-distribution overlap re-checked after both fixes

`scripts/phase0_chain2_blend_distributions_v2.py`-style Monte Carlo (20000 draws), same 3
training regimes as every prior check:

| | start_frac | duration_frac |
|---|---|---|
| train regime 0 | [0.740, 0.880] | 0.100 |
| train regime 1 | [0.120, 0.280] | [0.440, 0.600] |
| train regime 2 | [0.020, 0.140] | [0.080, 0.100] |
| eval v1 (pre-step-24) | [0.283, 0.495] | [0.051, 0.458] |
| eval v2 (dest-only fix) | [0.153, 0.405] | [0.097, 0.304] |
| **eval v3 (both fixes)** | **[0.265, 0.495]** | **[0.085, 0.255]** |

**Overlap: STILL 0.0% (0/20000), unchanged.** Symmetrization shifted the start-fraction
distribution back up slightly (since a longer, more front-loaded lead-in pushes blend_start
later as a fraction of the now-also-longer `seg_len`) but the fundamental mismatch is
unchanged: eval's blend duration (max 25.5%) never approaches train regime 1's requirement
(44-60%, blend must DOMINATE the window), and eval's start timing still never lands in
regime 0's or regime 2's narrow windows either. **Restated plainly, per instruction: this is
the next target, not fixed this session.** It has now persisted, unchanged at 0.0%, across
three independent formula revisions (v1, v2, v3) — strong evidence this is a structural
property of evaluating long, realistic, moderate-duration transitions against a training set
that only ever taught extreme-timing (early/late/dominant) blends, not something that
self-resolves as a side effect of unrelated generator tuning.

### Decision gate

**B — remaining failure is still mostly type-a boundary/blend-timing errors; the blend
distribution mismatch is the dominant issue and should be fixed BEFORE any STGT training
change.** Both observability fixes landed cleanly (source and destination OBS_NONE both
0.0%, chain-2 pair accuracy 18.7%→65.8%, more than tripled, zero retraining) and the
consolidation closes the silent-divergence risk that had already caused one real problem
(the chain-3+ regression, now resolved and statistically confirmed). But the refined
taxonomy is unambiguous: 100% of the 20 freshly-traced failures are driven by
boundary/blend-timing-concentrated misclassification, not clean, non-boundary misses, and
the train/eval blend-timing overlap remains exactly 0.0% after three independent formula
revisions. Decision A (STGT recognition limits) would require the taxonomy to show a
meaningful clean-miss share — it doesn't (0% at the trajectory level). Decision C doesn't
apply — the consolidation surfaced no further divergence beyond what was already disclosed,
and the chain-3+ movement is resolved, not a live bug blocking trust in A or B. **Next
experiment, NOT started this session**: fix the blend-timing training/eval mismatch
specifically — add training examples to `generate_dataset()`'s transitioning regime whose
blend timing matches eval's actual realized shape (start ~27-50% into the window, duration
~9-26%, i.e. a genuine 4th regime or a widened regime 1) — before any STGT retrain for
capacity/architecture reasons, since a capacity-focused retrain would not address a
distribution-mismatch-caused error pattern.

Full data: `evaluation/phase0_chain2_observability.json` (both-fixes result),
`evaluation/phase0_chain2_observability_deston.json` (dest-only-fix, preserved),
`evaluation/phase0_chainlength_breakdown.json` (both-fixes),
`evaluation/phase0_chainlength_breakdown_deston.json` (dest-only-fix, preserved).
`docs/V5_STATE.json` updated (`phase=0, step=26`). **STOP per instruction — no STGT
retraining, no blend-distribution fix started this session; reported and stopped per the
decision gate.**

### Step 27: the elephant in the room — eval_trajectories.py is evaluation-only, generate_dataset() is untouched

Every fix landed in steps 24-26 (dwell-time, source symmetrization, consolidation) lives in
`src/swarm_intent/eval_trajectories.py`, which builds long EVALUATION trajectories only.
`swarm_intent.data.generate_dataset()` — the function `scripts/generate_data.py` actually
calls to build STGT's training set — was last touched by strategy 5 (regime-margin
tightening, 2026-08-08) and has not changed since. Every Phase 0 ceiling number through step
26 measures the SAME frozen strategy-5 checkpoint, trained on the OLD (unchanged) regime
distribution, scored by an increasingly-accurate but entirely separate evaluation harness. No
training this session — establishing facts first.

**Full parameter diff, TRAIN (`generate_dataset()`/`generate_transition_sequence`) vs EVAL
(`eval_trajectories.py`, current, both fixes applied):**

| parameter | TRAIN (generate_dataset) | EVAL (eval_trajectories.py) | same? |
|---|---|---|---|
| example/segment length | fixed `n_timesteps=50` — the example IS the window | derived `seg_len` = lead_in+blend_duration+dwell, **80-132 timesteps** (1.6x-2.6x longer), later cut into multiple 50-step windows by `sliding_window_inference` | **NO** |
| blend timing | 3 discrete regimes, each a FRACTION of the fixed 50-step window (regime 0/2: blend very late/early + short, "pure"; regime 1: blend dominates >=45% of the window, "transitioning") | `lead_in~U(30,50)`, `blend_duration~U(10,25)`, `dwell~U(40,60)` sampled directly as ABSOLUTE timesteps, one continuous shape, no discrete regime split | **NO** (different parameterization AND different realized shape, quantified in step 28) |
| labeling | per-EXAMPLE single label (pure formation_a / "transitioning" / pure formation_b, chosen by regime) | per-TIMESTEP true label, consumed downstream by windowing — no single "regime" concept at generation time | **NO** (different granularity, though eval's labels are ground truth, not model input) |
| spread | `rng.uniform(0.7, 1.5)` | `rng.uniform(0.6, 1.8)` | NO (eval range is wider, ~26% lower min, ~20% higher max) |
| noise_std | `rng.uniform(0.3, 0.8)` | `rng.uniform(0.15, 1.4)` | NO (eval range is much wider — up to 1.75x train's max, down to half train's min) |
| dt | `0.5` | `0.5` (uses `generate_transition_sequence`'s own default, never overridden) | yes |
| per-timestep physics (cosine-ramp blend, acceleration, noise injection) | `generate_transition_sequence()` | same function, called directly (`from .data import generate_transition_sequence`) — not reimplemented | **yes, identical** — everything downstream of "which blend_start/blend_end/spread/noise_std get passed in" is the same code |
| formation-pair sampling | uniform over all `(a,b), a != b` ordered pairs | `sample_chain()`: first formation uniform, each subsequent one uniform over `BASE_FORMATIONS \ {previous}` — equivalent for a single hop | yes (for chain length 2) |

**The one thing NOT different**: the underlying per-timestep trajectory physics. Every
divergence is in what blend_start/blend_end/spread/noise_std get sampled and handed to that
shared, unmodified physics function — never in the physics itself. This matters for step 3's
framing: porting the fix is a training-DISTRIBUTION change, not a new-physics or
new-architecture change.

Script: none needed for this step (direct source read, `src/swarm_intent/data.py` lines
128-219 vs `src/swarm_intent/eval_trajectories.py` lines 62-127).

### Step 28: quantify what "porting the fix" would actually change

`scripts/phase0_train_eval_blend_divergence.py` -- same Monte Carlo methodology as step 26
(and its ancestor, `phase0_chain2_blend_distributions.py`), reusing `train_regime_fractions()`
unchanged and importing EVAL's ranges live from `eval_trajectories.py` (not duplicated, so
this can never silently drift the way the step-26 audit found `_v2.py` had). n=20000 draws
each side.

| | TRAIN regime 0 | TRAIN regime 1 | TRAIN regime 2 | EVAL (current) |
|---|---|---|---|---|
| start_frac | [0.740, 0.880] | [0.120, 0.280] | [0.020, 0.140] | **[0.265, 0.495]** |
| duration_frac | [0.100, 0.100] | [0.440, 0.600] | [0.080, 0.100] | **[0.085, 0.255]** |
| share of transitioning examples | ~33.1% | ~33.9% | ~33.0% | -- |

**Overlap: 0/20000 (0.0%) in ANY of the three regime boxes — individually and combined.**
Eval's realized region doesn't graze regime 0, 1, or 2's box even partially; it sits in the
gap between all three. Combined with example length being fundamentally different (train's
example IS a 50-step window; eval's hop is 80-132 steps, later cut into multiple windows),
"porting the fix" is not a 3-number parameter substitution -- naively assigning
`LEAD_IN_RANGE=(30,50)`/`BLEND_DURATION_RANGE=(10,25)`/`MIN_DWELL_RANGE=(40,60)` to
`generate_dataset()` would produce examples 1.6-2.6x longer than the fixed 50-step window
STGT is architected for (`PositionalEncoding`'s `max_len=50` buffer is baked into every
existing checkpoint). A genuine port means restructuring `generate_dataset()`'s transitioning-
example construction to sample a long hop the same way and extract windows from it the same
way `sliding_window_inference` does at eval time -- not a drop-in constant change.

**Answer to the question as posed**: if `generate_dataset()` adopted eval's timing
distribution, effectively **100% of currently-generated transitioning examples would exhibit
a blend-timing shape none of the 3 current regimes ever produce** -- this is not a partial-
overlap tuning question, it is a full distributional replacement.

Raw output: `evaluation/phase0_train_eval_blend_divergence.txt`.

### Step 29: do the step-26 numbers already answer "does the model need retraining"?

Restated plainly, per instruction, since the two readings are easy to conflate: **"chain-2
pair accuracy 18.7%→65.8%" and "does STGT generalize to the corrected blend-timing
distribution" are different claims, and only the first one is actually measured.**

What steps 24-26 changed: `eval_trajectories.py`'s `LEAD_IN_RANGE`/`MIN_DWELL_RANGE`, i.e. how
wide the pure-content margin is on either side of a hop's blend, before any window gets
carved out. Widening those margins does not change what a blend-dominant window looks like or
whether STGT classifies it correctly — it changes how MANY of the windows a long trajectory
produces are blend-dominant at all. With `lead_in>=30` and `dwell>=40` against a
`window_size=50` sliding grid, the large majority of windows in a fixed chain-2 hop are now
guaranteed to be dominated by pure formation_a or pure formation_b content; only a shrinking
minority straddle the blend region itself.

Two pieces of evidence already on the record settle this, not new measurement:

1. **The refined 20-case failure taxonomy (step 26, run AFTER both observability fixes)**:
   100% of remaining chain-2 failures are boundary/blend-timing-concentrated, 0% clean misses.
   If the observability fix had made STGT better at blend-dominant windows, failures would
   have become more evenly distributed, or concentrated elsewhere. They didn't — every
   remaining failure is still exactly the case type this section is asking about.
2. **The blend-overlap Monte Carlo (step 26 step 6, re-confirmed step 28)**: 0.0% overlap
   between train and eval blend-timing shapes, unchanged across three independent formula
   revisions including the observability fix itself. The observability fix changed WHICH
   windows get evaluated; it never touched WHAT those windows' blend content looks like
   relative to what STGT was trained on.

**Plain statement**: the step-26 numbers show "the model does okay once the eval harness
stopped handing it windows it was never trained to read" — a real, worthwhile fix, and not a
trivial one (3.5x on its own metric). They are NOT evidence that STGT generalizes to the
actual corrected/realistic blend-timing distribution, because no window in the post-fix
measurement population is *more* blend-dominant than before — observability fixes reduce
EXPOSURE to the hard case, they don't test performance ON it. The hard case (a genuinely
blend-dominant window, shaped like eval's real distribution) remains exactly as untested by
training as it was at step 1, because `generate_dataset()` has not changed.

### Step 30: decision gate, closed

Per the gate criteria set out in advance:

- **A** (port + retrain) — justified if step 28 shows substantial train/eval divergence AND
  step 29 shows the current checkpoint's gains are eval-artifact-driven, not genuine
  generalization.
- **B** (leave `generate_dataset()` as-is) — justified if step 29 instead showed the frozen
  checkpoint already generalizes well to corrected eval trajectories.
- **C** (insufficient evidence) — if neither is cleanly established.

Step 28: 0.0% overlap (0/20000), not partial — divergence is total, not a tuning gap. Step 29:
the step-26 gains are explained in full by reduced exposure to blend-dominant windows, with
the blend-dominant failure mode itself unmoved (100% boundary-concentrated failures, 0.0%
overlap, both measured AFTER the observability fix). Both of A's conditions hold and neither
of B's does. **Decision: A.** Full reasoning and implications for strategies 1-6's standing:
`HISTORY.md`'s 2026-08-10 decision entry. **Not executed this session — no training, no
`generate_dataset()` changes. STOP per instruction.** `docs/V5_STATE.json` updated
(`phase=0, step=30`).

### Steps 31-33: designing the port as three separable changes (combined write-up + implementation)

Decision A (step 30) said "port the fix." The parameter diff (step 27) surfaced two more
structural differences beyond blend timing: per-example vs per-timestep labeling granularity,
and fixed-50 vs derived-80-132 example length. Porting blend timing alone, without resolving
these, risks a dataset that's wrong in a new way. This turn designs and implements all three
as independently toggleable flags on `generate_dataset()` — no training yet. (Process note:
the design (steps 31-32 below) and its implementation (step 33) landed in one commit,
`8c7b0b7` — the design was validated by writing it; splitting further would have added
review risk without benefit. Noted here rather than silently claiming strict one-step-one-
commit throughout.)

**Step 31 — the labeling rule.** `generate_dataset()` currently assigns ONE discrete label to
an entire fixed 50-step example, implied directly by which of 3 regimes was chosen — the
regime IS the label, known a priori. Once blend timing is continuous (or example length is
variable), a window's content can be an arbitrary mix of pure-A/blend/pure-B in any
proportion, and the label must be DERIVED from realized content, not assumed from a
generation-time choice.

Two thresholds, both derived, neither copied uncritically:

- `PURE_LABEL_THRESHOLD = 0.70` (35/50). A window is labeled a pure endpoint formation only
  if that formation holds a COMFORTABLE majority of its content, not a bare one (bare majority
  is >=26/50 = 0.52). This reuses the exact 35/50 figure already derived for
  `MIN_DWELL_RANGE`/`LEAD_IN_RANGE` (steps 24-26: `D>=26+9=35`, the +9 a stride-slack margin).
  Reusing it is not laziness — it makes "a window eval's own observability logic would trust
  as reliably showing formation A" and "a window training confidently labels pure-A" the SAME
  window by construction, so train and eval agree on what "confidently observed" means, not
  just on raw blend timing.
- `TRANS_LABEL_MIN_BLEND_FRAC = 0.20` (10/50). Mirroring 0.70 for "transitioning" was
  considered and rejected: `BLEND_DURATION_RANGE`'s own max (25 timesteps) is exactly half of
  `WINDOW_SIZE` (50), so blend content can NEVER reach even a bare majority of any window,
  let alone 70% — requiring 0.70 would make "transitioning" structurally unreachable and
  silently delete the class from ported training data. Instead, "transitioning" requires blend
  to be the PLURALITY of the window's three content types (beats pure_a and pure_b
  individually) AND at least as large as `BLEND_DURATION_RANGE`'s own minimum (10 timesteps =
  10/50 = 0.20) — below that floor the window is grazing a blend edge without containing a
  meaningful chunk of it.
- **Windows meeting neither bar are EXCLUDED from training, not mislabeled.** `_label_window`
  (`src/swarm_intent/data.py`) returns `None` for this case; `generate_dataset` skips the
  window entirely rather than forcing a label onto genuinely ambiguous content.

**Step 32 — the windowing/architecture constraint.** Checked `src/swarm_intent/model.py` and
`config.py` directly (not assumed): `Config.max_seq_len=50`, consumed by
`PositionalEncoding(max_len=cfg.max_seq_len)`. Its `forward()` does
`x = x + self.pe[:, :x.size(1), :]` — a SLICE, not an assertion, so **T<=50 works for any T**
(the buffer just gets truncated), but **T>50 breaks**: `pe[:, :80, :]` on a 50-row buffer
returns 50 rows, and `x` (shape `[B,80,d]`) + a 50-row slice is a shape-mismatch crash.
Separately, `STGTModel.forward()` derives a single scalar `T = len(graph_sequences[0])` and
reshapes the WHOLE BATCH by it — sequences within one batch must share a length (though
different batches may differ). Combined with `window_size=50` being baked into
`sliding_window_inference`'s default and every downstream evaluation/bridge script,
**increasing `max_seq_len` is a full architecture change affecting every existing checkpoint
and the entire eval harness — out of scope for a data-generation port.** Decision: training
examples stay fixed at exactly 50 timesteps; a long (80-132-step) hop gets WINDOWED down to
50-step slices before becoming individual training examples, using the identical
`WINDOW_SIZE=50, STRIDE=10` grid `sliding_window_inference` already uses at eval time. This
is also what resolves the "does windowing reintroduce the dwell-time fix's boundary problem"
question directly: it doesn't, because each window is labeled from ITS OWN realized content
(step 31's rule) rather than inheriting one blanket label for the whole hop — the boundary
problem was specifically about a single label covering content it didn't match, which this
construction cannot do by design.

**Step 33 — implementation.** Three flags on `generate_dataset()`
(`src/swarm_intent/data.py`): `corrected_blend_timing`, `windowed_examples`,
`content_majority_labeling`, all default `False`. Verified bit-identical output (X, y, names)
against the pre-change generator at the same seed with all three off — zero behaviour change
for `scripts/generate_data.py`/`train_model.py`. `LEAD_IN_RANGE`/`BLEND_DURATION_RANGE`/
`MIN_DWELL_RANGE`'s canonical definition moved to `data.py` (training wants them now too);
`eval_trajectories.py` re-exports rather than duplicating, closing the exact silent-drift
risk step 26 already found once. Full 140-test suite passes unchanged.

### Step 34: label-sanity numbers for all four diagnostic datasets

`scripts/phase0_generator_port_diagnostics.py --n_transition 300` (seed=7, fresh, disjoint
from 0/1/42/999/2024), `return_diagnostics=True`. For every KEPT example: own-content
fraction distribution by assigned label, and whether the assigned label disagrees with the
PLURALITY of that example's own realized content (checked numerically, not assumed).

| run | hops | kept | excluded | disagree-with-own-content |
|---|---|---|---|---|
| baseline (all off) | 300 | 300 | 0 (0.0%) | **1/300 (0.3%)** |
| blend-timing only | 300 | 300 | 0 (0.0%) | 0/300 (0.0%) |
| windowing only | 300 | 1800 | 0 (0.0%) | **73/1800 (4.1%)** |
| labeling only | 300 | 297 | 3 (1.0%) | 0/297 (0.0%) |
| **all three combined (the proposed port)** | 300 | 886 | **928 (51.2%)** | **0/886 (0.0%)** |

**Two findings worth flagging plainly, not just the headline pass/fail:**

1. **"Windowing only" reintroduces mislabeling at 4.1%** — directly confirms the concern
   raised when this session was scoped ("does windowing reintroduce the boundary problem the
   dwell-time fix just solved?"). With `content_majority_labeling=False`, every window sliced
   from one hop inherits that hop's single regime-implied label regardless of what content
   that specific window actually contains — the exact per-hop-blanket-label bug pattern this
   whole line of work exists to fix, now happening per-window instead of per-hop. This is
   direct empirical evidence that **windowing must be paired with content-majority labeling**;
   shipping windowing alone would make training data worse, not better.
2. **The proposed port's exclusion rate is a real cost: 51.2% of candidate windows are
   thrown away, not mislabeled.** Structural cause: `BLEND_DURATION_RANGE`'s max (25/50=50%)
   means a window can be genuinely 3-way ambiguous (e.g. ~55% pure-A / ~25% blend / ~20%
   pure-B) — below `PURE_LABEL_THRESHOLD` for pure-A, and blend isn't a clean plurality
   either. The design deliberately excludes these rather than force a label (per step 31's
   stated goal), but at full training scale this means `n_transition` must be requested at
   roughly **2x** the desired final example count to compensate for yield loss — a planning
   /compute consideration, not a correctness problem, and disclosed here rather than
   discovered later.

**Also confirmed, not just claimed**: `pure_a`/`pure_b` own-content fractions in the combined
run range 0.700-1.000 (never below the 0.70 floor, by construction); `transitioning`'s range
0.360-0.460 (never below the 0.20 floor, always plurality). Both thresholds hold exactly as
designed under real generation, not just in the abstract derivation.

Raw output: `evaluation/phase0_generator_port_diagnostics.txt`.

### Step 35: characterizing the 51.2% exclusion — biased, or just ambiguous content?

`scripts/phase0_generator_port_diagnostics.py` extended to record `(frac_a, frac_blend,
frac_b, formation_a, formation_b)` for every EXCLUDED window too, not just kept ones
(`generate_dataset`'s `diag["excluded_examples"]`, new this turn). Same run as step 34
(seed=7, n_transition=300) — a breakdown of that exact population, not a fresh sample.

**Where in content-space**: excluded windows have `max(frac_a,frac_b)` in [0.340, 0.680]
(median 0.540) and `frac_blend` in [0.180, 0.460] (median 0.300) — entirely inside the
"near-miss" gap between the two thresholds (0.70 pure / 0.20+plurality transitioning), not
spread uniformly across the full space and not concentrated in some unexpected corner. The
2D histogram mass sits at `own_frac` 0.4-0.7 x `blend_frac` 0.2-0.4 — genuinely 3-way-mixed
windows near a blend boundary, exactly the case the design intends to exclude rather than
mislabel. No pathological shape found.

**By formation** (appears as either `a` or `b`): exclusion rate is essentially flat —
min 49.0% (`column`), max 53.7% (`v_shape`), **std 1.8 points**. No formation is being
starved.

**By ordered pair** (42 pairs): mean 51.1%, **std 5.1 points**, range 40.0%
(`diamond`->`dispersed`) to 66.7% (`encirclement`->`shield`). Wider spread than the
per-formation view, but every pair still yields a nonzero share of kept windows (worst case
still keeps 1/3), and per-pair sample sizes at this diagnostic scale are small (9-43
kept+excluded per pair) — a 5.1-point std across 42 small samples is consistent with sampling
noise, not necessarily a true per-pair effect. `(encirclement, shield)` is the one pair worth
watching at full scale, named explicitly rather than left as a future unexplained gap, but
not severe enough to block scaling on its own.

**Verdict: no severe systematic exclusion bias.** The 51.2% rate is explained by genuine
content ambiguity concentrated exactly where expected, formation-level exclusion is
near-uniform, and the one pair-level outlier found is disclosed and small enough to monitor
rather than block on. Raw output: `evaluation/phase0_exclusion_bias.txt`.

### Step 36: seed stability of the keep rate

`scripts/phase0_seed_stability.py`, 5 seeds (7 -- reused from steps 34/35 -- plus fresh 8, 9,
10, 11; disjoint from every seed already used elsewhere in this program: 0, 1, 42, 999,
2024), `n_transition=300` each, combined port.

| seed | hops | total windows | kept | excluded | keep rate |
|---|---|---|---|---|---|
| 7 | 300 | 1814 | 886 | 928 | 48.8% |
| 8 | 300 | 1830 | 920 | 910 | 50.3% |
| 9 | 300 | 1833 | 903 | 930 | 49.3% |
| 10 | 300 | 1837 | 892 | 945 | 48.6% |
| 11 | 300 | 1867 | 946 | 921 | 50.7% |

**keep_rate: mean 49.5%, std 0.8 points, range [48.6%, 50.7%].** Very stable — step 34's
single-seed 48.8% was representative, not an outlier. `windows_per_hop` is similarly stable
(mean 6.12, std 0.06).

**Compensation factor revised using the worst-case seed (48.6%), not the mean, for a
conservative margin: 1/0.486 = 2.06x.** Matches step 34's ~2x estimate closely — no revision
needed beyond tightening it slightly to 2.06x. Raw output:
`evaluation/phase0_seed_stability.txt`.
