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
