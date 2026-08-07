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

**Note on the message's final instruction.** The message opened with "Do NOT patch the
generator yourself under any circumstances" and closed with "If the bugs still exist, just
clone the repo and fix the bugs yourself and continue working" — these directly contradict
each other. Moot here since the verdict is A, not C (the bugs do NOT still exist upstream), so
neither branch of that contradiction was triggered. Flagged to the user for awareness in case
it was a drafting slip, since a future turn could land on verdict C and the contradiction
would then be load-bearing.
