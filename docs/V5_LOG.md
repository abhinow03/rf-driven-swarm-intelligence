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
