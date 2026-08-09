# Development history

This is a curated summary. The full, unedited, chronological record — including work that
didn't pan out — is preserved in three places, deliberately not rewritten into a single clean
narrative:

- **`AUDIT.md`** — a lettered, section-by-section research log (sections A through AH+)
  covering the LLM fine-tuning / adapter / coverage-routing arc.
- **`docs/V5_LOG.md`** and **`docs/CEILING.md`** — a numbered, step-by-step log of the STGT
  retraining program ("V5"), including every measured before/after number.
- **`docs/GAP_DIAGNOSIS.md`**, **`docs/UPSTREAM_ISSUES.md`**, **`docs/DEFENSE.md`** —
  supporting diagnostic documents referenced throughout V5.

This project does not use a "v1/v2/v3/v4/v5 pipeline version" numbering scheme end to end —
that would misrepresent how the work actually happened. There are two real, disclosed
versioning axes instead: **LLM adapter versions** (below) and the **V5 STGT retraining
program** (its own phase/step counter, currently phase 0). Read on for both.

## Phase 1 — package migration (2026-07-30)

The project started as three Jupyter notebooks (`capstone_with_llm.ipynb`,
`capstone_with eval.ipynb`, `models + data generation.ipynb`). The first commit in this
repository's history is a wholesale migration of that notebook code into an importable
package (`src/swarm_intent/`), removing hidden global state (`CFG`/`device`/`model_v2`
module-level globals) in favor of explicit `cfg`/`device` parameters threaded everywhere.
`MIGRATION_GUIDE.md` maps every old notebook function to its new module location. The
original notebooks are not present in this repository's current tree.

## Phase 2 — LLM reasoning layer and coverage routing (AUDIT.md sec A–AH)

Five QLoRA adapter iterations were trained against `Qwen/Qwen2.5-7B-Instruct`
(`docs/ADAPTER_VERSIONS.md` has the full comparison table):

| adapter | what changed | outcome |
|---|---|---|
| `qwen-swarm` (v1) | first fine-tune, 2700 rows | `assistant_only_loss=False` — a real bug, loss computed over the prompt too |
| v2 / v3a / v3b | iterative data/loss fixes | progressively closed the gap to the prompt-engineered baseline |
| v3b-fix | targeted fix for a specific low-threat confusion pattern | the strongest LLM-layer baseline used in later comparisons |
| v3c | epoch-matched control | ruled out under-training as the explanation for a `low`-threat-level collapse |
| v3d | further data revision | see `AUDIT.md` sec V onward for detail |

Alongside adapter iteration, this phase established the project's core evaluation
discipline (`docs/methodology.md`): independent ground truth for every battery, an
independent judge model (after an early version had the system grade its own output — 5/5
self-scores while objective accuracy was near 0%), and batched-generation validated
equivalent to unbatched before being used for large runs.

**Coverage routing (sec AE–AH):** measuring what fraction of real STGT output a plain
`(from,to) -> RULES` dictionary can resolve on its own led to `src/swarm_intent/coverage.py`
(bucket A/resolvable, B/guardable, C/unresolvable) and the three-layer `pipeline_v2.py`. A
follow-up investigation (sec AF) found the pipeline's real-world accuracy (10.4% correct,
69.1% over-abstention) traced to a **reduction brittleness bug**, not a property of RULES —
the bridge recognized only 3.6% of sequences as resolvable even though generator ground truth
said most were simple transitions. Sec AG built and tested a majority-vote "robust reduction"
fix for this and found — honestly reported, not hidden — that it **should not ship**: it
recovered more pairs, but mostly wrong ones (16.7% precision on newly-recovered cases),
because most recoveries still tripped an unconditional ambiguity guard on the way to a final
answer. That negative result is exactly what motivated Phase 3's guard audit, which found
the ambiguity guard itself was buggy (see below) — the "robust reduction" fix was solving the
wrong layer of the problem.

## Phase 3 — V5: STGT retraining program (docs/V5_LOG.md, current)

Triggered by sec AH's projection that fixing a real upstream generator bug could roughly
double Layer-1 firing. The bug (confirmed by diff against a teammate's separate repository,
commit `9158b081`, "added acceleration and split dispersed and converging logic"): the
synthetic data generator's `dispersed` and `converging` formations shared the exact same
geometry-generation branch, and drone velocity never varied within a trajectory. Both were
ported into this repo (commit `27adc23`), and the program has since worked through a
numbered sequence of phase-0 steps, each measured before/after against a fixed seed=999,
n=1000 population:

| step | what | headline result |
|---|---|---|
| 2 | regenerate + retrain on the fixed generator | test_acc 93.5%, but real ceiling still only 3.3% pair-level |
| 3–4 | root-cause the low ceiling | model over-predicts `transitioning` at a uniform ~20-28% false-positive rate across all classes — a calibration problem, not a classification-quality problem |
| 5 | targeted fix for the transitioning false-positive rate | pair-level ceiling roughly doubled (6.1% -> 12.2%) |
| 6–9 | training-schedule experiment (steadier LR schedule) | test_acc rose to 99.6% but the real ceiling roughly **halved** — reverted; standing rule recorded: judge STGT checkpoints on pair/threat ceiling, never test accuracy alone |
| 10–11 | audit the dispersed/converging ambiguity guard directly | confirmed: the guard fired on 75.8% of windows and was spurious 66.1% of the time — it tested raw probability closeness with no check that either class was actually competitive. **Single biggest gain of the program**: fixing the one-line condition took the threat ceiling from 13.0% to 52.3%/58.7%, no retraining |
| 12–20 | stratify by chain length; audit and fix two more guards of the same class (`oov_name` conflating a valid class with genuine OOV; `dominant_history_contradiction` testing a count-tie instead of the thing it claimed to test, later proven unreachable and kept as a defensive check) | threat ceiling to 60.3%/64.6%(robust) |
| 21 | measure real RULES coverage against the full realistic chain-length-3+ pattern space | 60.7% of chain-3+ trajectories are structurally distinct patterns — a future RULES extension would need a compositional rule, not enumeration; not attempted (requires sign-off) |
| 22 | stop pooling chain-1 and chain-2 accuracy into one misleading number | chain-1 ~87%, chain-2 ~19% — very different, hidden by pooling |
| 23–26 | diagnose and fix chain-2's real bottleneck: the destination formation was often never observed by any sliding window at all (a generator-parameter issue, not model or bridge) | chain-2 pair accuracy 18.7% -> 39.9% (destination-side fix) -> **65.8%** (source-side symmetrization), more than 3.5x, zero retraining |

**Current state (latest entry, step 26):** Decision **B** — the dominant remaining failure
mode is a train/eval blend-timing distribution mismatch (STGT was never trained on windows
whose blend timing matches what long, realistic evaluation trajectories actually produce),
confirmed unchanged across three independent formula revisions. This is the next scoped
target, not yet fixed. The program's internal 70%-pair-accuracy floor has not been reached.

Every step above followed the same discipline: a fixed seed/population, a before/after
measurement (not a claim), and — where a fix didn't help, or helped less than hoped, or made
something else worse — that result is reported exactly as measured (strategy 6's regression;
the blend-overlap remaining at 0.0% after three attempts) rather than omitted.
