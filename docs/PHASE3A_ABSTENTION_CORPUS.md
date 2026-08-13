# Phase 3a — Abstention Corpus Generation

Corpus-only session. No training happened here, and v5-a (`checkpoints/v5_sft/`) remains the
shipped baseline throughout — see AUDIT.md sec AN for why this phase exists: the no-retrain
attempt at closing the Layer-2 gap failed its own false-positive gate (51/498, 10.2%), so the
locked contingency plan is a retrain with abstention examples. This document is the
preregistered record of that corpus, produced BEFORE any retrain — the corpus is written to
`data/abstention_corpus.jsonl`, reviewed, and only merged into a training file on explicit
sign-off.

## 0. Safety copy (LOCKED CONFIG, done before anything else)

`checkpoints/v5_sft/` (v5-a) was copied to `checkpoints/v5_sft_v5a_PROTECTED/`, all 29 files
sha256-verified byte-identical, and **both** directories were made read-only
(`chmod -R a-w`, write attempts confirmed to fail with `Permission denied`). v5-a remains
loadable, evaluable, and citable exactly as-is, indefinitely. All Phase 3a work — and the
eventual retrain — happens under a new checkpoint name; neither copy of v5-a is ever written
to again. Re-verifiable any time: `python scripts/phase3a_verify_safety_copy.py`.

## 1. The ground-truth abstention signal (`src/swarm_intent/ground_truth_abstention.py`)

**Why not STGT's read**: AUDIT.md sec AN measured STGT's own per-window `formation_history`
(via `stgt_bridge.py`, `robust=False`, the shipped default) hallucinating a spurious 3rd
intermediate formation on 51/498 (10.2%) genuinely-2-formation trajectories. Any abstention
label derived from that read would teach a fine-tune to abstain on cases that are actually
answerable — exactly the failure this corpus exists to avoid.

**The function**: `classify_trajectory_ground_truth(chain, true_labels=None)`.
- `multi_hop` / `oscillation`: fully determined by `chain` alone (the list of distinct
  formations the simulator was TOLD to produce, known with certainty since the caller chose
  it) — `len(chain) >= 3`, then `oscillation` if `chain[0] == chain[-1]` else `multi_hop`.
- `terminal_transitioning`: needs `true_labels` (the simulator's own per-timestep label
  array, from `swarm_intent.eval_trajectories.build_long_sequence_labeled` —
  `TRANSITION_CLASS` during blend windows, also simulator-authored, never an STGT read).
  Fires when `true_labels[-1] == TRANSITION_CLASS`, i.e. the observation was truncated
  before the final hop's dwell period, so the destination formation was never confirmed.
  **Cannot occur under standard (untruncated) generation** — every hop always completes its
  dwell — so natural sampling gives it 0% frequency; it must be constructed deliberately
  (see step 3).
- Neither path imports `stgt_bridge.py` or touches any checkpoint.

**Validation against AUDIT.md sec AK's 502-case STGT-derived categorization**
(`llm_finetuning/validate_ground_truth_abstention.py`): **94.6% raw agreement (475/502)**.
All 27 disagreements trace to STGT's own read noise, not this function:
- 10 cases where STGT's read failed to detect the multi-hop structure at all (the known
  `bucket_A_misrouted`/`dispersed_converging_ambiguity` cases from sec AK/AN).
- 17 cases where a noisy intermediate read flipped `oscillation` vs `multi_hop`
  (e.g. true chain `['encirclement','dispersed','encirclement']` is genuinely oscillation,
  but STGT's noisy read never showed the return to `encirclement`, reading as `multi_hop`
  instead).

This function was correct in every one of the 27 disagreements — the validation demonstrates
exactly the independence from STGT's noise this phase requires, not merely asserts it.

12 unit tests (`tests/test_ground_truth_abstention.py`) cover both branches, the precedence
of the `len(chain)>=3` check over the `terminal_transitioning` check, and the no-`true_labels`
default.

## 2. Preregistered strata targets (`scripts/phase3a_step2_strata_targets.py`)

Re-scored under the ground-truth classifier (not sec AK's STGT-noisy numbers), the 502-case
population is 435 multi_hop (86.7%) / 67 oscillation (13.3%) / 0 terminal_transitioning
(0.0% — cannot occur naturally, see above).

| mechanism | target | reasoning |
|---|---|---|
| multi_hop | 780 | dominant mechanism; 900-row multi_hop+oscillation pool split at the exact 86.7:13.3 ratio measured above |
| oscillation | 120 | secondary mechanism; same ratio |
| terminal_transitioning | 100 | deliberately over-sampled relative to its 0.0% natural frequency — a real, distinct, structurally valid mechanism (an observer catching a single-hop transition mid-blend is an ordinary real-world shape) that would otherwise have zero training examples. Same precedent as Phase 1's own `STRATA_TARGETS` over-sampling the rare "critical" threat tier. |
| **total** | **1,000** | ~8.3% of the existing 12,001-row corpus — large enough to meaningfully teach the dominant real-world abstention behavior (multi_hop/oscillation are 98.0% of ALL real unanswerable observations per sec AK) without risking the over-abstention failure mode `PREREGISTRATION.md` already flags as gameable at low abstention volume. |

## 3. Generation (`llm_finetuning/build_abstention_corpus.py`)

Per row: sample the TRUE chain for the target mechanism (ground truth by construction) →
ground it in a REAL `swarm_intent.eval_trajectories` simulation (`build_long_sequence_labeled`,
same function Phase 0's ceiling measurements use — never STGT) → re-verify with step 1's
classifier before accepting the row → render a tactical-context string in Phase 1's own
narrative grammar (reusing `degradation.py`'s structural phrasing and `build_sft_dataset.py`'s
real-population-calibrated field sampling, `_sample_real`) → write a target matching
`pipeline_v2._layer2_abstain`'s schema exactly (`threat_level="unknown"`,
`likely_intent="unknown"`, `recommended_action="monitor"`), stating the ground-truth mechanism
reason in `threat_reasoning`/`key_indicators`.

**Terminal_transitioning construction**: appends one more, deliberately-unresolved hop to the
known prefix, generates it in full via the real simulator, then truncates strictly inside the
final hop's blend region (a random cut point within the last contiguous run of
`TRANSITION_CLASS` in `true_labels`) — the destination formation genuinely never gets
confirmed, by construction, not by narrative claim.

**Teacher**: `NVIDIA_API_KEY` is not set in this environment. Per explicit user confirmation,
generation ran with `--no-teacher` (rule-clean templated prose, the same fallback
`build_sft_dataset.py` itself supports for smoke tests) — **the answerability label was never
going to come from the teacher regardless** (see LOCKED CONFIG); only response prose quality
is affected. A future session with the key set can regenerate the same 1,000 rows with teacher
prose via `--append`-style accumulation, matching Phase 1's own workflow.

**Result**: exactly 780 / 120 / 100 = 1,000 rows generated, 0 resamples needed across all
1,000 (construction and the step 1 verifier agreed on every sample, every time).

## 4. Hard gates (`scripts/phase3a_step4_gates.py`) — ALL PASS

| gate | result |
|---|---|
| a. row count > 0, matches/exceeds target | 780/120/100, exact match — PASS |
| b. zero overlap with the existing 12,001-row corpus | 0 overlap, 0 internal duplicates (exact user-message-string dedup, same check `build_sft_dataset.py --append` uses) — PASS |
| c. 20-row-per-mechanism spot-check vs ground truth | 0 mislabeled. multi_hop/oscillation (chain length ≥3) independently re-verified from chain alone, 40/40 match. terminal_transitioning (chain length ≤2) cannot be re-verified from the persisted chain alone without `true_labels` (not currently persisted in the metadata) — disclosed as a real limitation, not silently passed; generation-time verification (the identical function, applied to the real simulated `true_labels` before the row was accepted) never failed once across all 100 rows — PASS |
| d. safety copy still matches original | re-verified byte-identical and read-only — PASS |

## 5. Final corpus (`evaluation/phase3a_final_corpus_report.json`)

1,000 new rows (`data/abstention_corpus.jsonl`, **not merged into any training file**) +
12,001 existing RULES rows = 13,001 combined corpus size if merged. Every number in this
report is pulled live from the persisted artifacts, not re-typed.

## What this phase did NOT do

No training. No modification to `checkpoints/v5_sft/` or the new
`checkpoints/v5_sft_v5a_PROTECTED/` safety copy (both read-only, re-verified after every
step). No edits to the existing 12,001 RULES rows — `data/abstention_corpus.jsonl` is
strictly additive and stands alone pending review. No merge into any training file.
