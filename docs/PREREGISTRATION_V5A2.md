# v5a2 Preregistration — locked before training

**This document, once committed, is immutable except via a dated erratum appended at the
bottom — same rule as `docs/PREREGISTRATION.md`.** This is the last planned attempt before
demo prep. Every known failure mode from v5-a's cycle (AUDIT.md secs AA–AO, `docs/V5_STATE.json`)
is closed here, before any v5a2 result exists to shape the bars.

**No training happened in this session.** Every number below is either (a) re-verified live
from an existing locked artifact, or (b) a preregistered floor/ceiling with its reasoning
stated in advance. Nothing here was fit to a result.

## 0. Upstream artifact re-verification (step 1)

Per the locked-config instruction: do not trust the last recorded hash blindly. All three
re-computed live, this session, right now.

| artifact | role | live sha256 | matches record? |
|---|---|---|---|
| `checkpoints/v5_sft_v5a_PROTECTED/` (29 files, tree hash via `scripts/phase3a_verify_safety_copy.py`) | protected baseline (v5-a, untouched since Phase 3a step 0) | `adapter_model.safetensors` = `79b71224e2d04a6149adf63ec3fcfc825d58007ce4bfe144e5d1f0e7cb89aad5` (all 29 files byte-identical to `checkpoints/v5_sft/`, all read-only) | **MATCH** — identical to the hash recorded at training completion (`docs/V5_STATE.json` `step0_freshness_check`, 2026-08-12) AND at safety-copy creation (commit `c7c77b9`, 2026-08-15) AND right now. Three independent timestamps, zero drift. |
| `data/sft_train_v5_phase3a_merged.jsonl` | training corpus (12,901 rows) | `5123a833274a168af2d420cc833f6c51b1493202a3e2b05e06b8e44fd8e2ab6b` | **MATCH** — identical to `docs/V5_STATE.json`'s `phase3a_corpus_finalization.step4_lock.merged_corpus_sha256`, locked in AUDIT.md sec AP step 4. |
| `eval_data/LOCKED_seed999_FINAL.json` | eval population (**secondary reference only** — see step 4 below, this file is never a bar's primary numerator/denominator) | `871a9dae4c6fdf08e1aed803592fa7c61b1a852c150693b5819fe2271717b96e` | **MATCH** — identical to `docs/V5_STATE.json`'s `rule0_seed999_lock.locked_file_sha256`, sec AM. |

No mismatch. Nothing halted.

**Non-overlap confirmed (step 1d)**: these are three structurally different artifacts, not
the same file counted twice — a PEFT adapter checkpoint directory (29 files, safetensors +
configs), a line-delimited JSONL training corpus, and a JSON trajectory-population file. Their
sha256 values are of entirely different byte content, and each backs a different, necessary
role (frozen comparison baseline; what v5a2 trains on; a population used only as a cross-check
reference, never a bar denominator — see step 4).

## 1. Non-proxy `pair_accuracy` (step 2)

**Root cause of the schema gap**: `OUTPUT_SCHEMA` (`src/swarm_intent/inference.py`) has no
field stating the literal `(from, to)` formation pair. `likely_intent` is a many-to-one
function of the pair via `RULES` (multiple pairs can share an intent), so matching intent was
always a lossy proxy, not pair identification — this was the old `AUDIT.md` "step2_pair_accuracy_fix"
number, 63.6% (n=497).

**Fix, computed from already-generated text, zero new inference**: the free-text fields
(`situation_summary`, `key_indicators`, `threat_reasoning`) narrate formation names literally
— e.g. *"maintained a diamond formation... transitioning to a shield formation"* — because
`build_llm_prompt`'s `key_windows` already puts formation names in front of the model.
`llm_finetuning/literal_pair_extraction.py` extracts the model's own literally-stated
`(from, to)` pair from this text: an explicit "from A to B (to C...)" transition-chain phrase
takes priority (handles non-chronological prose like *"maintained a shield formation
throughout... transitions occurred... from column to diamond and then to shield"*, which
states the endpoint before the history); otherwise the first two formation-qualified mentions
in reading order, where "qualified" means adjacent to the word "formation"/"configuration" —
this excludes a real vocabulary collision in this project's own tactical-context template
(*"dispersing spread (mean approach_rate=...)"* uses the same word as the `dispersed`
formation name to describe a spread-dynamics **trend**, unrelated to formation identity).

**Validated by hand before being trusted at scale** (`llm_finetuning/validate_literal_pair_extraction.py`,
n=30 known-pair cases, seed=7): 19/30 (63.3%) raw agreement. Every one of the 11 mismatches
was read by hand against the model's own text; **zero were extraction bugs** — every mismatch
is the extractor correctly reporting what the model actually said, and the model said the
wrong pair (e.g. narrating a steady destination-only state while missing the true origin
entirely). This is exactly what the metric is supposed to catch, not a flaw in it.

**Real number, full has_ground_truth=True population (n=498)**:

| | old proxy (likely_intent match) | **real, literal** |
|---|---|---|
| pair_accuracy | 63.6% (n=497) | **57.8% (288/498, 1 extraction failure)** |

The real number is lower than the proxy — expected, since literal pair identification is a
strictly harder bar than matching one of several possible intents that share a pair. **This
57.8% figure, not the old 63.6% proxy, is the correct v5-a baseline for every bar below that
references v5-a's pair accuracy** (bar b's floor, bar i's regression check). The instruction
that seeded this document cited "63.6%" for the regression check — that figure is the
superseded proxy; using it here would compare v5a2's real metric against a stale,
non-comparable proxy denominator. Corrected explicitly rather than silently carried forward.

Full artifacts: `evaluation/literal_pair_extraction_validation_sample.json` (hand-check),
`evaluation/v5a_real_pair_accuracy.json` (full-population run).

## 2. The bar set (step 3)

All bars use `phase4_eval_set.json` (seed=4321) as the primary population unless stated
otherwise — see step 4 below for the full stratification table and the explicit
non-crossing-the-population-boundary confirmation.

| | bar | target | v5-a's real number (baseline) | reasoning |
|---|---|---|---|---|
| a | `threat_accuracy_pooled_when_answerable` | **>= 55.0%** | 75.65% (n=497) | unchanged from v5-a's own preregistered floor — comfortably cleared before, no reason to expect the +900 abstention rows (7.5% of corpus) to erode the answerable-case task materially. |
| b | `pair_accuracy_pooled_when_answerable` (REAL, non-proxy) | **>= 45.0%** | 57.8% (n=498, see step 1) | first time this literal metric is used as a live bar rather than only reported — set modestly below v5-a's own real number to allow genuine first-run variance in a newly-introduced metric, per this project's own "deliberately modest relative to the ceiling" precedent (`docs/PREREGISTRATION.md`). |
| c | `ceiling_normalized_accuracy` | **>= 85.0%** | 91.67% (numerator: v5-a's 75.65% on phase4_eval_set.json n=498; denominator: 82.53% same-population STGT+bridge ceiling, `llm_finetuning/compute_same_population_ceiling.py`, ALSO on phase4_eval_set.json n=498 — same file, same 498 cases, both sides, per sec AM's lesson) | modest headroom below v5-a's 91.7%, expecting some dilution from the added abstention data. |
| d | `over_abstention_rate` | **<= 15.0%** | 0.2% (n=497) | unchanged from v5-a's preregistered ceiling. Paired with (e) — see below. |
| e | `under_abstention_rate` (**NEW**) | **>= 25.0%** | 0.0% (n=502) | **Definition, stated explicitly since the name alone is ambiguous**: the abstention rate measured *under* (conditioned on) the genuinely-unanswerable population — i.e. the fraction of has_ground_truth=False cases correctly abstained on. A FLOOR is the correct operator here: v5-a scored 0.0%, closing this exact gap is this retrain's whole purpose. Together with (d)'s ceiling, a system that never abstains fails (e); a system that always abstains fails (d) on the answerable subset — neither degenerate policy can pass both. |
| f | `correct_abstention_rate_multi_hop` | **>= 25.0%** | 0.0% (measured pooled only, per-mechanism never separately reported for v5-a — that pooling is exactly the masking pattern being closed here) | multi_hop is 83.9% of the real unanswerable population (421/502, `categorize_unanswerable_502.json`) and 86.7% of the abstention corpus (780/900 rows) — the dominant, best-resourced mechanism. |
| f | `correct_abstention_rate_oscillation` | **>= 20.0%** | 0.0% (same pooling caveat) | oscillation is the minority mechanism (13.9% of the real population, 70/502; 13.3% of the corpus, 120/900 rows) — a lower floor than multi_hop is a disclosed, deliberate asymmetry reflecting less training signal, not a lowered bar for its own sake. **f is reported as two SEPARATE bars, never pooled** — sec AK's masking pattern (a strong mechanism hiding a weak one inside one pooled number) must not repeat. |
| g | `escalation_direction` | **under_escalation_rate <= 25.0%** (real numeric ceiling, not qualitative) | 18.7% under / 5.6% over (n=498) | The old bar ("under-escalation must remain the larger direction") passes by construction whenever under-escalation is merely non-zero and larger than over-escalation, which it always has been — `docs/PREREGISTRATION.md`'s own step4 erratum documents this exact finding (sec AM). Reinstated as a real number: v5-a's 18.7% + a ~6.3pp buffer for expected noise from corpus dilution, chosen BEFORE any v5a2 result exists. |
| h | `schema_validity_rate` | **DROPPED** | 100.0% (v5-a); ~99.9%+ even for the untrained BASE model (`scripts/rule0_audit_2026_08_13.py`'s `check_3_preregistration_bars`, `schema_bar_passes_by_construction`) | shown to not discriminate capability — spread < 2pp across every system ever measured, including one that was never fine-tuned. Kept in this document as a reported diagnostic (a real schema break is still worth knowing about), but is **not a PASS/FAIL bar** — an unfalsifiable-by-construction bar counted toward "N/N bars pass" would misrepresent what was actually tested. |
| i | `regression_vs_v5a` (**NEW**) | threat_accuracy **>= 70.65%** (v5-a's 75.65% − 5.0pp); pair_accuracy (real) **>= 52.8%** (v5-a's real 57.8% − 5.0pp) | see above | measured on the SAME population as (a)/(b) — phase4_eval_set.json's 498 has_ground_truth=True cases IS "the original 49-RULES-pair answerable population v5-a was scored on," there is no separate file. A 5pp tolerance is deliberately tight: step 5 below explains why the chosen training approach (fresh full-corpus QLoRA, not continued-training) makes a large regression tolerance unnecessary — this bar exists specifically to catch catastrophic forgetting, and a small, real number is what makes it capable of catching it. |
| j | `memorization_vs_generalization` (re-applied) | overlap rate **< 15.0%** signal bar (unchanged); compared against a **freshly measured** null-hypothesis baseline of **0.74% (83/11,191)**, not v5-a's reused 0.6% | v5-a: 1.3% (7/534), "consistent with generalization" | Same method (`llm_finetuning/score_memorization.py`, leave-one-out TF-IDF cosine similarity >= 0.90, teacher-authored same-pair rows only) applied fresh to `data/sft_train_v5_phase3a_merged.jsonl`. **Confirmed empirically, not assumed**: the 900 new abstention rows contribute exactly 0 pair-keyed rows to this check (`_extract_pair_from_ctx` finds no `(form_a,form_b)` pair in an abstention row's context, since it describes a multi-hop/oscillation mechanism, not a single pair) — the small baseline shift (0.6% → 0.74%) reflects a stricter teacher-authored row count (11,191 here vs 10,080 originally — the pair-keyed row set itself is byte-identical, unchanged), not a corpus composition effect. Full derivation: `evaluation/v5a2_null_hypothesis_baseline.json`. |

**9 PASS/FAIL bars total (a, b, c, d, e, f×2, g, i, j)**, plus schema_validity_rate reported
as a non-bar diagnostic.

## 3. Stratification / population boundary (step 4)

| bar | locked file | exact n | notes |
|---|---|---|---|
| a, b, c (numerator), g, i | `evaluation/phase4_eval_set.json` | 498 (has_ground_truth=True) | seed=4321 |
| c (denominator) | `evaluation/phase4_eval_set.json` (same file, `llm_finetuning/compute_same_population_ceiling.py`'s regeneration) | 498 | **same population as c's numerator, confirmed live in the same file — this is the exact cross-population mistake sec AM found and fixed; not repeated here.** |
| d | `evaluation/phase4_eval_set.json` | 497 answerable-scored | seed=4321 |
| e, f (×2) | `evaluation/phase4_eval_set.json`, has_ground_truth=False subset | 502 | ground truth from the Phase 3a simulator-based classifier (`src/swarm_intent/ground_truth_abstention.py`'s `classify_trajectory_ground_truth`, step 1's function) applied to each case's `true_chain`/`true_labels` — **never from `stgt_bridge.py`'s read.** This is the exact contamination that produced the Layer-2 deterministic attempt's 10.2% false-positive rate (AUDIT.md sec AN) — it must not leak into v5a2's eval the same way, and this document commits to that in advance. |
| j (null baseline) | `data/sft_train_v5_phase3a_merged.jsonl` | 12,901 (11,191 teacher-authored, pair-keyed) | not a trajectory population — the training corpus itself. |
| j (overlap rate, post-training) | `evaluation/phase4_eval_set.json` answered cases | up to 498 | same population as a/b/g/i. |

**`eval_data/LOCKED_seed999_FINAL.json` (seed=999) is used nowhere above as a bar's primary
numerator or denominator.** Per sec AP's population identity check, it is a genuinely
different population from `phase4_eval_set.json` (different seed, different schema, different
generation script). Its only role in this document is the secondary sanity-check reference
already established in `docs/V5_STATE.json`'s `step3_same_population_ceiling` (the
cross-population 83.0%/91.2% figures) — reported if useful, never scored as a bar. No bar
above silently pools across this boundary.

## 4. Training approach decision (step 5)

**Decision: fresh QLoRA on the full 12,901-row merged corpus, same hyperparameters as v5-a's
own run** (r=32, lora_alpha=64, all 7 target modules, base=Qwen/Qwen2.5-7B-Instruct,
assistant-only loss, effective batch 8×4=32 — `docs/V5_STATE.json`'s `step0_freshness_check`).
Step count scales proportionally with the +900/12,001 (7.5%) row increase; all else identical,
per `CLAUDE.md`'s own standing rule that v5-a's locked batch config is the default for future
runs unless a numerics-affecting change is being deliberately made (none is, here).

**Rejected alternative: continued training from v5-a's adapter on only the 900 new rows.**
Reasoning:
- **Catastrophic forgetting risk is real and structural, not hypothetical.** A second LoRA
  phase trained exclusively on 900 abstention rows (100% `threat_level="unknown"`,
  `recommended_action="monitor"`) with zero interleaved exposure to the original 12,001 RULES
  examples is the textbook setup for overwriting previously-learned behavior — there is no
  rehearsal of the answerable-case task during this hypothetical second phase at all.
- **Fresh full-corpus training interleaves abstention examples throughout the original task**,
  which is the standard mitigation for exactly this failure mode, and is also simply v5-a's
  own proven recipe with 900 more rows — the smallest, best-understood change available.
- **Compute cost is not a real factor**: 900 extra rows is a ~7.5% increase in per-epoch steps
  over v5-a's exact run (1,014 steps, 3 epochs) — negligible.
- **No validated continued-training precedent exists in this project.** v5-a itself was
  trained fresh, not continued from an earlier checkpoint. Introducing continued-training now
  — on the last attempt before demo prep — would be a new, numerics-affecting methodology with
  no equivalence check behind it, which `CLAUDE.md`'s standing rule requires before trusting
  such a change regardless of time pressure. That check was not done and there is no budget to
  do it responsibly in this session.

**This decision is why bar (i)'s regression tolerance is a tight 5pp, not a generous one**: a
continued-training approach would carry a real, expected forgetting risk that would justify
(and require) a much larger tolerance to avoid the bar failing on an accepted trade-off. Having
chosen the lower-risk fresh-training path instead, a small, strict tolerance is both
appropriate and actually capable of catching a real regression if one occurs.

## 5. Rule 0 self-check (step 6)

`scripts/rule0_audit_v5a2_preregistration.py` confirms every bar in section 2's table above
traces to a live, non-zero assertion inside `scripts/check_preregistration_v5a2.py` (not a
prose-only table entry), and that `tests/test_check_preregistration_v5a2.py` includes at least
one test per bar that FAILS when the condition making that bar meaningful is absent (e.g. a
synthetic 0%-abstention results blob must fail bar e, proving it isn't vacuous). See
AUDIT.md sec AQ step 6 for the run output.

## 6. Lock

`docs/PREREGISTRATION_V5A2.md` sha256 recorded in `docs/V5_STATE.json` the moment this file is
finalized — the timestamp that proves this document predates any v5a2 result. See
`docs/V5_STATE.json`'s `v5a2_preregistration_lock` block.

**Training was not run in this session.** The training prompt is a separate, later step,
pending sign-off on this document.

---

## Erratum — 2026-08-16 (post-lock, three definitional gaps found on review)

Append-only, per this document's own rule (section header 0). Nothing above this line was
edited. Original file lock (`docs/V5_STATE.json`'s `v5a2_preregistration_lock.file_sha256_LOCK`
= `af62a867a753a1e43993cc734a056566c3160ac8dde8f47150a9e21c6056c322`) is left untouched — it
correctly records what the document said BEFORE this erratum, which is exactly its purpose.

### 1. `under_abstention_rate` — was pooled, mathematically redundant with bar f, renamed and dropped as a scored bar

**Exact formula as implemented** (`scripts/check_preregistration_v5a2.py`,
`correct_abstention_and_per_mechanism`, quoted verbatim):
```python
gt_false = [it for it in items if not it["has_ground_truth"]]
by_mechanism: dict[str, list[bool]] = {}
pooled = []
for it in gt_false:
    parsed = parsed_by_case.get(it["name"])
    if parsed is None:
        continue
    mechanism = classify_trajectory_ground_truth(it["true_chain"], true_labels=None)
    if mechanism is None:
        continue  # not actually unanswerable by this classifier -- skip, don't misscore
    correct = is_abstention(str(parsed.get("likely_intent", "")))
    pooled.append(correct)
    by_mechanism.setdefault(mechanism, []).append(correct)
# rate(pooled) = sum(pooled) / len(pooled)
```

**Confirmed (a): pooled abstention frequency across all has_ground_truth=False cases,
regardless of mechanism** — computed live against `evaluation/phase4_eval_set.json`: **502**
has_ground_truth=False cases total, all 502 classify cleanly as either `multi_hop` (435) or
`oscillation` (67) via `classify_trajectory_ground_truth` (0 dropped as `mechanism=None`,
since every has_ground_truth=False case in this file has `len(true_chain)>=3` by
construction, which the `len(chain)>=3` branch always resolves). This corrects a smaller
inaccuracy in the original document's prose, which cited 421/70 (a different artifact,
`categorize_unanswerable_502.json`'s STGT-**bucket**-based sub-categorization, not the
ground-truth-chain classifier this script actually runs) — 435/67 is the right figure, and
also matches `evaluation/phase3a_strata_targets.json`'s own `prior_multi_hop`/`prior_oscillation`
fields, confirming it as the correct lineage.

**Confirmed: `pooled` is exactly the concatenation of `by_mechanism[MULTI_HOP]` and
`by_mechanism[OSCILLATION]`** (since `mechanism` is never `None` for this population, per
above) — so `rate(pooled)` is algebraically identical to the n-weighted average of
`correct_abstention_rate_multi_hop` and `correct_abstention_rate_oscillation`:
`(435*rate_mh + 67*rate_osc) / 502`. **It is not independent information — it is fully
determined by bar f's two numbers plus their fixed weights.** Keeping it as a third scored
bar alongside f's two components adds no anti-gaming protection (if a system never abstains,
BOTH f floors already fail independently — verified by
`tests/test_check_preregistration_v5a2.py::TestAbstentionBarsAreNotVacuous::test_never_abstaining_fails_both_mechanism_bars`)
and reintroduces exactly the "a strong mechanism can hide a weak one in one pooled number"
risk (sec AK) this document elsewhere commits to avoiding.

**Name was also misleading independent of the redundancy finding**: "under_abstention_rate"
was intended to read as "abstention rate, measured **under** (conditioned on) the
unanswerable population" — an atypical, easily misread use of "under" (the more natural
reading is "the rate of under-abstaining," i.e. failing to abstain, which would want a
CEILING, not the floor this bar used). **Corrected: renamed to
`correct_abstention_rate_pooled`**, matching bar f's own naming convention for its two
components.

**Fix applied**: removed from `BARS` (no longer scored into `OVERALL`); still computed and
reported in every run, now as a diagnostic-only row (`status: "N/A"`), same treatment as
`schema_validity_rate`. **The document's original bar table (section 2) is left as originally
written** — readers should treat its `under_abstention_rate` row as superseded by this
erratum, not edited in place.

### 2. Bar j's cited v5-a baseline — corrected in prose, confirmed no live bug

**Corrected figure: v5-a's real, re-verified memorization overlap is 0.5% (3/644, precisely
0.4658%)**, not the 1.3% (7/534) this document's section 2 table and `AUDIT.md` sec AQ cited.
The 534-case denominator predates a real pair-extraction bug fix (`AUDIT.md`, "v5-hardening-step1",
commit `5d5ea61`: *"rate moved from 1.3%/n=534 to 0.5%/n=644, verdict unchanged"*) — the
original document cited the stale, pre-fix figure by checking an earlier `V5_STATE.json`
entry (`step2_memorization_verdict`) without following the correction forward into `AUDIT.md`'s
own later record of it.

**Confirmed via `grep -n "534\|0\.013\|1\.3%" scripts/check_preregistration_v5a2.py` — zero
matches.** The stale figure appears nowhere in any executable code path. The script's actual
`check_memorization()` function compares a future v5a2 run's measured overlap rate against
exactly two live constants: `FRESH_NULL_HYPOTHESIS_BASELINE = 0.0074` (v5a2's own
corpus-derived chance baseline, unrelated to v5-a's historical number) and the fixed `0.15`
signal bar. **This was a display/reporting artifact in this document's prose and in
`AUDIT.md`'s narrative, not a live bug** — no comparison, threshold, or scored value was ever
computed from the wrong number. Fixed: `scripts/check_preregistration_v5a2.py`'s module
docstring now cites 0.5%/n=644 explicitly, with a note explaining the correction and pointing
here.

### 3. File binding for bars f (and the now-diagnostic pooled figure) — confirmed, NOT switched, flagged as an open question

**Confirmed by reading the code** (`PHASE4_EVAL_SET` constant, `load_phase4_items()`,
`scripts/check_preregistration_v5a2.py`): bars d/f/g/i and the diagnostic pooled figure are
scored against **`evaluation/phase4_eval_set.json` (seed=4321)**, with
`classify_trajectory_ground_truth` applied fresh to each case's `true_chain` — **NOT**
`eval_data/LOCKED_seed999_FINAL.json` (seed=999). This is confirmed as the CURRENT, ACTUAL
behavior, not assumed.

**This was NOT changed, and no change is made in this erratum**, for two reasons stated
plainly rather than silently deferred:

1. **It matches this document's own explicit design requirement**, set in the prompt that
   produced it: *"confirm no bar silently pools across the seed=999/seed=4321 population
   boundary sec AP found."* Switching bars f to `LOCKED_seed999_FINAL.json` while every other
   bar stays on `phase4_eval_set.json` would itself CREATE a cross-population split within
   the same bar set — the exact failure mode sec AM's lesson exists to prevent, not a fix for
   it.
2. **It is not a file-path swap.** `eval_data/LOCKED_seed999_FINAL.json` contains only raw
   generator trajectories (`chain`/`positions`/`true_labels`/`n_timesteps`) — zero model
   output, zero precomputed `ctx`/`key_windows` tactical-context text. Scoring bars f against
   it would require building and validating an entirely new STGT+bridge+context-generation
   inference pipeline over that population from scratch — substantive new work, not a bug fix,
   and out of scope for a no-training erratum review.

**A real, adjacent concern is disclosed instead of silently resolved either way**:
`evaluation/phase3a_strata_targets.json`'s 780/120 multi_hop/oscillation corpus mixture was
itself derived from `phase4_eval_set.json`'s own category proportions (via
`categorize_unanswerable_502.json`, confirmed same seed=4321 lineage). Scoring bars f on the
same population whose distribution shaped the training corpus's stratum ratio is a soft,
population-level methodological concern (not case-level literal leakage — no training row is
a real eval case) worth being honest about, not something this erratum can resolve
unilaterally. **Left open, explicitly, for a decision**: either (a) accept
`phase4_eval_set.json` for bars f with this caveat now disclosed, or (b) commission the new
`LOCKED_seed999_FINAL.json` eval pipeline as prerequisite work before v5a2 training. Not
decided here.

### Verification after the code fix

`scripts/rule0_audit_v5a2_preregistration.py` re-run: **PASS** (updated `DOCUMENT_BARS` list,
9 scored bars, no `under_abstention_rate`). Full test suite
(`python -m unittest discover -s tests`): **220/220 pass** (18 in
`test_check_preregistration_v5a2.py`, up from 17 — added a dedicated test proving the pooled
figure is reported as a diagnostic and numerically equals the n-weighted average of bar f's
two components).

---

## Erratum, part 2 — 2026-08-16 (bar-f file binding resolved)

Append-only, per this document's own rule. Nothing above this line, including erratum part 1,
was edited. Original file lock is left untouched.

### Step 1: CONFIRMED — a single v5a2 run over the seed=999 population can produce both splits, and building the eval-set-shaped file needed to do it costs ~6 minutes, CPU-only, zero new methodology

**Yes**, with one precise clarification. `eval_data/LOCKED_seed999_FINAL.json` does NOT itself
contain `ctx`/`key_windows`/`has_ground_truth` (only raw `chain`/`positions`/`true_labels`/
`n_timesteps` — confirmed in erratum part 1) — so a v5a2 LLM run cannot point at that file
directly, the same way it points at `phase4_eval_set.json`. But producing the missing fields
requires **zero new methodology**: `scripts/rule0_am_reproduce_headline.py` already runs
`sliding_window_inference` + `classify_observation` directly on this exact file's own
`positions` array (proven, already-committed code, run repeatedly for ceiling measurements).
`llm_finetuning/build_seed999_eval_set.py` (new, this session) does exactly that, mirroring
`generate_phase4_eval_set.py`'s item-construction logic verbatim but **loading** `chain`/
`positions` from the locked file instead of **generating** them fresh — no `sample_chain`,
no `build_long_sequence`, no new trajectories. Run: integrity-gated (refuses if the locked
file's sha256 has drifted), CPU-only, **6m10s wall-clock for all 1000 records**
(rate ≈2.7/s, matching `rule0_am_reproduce_headline.py`'s own precedent timing). Output:
`evaluation/seed999_eval_set.json` (sha256 `00aca8a552fce9beb0316bc929c66d825955561bcc577899f56b4d5c3b96deaf`),
in the exact same item shape `phase4_eval_set.json` uses (`name`/`true_chain`/`ctx`/
`key_windows`/`bucket`/`has_ground_truth`).

**Once this file exists, one v5a2 LLM generation pass over its 1000 items would naturally
split into both populations in the same run** — 494 has_ground_truth=True / 506
has_ground_truth=False — exactly the same mechanism `phase4_eval_set.json`'s own run used to
produce 498/502. **So: not literally zero new work (the conversion script + its ~6-minute
run), but zero new inference *methodology*, and well inside a "20-minute fix" — not the
"new pipeline" erratum part 1 correctly flagged as out of scope for a same-file switch.**
The distinction erratum part 1 drew (STGT+bridge reuse vs. a whole new pipeline) was the
right one; what was missing was recognizing the STGT+bridge half could be assembled from
already-proven pieces this cheaply.

### Step 2: rewired — bars f now score against `evaluation/seed999_eval_set.json`

`scripts/check_preregistration_v5a2.py` now takes a **separate** `--seed999-results` argument
(a v5a2 eval results JSON over `seed999_eval_set.json`'s items) — bars
`correct_abstention_rate_multi_hop`/`_oscillation` (and the diagnostic-only
`correct_abstention_rate_pooled`) are computed from it exclusively. **Bars a/b/c/g/i are
unchanged, still on `phase4_eval_set.json`.** Without `--seed999-results`, bars f report
`MISSING` — confirmed by a new test
(`test_no_seed999_results_reports_missing_not_silently_scored_from_main_results`) that the
bars do NOT silently fall back to scoring the wrong population even when the main
`results_json` happens to also contain a `parsed_by_case`.

**Mechanism counts, seed=999 unanswerable population (n=506), compared to `phase4_eval_set.json`'s
(n=502) — confirmed similar, not assumed:**

| | seed=999 (n=506) | seed=4321 / `phase4_eval_set.json` (n=502) |
|---|---|---|
| multi_hop | 443 (87.5%) | 435 (86.7%) |
| oscillation | 63 (12.5%) | 67 (13.3%) |
| chain length 3 | 248 (49.0%) | 269 (53.6%) |
| chain length 4 | 258 (51.0%) | 233 (46.4%) |

Mechanism split within ~1pp on both dimensions — close enough that bar f's floors (25%/20%)
were not re-tuned for this population; chain-length composition differs a bit more (~5pp) but
in a direction that, if anything, makes seed=999 slightly harder (more length-4 chains), not
easier — no adjustment made, flagged for visibility.

The bar set now deliberately spans two populations by design: a/b/c/g/i on seed=4321, f on
seed=999. **This is the intended fix for erratum part 1's disclosed concern** (seed=4321's own
category proportions shaped the abstention corpus's strata mixture; seed=999 was never
involved in that derivation) — not a new inconsistency, the resolution to one.

### Verification

Full test suite: **221/221 pass** (19 in `test_check_preregistration_v5a2.py`, up from 18 —
`TestAbstentionBarsAreNotVacuous` rewired to route through `seed999_results`/`seed999_items`,
plus one new MISSING-without-seed999-results test). `scripts/rule0_audit_v5a2_preregistration.py`
re-run: **PASS**.

---

## Erratum, part 3 — 2026-08-16 (full population unification)

Append-only, per this document's own rule. Nothing above this line, including erratum parts
1 and 2, was edited. All prior locks are left untouched.

### Step 1: premise corrected — no v5a2 inference exists at all

**The prompt that produced this erratum assumed a "v5a2 bars f inference run" already existed
over the seed=999-derived population. It does not.** v5a2 has not been trained in any session
— every prior erratum, including part 2, was explicit that no training has happened. The only
artifact that existed over `evaluation/seed999_eval_set.json` before this erratum was the
STGT+bridge population file itself (`ctx`/`key_windows`/`has_ground_truth`, zero LLM output).
There is no v5a2 per-case output anywhere to extract a/b/c from — step 1 as posed cannot
proceed, and nothing was "zero new LLM inference" here because there was no existing LLM
inference to reuse. Stated plainly rather than fabricating a result or silently reinterpreting
the question.

### Step 2: v5-a re-run on the seed=999 population — real GPU inference, not v5a2, not training

`checkpoints/v5_sft_v5a_PROTECTED/` sha256 re-verified first (`scripts/phase3a_verify_safety_copy.py`):
all 29 files byte-identical, read-only — **PASS**, unchanged from every prior check this
session. Evaluating v5-a (already trained, frozen, protected) over a new population is not
v5a2 training and not new training of any kind — the same category of work that already
produced `evaluation/phase4_v5a_results.json`, now reused on a different input file
(`llm_finetuning/eval_sft_v5.py --real --adapter checkpoints/v5_sft_v5a_PROTECTED/
--phase4-set evaluation/seed999_eval_set.json`, unmodified — that script already accepted an
arbitrary `--phase4-set` path). Ran in tmux, 1000 real generations, GPU (LocalHFClient,
4-bit, greedy) — result: `evaluation/v5a_seed999_results.json`.

**Comparison, seed=4321 (existing) vs seed=999 (new), same frozen v5-a model:**

| metric | seed=4321 (existing) | seed=999 (new) | gap | statistical significance |
|---|---|---|---|---|
| threat_accuracy_pooled_when_answerable | 75.65% (n=497) | **78.9% (n=493)** | +3.25pp | two-proportion z-test p=0.22 — NOT significant |
| pair_accuracy (REAL, literal) | 57.83% (288/498) | **63.56% (314/494)** | +5.73pp | two-proportion z-test p=0.065 — NOT significant at 0.05, borderline |
| abstention_rate_when_unanswerable | 0.0% (n=502) | 0.0% (n=506) | 0pp | identical |
| over_abstention_rate | 0.2% | 0.2% | 0pp | identical |
| under_escalation_rate | 18.7% | 18.4% | -0.3pp | negligible |
| over_escalation_rate | 5.6% | 2.6% | -3.0pp | not tested, smaller absolute magnitude |
| schema_validity_rate | 100.0% | 100.0% | 0pp | identical |

Both accuracy metrics are HIGHER on seed=999, consistently (not a mixed result), but neither
crosses conventional statistical significance at n≈500 each (Wilson 95% CIs overlap on both:
threat [71.7%,79.2%] vs [75.1%,82.3%]; pair [53.5%,62.1%] vs [59.2%,67.7%]). The pair_accuracy
gap (p=0.065) is the closest to significant and is flagged, not glossed over.

**A live, freshly-reproduced same-population STGT ceiling for seed=999 was also required**
(bar c's future denominator) — re-ran `scripts/rule0_am_reproduce_headline.py` fresh rather
than trusting a cited figure: **83.0% threat / 77.3% pair, n=494 pair-eligible** (exact match
to `docs/CEILING.md`'s own cited headline, confirming the CURRENT locked file, not a stale
population — a stray "509 pair-eligible" figure floating in some older `V5_STATE.json` prose
was checked directly against the live locked file and found to be from a pre-lock population;
the live, current count is 494, matching `seed999_eval_set.json`'s own `has_ground_truth=True`
count exactly).

### Steps 3/4 decision: PROCEED with unification, with one disclosed adjustment

Per the two-proportion z-tests (p=0.22, p=0.065 — neither below the conventional 0.05
threshold) and the fully overlapping Wilson 95% CIs, **the two populations are not
statistically distinguishable for v5-a's measured accuracy at this sample size** — this
satisfies "close," and unification proceeds (step 3), not the STOP branch (step 4).

**One disclosed, load-bearing adjustment**: the pair_accuracy regression tolerance is widened
from 5.0pp to **8.0pp**. Reasoning: this measurement showed that re-scoring the IDENTICAL,
frozen v5-a model on two different-but-similar populations produces a 5.73pp swing in
pair_accuracy on its own, with zero real model change. A 5pp regression tolerance would
therefore have been at real risk of firing on population-selection noise alone, not genuine
degradation — an 8pp tolerance keeps ~2.3pp of margin above the measured noise floor while
still catching a real, large regression (verified:
`test_widened_pair_tolerance_still_catches_a_real_regression`, a 0%-correct model still fails
the widened floor). Threat_accuracy's tolerance is left at 5.0pp — its observed cross-
population gap (3.25pp) was comfortably inside it already.

### Step 3: implementation — full unification

`scripts/check_preregistration_v5a2.py` fully rewritten: ALL bars (a/b/c/d/f/g/i, plus the
diagnostic pooled/schema rows) now score against `evaluation/seed999_eval_set.json` and a
SINGLE results JSON — the `--seed999-results`/dual-population plumbing from erratum part 2 is
removed entirely (no longer needed once every bar shares one population).
`evaluation/phase4_eval_set.json` is no longer read by this script at all (it remains a valid,
separately-locked artifact for other purposes elsewhere in the project, untouched).

| constant | old (seed=4321) | new (seed=999) |
|---|---|---|
| `SAME_POPULATION_CEILING` | 0.8253 | **0.830** |
| `V5A_REAL_THREAT_ACCURACY` | 0.7565 | **0.789** |
| `V5A_REAL_PAIR_ACCURACY` | 0.5783 | **0.6356** |
| threat regression tolerance | 5.0pp | 5.0pp (unchanged) |
| pair regression tolerance | 5.0pp | **8.0pp** |

Smoke-tested end-to-end against the real `evaluation/v5a_seed999_results.json` (v5-a scoring
itself — a real, not synthetic, sanity check): threat/pair/ceiling/regression bars all PASS
(as expected, v5-a vs its own baseline); bars f correctly FAIL (v5-a's known 0.0%
correct-abstention gap, now confirmed present on this population too, not an artifact of the
seed=4321 corpus-derivation concern erratum part 1 raised).

### Verification

Full test suite: **222/222 pass** (20 in `test_check_preregistration_v5a2.py`, up from 19 —
`TestAbstentionBarsAreNotVacuous` simplified back to the single-population form, plus one new
test confirming the widened pair tolerance still catches a real regression).
`scripts/rule0_audit_v5a2_preregistration.py` re-run: **PASS**.
