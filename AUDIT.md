# AUDIT — measured from disk, not from the handoff doc

Every number below was produced by a script that reads files directly (`ast.parse` +
`ast.literal_eval` for `RULES`/`OUTPUT_SCHEMA`/`INTENT_FAMILIES`/`THREAT_FAMILIES`/
`TEST_CASES` — no `import swarm_intent`, so no torch/groq needed for those checks;
`torch.load` is used only inside the adapter-forensics check, wrapped so its absence
doesn't kill the run). Each of the 7 checks (A-G) ran in its own try/except; all 7
reported `ok=true` on this run. Raw machine-readable output: `audit.json` (repo root).
This document does not change any source file — measurement only.

---

## A. Inventory

| Claimed path | Exists? |
|---|---|
| `swarm_data/best_model.pt` | **False** |
| `swarm_data/norm_stats.npy` | **False** |
| `src/swarm_intent/llm/pipeline.py` | **False** |
| `evaluation/finetuned_eval.json` | **False** |

**Handoff claim confirmed: all 4 do not exist.**

Supplementary (not requested but free to check): `swarm_data/` itself doesn't exist
either (no partial artifacts). `.git` exists. All three adapter dirs
(`adapters/qwen-swarm`, `adapters/qwen-swarm-v2`, `adapters/smoke-test`) exist. The
three legacy notebooks named in `CLAUDE.md` (`capstone_with_llm.ipynb`, `capstone_with
eval.ipynb`, `models + data generation.ipynb`) do **not** exist on disk.
`llm_finetuning/configs/qlora_qwen2.5-7b.yaml` exists (its emptiness of actual effect
on training is a separate, previously-noted finding, not re-verified in this run).

---

## B. Adapter forensics

**Handoff claim: `assistant_only_loss` is `False` on both real training runs — confirmed, not refuted, on all three adapters.**

| Adapter | `assistant_only_loss` | epochs | batch×grad_accum | lr | bf16/fp16 | packing | LoRA r/α | final ckpt | final train loss | final eval loss | mean_token_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `smoke-test` | **False** ⚠️ | 1.0 | 1×8 | 2e-4 | True/False | False | 16/32 | checkpoint-3 | n/a (1 log entry, eval only) | 1.567 | — |
| `qwen-swarm` | **False** ⚠️ | 3.0 | 1×8 | 2e-4 | True/False | False | 16/32 | checkpoint-1014 | 0.0528 | 0.0529 | 0.9778 |
| `qwen-swarm-v2` | **False** ⚠️ | 3.0 | 1×8 | 2e-4 | True/False | False | 16/32 | checkpoint-306 | 0.0725 | 0.0756 | 0.9712 |

All three: `target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`,
`base_model_name_or_path = Qwen/Qwen2.5-7B-Instruct`, `dataset_text_field = "text"`.

`qwen-swarm` eval loss by epoch: 0.0535 → 0.0532 → 0.0529 (flat from epoch 1).
`qwen-swarm-v2` eval loss by epoch: 0.0831 → 0.0770 → 0.0756 (also flat, slightly worse than v1).

**Reading this straight:** `assistant_only_loss=False` on every real run means TRL
computed loss over the full templated prompt + JSON answer, not the answer alone.
Loss/accuracy converging this fast (2-3 epochs, ~500-800 rows) is consistent with the
model memorizing a highly repetitive prompt template rather than learning to reason —
the loss numbers by themselves cannot distinguish "learned good tactical reasoning"
from "learned to reproduce a templated prompt." This audit does not attempt to resolve
that ambiguity; it only confirms the flag's value and the loss trajectory.

---

## C. Dataset ledger

| File | rows | malformed lines | assistant-JSON parse fail | distinct user prompts | distinct summaries | distinct follow_up_watch | templated-fallback rows |
|---|---|---|---|---|---|---|---|
| `sft_train.jsonl` | 2700 | 0 | 0 | 2700 | 49 | 1 | 2700 (100.0%) |
| `sft_train_val.jsonl` | 300 | 0 | 0 | 300 | — | — | — |
| `sft_train_v2.jsonl` | 810 | 0 | 0 | 810 | 430 | 355 | 404 (49.9%) |
| `sft_train_v2_val.jsonl` | 90 | 0 | 0 | 90 | — | — | — |
| `sft_train_v3.jsonl` | 406 | 0 | 0 | 406 | 406 | 406 | 0 (0.0%) |
| `sft_train_v3_val.jsonl` | 47 | 0 | 0 | 47 | — | — | — |
| `sft_train_final.jsonl` | 234 | 0 | 0 | 234 | 227 | 230 | 0 (0.0%) |
| `sft_train_final_val.jsonl` | 26 | 0 | 0 | 26 | — | — | — |
| `smoke_train.jsonl` | 20 | 0 | 0 | 20 | — | — | 20 (100.0%) |
| `smoke_val.jsonl` | 4 | 0 | 0 | 4 | — | — | — |

Zero malformed lines and zero assistant-JSON-parse failures across every file — the
generation pipeline's JSON hygiene is clean, no corruption. `sft_train.jsonl`'s 49
distinct summaries against 2700 rows is expected — 49 = the number of distinct `(threat,
intent)`-driven template strings possible from `RULES`' 49 `(from,to)` pairs, and
`distinct_follow_up_watch=1` because the templated fallback's `follow_up_watch` string
is a hardcoded constant regardless of scenario.

**Train/val leakage: zero shared user prompts in every one of the 5 train/val pairs**
(`sft_train`↔`sft_train_val`, `v2`↔`v2_val`, `v3`↔`v3_val`, `final`↔`final_val`,
`smoke_train`↔`smoke_val`). No leakage detected.

**RULES coverage in `sft_train_final.jsonl`:** the 49 `(from,to)` rule entries collapse
to **21 distinct `(threat, intent, action)` triples** (many formation-pairs share the
same decision triple). All **21/21** distinct triples appear at least once in
`sft_train_final.jsonl`'s 234 rows — **0 missing.** Full decision-triple coverage
despite the small row count.

---

## D. Vocab 3-way diff

**`threat_level`** — perfect match, no diffs anywhere:
`RULES_emits == OUTPUT_SCHEMA_allows == THREAT_FAMILIES_keys == {low, medium, high, critical}`.

**`likely_intent`:**
- `RULES` emits 14 distinct intents — every one of them **is** covered by `INTENT_FAMILIES`
  (`RULES_minus_INTENT_FAMILIES = []`). **No RULES-trained intent is at risk of being
  scored as a hallucination by `is_hallucination()`.**
- `OUTPUT_SCHEMA` allows 15 values (`RULES`' 14 + `"unknown"`). `"unknown"` is legal per
  schema but appears in **neither** `RULES` nor `INTENT_FAMILIES` — if the model ever
  emits `"unknown"`, `is_hallucination()` has no family entry to match it against and
  will flag it, even though the schema explicitly allows it as a value.
- `INTENT_FAMILIES` has one extra key, `"attack"`, that `RULES` never emits — dead
  vocabulary in the family dict, not itself a bug, just unused.

**`recommended_action`:**
- `RULES` emits 4 actions, all legal per `OUTPUT_SCHEMA` (`RULES_minus_OUTPUT_SCHEMA = []`).
- `OUTPUT_SCHEMA` allows a 5th value, `"intercept"`, that `RULES` never emits — legal but
  untrained, the mirror case of `"unknown"` above.
- **No `ACTION_FAMILIES` dict exists anywhere in `prompts.py`** — there is no fuzzy
  matcher for `recommended_action` at all, so this field's hallucination-risk can't be
  assessed the way threat/intent can.

**Net finding:** every value `RULES` actually produces is legally schema-compliant and
has a matching evaluator family — the “trained-for but unscorable” failure mode this
check was built to catch does **not** occur. The only gap running the other direction:
`"unknown"` (intent) is schema-legal but has no family, so a correct abstention would be
mis-scored as a hallucination if the model ever produces it.

---

## E. Prompt drift

`grep -rn "Return ONLY valid JSON" **/*.py` (excluding `.venv`) — 3 hits:

| File:line | Context |
|---|---|
| `src/swarm_intent/llm/client.py:87` | `GroqClient.generate` — prepended to every prompt sent to the hosted baseline |
| `src/swarm_intent/llm/client.py:123` | `LocalHFClient.generate` — prepended to every prompt sent to the local/fine-tuned model |
| `src/swarm_intent/llm/prompts.py:83` | Inside `JUDGE_PROMPT` — unrelated, this is the independent judge's instruction, not the SFT/inference pipeline |

**`llm_finetuning/build_sft_dataset.py` has zero hits.** The SFT training rows save
`build_llm_prompt(...)`'s raw output as the user turn with no such prefix; both
`GroqClient` and `LocalHFClient` add it at generation time. Confirmed: the prefix is
present at inference/eval time and absent at training time — a real, measured train/
inference input mismatch (2 of the 3 hits are the pipeline; the third is the judge,
a different consumer entirely).

---

## F. Scale check

**Synthetic ranges parsed out of `synth_context()`** (`llm_finetuning/build_sft_dataset.py`):

| label | low | high |
|---|---|---|
| `approach` | -1.5 | 0.5 |
| `delta_v` | -1.0 | 2.0 |
| `mean_conf` | 0.7 | 0.98 |
| `mean_stab` | 0.6 | 0.95 |
| `velocity` | 0.0 | 0.1 |

**Real values collected from `evaluation/llm_run_output.json` +
`evaluation/sliding_window_predictions_demo.json`** (recursive key-match on the exact
names `centroid_velocity`, `approach_rate`, `stability`, `confidence`, `velocity`,
`approach`, `delta_velocity`):

| key | n | min | max | mean |
|---|---|---|---|---|
| `centroid_velocity` | 18 | 0.04 | 0.072 | 0.0628 |
| `approach_rate` | 18 | -0.001 | -0.0 | -0.00089 |
| `delta_velocity` | 1 | 0.018 | 0.018 | 0.018 |
| `stability` | 0 | — | — | — |
| `confidence` | 0 | — | — | — |
| `velocity` | 0 | — | — | — |
| `approach` | 0 | — | — | — |

`stability`/`confidence`/`velocity`/`approach` come back **n=0** — not because the real
pipeline doesn't produce these numbers, but because the captured demo files use the
prefixed field names `formation_stability` / `formation_confidence` / `centroid_velocity`
/ `approach_rate`, not the short names `synth_context()`'s `key_windows` dict uses. That
naming mismatch between what the real prediction dict emits and what the synthetic
`key_windows` builder labels its own fields is itself worth registering, separate from
the magnitude question below.

**Magnitude ratios** (synthetic max |value| ÷ real max |value|, only computed where real n≥3):

| synthetic label | real key | synthetic range | real n | real max\|·\| | ratio |
|---|---|---|---|---|---|
| `approach` | `approach_rate` | [-1.5, 0.5] | 18 | 0.001 | **1500.0×** |
| `velocity` | `centroid_velocity` | [0.0, 0.1] | 18 | 0.072 | **1.4×** |
| `delta_v` | `delta_velocity` | [-1.0, 2.0] | **1** | — | **N/A — real n=1 < 3, ratio not computed** |
| `mean_conf` | `confidence` | [0.7, 0.98] | **0** | — | **N/A — real n=0 < 3, ratio not computed** |
| `mean_stab` | `stability` | [0.6, 0.95] | **0** | — | **N/A — real n=0 < 3, ratio not computed** |

### Verdict

- **Do ALL real `delta_velocity` values fall inside ±0.5 (the "steady" band)?** Yes —
  but on **n=1** sample only. Not a robust claim; it's the one number captured on disk.
- **Are ALL real `approach_rate` values above -0.1 (the "converging" threshold)?** Yes,
  on all **n=18** samples (range -0.001 to -0.0 — essentially zero, nowhere near the
  threshold).
- **`velocity_and_converging_branches_dead_in_production: true`** for the data measured.
  Every captured real `delta_velocity` sample sits inside the "steady" band and every
  real `approach_rate` sample sits above the "converging" threshold. In the file(s)
  actually on disk, `vel_trend` can never read `accelerating`/`decelerating` and
  `spread_dynamics` can never read `converging`/`dispersing`. This is n=1 and n=18
  samples respectively from 2 files — it is evidence from what was captured, **not**
  proof for all possible real sensor inputs, since only one real pipeline run has ever
  been saved to disk (see `evaluation/finetuned_eval.json` non-existence in §A).
- The one ratio computed with adequate sample size (`approach` vs `approach_rate`, n=18)
  shows the synthetic sampling range is **1500× larger in magnitude** than every real
  value observed. The `delta_v`/`mean_conf`/`mean_stab` ratios could not be computed —
  too few or zero real samples exist to trust a ratio, exactly per the n<3 guard.

---

## F2. Closing the stability/confidence gap (follow-up to §F)

§F reported n=0 for `stability` and `confidence` because those exact key strings don't
appear in the two real capture files. Inspected the files directly before writing
anything (no guessing): the actual keys present are `formation_stability`,
`formation_confidence`, `mean_stability`, `mean_confidence` (plus `confidence_in_assessment`,
a string enum, and `stability_trend`, a string label — both excluded, not numeric).

| key | n | min | max | mean |
|---|---|---|---|---|
| `formation_stability` | 18 | 0.7922 | 0.9562 | 0.8333 |
| `formation_confidence` | 18 | 0.6286 | 0.9679 | 0.8991 |
| `mean_stability` | 1 | 0.833 | 0.833 | 0.833 |
| `mean_confidence` | 1 | 0.899 | 0.899 | 0.899 |

Combined `formation_stability` + `mean_stability` (n=19): **every value is above 0.7**
(min 0.7922).

**Correction to the premise before answering:** the fixed `0.7` "holding" threshold does
**not** live in `build_tactical_context` (the function that actually narrates production
output, `src/swarm_intent/inference.py:122`). That function uses a *relative* rule with
no absolute cutoff at all:
```python
stab_trend = "degrading" if late < early - 0.1 else "improving" if late > early + 0.1 else "holding"
```
(compares the mean of the first half of a window sequence against the second half; a
swing of more than ±0.1 between halves is what flips it out of "holding" — the absolute
level never enters the comparison). The literal `mean_stab > 0.7` cutoff exists
only in `synth_context()` (`llm_finetuning/build_sft_dataset.py:143`), i.e. it drives the
**synthetic training-data narrative**, not the real inference-time narrative. These are
two different formulas that happen to both currently be labeled "holding" — not the same
code path.

**So, answered precisely:**
- Applying `synth_context`'s literal `mean_stab > 0.7` rule to every real stability value
  captured on disk (n=19): **all 19/19 would read "holding."** If this rule is what the
  fine-tune's training distribution is anchored to, it is trained overwhelmingly (see §F's
  `mean_stab` sampling range `[0.6, 0.95]`, which straddles 0.7 without matching real
  data's much narrower and consistently-higher band) on a threshold that real data almost
  always sits on one fixed side of.
- Production's actual mechanism (the ±0.1 relative-delta rule in `build_tactical_context`)
  is a different, independently-dead-or-not question: the one captured real run's stored
  `stability_trend` field is `"holding"` — but that's the outcome of the relative-delta
  formula on that run's specific early/late split, not evidence the branch is structurally
  unreachable the way `vel_trend`/`spread_dynamics` were shown to be in §F (those had a
  fixed numeric threshold the real magnitude range could never cross; this one is
  relative, so it can in principle fire given a big enough within-window swing — this
  audit did not attempt to re-derive early/late means from the raw per-window sequence to
  check how close that one run came to the ±0.1 boundary).
- Net: the *synthetic-data-generation* narrative branch (`synth_context`'s absolute 0.7
  cutoff) is dead against every real sample measured, same pattern as §F. The
  *production* narrative branch (`build_tactical_context`'s relative rule) is not shown
  dead by this data — it's a different mechanism this audit did not exercise.

---

## G. Eval coverage

`TEST_CASES` in `src/swarm_intent/llm/prompts.py`: **6 cases.**

| case | (formation_a, formation_b) | expected_intent | expected_threat | RULES threat/intent/action | contradiction? |
|---|---|---|---|---|---|
| Converging Attack | (dispersed, converging) | approach | high | high / approach / alert_operator | No |
| Stable Patrol | (column, column) | patrol | low | low / patrol / monitor | No |
| Encirclement Behavior | (v_shape, encirclement) | encircle | high | high / encircle / alert_operator | No |
| Defensive Shield | (shield, shield) | defensive | medium | medium / defensive / monitor | No |
| Area Search | (diamond, dispersed) | area_search | medium | medium / area_search / monitor | No |
| Breaking Contact | (converging, dispersed) | withdraw | low | low / withdraw / monitor | No |

**0/6 contradictions** — every test case's stated expectation matches `RULES[(a,b)]`
exactly (both threat and intent).

**Coverage: 6/49 distinct rule pairs touched** (12%). **2/6 cases are de-escalation**
(rule threat == `low`): Stable Patrol, Breaking Contact.

---

## Summary of what this run measured vs. did not measure

Measured and confirmed from disk: the 4 handoff-claimed-missing paths are indeed
missing; `assistant_only_loss=False` on all 3 real training runs; dataset JSON hygiene
is clean with zero train/val prompt leakage; the RULES vocabulary is fully covered by
both the output schema and the evaluator's fuzzy-match families (with one asymmetric gap
around `"unknown"`); the `"Return ONLY valid JSON"` prefix is added at inference but not
training time; the one real captured pipeline run shows `approach_rate` magnitudes ~1500×
smaller than what the synthetic data generator samples; and all 6 eval test cases agree
with the canonical rule table with zero contradictions, covering 12% of the rule space.

Not measured here (out of scope for these 7 checks, flagged for awareness only): whether
the loss-masking issue actually degrades real generation quality (would require running
inference, which this session did not do), and whether the scale mismatch holds beyond
the single captured real run (no second real pipeline run exists on disk to check against).

---

## H. Mechanism diff — `build_tactical_context()` vs `synth_context()`

Read-only. `build_tactical_context()` (`src/swarm_intent/inference.py:100-157`) is what
actually narrates production/inference-time output. `synth_context()`
(`llm_finetuning/build_sft_dataset.py:122-158`) is what narrates every SFT training
example. F2 already established these differ for `formation_stability`. This section
checks the same question for every other narrative field both functions emit.

| field | production rule (`build_tactical_context`) | synth rule (`synth_context`) | absolute or relative | same mechanism? |
|---|---|---|---|---|
| **Velocity trend** | `delta_v` = mean(late-half `centroid_velocity`) − mean(early-half); `accelerating` if `delta_v>0.5`, `decelerating` if `<-0.5`, else `steady` (lines 116-117) | `delta_v` sampled directly `~U(-1.0,2.0)`; same `>0.5`/`<-0.5`/else formula applied to the sampled value (line 131-132) | both: absolute ±0.5 threshold on a `delta_v` value | **Y** — classification formula is identical; only the *source* of `delta_v` differs (derived-from-data vs directly-sampled), which is the input-scale mismatch §F already covers, not a separate mechanism gap |
| **Spread dynamics** | 3-way: `<-0.1` → `converging`, `>0.1` → `dispersing`, else `stable spread` (lines 126-128) | 2-way only: `<-0.1` → `converging`, **else `stable spread`** — no `dispersing` branch exists at all (line 144) | both: absolute threshold on `approach`/`approach_rate`, same −0.1 cutoff | **N** — synth is missing an entire branch. Its `approach` range is `U(-1.5,0.5)`, which *can* land above +0.1; any such row is written into training data as `"stable spread"` even though production would call the identical value `"dispersing"`. Not found in earlier sections. |
| **Formation stability** | relative early-vs-late delta, `±0.1` (line 122) — no absolute level ever enters it | absolute cutoff `mean_stab > 0.7` (line 143) | production: relative / synth: absolute | **N** (confirmed in §F2, restated here for completeness) |
| **Classifier confidence** | reports `mean_confidence` **and** a count of low-confidence windows, `conf<0.6` (line 132, 147: `"... ({low_conf} low-confidence windows)"`) | reports only the mean — no low-confidence count sub-statistic exists anywhere in `synth_context` (line 146: `"Classifier confidence: mean={mean_conf:.2f}"`) | N/A (not a branch; a whole sub-statistic) | **N** — the model has never seen a "N low-confidence windows" phrase during training; production emits one on every single run |
| **Role differentiation** | unconditionally emits `"Role differentiation: present/not prominent"`, from `role_flag` = majority of windows with `role_differentiation=True` (lines 130, 146) | **field does not exist anywhere in `synth_context`** — no line, no variable, nothing | N/A — absent, not just differently-thresholded | **N**, and the most severe gap in this table: every real production prompt contains this line; **zero** SFT training examples do |
| **Formation history** | `' -> '.join(dict.fromkeys(formation_seq))` — arbitrary-length deduplicated sequence of whatever the model actually predicted per window, can be 1..N distinct formations, `"transitioning"` only appears if that was a genuine predicted class (line 138, 150) | fixed template: either just `form_a`, or exactly `"{form_a} -> transitioning -> {form_b}"` — capped at 2 real formations, and the literal word `"transitioning"` is always hardcoded into the middle regardless of whether any class prediction produced it (line 134, 139) | N/A — structural/generative, not numeric | **N** — production is a data-driven sequence of arbitrary length; synth is a fixed ≤3-token template |
| **Transition lines** | computes every consecutive-window formation change, each with a real timestamp from that window's actual `time_start_s`; can be 0, 1, or many; format `"Transition at t={t}s: {from} -> {to}"` (lines 107-110, 140-141) | at most ONE transition, always at the hardcoded literal `t=20.0s` regardless of any sampled parameter; format `"Transition detected at t=20.0s: {form_a} -> {form_b}"` — note the wording itself differs too ("Transition at" vs "Transition detected at") (line 140) | N/A — structural | **N** — production supports multiple real-timestamped transitions; synth hardcodes exactly one transition at a constant that means nothing |

**Net for section H:** of the 7 fields checked, only **velocity trend** uses the literal
same classification mechanism in both functions (the mismatch there is purely upstream
input scale, already covered by §F). The other 6 — spread dynamics, formation stability,
classifier confidence, role differentiation, formation history, and transition lines —
all differ, and not uniformly in the same way: some are absolute-vs-relative threshold
swaps (stability), some are missing branches (spread dynamics), some are missing
sub-statistics (confidence), and two are fields/structures **entirely absent** from every
training example while appearing in every production prompt (role differentiation,
and the multi-transition/arbitrary-length-history structure). This scopes Phase 3 exactly
as intended — nothing here has been fixed.

---

## I. Qualitative failure analysis — v3b on the two held-out shapes it does NOT abstain on

Read-only, no new generations. Source: `evaluation/holdout_v3b.json` (already on disk from
the held-out-shapes session), specifically `shapes.oov_formation.per_case` and
`shapes.dominant_mismatch.per_case`. Both shapes are `n_runs=5`, `has_ground_truth=False`
for all 6 cases (one per `ORIGINAL_TEST_CASES` base) — correct behaviour is abstention, and
v3b abstains 0% of the time on both.

**Data-availability caveat:** `evaluate_llm` (`src/swarm_intent/llm/evaluate.py`) persists
only the per-case *majority* `intent`/`threat`/`action` (the mode across the 5 runs), not
the full raw JSON assessment (`situation_summary`, `threat_reasoning`, `key_indicators`
prose) for any run. That prose was never written to disk, so it cannot be quoted here
without a new generation — which this section is explicitly barred from running. The
analysis below is therefore over the structured decision fields only (what "3 examples" can
mean from the data that actually exists), not literal prose quotes.

### oov_formation: does it map the unfamiliar formation onto a known one?

Context shape: `"Dominant formation: {form_a}"` / `"Transition detected at t=20.0s:
{form_a} -> phalanx"` — "phalanx" is not in `BASE_FORMATIONS`, not in `RULES`, not in any
training row (verified in the held-out-shapes session). A clean-looking, resolvable
transition line where the destination is vocabulary RULES cannot key on.

| base case | majority intent / threat / action | matches `RULES[form_a -> X]` for any real `X`? |
|---|---|---|
| Converging Attack (form_a=dispersed) | `defensive_transition` / `medium` / `increase_surveillance` | **X = shield** (`RULES[dispersed,shield] = (medium, defensive_transition, monitor)` — action differs, rest matches) |
| Stable Patrol (form_a=column) | `defensive_transition` / `medium` / `monitor` | **X = shield**, exact match: `RULES[column,shield] = (medium, defensive_transition, monitor)` |
| Encirclement Behavior (form_a=v_shape) | `defensive` / `medium` / `monitor` | no exact `RULES[v_shape,X]` match for intent=`defensive` |
| Defensive Shield (form_a=shield) | `defensive_transition` / `medium` / `monitor` | no exact `RULES[shield,X]` match (shield has no outgoing `defensive_transition` entry) |
| Area Search (form_a=diamond) | `defensive_transition` / `medium` / `increase_surveillance` | **X = shield**, intent matches: `RULES[diamond,shield] = (medium, defensive_transition, monitor)` |
| Breaking Contact (form_a=converging) | `defensive` / `medium` / `monitor` | **X = shield**, intent matches: `RULES[converging,shield] = (medium, defensive, monitor)` |

4 of 6 cases produce an intent that exactly matches `RULES[form_a -> shield]`, and every
single case (6/6) answers with either `defensive` or `defensive_transition` at `medium`
threat — the two intents `RULES` reserves almost exclusively for transitions into `shield`.
**Reading: v3b is not recognizing "phalanx" as an unfamiliar/unanswerable token — it is
silently treating it as if it were "shield"** (plausibly the nearest embedding neighbor, or
the formation most associated with unfamiliar/defensive-sounding names), and then answering
RULES' shield-transition logic confidently. This is a hallucination of a specific,
identifiable kind (vocabulary substitution), not random noise: `hallucination_rate=0.0` in
the aggregate because the *output vocabulary* is schema-legal throughout — `is_hallucination`
only checks token validity, not whether the input was answerable, so this failure mode is
invisible to every metric this project has computed so far except this manual read.

### dominant_mismatch: which contradictory line does the assessment follow?

Context shape: `"Dominant formation: {X}"` (X != form_a, != form_b) paired with a
self-consistent `"Formation history: {form_a} -> transitioning -> {form_b}"` and
`"Transition detected at t=20.0s: {form_a} -> {form_b}"` — the ONE line that's wrong is
"Dominant formation", contradicting two other lines that agree with each other.

| base case | dominant (contradiction) says | history (correct) says | model majority | intent follows history? | threat/action follows history? |
|---|---|---|---|---|---|
| Converging Attack | v_shape steady: medium/surveillance/increase_surveillance | dispersed->converging: high/approach/alert_operator | approach / **medium** / **increase_surveillance** | Y | **N — threat AND action both pulled fully to the dominant line's values** |
| Stable Patrol | v_shape steady: medium/surveillance/increase_surveillance | column->column: low/patrol/monitor | patrol / low / monitor | Y | Y |
| Encirclement Behavior | column steady: low/patrol/monitor | v_shape->encirclement: high/encircle/alert_operator | encircle / high / alert_operator | Y | Y |
| Defensive Shield | v_shape steady: medium/surveillance/increase_surveillance | shield->shield: medium/defensive/monitor | defensive / **low** / monitor | Y | partial — threat dropped to `low`, matching neither line exactly |
| Area Search | v_shape steady: medium/surveillance/increase_surveillance | diamond->dispersed: medium/area_search/monitor | area_search / medium / monitor | Y | Y (threat coincides with both lines here; action follows history) |
| Breaking Contact | v_shape steady: medium/surveillance/increase_surveillance | converging->dispersed: low/withdraw/monitor | withdraw / **medium** / monitor | Y | partial — threat pulled toward dominant's `medium`, action stayed correct |

**`likely_intent` follows the correct "Formation history"/"Transition detected" line in
6/6 cases (100%) — it never once adopts the contradictory "Dominant formation" line's
implied intent.** But `threat_level` is measurably contaminated by the wrong line in 3/6
cases, most cleanly in "Converging Attack" where BOTH `threat_level` and
`recommended_action` snap exactly onto the contradictory dominant formation's own
steady-state RULES answer (`medium`/`increase_surveillance`) instead of the correct
`high`/`alert_operator` the actual transition warrants — a threat-level miss in the
dangerous direction (under-stating risk on what RULES calls a `high`-threat scenario).
The other 2 affected cases show partial contamination (threat pulled toward the
contradictory line's value without a full match). Reading: **the fields are not decided
from one unified read of the context — intent classification appears to key primarily off
the explicit "Transition detected" line, while threat-level assessment is more diffuse and
can be pulled off by an unrelated, contradictory line elsewhere in the same context.**

### Net for section I

Both failure modes are real and distinct, and both would be invisible to every metric this
project has reported except a manual per-case read: `oov_formation` is a **vocabulary
substitution** hallucination (confident, schema-legal, wrong because the input token was
silently replaced with a known one); `dominant_mismatch` is a **selective-field
contamination** — the model resolves `likely_intent` correctly from the right source line
but lets `threat_level`/`recommended_action` leak information from a contradictory,
irrelevant line in the same context. Neither shows up as `hallucination_rate>0` or
`over_abstention_rate>0` in the aggregate; both are only visible by reading per-case
majority-vote fields against `RULES` by hand, which this section did but no automated
metric in this codebase currently does.
