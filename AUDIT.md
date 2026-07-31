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

## J. The 55-case battery is in-distribution — its 100% scores are recall, not generalization

`qwen-swarm-v2` scored 100.00% ± 0.00% `accuracy_when_answerable` on the 55-case
`TEST_CASES` battery (identical to `rules_lookup`, the symbolic oracle) across all 5
runs. Before citing that number, `llm_finetuning/check_training_coverage.py` was
written to check the obvious confound: does v2's training data already contain every
`(formation_a, formation_b)` pair the battery tests?

The script parses `Dominant formation:` / `Formation history:` out of each row's user
message in the three SFT files (`build_sft_dataset.py`'s `synth_context()` writes both
deterministically, so the exact `(form_a, form_b)` pair each training row was sampled
for can be recovered exactly, not inferred) and cross-references against the 49 keys
of `RULES` (a complete function over `BASE_FORMATIONS × BASE_FORMATIONS`, 7×7=49 — see
`CLAUDE.md`).

| training file | rows | RULES pairs covered | examples per covered pair (min/mean/max) |
|---|---|---|---|
| `sft_train_v2.jsonl` (trains v2) | 810 | **49/49 (100%)** | 8 / 16.5 / 28 |
| `sft_train_final.jsonl` (trains v3a, v3a-nomask) | 234 | **49/49 (100%)** | 1 / 4.8 / 11 |
| `sft_train_final_abstain.jsonl` (trains v3b) | 270 (32 rows are OOV/held-out-shape, non-pair) | **49/49 (100%)** | 1 / 4.9 / 11 |

**All three files — meaning every adapter evaluated in section headline tables,
not just v2 — already cover 100% of the 49 RULES pairs the 55-case `TEST_CASES`
battery draws on.** Every `RULES_COVERAGE_CASES` entry has a same-pair counterpart
somewhere in that adapter's training data (though not a byte-identical row: the
random `delta_v`/`approach_rate`/`stability` floats and window text differ per
`synth_context()` call, so this is rule-table-pair recall under a noisy surface
form, not verbatim example memorization).

**Conclusion: `TEST_CASES` (`ORIGINAL_TEST_CASES + RULES_COVERAGE_CASES`) is a
rule-table RECALL battery for every fine-tuned system in this project, not a
generalization test.** v2's 100% — and the other adapters' `accuracy_when_answerable`
figures on this same battery (section headline tables, step 3/4) — should be reported
as "recall of the trained rule table," not cited as evidence of generalization. This
caveat is now recorded in code comments at `src/swarm_intent/llm/prompts.py`
(`RULES_COVERAGE_CASES`) and `llm_finetuning/run_headline_eval.py`. The actual
generalization test in this project remains `llm_finetuning/holdout_shapes.py` (AUDIT.md
secs G/I), whose three shapes are verified absent from training data by string search —
`TEST_CASES` makes no such claim and never has.

## K. Stratified abstention re-test — does ~100% hold beyond the original 6 base cases?

The `multi_hop`/`terminal_transitioning` ~100% abstention finding (AUDIT.md secs G/I,
`degradation_v3b.json`) was measured on perturbations of only the original 6
`ORIGINAL_TEST_CASES`. `llm_finetuning/run_stratified_abstention_retest.py` re-runs
just those two axes against a stratified 15-case sample drawn from the full 55-case
pool: both `critical`-threat cases forced in, the remaining 13 slots allocated
proportionally across `low`/`medium`/`high` (4/6/3), and within each threat stratum
selection round-robins across distinct `formation_a` values for family diversity
(6/7 `BASE_FORMATIONS` represented as `formation_a`, deterministic seed 42). 90
perturbed cases (45 per axis) × 5 runs = 450 generations against `qwen-swarm-v3b`
(~39 min actual).

| axis | abstention_rate_when_unanswerable (mean ± std across runs) | original 6-case-battery figure |
|---|---|---|
| `multi_hop` (sev 3–4 unanswerable) | **100.00% ± 0.00%** | 100% |
| `terminal_transitioning` (all severities unanswerable) | **94.67% ± 1.78%** | 100% |

`over_abstention_rate` on `multi_hop`'s answerable sev-2 cells stayed 0.00% (no
new over-triggering introduced by the larger sample).

**Reading: `multi_hop` abstention survives unchanged at 100% on the more diverse
15-case sample. `terminal_transitioning` abstention drops slightly, from 100% to
94.67% ± 1.78% — real but small, and non-zero variance across runs (the original
6-case measurement had none, at that scale one flip is enough to look categorical).**
This does not overturn sec G/I's conclusion (`terminal_transitioning` abstention is
still overwhelmingly reliable, and is a within-mechanism check, not evidence of
transfer to a structurally different unanswerability mechanism like contradiction or
OOV vocabulary, which sec G already found does NOT transfer) — but it should be
reported as ~95%, not ~100%, going forward.

## L. Masking effect as a gradient — scales with distribution shift

`llm_finetuning/report_masking_gradient.py` puts v3a-vs-v3a-nomask (the clean
masking 2×2 isolating `assistant_only_loss`, sec H) side by side on the
in-distribution clean 55-case battery (sec J — recall, not generalization) and the
out-of-distribution degradation battery (secs C/H), pooling severities per axis
weighted by `n_cases_with_ground_truth`.

| battery | v3a acc | v3a-nomask acc | delta |
|---|---|---|---|
| clean 55-case [in-distribution] | 68.7% ± 2.7% (across runs) | 55.3% ± 3.7% (across runs) | **+13.5 pts** |
| degradation: `multi_hop` sev2 [perturbed] | 83.3% ± 18.0% (across cases) | 66.7% ± 29.8% (across cases) | +16.7 pts |
| degradation: `confidence_decay` [perturbed] | 84.0% ± 26.5% (across cases) | 60.0% ± 36.9% (across cases) | +24.0 pts |
| degradation: `dropped_lines` [perturbed] | 81.1% ± 24.5% (across cases) | 70.0% ± 34.2% (across cases) | +11.1 pts |
| degradation: `contradictory_cues` [perturbed] | 91.1% ± 15.2% (across cases) | 55.6% ± 36.9% (across cases) | +35.6 pts |
| **mean of the 4 perturbed axes** | | | **+21.8 pts** |

Std caveat: the clean-battery std is across RUNS (5 independent replicate
whole-battery measurements — `evaluate.py`'s `*_std_across_runs`, added mid-session).
The degradation-battery std is across CASES within an axis (`degradation_v3a*.json`
predates that instrumentation; retroactively getting run-level std there would mean
re-running the ~1080-generation battery for both adapters, not undertaken without being
asked). The two stds measure different kinds of dispersion and should not be read as
directly comparable error bars — only the deltas are being compared here.

**Finding: the masking effect is larger under perturbation (+11.1 to +35.6 pts,
mean +21.8) than on the clean in-distribution battery (+13.5 pts). Masking's benefit
scales with distribution shift — it is not a fixed constant, and reporting a single
number for "the masking effect" understates what it does specifically under the kind
of input degradation (dropped lines, decayed confidence, contradictory cues, deeper
chains) this project's degradation battery was built to simulate.**

## M. Threat-level confusion matrices — the low-threat collapse is over-escalation to `medium`

Sec (per-class breakdown, prior session) found `threat_level` accuracy on `low` cases
collapses (v3a 2.7%, v3a-nomask 0.0%, v3b 13.3%) despite the pairs being in-distribution
(sec J). `llm_finetuning/report_confusion_matrices.py` builds predicted-vs-expected
confusion matrices from the existing 55-case battery JSONs (one vote per case — its
majority label across the 5 runs, not per-run; `evaluate_llm` never persisted individual
run predictions, see the script's docstring for the full caveat) to find out what `low`
is being predicted AS.

**`low` cases collapse to `medium`, specifically — never to `high` or `critical`:**

| system | low→low | low→medium | low→high | low→critical | n |
|---|---|---|---|---|---|
| v2 | 13 | 2 | 0 | 0 | 15 |
| v3a | **0** | **15** | 0 | 0 | 15 |
| v3a-nomask | **0** | **15** | 0 | 0 | 15 |
| v3b | 2 | 13 | 0 | 0 | 15 |

v3a and v3a-nomask predict `medium` on literally every one of the 15 `low` cases — this
is a systematic bias, not scattered noise. **This is over-escalation** (the system
reports more threat than is warranted, in the direction a human operator would rather
err toward, but still a real accuracy failure the demo needs to know about).

The same central-tendency pull shows up at the other end of the scale, in the opposite
direction: both `critical` cases (n=2, not statistically meaningful on their own, but
consistent with the pattern) get predicted `high` for v3a/v3a-nomask/v3b — under-escalation
toward the center. And v3a/v3a-nomask under-escalate part of `high` toward `medium` too
(v3a: 5/14 high cases → medium; v3a-nomask: 3/14 → medium). **Reading: these adapters are
not learning four distinct threat thresholds — they are regressing toward `medium` as a
default/central answer, pulling both extremes inward.** v2 (810 rows) is the only adapter
that does NOT show this pattern (13/15 low correct, all high/critical correct) — see sec N
for whether that's a training-data-size or data-composition effect.

`recommended_action` shows the same directional bias on the `monitor` action (the action
tied to `low`/some `medium` cases): v2 31/31 correct; v3a 24/31 pulled to
`increase_surveillance`, v3a-nomask 19/31, v3b 11/31 — same escalation direction,
consistent with the threat-level collapse rather than an independent failure.
`likely_intent` does NOT show this pattern — it is highly diagonal for all four systems
(few off-diagonal cells, no consistent directional pull), meaning **the collapse is
specific to `threat_level` (and its downstream `recommended_action`), not a general
inability to read the context.** Full per-system intent/action matrices in
`evaluation/confusion_matrices.json`.

## N. Training-data class balance — proportional, not underrepresented (correction: RULES is 13/49 low, not 15/49)

Correction first: the session's opening premise cited "RULES' own 15/49" low-threat
share. That 15 is the pooled 55-case `TEST_CASES` figure (`RULES_COVERAGE_CASES` + the
6 `ORIGINAL_TEST_CASES`, 2 of which are themselves `low`). `RULES` alone (49 pairs) is
**13/49 (26.5%) low**, 22/49 (44.9%) medium, 12/49 (24.5%) high, 2/49 (4.1%) critical
(`llm_finetuning/report_class_balance.py`) — the 2-case difference doesn't change the
conclusion below, but the number should be cited correctly going forward.

`llm_finetuning/report_class_balance.py` parses the `threat_level` field out of every
row's ASSISTANT TARGET (not the input context) in the three SFT files and compares
each file's share to RULES' 13/22/12/2 baseline:

| file (trains) | low | medium | high | critical |
|---|---|---|---|---|
| RULES (baseline) | 26.5% | 44.9% | 24.5% | 4.1% |
| `sft_train_v2.jsonl` (v2) | 26.7% (216 rows) | 43.6% | 25.9% | 3.8% |
| `sft_train_final.jsonl` (v3a, v3a-nomask) | 24.8% (**58 rows**) | 47.0% | 24.8% | 3.4% |
| `sft_train_final_abstain.jsonl` (v3b) | 21.5% (58 rows) | 54.1% | 21.5% | 3.0% |

**All three files are within a few points of RULES' own class balance — low-threat
rows are NOT severely underrepresented relative to RULES.** `sft_train_final_abstain.jsonl`
(v3b) shows the largest skew (low −5.0pts, medium +9.2pts) but it's still a mild skew,
not the kind of order-of-magnitude imbalance that would on its own explain a collapse
from ~27% representation to 2.7%/0.0%/13.3% recall.

**This confirms the session's opening hypothesis: with 58 low-threat training rows
(sft_train_final.jsonl, proportionally represented) producing 2.7% (v3a) / 0.0%
(v3a-nomask) recall on low-threat test cases, class imbalance is not the explanation.**
The failure mode is the sec M over-escalation-toward-`medium` bias — something about
*how* these ~58 rows are learned (see sec O for whether v2's 216 low-threat rows, same
proportion but ~4x the count, are diverse examples or near-duplicates), not how many of
them there are.

## O. Threshold hypothesis (memorization capacity vs data quality) — NOT supported by the data

The hypothesis under test: v2 (810 rows, doesn't collapse) differs from
`sft_train_final.jsonl` (234 rows, collapses) in both size AND teacher composition
(CLAUDE.md: v2 is ~50% teacher/50% template fallback; `final` is ~100% teacher). If
v2's advantage is really "16 near-identical template rows per pair," that's
memorization capacity, not data quality, and should be described that way.
`llm_finetuning/report_template_fallback.py` tests this directly by counting DISTINCT
`situation_summary` strings per `(form_a, form_b)` pair — `build_sft_dataset.py`'s
teacher-fallback template is a fixed string per pair regardless of context noise, so
"1 distinct summary across N rows of a pair" is a template/memorization signature, while
teacher-written rows vary row to row even for the same pair.

| file | template-fallback rows | pairs where ALL rows are template | mean distinct summaries/pair | mean examples/pair |
|---|---|---|---|---|
| `sft_train_v2.jsonl` | 404/810 (49.9%) | **0/49** | 8.78 | 16.5 |
| `sft_train_final.jsonl` | **0/234 (0.0%)** | 0/49 | 4.63 | 4.8 |

**The hypothesis is not supported.** `sft_train_v2.jsonl` is not "16 near-identical
rows per pair" — every one of its 49 pairs has at least 4 distinct teacher-written
summaries, mean 8.78/16.5 ≈ 53% of its rows per pair are unique prose. And
`sft_train_final.jsonl` (the file that DOES collapse) is *more* purely teacher-written
than v2 (0% template fallback vs v2's 49.9%), not less — if template-fallback rows were
memorization filler, `final.jsonl` should be the higher-quality file by this measure,
yet it's the one that fails.

Breaking down by threat class within each file also rules out `low` being
selectively template-heavy or duplicate-heavy relative to `medium`/`high` in either
file — `low` pairs have comparable or slightly *higher* example/distinct-summary counts
than `medium` in both files (v2: low 16.62/9.15 vs medium 16.05/8.45; `final`: low
4.46/4.46 vs medium 5.00/4.73).

**What's left, by elimination: the two files differ materially only in raw example
count per pair (16.5 vs 4.8, a ~3.4x difference) — not teacher-vs-template composition,
not diversity, not class-specific representation.** The evidence points at plain
sample-size sensitivity (v3a/v3a-nomask/v3b all sit at ≤4.8 examples/pair and collapse
on `low`; v2 sits at 16.5/pair and doesn't) rather than any of the compositional
explanations this step set out to test — stated plainly per the session's instruction,
since the hypothesis being tested was not confirmed. This is inferred from elimination
across two data points (not a controlled sweep across intermediate example-per-pair
counts), and confirming it further would require training an intermediate-size adapter,
which this session was explicitly told not to do.

## P. Does the collapse show up under perturbation too? — yes for v3a/v3a-nomask, mixed for v3b

`llm_finetuning/report_degradation_by_class.py` re-cuts the existing degradation-battery
JSONs (perturbations of the 6 `ORIGINAL_TEST_CASES` along 5 axes — only 2 of the 6 are
`low`: "Stable Patrol", "Breaking Contact"; there is no `critical` base case in this
battery at all) by the base case's `expected_threat`, giving `threat_accuracy` under
perturbation per class.

| system | low threat_accuracy | medium threat_accuracy | high threat_accuracy |
|---|---|---|---|
| rules_lookup | 100.0% (n=23) | 100.0% (n=25) | 100.0% (n=24) |
| v2 | **100.0%** | 64.0% | 100.0% |
| v3a | **13.0%** | 96.0% | 50.0% |
| v3a-nomask | **0.0%** | 98.4% | 50.8% |
| v3b | 52.2% | 76.8% | 52.5% |

Caveat: n=23/25/24 here are perturbations of only 2/2/2 underlying base cases each —
correlated samples from a handful of scenarios, not 23+ independent draws; treat the
spread (std, not shown in this table but in the script output) accordingly.

**For v3a and v3a-nomask, the collapse is systemic — it shows up under perturbation at
the same order of magnitude as the clean 55-case battery (v3a: 13.0% perturbed vs 2.7%
clean; v3a-nomask: 0.0% both), not an artifact specific to the 55-case battery's
particular pairs.** v2 does not collapse under perturbation either, consistent with sec
M/N/O. **v3b is the outlier: 52.2% low-threat accuracy under perturbation, much higher
than its 13.3% on the clean battery** — with only 2 underlying base cases this could be
those 2 specific scenarios happening to perturb more forgivingly for v3b, not a real
reversal of the collapse; not enough independent samples here to resolve which.

**Every pooled degradation-battery number reported previously in this project (secs C,
H, L) needs this per-class footnote: the aggregate `accuracy_when_answerable` figures
average over a threat-level mix that, for v3a/v3a-nomask at least, is masking a
near-total `low`-threat failure underneath a `medium`/`high` accuracy that looks fine.**

## Q. Demo impact — can the system say "low" on a benign steady-state formation?

Concrete answer for the literal capstone-demo scenario ("Stable Patrol" /
`column->column` — a drone group holding a steady column formation, RULES says
`low`/`patrol`/`monitor`, the textbook benign case):

| adapter | `threat_level` on this exact scenario | across all 55 cases, how often does it ever output `low`? |
|---|---|---|
| `qwen-swarm-v2` | **`low`** (correct, 5/5 runs) | 14/55 |
| `qwen-swarm-v3a` | `medium` (wrong, 2/5 runs said `low`) | **0/55 — never** |
| `qwen-swarm-v3a-nomask` | `medium` (wrong, 0/5 runs said `low`) | **0/55 — never** |
| `qwen-swarm-v3b` | `medium` (wrong, 1/5 runs said `low`) | 2/55 |

**If the capstone demo runs a benign, steady-state, low-threat formation through
`qwen-swarm-v3a` or `qwen-swarm-v3a-nomask`, the system will say `medium` threat
(`increase_surveillance`), not `low` (`monitor`) — and it will do this on essentially
every low-threat input, not occasionally. `v3a`/`v3a-nomask` cannot currently emit
`low` at all on this battery (0/55).** `v3b` can, rarely (2/55, ~13% correct on
low-threat cases per sec M). **`qwen-swarm-v2` is the only adapter that behaves
correctly here** (`low` on the benign case, `low` predicted 14/55 times, matching the
RULES base rate).

This is a visible, easily-triggered demo behavior, not an edge case: any operator
running an idle/patrolling swarm through `v3a` or `v3a-nomask` during a live demo will
see the system call it `medium` threat and recommend `increase_surveillance` on a
scenario RULES and the training data both label `low`/`monitor`. **Recommendation: if
the demo needs a fine-tuned (not base/rules_lookup) system, use `qwen-swarm-v2` — it is
the only adapter without this failure mode** (secs M–P all converge on v2 not
collapsing, most plausibly for the sample-size reasons in sec O). If `v3b`'s
abstention behavior (secs G/I/K) is specifically what's being demonstrated, be aware it
inherits the same near-total low-threat failure as v3a — pick a demo scenario
deliberately, not one drawn at random from `low`-threat inputs.

## S. Field-structure check — does `likely_intent` survive low-data fine-tuning better than `threat_level`/`recommended_action`?

(Note on lettering: the requesting session asked for this to be appended as "section
Q" — that letter was already used for sec Q above, "Demo impact," written earlier in
the same day's work before this request arrived. Using the next free letter, S,
instead of renumbering what's already committed.)

`llm_finetuning/report_field_structure.py` puts `mean_intent_accuracy`,
`mean_threat_accuracy`, `mean_action_accuracy` side by side for every system
evaluated on the 55-case battery, to test whether `likely_intent` (whose correct
values often lexically echo the input formation names — e.g. `encircle` for an
`encirclement` transition) is systematically easier to learn from limited data than
`threat_level`/`recommended_action` (which require applying the full arbitrary RULES
mapping with no lexical shortcut).

| system | intent | threat | action | intent > threat | intent > action |
|---|---|---|---|---|---|
| `rules_lookup` | 100.0% | 100.0% | 100.0% | n/a (ceiling) | n/a (ceiling) |
| `base` | 25.8% | 50.5% | 19.6% | **no** | yes |
| `rules_in_prompt` | 59.3% | 57.5% | 69.5% | yes | **no** |
| `qwen-swarm-v2` | 100.0% | 94.5% | 99.3% | yes | yes |
| `qwen-swarm-v3a` | 68.7% | 56.4% | 44.7% | yes | yes |
| `qwen-swarm-v3a-nomask` | 55.3% | 57.1% | 50.9% | **no** (by 1.8pts) | yes |
| `qwen-swarm-v3b` | 77.5% | 61.1% | 63.3% | yes | yes |

**The pattern does not hold universally — it is not a clean rule.** `intent > threat`
in 4/6 non-oracle systems (fails for `base` and `v3a-nomask`, the latter only by 1.8
points); `intent > action` in 5/6 (fails for `rules_in_prompt`). It holds for every
*fine-tuned* adapter except `v3a-nomask`'s near-tie on `threat`, and fails for both
non-fine-tuned baselines in different directions (`base` has weaker intent than
threat; `rules_in_prompt` has weaker intent than action) — so the "lexically-
recoverable fields survive" story is a reasonable read **for the fine-tuned adapters
specifically**, not a fact about the task's field structure in general that would hold
for any system reading these prompts.

## R. Epoch-matched control (`qwen-swarm-v3c`) — under-training ruled out, `low` does NOT recover

Sec O's "examples-per-pair" explanation was confounded the whole time with optimizer
steps: `llm_finetuning/report_step_counts.py` reads `global_step` straight from each
adapter's `trainer_state.json` —

| adapter | rows | epochs | global_step | steps/epoch |
|---|---|---|---|---|
| `qwen-swarm-v2` | 810 | 3.0 | **306** | 102 |
| `qwen-swarm-v3a` | 234 | 3.0 | **90** | 30 |
| `qwen-swarm-v3a-nomask` | 234 | 3.0 | 90 | 30 |
| `qwen-swarm-v3b` | 270 | 3.0 | 102 | 34 |

v2:v3a step ratio (306:90 = 3.40:1) is nearly identical to sec O's examples-per-pair
ratio (16.5:4.8 = 3.44:1) — not a coincidence: epochs and effective batch size (8) are
constant, so step count is a direct linear function of row count. Every sec M–P finding
attributed to "examples-per-pair" was equally attributable to "fewer optimizer steps,"
undistinguished.

**The decisive test:** `train_qlora.py` was extended with `--load-best-model`
(`load_best_model_at_end`, `metric_for_best_model="eval_loss"`) and `--progress-task`
(a `Reporter`-driven `TrainerCallback`), then `qwen-swarm-v3c` was trained on
`sft_train_final.jsonl` — v3a's exact 234-row file, `assistant_only_loss=True`, same
hyperparameters — for `--epochs 10.2` (`306 target steps / 30 steps-per-epoch`,
computed exactly, not estimated) to match v2's 306 steps.

**Eval loss by epoch — a clean overfitting curve, minimum at epoch 2:**

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 10.2 (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_loss | 0.550 | **0.495** | 0.497 | 0.532 | 0.589 | 0.641 | 0.780 | 0.836 | 0.874 | 0.876 | 0.876 |

Eval loss nearly **doubles** from its epoch-2 minimum (0.495) to the epoch-10 endpoint
(0.876) — training past ~epoch 2–3 on this 234-row file actively overfits, it does not
converge toward v2-like performance. `--load-best-model` correctly rolled back and
saved **`checkpoint-60` (epoch 2, 0.495 eval_loss)** as the final `qwen-swarm-v3c`
adapter — meaning the "best" version of this epoch-matched run ends up trained for
*fewer* effective steps than even v3a itself (60 vs v3a's 90), because nothing past
that point improves on held-out loss.

**v3c vs v3a on the 55-case battery, per threat class (`--n-runs 5`):**

| system | overall acc | low threat_acc | medium threat_acc | high threat_acc | critical threat_acc | low predicted as |
|---|---|---|---|---|---|---|
| v3a | 68.7% ± 2.7% | 2.7% | 92.5% | 60.0% | 0.0% | 15/15 → `medium` |
| v3c | 68.0% ± 1.9% | **1.3%** | 80.8% | 81.4% | 0.0% | **15/15 → `medium`** |

(per-case `threat_accuracy`, mean within each expected-threat stratum, both computed
identically — directly comparable.)

**Verdict, stated plainly per the session's instruction: `low` stays at ~0%. v3c does
NOT recover.** More optimizer steps on the same 234-row file does not fix the collapse
— it cannot, because the model already overfits and eval loss climbs well before it
would reach v2's step count. **This rules out under-training as the explanation.** By
elimination (class balance ruled out in sec N, template/memorization composition ruled
out in sec O, and now step count ruled out here), what's left is genuine **data
diversity**: v2's ~16.5 examples per rule-pair, even with roughly the same ~50%
template-fallback content sec O already showed isn't the mechanism, appears to give the
model enough *varied restatements* of what "this specific pair is low-threat" looks
like to actually learn the mapping, where v3a/v3c's ~4.8 distinct-but-few examples per
pair do not — more steps over the same few examples just memorizes/overfits those
specific rows instead of generalizing the class.

**Per the session's explicit instruction, `qwen-swarm-v4` was NOT trained** — step 3 was
conditional on v3c recovering `low`, and it did not. The path to fixing this failure
mode is more/more-diverse low-threat training examples, not more training steps or a
masking change; training `v4` on the existing `sft_train_final_abstain.jsonl` (which
has the same ~4.8-examples-per-pair ceiling) would not be expected to fix it either,
and was not attempted on that basis.

## T. GPU memory benchmark — measured, not estimated (RTX 4090, 24GB)

`scripts/bench_memory.py` measures `torch.cuda.max_memory_allocated()` for the two
real workloads this box runs. Generation runs through the SAME `LocalHFClient`
4-bit/fp16 path production eval uses; training runs through the SAME `SFTTrainer`
path `train_qlora.py` uses (not a hand-rolled forward/backward — see the debugging
note below on why that distinction mattered).

| phase | grad_checkpointing | batch | peak GB | fits ≥15% headroom (≤19.98GB)? |
|---|---|---|---|---|
| train | False | 1 | 9.14 | YES |
| train | False | 2 | 9.46 | YES |
| train | False | 4 | 10.19 | YES |
| train | False | 8 | 13.09 | YES |
| train | True | 1 | 9.14 | YES |
| train | True | 2 | 9.46 | YES |
| train | True | 4 | 10.20 | YES |
| train | True | 8 | 13.10 | YES |
| generate | n/a | 1 | 5.76 | YES |
| generate | n/a | 8 | 5.86 | YES |
| generate | n/a | 16 | 6.11 | YES |
| generate | n/a | 32 | 6.60 | YES |
| generate | n/a | 64 | 7.58 | YES |

**Every configuration tested fits comfortably** — even train batch=8 (13.1GB) and
generate batch=64 (7.6GB) leave well over 15% headroom on a 24GB card. Nothing here
needed to be estimated or extrapolated; all 13 rows are real measurements.

**Two real bugs surfaced and were fixed while building this benchmark, both worth
recording since they'd silently corrupt any future memory sweep that reused the same
mistakes:**

1. A hand-rolled forward/backward reimplementation (instead of going through the real
   `SFTTrainer` path) used `padding="max_length"` fixed at 1024 tokens for every batch
   regardless of actual content length. This alone produced a reproducible false OOM at
   **batch=1** (22GB+ for a single sequence) — contradicting the fact that v2/v3a/
   v3a-nomask/v3b/v3c had all just trained successfully at batch=1 on this exact GPU
   minutes earlier. Real training rows are 668–813 tokens (mean 734); forcing every
   benchmark sequence to 1024 combined with subtly different loss-computation memory
   behavior outside the real trainer path produced numbers that had nothing to do with
   reality. Fixed by routing the training benchmark through the actual `SFTTrainer`
   class (same dynamic-padding collator as production).
2. Reusing one Python process across all 8 (grad_checkpointing, batch_size) configs —
   reloading a fresh model object between configs but staying in the same CUDA context —
   leaked memory specifically **after a config OOM'd mid-step**: the next config's model
   reload itself then OOM'd during `prepare_model_for_kbit_training` (21.95GB already "in
   use" before the new model finished loading), because `del`/`gc.collect()`/
   `empty_cache()` could not fully reclaim a trainer that failed mid-backward. Caught by
   cross-checking one contaminated result (`grad_checkpointing=True batch=8`, which
   crashed during the *next* config's load) against an isolated re-measurement — 13.10GB
   clean vs an apparent OOM contaminated by the leak. Fixed by giving **every config its
   own subprocess** (fresh CUDA context, not just a fresh model object) — costs a few
   extra seconds of reload time per config but is the only way every number here is
   actually trustworthy.

**Interesting real finding, not guessed:** at this project's actual sequence lengths
(~734 tokens mean) and LoRA-only backprop (base model frozen), `gradient_checkpointing`
makes essentially **no measurable memory difference** (9.14/9.46/10.19–10.20/13.09–13.10
GB, matching to within rounding at every batch size). This is plausible precisely
*because* only ~40M LoRA parameters need gradients out of 7.6B total — there's
comparatively little activation memory to trade away by checkpointing in the first
place, unlike full fine-tuning where checkpointing's savings are much larger. This does
not mean `gradient_checkpointing=True` should be dropped (step 5 keeps it per the
session's request, and it costs nothing measurable here), just that it isn't doing the
heavy lifting some might assume.

## X. Coverage grid and teacher source, before committing to a v4 coverage-aware pipeline

(Note on lettering: the requesting session asked for "section T" — already claimed
earlier the same day by the throughput-optimization session's memory benchmark (and
U/V/W are referenced-but-pending from that same in-flight session). Using the next
genuinely free letter, X.)

**(a) 49/49 RULES-pair coverage, re-confirmed.** `llm_finetuning/report_coverage_grid.py`
part (a) reruns sec J's check: `sft_train_v2.jsonl`, `sft_train_final.jsonl`,
`sft_train_final_abstain.jsonl` all still cover 49/49. Unchanged.

**(b) The 49×54 = 2,646-cell grid is 6.39% populated — but only 12/54 combinations per
pair are even reachable, independent of sample size.** `sft_train_final.jsonl`
populates **169/2,646 cells (6.39%)**, mean 1.38 rows/cell. But `context_spec.py`
defines 3×3×3×2=54 theoretical combinations per pair while `synth_context()`
(`build_sft_dataset.py`, the only generator that ever produced this file) can only ever
reach a fraction of them:
- `velocity_trend`: all 3 values reachable and realized (`accelerating`/`decelerating`/`steady`)
- `spread_dynamics`: only 2/3 reachable — `synth_context()`'s condition is a **binary**
  `if approach < -0.1: converging else: stable`; `"dispersing (drones spreading out)"`
  is never producible by this generator regardless of how much data you sample
- `stability_trend`: only 2/3 reachable — binary `if mean_stab > 0.7: holding else:
  degrading`; `"improving"` is never producible
- `role_differentiation`: **never rendered in the training prompt text at all**. It's
  hardcoded `False` in every row's `key_windows` (`build_sft_dataset.py:268`), the
  `build_llm_prompt` key-window JSON renderer excludes that field regardless
  (`inference.py` lines ~189–193), and `summary={}` is passed empty for every SFT row —
  so this "dimension" carries zero information for the model to condition on, in any
  training row, ever.

Reachable grid: 3 × 2 × 2 × 1 = **12 combinations/pair**, not 54 — so relative to what
this data-generation pipeline can even produce, coverage is 169 / (49×12) = **28.7%**,
not 6.39%. Both numbers are real; which one is the relevant denominator depends on
whether a v4 pipeline would use a *different* generator (in which case the full
54-cell/2,646-cell grid is the right target) or extend the *existing* `synth_context()`
(in which case 12-cell/588-cell is what's actually achievable without first fixing the
generator's binary thresholds).

**(c) RULES ignores all four dimensions — confirmed structurally, not by sampling.**
`gold_assessment()`'s only rule lookup is `RULES.get((form_a, form_b), DEFAULT_RULE)` —
read directly from the function source, not inferred from row statistics. `RULES` has
no other inputs. Every one of the 54 (or 12 reachable) velocity/spread/stability/role
combinations for a fixed pair maps to an **identical** `(threat_level, likely_intent,
recommended_action)` triple by construction — this cannot be otherwise, since the
function that assigns labels literally cannot see those fields.

**Conclusion, stated as precisely as the session asked: the label space (49 RULES
pairs → 21 distinct decision triples, sec C) is fully covered in every training file.
What is sparse in the 54-cell-per-pair grid is LABEL-INVARIANT INPUT VARIATION —
narrative phrasing/metadata the model was never asked to condition its answer on — NOT
missing decision regions.** A "coverage-aware" pipeline proposal built on this grid
should be framed as targeting **input diversity / robustness to narrative phrasing**,
not label coverage, which is already complete and was never the sparse resource.

**Teacher source audit.** `data/sft_train_final.jsonl` carries no per-row model
metadata (`build_sft_dataset.py`'s `main()` only ever writes `{"messages": [...]}`
per row — no teacher-identifying field exists to record). The file itself predates
this repository's traceable git history: `git log --all -- data/sft_train_final.jsonl`
shows exactly one commit, the initial handoff snapshot (`2653c96`) — it was never
generated by a `build_sft_dataset.py` invocation this engagement can see or replay, so
no `--teacher-model` flag value is recoverable from commit history either. (Contrast
with `sft_train_final_abstain.jsonl`, sec commit `5ddcec8`, built LIVE this engagement,
whose 234 base rows are confirmed byte-identical carries from `sft_train_final.jsonl` —
no new teacher calls for those.)

Given no direct evidence is recoverable, the honest answer rests on two pieces of
circumstantial-but-consistent evidence: (1) `CODE_REVIEW.md` (written independently,
before this engagement, describing the *original* pre-existing pipeline) states the
project's LLM layer used exactly one model, **`llama-3.3-70b-versatile`** via Groq —
which is also `GroqClient`'s hard-coded default in this codebase; (2) a stylistic scan
of all 234 `situation_summary` values shows a single, narrow voice — 88% (207/234)
open with "A" or "The", a unimodal length distribution (mean 190 chars, std 48.7, no
second cluster), and consistent phrasing patterns ("A swarm of UAVs is currently...",
"The UAV swarm is transitioning from...") repeated with only scenario-level variation,
not the kind of register-switching multiple distinct models typically produce.

**Verdict: best available evidence — independent documentation plus a stylistic
homogeneity check, not certainty — indicates `sft_train_final.jsonl` is single-source
(`llama-3.3-70b-versatile`). No evidence of multiple source models was found.** Per the
session's framing (Synthetic Eggs, arXiv 2511.01490, concerns *source-model* diversity,
not writing-style diversity): if this verdict holds, multi-teacher generation would be
a **real, untried intervention** for a v4 pipeline, not a no-op — the current data has
never had more than one model's failure modes baked into it. This is inference from
absence of contrary evidence, not a proven fact; a v4 decision resting heavily on this
point should treat it as "no evidence of multi-source found," not "confirmed
single-source."

## U. Batched generation, and its equivalence check — PASS, ~2.8x speedup

`LocalHFClient.generate_batch()` (`src/swarm_intent/llm/client.py`) adds left-padded,
length-sorted batched generation alongside the existing single-prompt `generate()`
(unchanged, still the default everywhere `--batch-size` isn't passed).
`llm_finetuning/baselines.py`'s new `make_batched_run_case()` pre-generates every
`(case, run)` completion for a battery in one batched pass, in the exact case-major/
run-minor order `evaluate_llm` calls `run_case` — so `ctx` text is byte-identical to
the unbatched path (same seeded RNG draws), and it's a drop-in replacement wherever
`make_rules_in_prompt_run_case` was used. Wired into `run_headline_eval.py` via
`--batch-size` (default 1 = exact reproduction).

**Equivalence check (`llm_finetuning/run_batching_equivalence_check.py`,
`qwen-swarm-v3a`, 55-case battery, `--n-runs 5`, batch_size=8 from sec T's memory
bench):**

| threat class | unbatched threat_acc | batched threat_acc | \|delta\| | within 1 unbatched-std? |
|---|---|---|---|---|
| low | 5.3% (std 13.6%) | 4.0% (std 10.8%) | 1.3% | yes |
| medium | 92.5% (std 20.7%) | 91.7% (std 20.7%) | 0.8% | yes |
| high | 60.0% (std 38.5%) | 61.4% (std 39.6%) | 1.4% | yes |
| critical | 0.0% (std 0.0%) | 0.0% (std 0.0%) | 0.0% | yes |

Overall `accuracy_when_answerable`: unbatched 66.18%±4.96%, batched 68.00%±3.37% —
within one std. **PASS in every class.** (Note: this run's unbatched `low` figure,
5.3%, differs slightly from sec M's 2.7% for the same `qwen-swarm-v3a` — expected
run-to-run sampling noise at `temperature=0.3`, not a discrepancy between this check
and prior sections; both are well within the collapsed regime this whole session's
diagnosis concerns.)

**Wall-clock: unbatched 22m36s → batched 8m07s, a 2.78x speedup** for the full
55-case×5-run battery on one system — real, substantial, and below the naive 8x
(batch_size) ceiling as expected (padding waste on the longest sequence per batch,
and generation is only partially parallel-scaling since it's still autoregressive per
token). **Batching is confirmed safe to adopt elsewhere in this project's eval/
degradation runners** — this check was the gate for doing so, per the session's
explicit instruction not to mix batched and unbatched results in the same comparison
table without it passing first.

## Y. Logit inspection — the low-threat collapse is MIXED: calibration for v3b, partly genuine gap for v3a

(Note on lettering: the requesting session asked for "section T" — already claimed by
sec T earlier the same day (throughput session). Using the next free letter, Y.)

`llm_finetuning/logit_inspection.py` greedily generates each case's real completion,
locates the exact token position where `"threat_level": "`'s value begins, and does
sequence-scored teacher-forcing over the four candidate strings (handles multi-token
candidates correctly) to get `P(low)/P(medium)/P(high)/P(critical)` — a distribution
over the four candidates, not the full vocabulary. Then applies a logit-prior
correction (`corrected_logP(c) = raw_logP(c) - log(train_freq(c))`, each model's own
training file's class frequency) and re-scores.

**v3a — 15 low-threat cases:**

| case | P(low) | P(medium) | raw | corrected |
|---|---|---|---|---|
| Stable Patrol | 51.1% | 48.8% | low | low |
| Breaking Contact | 73.7% | 26.3% | low | low |
| column→column | 51.1% | 48.8% | low | low |
| diamond→diamond | 56.1% | 43.7% | low | low |
| dispersed→dispersed | 70.2% | 29.7% | low | low |
| converging→dispersed | 73.7% | 26.3% | low | low |
| diamond→column | 43.4% | 56.6% | medium | **low** (flipped) |
| encirclement→column | **7.0%** | 84.3% | medium | medium |
| encirclement→dispersed | **13.7%** | 84.2% | medium | medium |
| converging→column | 15.5% | 83.7% | medium | medium |
| v_shape→column | 27.7% | 71.9% | medium | medium |
| v_shape→dispersed | 32.6% | 66.9% | medium | medium |
| column→dispersed | **10.9%** | 88.2% | medium | medium |
| dispersed→column | **8.6%** | 86.6% | medium | medium |
| shield→dispersed | **9.3%** | 89.3% | medium | medium |

v3a low-threat accuracy: raw (greedy) **40.0%** → corrected **46.7%** (+6.7pts, n=15).

**v3b — same 15 cases:** raw (greedy) **53.3%** → corrected **86.7%** (+33.4pts, n=15) —
only `encirclement→column` and `encirclement→dispersed` remain wrong after correction
(both stuck with `P(low)` in the 19–22% range, not competitive even after the prior
shift).

**Critical (n=2, both models):** raw 0.0% → corrected 50.0%. `converging→encirclement`
flips to the correct `critical` after correction (both models). `encirclement→converging`
stays `high` in both models even after correction — `P(critical)` raw is only 5.4–7.7%,
nowhere close to competitive.

**Verdict — stated plainly, and it does NOT collapse into a clean binary:**

- **For v3b, this is predominantly a decoding/calibration problem.** A simple, free
  prior correction recovers low-threat accuracy from 53.3% to 86.7% — 13/15 cases. The
  v4 coverage-aware data pipeline is **not justified by this evidence** as the fix for
  v3b's low-threat collapse; a cheap decoding-time correction gets most of the way
  there already.
- **For v3a, the picture is genuinely mixed, not purely calibration.** 6/15 cases
  already have `P(low)` winning even under raw greedy decoding (never showed up in the
  n_runs=5 sampled-decode collapse because temperature=0.3 sampling doesn't always pick
  the argmax). Of the 9 that fail raw, correction only flips 1 (`diamond→column`). The
  remaining **8 cases have `P(low)` genuinely crushed to 7–33%** against `medium`'s
  67–89% — several of these (7.0%, 8.6%, 9.3%, 10.9%) are far too lopsided for any
  single constant log-prior shift to fix (the correction here is a fixed
  `log(0.470/0.248) ≈ 0.64` nats added to `low`'s log-odds against `medium` — nowhere
  near enough to overturn a 7%/84% split). **For v3a specifically, roughly half the
  low-threat failures look like a genuine knowledge gap, not a decoding artifact** — the
  v4 program has real justification here, though scoped to a subset of pairs
  (`encirclement→*`, `*→column`, `*→dispersed` are overrepresented among the
  unrecoverable cases), not the whole low-threat class uniformly.
- **Critical-threat evidence (n=2) is too thin to generalize**, but is directionally
  consistent: one direction (`converging→encirclement`) is calibration-fixable, the
  other (`encirclement→converging`) shows a genuine near-zero `P(critical)` even under
  correction.

**Net: this test does not give a single "build v4" / "don't build v4" answer — it gives
a much more precise one. The v4 case is weak for v3b (mostly free via calibration) and
real but narrower than "the whole low-threat class" for v3a (concentrated in specific
pair patterns the model's raw distribution treats as confidently wrong, not merely
under-preferred).**

## AA. Counterintuitive-rule hypothesis — does NOT hold; only 3/8 pairs match

Prediction under test: if v3a's 8 unrecoverable low-threat failures (sec Y) are caused
by a pretraining semantic prior conflicting with a counterintuitive rule (e.g.
`encirclement->column` being `low` reads as surprising), then `rules_in_prompt` — base
Qwen2.5-7B-Instruct + `RULES.txt` in the system prompt, **zero fine-tuning** — should
fail on the same 8 pairs, since the conflict would be present in the base model
regardless of any training. `llm_finetuning/report_prior_hypothesis.py` checks this
against the already-collected `evaluation/eval_expanded_rules_in_prompt.json` (no new
generations).

| v3a-failing pair | rules_in_prompt majority | threat_accuracy | matches v3a's failure? |
|---|---|---|---|
| encirclement→column | medium | 20.0% | **YES** |
| encirclement→dispersed | low | 100.0% | no |
| converging→column | low | 60.0% | no |
| v_shape→column | medium | 20.0% | **YES** |
| v_shape→dispersed | low | 80.0% | no |
| column→dispersed | low | 100.0% | no |
| dispersed→column | medium | 40.0% | **YES** |
| shield→dispersed | low | 100.0% | no |

**Only 3/8 (37.5%) match.** `rules_in_prompt` — with zero fine-tuning, RULES text just
pasted into its context — correctly resolves `low` on 5 of the 8 pairs v3a cannot
recover even after logit-prior correction, several at high confidence
(`encirclement->dispersed`, `column->dispersed`, `shield->dispersed`: 100% accuracy).
The 4 pairs v3a succeeds on (steady-state ×3 + `converging→dispersed`) are also handled
well by `rules_in_prompt` (60–100%), consistent, but that's not the discriminating test.

**Hypothesis does NOT hold. The mechanism is not simply "a pretraining prior conflicts
with the rule table" — if it were, the base model would show the same failure pattern
independent of fine-tuning, and it mostly doesn't.** Only `encirclement→column`,
`v_shape→column`, and `dispersed→column` show a pattern consistent with both systems
struggling (worth separate scrutiny — all three end in `column`, a possible narrower
lead), while the other 5 of v3a's 8 unrecoverable failures reflect something
fine-tuning-specific: either the small-dataset diversity effect from secs N–O, or an
interaction between fine-tuning and this particular subset of pairs not yet isolated.
This narrows, rather than confirms, the "pretraining prior" explanation from sec Y.

## BB. Fixed two real train/serve mismatches in `synth_context()` — generator only, no dataset regenerated

Per sec X, the sparse 54-cell-per-pair grid was traced to `synth_context()`
(`llm_finetuning/build_sft_dataset.py`) itself being unable to reach the full
`context_spec.py` vocabulary, independent of sample size. Two concrete bugs, fixed
this session (RULES untouched, no dataset regenerated — generator fix only, per the
session's explicit instruction):

**(a) `role_differentiation` was never rendered into the training prompt text at
all**, despite `build_tactical_context()` (production) always emitting a
`"Role differentiation: ..."` line (`inference.py:155`). Fixed: `synth_context()` now
samples a `role_present` boolean (`rng.random() < 0.3`, matching production's
minority-case framing — a role split only occurs when one drone strays >2x the
group's median distance) and renders `f"Role differentiation: {role_str}"` as an
8th context line, consistent with `key_windows`' `role_differentiation` field
(previously hardcoded `False` in both `synth_context()`'s output and
`build_sft_dataset.py`'s `main()` prediction construction — both now use the real
sampled value).

**(b) `spread_dynamics` and `stability_trend` were binary in the generator despite a
3-way production vocabulary**, matching `calibration.py`'s `AbsoluteCalibrator`
thresholds exactly (not invented — read directly from the production calibrator):
- `spread_dynamics`: was `if approach < -0.1: converging else: stable` (no
  `dispersing` branch at all). Now: `< -0.1` → converging, `> 0.1` → dispersing,
  else stable — and `approach`'s sampling range widened from `U(-1.5, 0.5)` to
  `U(-1.5, 1.5)` so the new `dispersing` branch is actually reachable, not just
  theoretically present in the code.
- `stability_trend`: was a single scalar (`mean_stab`) compared against one cutoff
  (`> 0.7` → holding, else degrading) — structurally could never produce
  `improving`, because production's real logic compares an EARLY-window value
  against a LATE-window value (`calibration.py`'s `stability_trend(early, late)`),
  not one scalar against a threshold. Fixed: `synth_context()` now samples
  `stab_early`/`stab_late` independently and applies the identical
  early/late-delta-vs-0.1 logic production uses; `key_windows`' two windows now
  carry `stab_early`/`stab_late` respectively (previously both windows shared one
  identical `mean_stab` value, itself a minor fidelity gap beyond just the
  trend-label bug).

**Test added** (`tests/test_synth_context_coverage.py`, 6 tests, 500 sampled
`(form_a, form_b)` draws): asserts every value in `VELOCITY_TREND_VALUES`,
`SPREAD_DYNAMICS_VALUES`, `STABILITY_TREND_VALUES`, `ROLE_DIFFERENTIATION_VALUES` is
now reachable; asserts the rendered `role_str` line and the `key_windows` boolean
stay consistent; asserts `RULES` (49 pairs, exact `BASE_FORMATIONS × BASE_FORMATIONS`
set) is byte-for-byte untouched by this change, as an explicit scope guard. All 6
pass; full suite (42 tests) passes.

**Scope note: this is a generator fix only.** `data/sft_train_final.jsonl` and every
other existing training file are untouched — they were generated by the OLD,
narrower `synth_context()` and still reflect sec X/Y/AA's findings exactly as
measured. Regenerating a dataset from the fixed generator (and re-training on it) is
a separate, larger decision this session was explicitly told not to make.
