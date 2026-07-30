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
