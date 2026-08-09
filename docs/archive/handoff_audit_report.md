# LLM Fine-Tuning Handoff Audit

Scope: `llm_finetuning/`, `data/*.jsonl`, `adapters/`, `evaluation/`, and the parts of
`src/swarm_intent/` (`inference.py`, `llm/client.py`, `llm/prompts.py`, `llm/evaluate.py`)
that the fine-tuning pipeline directly imports. Findings below are verified against
on-disk artifacts (trainer_state.json, adapter_config.json, actual `.jsonl` row content) —
not just the code — wherever a claim could be checked that way.

Repo is **not under version control** (no `.git`). Everything in `adapters/` and `data/`
exists only on this local disk; `.gitignore` already excludes `adapters/`, `swarm_data/`,
`*.safetensors`, `*.pt`, so even if this becomes a git repo later, none of the three
trained checkpoints will be captured. Back them up out-of-band before this machine is reset.

---

## 1. File Manifest & Architecture Map

### `llm_finetuning/`

| File | Responsibility |
|---|---|
| `build_sft_dataset.py` (326 lines) | Generates the SFT training set. Fabricates synthetic tactical-context scenarios (`synth_context`), optionally calls a Groq teacher model for fluent prose, then **overrides** `threat_level`/`likely_intent`/`recommended_action` with the canonical `RULES` dict (line 47) so decision fields are consistent regardless of teacher quality. Writes chat-format JSONL. |
| `train_qlora.py` (113 lines) | QLoRA SFT of a 7B instruct model (default Qwen2.5-7B-Instruct) via `peft` + `trl.SFTTrainer`. 4-bit NF4, LoRA r=16/alpha=32 on all 7 attn+MLP projections, batch=1 + grad_accum=8, target: free Colab T4. |
| `evaluate_finetuned.py` (75 lines) | Runs the fine-tuned model (as `LocalHFClient`) against `TEST_CASES`, scores objective intent/threat accuracy plus an *independent* Groq judge, via `evaluate_llm`. |
| `extract_teacher_rows.py` (47 lines) | One-off salvage utility: strips templated (non-teacher) rows out of a dataset file by regex-matching the fixed `gold_assessment` fallback sentence, so a quota-starved build can be cleaned and re-grown with `--append --teacher-only`. |
| `RULES.txt` (85 lines) | Human-readable dump of the 49 `RULES` entries (7 steady-state + 12 escalating + 20 medium + 10 de-escalating). Explicitly documented as **derived from, not the source of**, `RULES` in `build_sft_dataset.py:47-118`. |
| `configs/qlora_qwen2.5-7b.yaml` | Documents the intended hyperparameters. **Not read by any script** — see §2. |
| `README.md` | Pipeline overview + Colab quickstart. |

### `data/` (all outputs of `build_sft_dataset.py`, chronological by mtime)

| File | Rows (train/val) | Teacher-prose share | Consumed by |
|---|---|---|---|
| `smoke_train.jsonl` / `smoke_val.jsonl` | 20 / 4 | 0% (`--no-teacher`) | `adapters/smoke-test` |
| `sft_train.jsonl` / `sft_train_val.jsonl` (**v1**) | 2700 / 300 | **0%** | `adapters/qwen-swarm` |
| `sft_train_v2.jsonl` / `_val` (**v2**) | 810 / 90 | 50% | `adapters/qwen-swarm-v2` |
| `sft_train_v3.jsonl` / `_val` (**v3**) | 406 / 47 | **100%** | *unused — no adapter trained on it* |
| `sft_train_final.jsonl` / `_val` (**final**) | 234 / 26 | **100%** | *unused — no adapter trained on it, and never evaluated* |

(Teacher-prose share computed by matching `situation_summary` against the exact fallback
template regex from `extract_teacher_rows.py:19-20` — every row that doesn't match came
from the Groq teacher.)

### `adapters/` (LoRA weights, all base model `Qwen/Qwen2.5-7B-Instruct`, r=16/alpha=32)

| Adapter | Trained on | Epochs done | Final step | Final train loss / eval loss | Token accuracy (last log) |
|---|---|---|---|---|---|
| `smoke-test` | `smoke_train.jsonl` (20 rows) | 1.0 | 3 | — / 1.567 | — |
| `qwen-swarm` | `sft_train.jsonl` (v1, **100% templated**) | 3.0 (complete) | 1014 | 0.053 / 0.053 | 97.8% |
| `qwen-swarm-v2` | `sft_train_v2.jsonl` (v2, 50/50 mix) | 3.0 (complete) | 306 | 0.073 / 0.076 | 97.1% |

Both real runs finished cleanly (no crashed/partial checkpoint, no early stopping) — this
is not a "training failed" situation. See §2/§3 for why the loss numbers are misleadingly good.

### `evaluation/`

| File | Content |
|---|---|
| `llm_run_output.json` | One real end-to-end run: ML sliding-window output + `llm_assessment` from the **prompt-engineered Groq baseline** (pre-fine-tuning). This is the only ground truth for what real pipeline numbers look like. |
| `sliding_window_predictions_demo.json` | The `predictions` list from the same run, standalone. |
| `llm_eval_summary.png`, `ml_confusion_matrix.png` | Plots from the STGT classifier / baseline LLM eval (outside fine-tuning scope). |
| `finetuned_eval.json` | **Does not exist.** `evaluate_finetuned.py` has never been run against any of the three adapters. |

### Shared dependencies in `src/swarm_intent/`

| File | Role in this pipeline |
|---|---|
| `inference.py` | `build_llm_prompt` / `OUTPUT_SCHEMA` (imported by `build_sft_dataset.py:40` and `evaluate_finetuned.py:29`) is the single source of truth for the prompt+schema. `build_tactical_context` computes the trend narrative fed into the real (non-synthetic) prompt. |
| `llm/client.py` | `GroqClient` (teacher + judge), `LocalHFClient` (loads base model + LoRA adapter for eval/deployment). |
| `llm/prompts.py` | `TEST_CASES` (6 scenarios), `INTENT_FAMILIES`/`THREAT_FAMILIES` fuzzy-matchers, `JUDGE_PROMPT`. |
| `llm/evaluate.py` | `evaluate_llm` (objective headline metric) / `evaluate_ml_model` (STGT classifier, unrelated to fine-tuning). |

**Documentation/code mismatch:** `CLAUDE.md` and `MIGRATION_GUIDE.md` both describe a
`src/swarm_intent/llm/pipeline.py` with a `run_full_pipeline(model, raw_sequence, ...,
llm_client)` entry point that stitches sliding-window inference → context → prompt →
LLM call into one function. **This file does not exist anywhere in the repo**
(confirmed by repo-wide search). Instead, `build_sft_dataset.py:246-256` and
`evaluate_finetuned.py:48-57` each hand-roll their own near-identical
`predict → prediction-dict → build_llm_prompt` glue. Any future change to that glue
(e.g. a new field in the prediction dict) has to be made in both places by hand.

---

## 2. Pipeline Data & Logic Audit

### 2.1 End-to-end data flow

```
build_sft_dataset.py:synth_context()          [FABRICATED numbers, not the real model]
        │  random tactical-context + 2 "key windows"
        ▼
inference.py:build_llm_prompt()                [real prompt-assembly code, shared]
        │  schema + tactical context + key windows → prompt string
        ▼
gold_assessment(): RULES[(form_a,form_b)] overrides threat/intent/action
        │  optional Groq teacher fills prose (situation_summary, threat_reasoning, ...)
        ▼
{"messages":[{"role":"user","content":prompt}, {"role":"assistant","content":json.dumps(gold)}]}
        ▼
train_qlora.py: tok.apply_chat_template(messages) → "text" column → SFTTrainer
```

Note the dataset **never calls the trained STGT model** — `synth_context` (lines
122–158) invents `approach_rate`, `centroid_velocity`, and `delta_v` from hand-picked
`random.uniform` ranges. This is reasonable for bootstrapping SFT data before a model
exists, but as of now a trained STGT model *does* exist and produces real numbers
(`evaluation/llm_run_output.json`) that are wildly different in scale from what
`synth_context` samples — see §2.3, the most consequential finding in this audit.

### 2.2 Loss masking — confirmed NOT applied to assistant turns only

`train_qlora.py:72-76` converts each `{"messages": [...]}` example to a flat `"text"`
string via `tok.apply_chat_template(ex["messages"], tokenize=False, ...)`, then hands
`SFTConfig`/`SFTTrainer` a dataset whose only relevant column is `"text"` (line 78-96,
98-104). This is TRL's "standard" (non-conversational) code path
(`sft_trainer.py:667-671` in the installed trl==1.8.0), which computes loss over **every
token in the string, prompt included** — not the "conversational" path that would
auto-derive an `assistant_masks` mask from a `"messages"` column
(`sft_trainer.py:1236-1244,1493`).

`SFTConfig.assistant_only_loss` defaults to `False` and is never set in `train_qlora.py`.
Confirmed directly by inspecting the saved `training_args.bin` for both real runs:

```
qwen-swarm:     assistant_only_loss = False   (dataset_text_field='text', packing=False)
qwen-swarm-v2:  assistant_only_loss = False
```

**Effect:** the model is being trained to predict the ~500-700 token tactical-context
prompt (which is highly templated/repetitive — see `synth_context`'s fixed sentence
skeleton) in addition to the ~150-token JSON assessment. Because the prompt is far more
predictable than the assessment, it dominates the average loss/accuracy. This is the
most likely explanation for the suspiciously good numbers in §3 (loss 0.05, 97.8% token
accuracy after only 2 epochs) — the trainer isn't reporting how well the model writes
tactical assessments, it's reporting how well it reproduces a template it has seen
hundreds of times. **Fix:** either pass `"messages"` (unflattened) directly to
`SFTTrainer` with `assistant_only_loss=True` and a chat template that has `{% generation
%}` markers (trl requires this, `sft_trainer.py:1244`), or use `DataCollatorForCompletionOnlyLM`.

### 2.3 Training-data distribution vs. real pipeline output — confirmed mismatch

`synth_context()` (`build_sft_dataset.py:130-131`) samples:
```python
approach = round(rng.uniform(-1.5, 0.5), 3)
delta_v  = round(rng.uniform(-1.0, 2.0), 2)
```
These feed narrative thresholds in `inference.py:build_tactical_context` (lines 117, 126-128):
```python
vel_trend = "accelerating" if delta_v > 0.5 else "decelerating" if delta_v < -0.5 else "steady"
approach_summary = "converging..." if mean_approach < -0.1 else "dispersing..." if mean_approach > 0.1 else "stable spread"
```
With `delta_v ~ U(-1.0, 2.0)`, roughly **2/3 of training rows narrate "accelerating" or
"decelerating"**, and with `approach ~ U(-1.5, 0.5)` the great majority narrate
"converging" or "dispersing" (only a thin band near zero reads "stable spread").

The one real captured pipeline run (`evaluation/llm_run_output.json`) shows actual model
output on this scale:
```
centroid_velocity : 0.04 – 0.072      (synth_context velocity field samples 0.0–0.1, OK)
approach_rate      : -0.001 to -0.0    (synth_context samples -1.5 to 0.5 — ~1000x off)
delta_velocity      : 0.018            (synth_context samples -1.0 to 2.0 — ~50-100x off)
→ velocity_trend    : "steady"
→ spread_dynamics   : "stable spread"
```
This matches `CLAUDE.md`'s documented units caveat (regression labels are in
normalised-space, so real `centroid_velocity` ≈ 0.05, not the README's `4.3` example).
**Consequence:** in real/demo use, `vel_trend` and `spread_dynamics` will almost always
read "steady" / "stable spread" — the "accelerating/decelerating" and
"converging/dispersing" branches are structurally near-dead in production — yet the
fine-tuned model has been trained mostly on scenarios where those branches fire. The
model has learned reasoning patterns (and threat escalation language, e.g. "accelerating
velocity" cited as a threat indicator in a sample training row) for a regime the real
sensor pipeline essentially never produces. This is very likely to show up as confident
but wrong reasoning at demo time. **Fix:** rescale `synth_context`'s `approach`/`delta_v`
sampling to match the actual trained model's regression output range (pull real ranges
from a `sliding_window_inference` run over `swarm_data`, not hand-picked constants), or
resolve the underlying units bug (`CODE_REVIEW.md`) so both sides are on the same scale.

### 2.4 Train/inference prompt-wrapping mismatch

Training rows save the raw `build_llm_prompt(...)` output as the `user` message
(`build_sft_dataset.py:269-270`) — the prompt itself already ends with "Respond with
ONLY the JSON object. No preamble..." (`inference.py:199`).

At inference/eval time, both `GroqClient.generate` (`client.py:86-87`) and
`LocalHFClient.generate` (`client.py:123`) wrap that same prompt with an **extra**
prefix line before sending it to the model:
```python
messages = [{"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"}]
```
So `evaluate_finetuned.py` (and any future deployment using `LocalHFClient`) feeds the
fine-tuned model a slightly different input distribution than it was trained on — a
short generic instruction line prepended that never appeared during SFT. This is a real
train/inference skew, though a mild one (Qwen models are fairly robust to this). It's
also functionally redundant: the schema instruction is already in `build_llm_prompt`.
**Fix:** drop the prefix in `client.py`, or bake it into `build_llm_prompt` so both paths match exactly.

### 2.5 Minor: evaluation vocabulary gaps (affects reported hallucination rate)

`llm/prompts.py`'s `INTENT_FAMILIES` includes an `"attack"` family (line 8) that is
**not a valid value** anywhere in `OUTPUT_SCHEMA`/`RULES` (the closest valid value is
`attack_preparation`) — dead vocabulary. Conversely, `OUTPUT_SCHEMA` allows
`likely_intent: "unknown"` (`inference.py:167`) but `INTENT_FAMILIES` has no `"unknown"`
entry, so `is_hallucination()` (`prompts.py:46-50`) will flag any legitimate
`"unknown"` response as a hallucination. Low severity, but will quietly inflate the
hallucination-rate metric if the model ever (correctly) abstains.

### 2.6 Orphaned config file

`llm_finetuning/configs/qlora_qwen2.5-7b.yaml` documents hyperparameters "to keep in
sync" with `train_qlora.py`, but no script in the repo ever parses YAML (`grep -r
"yaml"` across all `.py` files returns nothing outside `.venv`). All actual
hyperparameters come from `train_qlora.py`'s `argparse` defaults. The two happen to
agree today (r=16/alpha=32/lr=2e-4/epochs=3/max_length=1024 by TRL default), but nothing
enforces that — editing the yaml does nothing.

---

## 3. Current State & Completion Checklist

**Dataset generation** — done, but iterated messily; latest/best version is unused:
- [x] v1 generated (2700 rows) — **0% teacher prose**, pure `RULES` template filler (both `situation_summary` sentences are one of two fixed strings).
- [x] v2 generated (810 rows) — 50/50 templated/teacher mix (the "bug that produced a fully-templated v2 dataset" referenced in `build_sft_dataset.py:165` comments was mid-fixed here).
- [x] v3 generated (406 rows) via `extract_teacher_rows.py` salvage + `--append --teacher-only` — 100% teacher prose, internally consistent (teacher sees the ground-truth rule and writes supporting prose per the `gold_assessment` fix at line 174-183).
- [x] "final" generated (234 rows) — 100% teacher prose, smallest and most recent.
- [ ] **v3 and final have never been used to train an adapter.** They're the highest-quality data on disk and are sitting idle.
- [ ] No documented target row count anywhere (README suggests "start ~600, scale up" — v1 alone already exceeded that, but with 0% teacher quality).

**Training runs** — 3 completed, all functional, none evaluated:
- [x] `smoke-test`: 20 rows, 1 epoch, 3 steps — sanity check only, expected.
- [x] `qwen-swarm`: v1 data, 3 epochs, step 1014, clean loss curve — but trained on 100%-templated targets with full-sequence (unmasked) loss (§2.2). Likely learned to reproduce 2 boilerplate sentence patterns plus RULES-correct labels, not genuine reasoning.
- [x] `qwen-swarm-v2`: v2 data, 3 epochs, step 306, clean loss curve — better data (50% teacher prose) but same masking issue.
- [ ] No `qwen-swarm-v3` / `qwen-swarm-final` trained on the two best datasets.

**Evaluation and logging** — the biggest gap:
- [ ] `evaluate_finetuned.py` has **never been run** against any of the 3 adapters — `evaluation/finetuned_eval.json` does not exist. There are currently **zero objective accuracy numbers for any fine-tuned checkpoint.**
- [x] The pre-fine-tuning Groq baseline was evaluated once (`evaluation/llm_run_output.json`, `llm_eval_summary.png`) — this is your only comparison point today, and it's from the prompt-engineered API baseline, not a fine-tune.
- [ ] `TEST_CASES` has only 6 scenarios; `prompts.py:54-55` self-flags this: *"expand this list substantially (>=50, incl. ambiguous/adversarial cases)... Six cases is not enough"* before reporting any headline number.
- [ ] `evaluate_llm`'s independent-judge design (`llm/evaluate.py`) is correctly implemented and ready to use — it just hasn't been pointed at a fine-tuned model yet.

---

## 4. Actionable Next Steps (priority order)

1. **Fix loss masking before doing any more training.** Set the SFT trainer up for
   assistant-only loss (`assistant_only_loss=True` on a conversational `"messages"`
   dataset with a chat template that has `{% generation %}` markers, or a completion-only
   collator) in `train_qlora.py`. Until this is fixed, loss/accuracy numbers from any run
   are not trustworthy signals of assessment quality. (§2.2)

2. **Rescale `synth_context`'s numeric ranges to match the real model's output.** Pull
   actual `approach_rate`/`centroid_velocity`/regression ranges from a real
   `sliding_window_inference` run (you already have one data point in
   `evaluation/llm_run_output.json`: velocity ~0.04-0.07, approach ~0.0) instead of the
   current `U(-1.5,0.5)` / `U(-1.0,2.0)` guesses. This is the difference between a model
   that reasons correctly about real sensor output and one that's confidently wrong at
   demo time. Do this *before* generating any more training data — items 3-4 are wasted
   effort if the input distribution is still off. (§2.3)

3. **Regenerate a clean dataset with the two fixes above baked in**, targeting the
   README's "500-1000 diverse, correctly-labelled" guidance with `--teacher-only` so
   every row is genuinely teacher-distilled (you already have the tooling —
   `build_sft_dataset.py --append --teacher-only`). Don't reuse v1 (100% templated) or v2
   (50% templated) as-is.

4. **Train `qwen-swarm-v3`** on the corrected dataset with assistant-only loss, then
   **actually run `evaluate_finetuned.py`** against it, the earlier adapters, and the
   Groq baseline, so you have a real before/after comparison table for the panel. Right
   now you have three trained checkpoints and zero accuracy numbers for any of them —
   that's the single highest-leverage gap for a defensible capstone result.

5. **Expand `TEST_CASES` past 6** (`llm/prompts.py:56`) before quoting any headline
   accuracy number, including a few ambiguous/adversarial cases (the README itself calls
   this out as where a fine-tune should earn its keep over prompting).

6. **Fix the small correctness issues** while you're in the code: drop the
   `"Return ONLY valid JSON.\n\n"` prefix mismatch between training and inference
   (`client.py:86-87,123` vs `build_sft_dataset.py:269-270`, §2.4); add `"unknown"` to
   `INTENT_FAMILIES` or special-case it in `is_hallucination` (§2.5); either wire
   `configs/qlora_qwen2.5-7b.yaml` into `train_qlora.py` (`argparse` reading defaults
   from it) or delete it so it stops implying a config path that doesn't exist (§2.6).

7. **Decide whether to build the documented `pipeline.py`/`run_full_pipeline`**
   (referenced in `CLAUDE.md` and `MIGRATION_GUIDE.md` but absent from the repo) to
   de-duplicate the prediction-list-comprehension glue currently copy-pasted between
   `build_sft_dataset.py` and `evaluate_finetuned.py` — low urgency, but worth doing
   before a third caller needs the same glue.

8. **Back up `adapters/`** somewhere outside this machine — there is no git repo here,
   and `.gitignore` excludes `adapters/`/`*.safetensors` even if one is initialized
   later. `qwen-swarm` and `qwen-swarm-v2` represent real GPU time; losing them means
   redoing the training runs from scratch.
