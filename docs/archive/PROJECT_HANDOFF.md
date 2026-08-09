# Project Handoff — UAV Swarm Intent (Counter-UAV Capstone)

Paste this whole document into a new chat to bring it up to speed. It covers what the
project is, how it's architected, exactly where every artifact lives on disk, what's
done vs. not done, and the known gaps worth planning around. Repo root:
`/home/pw26_akp_01/aadhya/uav-llm` (no git — this machine is the only copy of everything
below; nothing is backed up).

---

## 1. What this project is

University capstone: **"LLM-Driven Semantic RF Analysis for Detection and Visualisation
of UAV Swarm Communication."** Team and guide names redacted for public release.

**Pitch:** go from raw drone RF signals to a human-readable tactical assessment
("V-formation, accelerating, converging → likely attack pattern, alert operator") in
real time, by chaining RF sensing → multilateration → a graph/transformer swarm model →
an LLM reasoning layer → an AR dashboard.

**Reality check (important for planning):** the README describes 4 components; only 2
have code in this repo:

| # | Component | Status |
|---|---|---|
| 1 | RF fingerprinting (on-drone DL model) | **Not in this repo.** Lives in a teammate's repo per `MIGRATION_GUIDE.md`; planned to land under `src/swarm_intent/rf/`. |
| 2 | EKF / multilateration tracker | **Not in this repo.** Same story, planned under `src/swarm_intent/tracking/`. |
| 3 | Swarm behavior model (STGT: GATv2 + Transformer) | **Code exists**, but **no trained model exists on this disk** — `swarm_data/` (where `best_model.pt` + `norm_stats.npy` would live) doesn't exist. It has never been run in this environment. |
| 4 | LLM interpretation layer + fine-tuning | **Code + trained artifacts exist.** This is the most mature part of the repo — see §3-5. |
| 5 | AR/3D dashboard | **Not in this repo.** |

An independent code review (`CODE_REVIEW.md`) is blunt about this: *"the repository
delivers roughly half of the system the README advertises... as a research result it is
not yet defensible."* Known specific issues called out there: all validation is on
hand-generated synthetic geometry (no real trajectories), regression labels are
circular/self-referential, and there's a documented **units bug** — the STGT model's
`centroid_velocity`/`approach_rate` regression outputs are in normalized-space, not
physical m/s (real magnitude ~0.05, not the README's example value of 4.3). That units
bug matters directly for the LLM fine-tuning work — see §6.2.

---

## 2. Full architecture (as designed)

```
UAV RF Signals
     │
     ▼
[1] On-Drone DL Model (RF Fingerprinting)            ─┐
     Extracts AoA, TDoA, RSSI, Doppler, Bandwidth      │  NOT IN THIS REPO
     Output: 128-dim hardware fingerprint per drone     │
     ▼                                                 ─┘
[2] Base Station — Multilateration & Swarm Modeling
     │  EKF fuses features across drones → 3D position + velocity   ← tracker NOT in this repo
     │
     │  Swarm graph: nodes = drones (x,y,z), edges = spatial proximity
     │  GATv2 (spatial) + Transformer (temporal)
     │  → formation type (7-8 classes), stability, approach_rate,
     │    centroid_velocity, role_differentiation
     ▼                                                    STGT model — code exists,
[3] LLM Interpretation Layer                              NOT trained in this env
     │  Input: structured JSON from the STGT model
     │  sliding_window_inference → build_tactical_context → build_llm_prompt
     │  → LLM (Groq baseline OR local QLoRA-fine-tuned Qwen2.5-7B)
     │  Output: {threat_level, likely_intent, recommended_action, ...}   ← MOST MATURE PART
     ▼
[4] Visualisation (3D / AR Dashboard)                  ─┐  NOT IN THIS REPO
                                                         ─┘
```

### 2a. STGT model (`src/swarm_intent/model.py`) — swarm formation classifier

Fixed input shape: **6 drones × 3 coords (x,y,z) × 50 timesteps**, hardcoded throughout.

```
(50, 6, 3) sequence
  → sequence_to_graphs (graph.py): one proximity graph per timestep
       nodes = drone (x,y,z); edges connect pairs closer than cfg.edge_threshold (normalized units)
       edge_attr = (dx,dy,dz,dist); self-loops if isolated
  → SpatialGAT: 2× GATv2Conv + global_mean_pool → (50, d_model) per-timestep vectors
  → TemporalTransformer: positional encoding + 4× encoder layers + mean-pool → (d_model,)
  → shared head → two heads:
       cls_head: formation classifier (7 or 8 classes)
       reg_head: 3 regression values [centroid_velocity, approach_rate, stability]
```

Formation vocabulary (`config.py:24-33`, index = integer label):
`v_shape, encirclement, column, diamond, dispersed, converging, shield` (+ `transitioning` as
an 8th class when `Config(n_classes=8)`).

Config invariants: `d_model == gat_out_dim` (both 128). One `Config` dataclass is the
single source of truth (`config.py`) — no more global `CFG`/`device` dicts.

Commands to produce it (not yet run here):
```bash
python scripts/generate_data.py --per-formation 1000 --transitions 2000   # → swarm_data/
python scripts/train_model.py --classes 8 --epochs 80                     # → swarm_data/best_model.pt
```

### 2b. LLM interpretation layer (`src/swarm_intent/llm/` + `src/swarm_intent/inference.py`)

```
raw (T,6,3) sensor stream
  → sliding_window_inference (inference.py)   [normalizes each window w/ TRAIN stats, runs STGT predict()]
  → build_tactical_context (inference.py)      [summarizes window sequence → narrative + summary dict]
  → build_llm_prompt (inference.py)            [narrative + schema + key windows → prompt string]
  → LLMClient.complete(prompt)                  [GroqClient (hosted baseline) or LocalHFClient (QLoRA fine-tune)]
  → JSON: {situation_summary, threat_level, threat_reasoning, likely_intent,
           recommended_action, confidence_in_assessment, key_indicators, follow_up_watch}
```

`OUTPUT_SCHEMA` in `inference.py` is the single source of truth for that JSON shape —
used by both prompt assembly and evaluation.

**Note:** `CLAUDE.md`/`MIGRATION_GUIDE.md` describe a `src/swarm_intent/llm/pipeline.py`
with a `run_full_pipeline(...)` that would wire the above into one call. **It doesn't
exist.** `build_sft_dataset.py` and `evaluate_finetuned.py` each hand-roll the same glue
separately — worth building for real once a third caller needs it.

### 2c. LLM fine-tuning pipeline (`llm_finetuning/`) — turns the Groq baseline into a local QLoRA-tuned model

```
build_sft_dataset.py   →  data/*.jsonl (chat-format SFT rows)
        │  synth_context() fabricates tactical scenarios (NOT from the real trained STGT
        │  model — hand-picked random ranges); RULES dict (49 entries, one per
        │  formation-pair transition) is the canonical ground truth for threat/intent/
        │  action; optional Groq teacher (llama-3.3-70b) supplies fluent prose which gets
        │  overridden on the decision fields
        ▼
train_qlora.py          →  adapters/<name>/  (LoRA adapter via peft+trl SFTTrainer)
        │  4-bit NF4 quant, LoRA r=16/alpha=32 on all 7 attn+MLP proj, batch=1+grad_accum=8
        │  base model: Qwen/Qwen2.5-7B-Instruct
        ▼
evaluate_finetuned.py   →  evaluation/finetuned_eval.json
           LocalHFClient (system under test) vs. independent Groq judge (llama-3.3-70b);
           headline metric = objective intent/threat accuracy, NOT the judge's self-score
```

---

## 3. Exact file/model locations (everything on disk right now)

### Code
| Path | What |
|---|---|
| `src/swarm_intent/config.py` | `Config` dataclass, `BASE_FORMATIONS`, `formation_names()` |
| `src/swarm_intent/model.py` | STGT model (SpatialGAT + TemporalTransformer) |
| `src/swarm_intent/graph.py` | `sequence_to_graphs` |
| `src/swarm_intent/data.py`, `dataset.py` | synthetic data generation + `SwarmDataset` |
| `src/swarm_intent/train.py` | STGT training loop |
| `src/swarm_intent/inference.py` | `predict`, `sliding_window_inference`, `build_tactical_context`, `build_llm_prompt`, `OUTPUT_SCHEMA` |
| `src/swarm_intent/llm/client.py` | `GroqClient`, `LocalHFClient` |
| `src/swarm_intent/llm/prompts.py` | `TEST_CASES` (6 scenarios), intent/threat fuzzy matchers, `JUDGE_PROMPT` |
| `src/swarm_intent/llm/evaluate.py` | `evaluate_llm` (objective metric), `evaluate_ml_model` |
| `llm_finetuning/build_sft_dataset.py` | SFT dataset builder + `RULES` (canonical decision policy) |
| `llm_finetuning/train_qlora.py` | QLoRA training script |
| `llm_finetuning/evaluate_finetuned.py` | Fine-tune evaluation script |
| `llm_finetuning/extract_teacher_rows.py` | Dataset-cleaning salvage utility |
| `llm_finetuning/configs/qlora_qwen2.5-7b.yaml` | Documented hyperparams — **not actually loaded by any script**, decorative only |
| `scripts/generate_data.py`, `scripts/train_model.py` | STGT model CLI entry points |

### Trained artifacts (LoRA adapters — base model `Qwen/Qwen2.5-7B-Instruct`, r=16/alpha=32)

| Path | Trained on | Epochs | Final step | Train/eval loss | Notes |
|---|---|---|---|---|---|
| `adapters/smoke-test/` | `data/smoke_train.jsonl` (20 rows) | 1.0 | 3 | — / 1.567 | Sanity check only |
| `adapters/qwen-swarm/` | `data/sft_train.jsonl` — **v1, 100% templated (0% teacher prose)** | 3.0 (complete) | 1014 | 0.053 / 0.053 | Loss numbers misleading — see §6.1 |
| `adapters/qwen-swarm-v2/` | `data/sft_train_v2.jsonl` — **v2, 50/50 templated/teacher** | 3.0 (complete) | 306 | 0.073 / 0.076 | Better data, same masking issue |
| *(none)* | `data/sft_train_v3.jsonl` (406 rows, 100% teacher) | — | — | — | **Built but never trained** |
| *(none)* | `data/sft_train_final.jsonl` (234 rows, 100% teacher) | — | — | — | **Built but never trained — highest quality data, unused** |

STGT model checkpoint (`swarm_data/best_model.pt`): **does not exist.** Never trained in this environment.

### Datasets (`data/*.jsonl`, all outputs of `build_sft_dataset.py`)

| File | Rows train/val | Teacher-prose % | Used by |
|---|---|---|---|
| `smoke_train.jsonl` / `smoke_val.jsonl` | 20/4 | 0% | `adapters/smoke-test` |
| `sft_train.jsonl` / `sft_train_val.jsonl` (v1) | 2700/300 | 0% | `adapters/qwen-swarm` |
| `sft_train_v2.jsonl` / `_val` (v2) | 810/90 | 50% | `adapters/qwen-swarm-v2` |
| `sft_train_v3.jsonl` / `_val` (v3) | 406/47 | 100% | unused |
| `sft_train_final.jsonl` / `_val` (final) | 234/26 | 100% | unused |

### Evaluation artifacts (`evaluation/`)

| File | Content |
|---|---|
| `llm_run_output.json` | The **only** real end-to-end pipeline run on disk: STGT sliding-window output + Groq baseline `llm_assessment`. Real regression scale: `centroid_velocity` 0.04-0.072, `approach_rate` ≈ 0.0, `delta_velocity` 0.018. |
| `sliding_window_predictions_demo.json` | Same run's raw predictions list |
| `llm_eval_summary.png`, `ml_confusion_matrix.png` | Plots from STGT/baseline eval |
| `finetuned_eval.json` | **Does not exist** — `evaluate_finetuned.py` has never been run against any adapter |

### Docs
`CLAUDE.md` (guidance for AI assistants working in this repo), `CODE_REVIEW.md`
(independent honest assessment, scores, top-10 issues), `MIGRATION_GUIDE.md` (maps old
notebook functions → new module locations), `llm_finetuning/README.md`,
`llm_finetuning/RULES.txt` (human-readable dump of the 49 decision rules).

Three original Jupyter notebooks still sit at repo root (`capstone_with_llm.ipynb`,
`capstone_with eval.ipynb`, `models + data generation.ipynb`) — superseded by `src/`, not
maintained, don't build on them.

---

## 4. Current progress snapshot

**Done:**
- STGT model architecture fully coded and documented (never trained/checkpointed here).
- LLM prompt/schema/context pipeline fully coded and works — verified by one real captured run (`evaluation/llm_run_output.json`) using the Groq baseline.
- SFT dataset generation pipeline (teacher-distillation + rule-cleaning) built and iterated 5 times (smoke, v1, v2, v3, final).
- QLoRA training pipeline works end-to-end — 2 real training runs completed cleanly (`qwen-swarm`, `qwen-swarm-v2`), no crashes, no OOM, sensible loss curves.
- Objective (non-self-judging) evaluation framework built (`evaluate_llm`, independent Groq judge) and its scenario set (`TEST_CASES`).

**Not done:**
- STGT model has never actually been trained in this environment — no `swarm_data/`, no checkpoint. Everything downstream (real regression numbers, real demo) has exactly one data point (`llm_run_output.json`) to go on.
- The best 2 datasets (v3, final — both 100% teacher-distilled) have never been used to train an adapter.
- `evaluate_finetuned.py` has never been run — **zero accuracy numbers exist for any of the 3 trained adapters.** All you have is the Groq baseline's eval, which isn't a fine-tune at all.
- RF fingerprinting, EKF/tracking, dashboard: not started, not in this repo.
- No git repo, no backups — `adapters/` (the only trained model weights) exists only on this local disk.

---

## 5. Known bugs / gaps worth fixing before more training runs (full detail in `handoff_audit_report.md`, root of repo)

1. **Loss masking is off.** `train_qlora.py` never sets `assistant_only_loss=True` on
   `SFTConfig` (confirmed by inspecting `training_args.bin` on both real runs — it's
   `False`). Loss is computed over the *entire* prompt+response text, not just the
   assistant's JSON answer. Since the prompt is highly templated/repetitive, this
   dominates the loss/accuracy numbers and makes them look better than actual reasoning
   quality warrants (0.05 loss / 97.8% token accuracy after 2 epochs is a red flag, not a
   win).

2. **Training data is off-scale vs. real model output.** `synth_context()` in
   `build_sft_dataset.py` samples `approach_rate ~ U(-1.5, 0.5)` and
   `delta_v ~ U(-1.0, 2.0)` to build training scenarios. The one real pipeline run shows
   actual values ~1000x smaller (`approach_rate ≈ -0.001`, `delta_v ≈ 0.018`). This means
   the model is trained mostly on "accelerating"/"converging" narratives that will almost
   never actually fire in production — ties directly to the documented STGT units bug
   (§1). Rescale `synth_context` (or fix the units bug) before generating more data.

3. **Train/inference prompt mismatch.** Training rows save the raw prompt as-is, but both
   `GroqClient.generate` and `LocalHFClient.generate` prepend an extra
   `"Return ONLY valid JSON.\n\n"` line at inference/eval time that was never present
   during training — minor but real distribution skew.

4. **`TEST_CASES` has only 6 scenarios** — the code itself flags this
   (`prompts.py:54-55`: *"expand this list substantially (>=50...) before reporting any
   headline LLM accuracy. Six cases is not enough."*).

5. Minor: `configs/qlora_qwen2.5-7b.yaml` is decorative (never loaded); `INTENT_FAMILIES`
   vocabulary has a small mismatch with `OUTPUT_SCHEMA` (`"attack"` is dead, `"unknown"`
   is missing, so a correct `"unknown"` response gets flagged as a hallucination);
   `pipeline.py`/`run_full_pipeline` is documented but doesn't exist (duplicated glue code
   in two scripts instead).

---

## 6. What to ask the next chat to help plan

Suggested framing for whoever picks this up: the LLM fine-tuning pipeline is real,
functional, and has completed training runs — but has **zero evaluation numbers** and
**two known distribution-mismatch bugs** that should be fixed before more GPU time is
spent. The STGT model itself has never been trained in this environment, so all
"training data realism" work is currently anchored on a single captured example. Good
planning questions:
- Fix loss masking + rescale `synth_context` first, or train the STGT model first (so
  `synth_context` can be replaced with real `sliding_window_inference` output entirely)?
- Is there GPU time budgeted for a `qwen-swarm-v3` run, and when — before or after the
  STGT model exists?
- What's the capstone panel deadline, and does that force prioritizing "get *any*
  evaluation number published" over "get the data pipeline fully correct first"?
