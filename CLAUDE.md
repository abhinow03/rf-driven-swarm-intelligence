# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually contains

This is a university capstone. The full stated project is a **four-component** counter-UAV
system (RF fingerprinting → EKF multilateration → swarm behavior model → LLM intent layer →
AR dashboard, see `docs/architecture.md`), but **only two of those components have code
here**:

- **Swarm behavior model (STGT)** — a Spatial-Temporal Graph Transformer that classifies swarm formation from drone *positions* (not RF signals).
- **LLM interpretation layer** — sliding-window inference → rule-based tactical context → prompt → LLM tactical assessment (JSON).

The RF fingerprinting model, EKF/multilateration tracker, ONNX export, and AR dashboard are **not in this repo** (per `MIGRATION_GUIDE.md` they live in a teammate's repo, to be added under `src/swarm_intent/rf/` and `.../tracking/`). When any documentation and the code disagree, trust the code. See `CODE_REVIEW.md` for an honest assessment of known limitations (synthetic-only data, circular regression labels, units bug).

The package was migrated from three Jupyter notebooks (`capstone_with_llm.ipynb`, `capstone_with eval.ipynb`, `models + data generation.ipynb`) in the initial commit; the notebooks themselves are no longer present in this repository's tree. **`src/swarm_intent/` is the source of truth** — do not edit or revive the notebooks for new work; `MIGRATION_GUIDE.md` maps old notebook functions to new module locations.

## Commands

```bash
# Install (torch + torch-geometric must be installed per-platform — see requirements.txt)
pip install -e .
pip install -r requirements.txt

# 1. Generate synthetic dataset → saves splits + norm_stats to swarm_data/
#    Omit --transitions for the 7-class model; pass it to enable the 8th "transitioning" class.
python scripts/generate_data.py --per-formation 1000 --transitions 2000

# 2. Train STGT model → saves swarm_data/best_model.pt (with cfg + reg stats embedded)
python scripts/train_model.py --classes 8 --epochs 80

# LLM baseline / pipeline requires a Groq key:
export GROQ_API_KEY=...   # never hardcode; GroqClient reads it from the env
```

LLM fine-tuning (QLoRA) is a separate workflow run on a GPU box / Colab T4 — see `llm_finetuning/README.md`. Its deps are the `finetune` extra (`pip install -e ".[finetune]"`).

There is **no test suite, linter config, or CI** in this repo. There are no saved model weights or dataset artifacts committed — you must run `generate_data.py` then `train_model.py` to produce them. Smoke-test changes with a tiny dataset (`--per-formation 20`, 1–2 epochs).

## Architecture and data flow

### The model: STGT (`src/swarm_intent/model.py`)

Fixed input shape: **6 drones × 3 coords (x,y,z), 50 timesteps**. These dimensions are hardcoded throughout — changing drone count or window length means touching multiple files.

```
(50, 6, 3) sequence
  → sequence_to_graphs (graph.py): one proximity graph per timestep
       nodes = drone (x,y,z); edges connect pairs closer than cfg.edge_threshold
       (NORMALISED units); edge_attr = (dx,dy,dz,dist); self-loops if isolated
  → SpatialGAT: 2× GATv2Conv + global_mean_pool → (50, d_model) per-timestep vectors
  → TemporalTransformer: positional encoding + 4× encoder layers + mean-pool → (d_model,)
  → shared head → two heads:
       cls_head: formation classifier (7 or 8 classes)
       reg_head: 3 regression values [centroid_velocity, approach_rate, stability]
```

The model **infers its device from its own parameters** (no global `device`). `forward()` takes a `list[batch]` of `list[T]` PyG `Data` graphs and batches them internally.

### Config is the single source of truth (`config.py`)

One `Config` dataclass replaces the old `CFG`/`CFG_V2` dicts. Key invariants:
- `d_model` **must equal** `gat_out_dim` (both 128).
- `n_classes`: 7 (base formations) or 8 (adds `"transitioning"`). The formation vocabulary order in `BASE_FORMATIONS` **is** the integer label mapping — index = label. Use `formation_names(cfg)` to get the active list.
- `Config` supports dict-style access (`cfg["lr"]`, `cfg.get(...)`) for legacy compatibility.

### Normalization discipline (important, easy to get wrong)

- Position normalization stats (`train_mean`/`train_std`) are computed from the **TRAIN split only** and applied to all splits (`data.split_and_normalize`). Saved to `swarm_data/norm_stats.npy`.
- Regression-label stats (`reg_mean`/`reg_std`) are likewise computed on train and **passed into** the val/test `SwarmDataset`. The training checkpoint embeds `reg_mean`/`reg_std`.
- At inference, every raw window must be normalized with the **same train stats** before `predict()`. `sliding_window_inference` does this per window.

> **Known units caveat** (`dataset.py`, `CODE_REVIEW.md`): regression labels are computed on *normalised* positions, so `centroid_velocity` is in normalised-space, not physical m/s. The README's worked example (e.g. `4.3`) does not match actual pipeline output (~0.05). Do not present these as m/s without rescaling.

### LLM layer (`src/swarm_intent/llm/`)

- **`client.py`** — provider-agnostic. `GroqClient` (hosted baseline, default `llama-3.3-70b-versatile`) and `LocalHFClient` (local/QLoRA-fine-tuned HF model). Both implement `generate(prompt) -> str`; the base `complete()` adds retry/backoff and tolerant JSON extraction. Swap the client and the rest of the pipeline is identical.
- **`pipeline.py`** — `run_full_pipeline(model, raw_sequence, ..., llm_client)`: sliding-window ML inference → `build_tactical_context` → `build_llm_prompt` → `llm_client.complete()`.
- **`inference.py`** — `OUTPUT_SCHEMA` is the single source of truth for the JSON the LLM must return; it's used by both prompt assembly and evaluation. Edit it there, not in copies.
- **`evaluate.py`** — `evaluate_llm` takes a `run_case` callback and reports **objective intent/threat accuracy as the headline metric**. The LLM-as-judge (`judge_client`) must be an **independent** client/model from the system under test — the original eval had a model grade itself (5/5 self-scores while objective accuracy was ~0); do not reintroduce that. `evaluate_ml_model` leads with macro-F1 (robust to `transitioning`-class imbalance).

### LLM fine-tuning (`llm_finetuning/`)

`build_sft_dataset.py` → `train_qlora.py` → `evaluate_finetuned.py`. The `RULES` dict in `build_sft_dataset.py` is the **canonical domain decision logic** the model learns — teacher-distilled prose is overridden with rule-clean `threat_level`/`likely_intent`/`recommended_action` labels so the model learns a consistent policy, not teacher noise. Curating `RULES` is the highest-leverage knob; edit it with domain input.

## Conventions

- The migration deliberately removed hidden global state. Pass `cfg`, `device`, graph thresholds, and normalization stats **explicitly** — do not reintroduce module-level globals like the old `CFG`/`device`/`model_v2`.
- Reproducibility: a single seeded `np.random.Generator` (from `cfg.seed`) is threaded through **all** sampling in `data.py`. Keep it that way — do not call `np.random.*` or `default_rng()` without the threaded `rng`.
- Secrets come from the environment (`GROQ_API_KEY`). Never commit a key or a placeholder fallback key.
