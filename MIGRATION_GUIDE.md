# Migration Guide — from 3 notebooks to a clean package

This converts the repetitive notebook code into one importable package and adds
the LLM fine-tuning pipeline. Nothing was deleted — the original notebooks are
kept so you can diff. The new code is the source of truth going forward.

## New layout

```
src/swarm_intent/            ← the package (import swarm_intent)
  config.py                  ← ONE Config dataclass (replaces CFG / CFG_V2)
  formations.py              ← get_formation_offsets (+ seedable RNG fix)
  data.py                    ← generate_swarm_sequence, transitions, dataset, split+normalize
  graph.py                   ← build_graph, sequence_to_graphs
  model.py                   ← SpatialGAT, PositionalEncoding, TemporalTransformer, STGTModel
  dataset.py                 ← SwarmDataset, collate_fn, compute_regression_labels
  train.py                   ← train_one_epoch, evaluate, train
  inference.py               ← predict, sliding_window_inference, tactical context, prompt
  llm/
    client.py                ← GroqClient + LocalHFClient (one provider-agnostic interface)
    prompts.py               ← intent/threat families, matching, test cases, judge prompt
    pipeline.py              ← run_full_pipeline
    evaluate.py              ← evaluate_llm (independent judge) + evaluate_ml_model

scripts/
  generate_data.py           ← python scripts/generate_data.py --per-formation 1000 --transitions 2000
  train_model.py             ← python scripts/train_model.py --classes 8

llm_finetuning/              ← YOUR area (see its README)
  build_sft_dataset.py · train_qlora.py · evaluate_finetuned.py · configs/

requirements.txt · pyproject.toml · .gitignore
```

## What each notebook function maps to

| Old (any notebook) | New location |
|---|---|
| `CFG`, `CFG_V2` | `config.Config` (one class; `n_classes=8` for transitions) |
| `get_formation_offsets` | `formations.get_formation_offsets` |
| `generate_swarm_sequence`, `generate_dataset`, transition generators | `data.py` |
| `build_graph`, `sequence_to_graphs` | `graph.py` |
| `SpatialGAT`/`PositionalEncoding`/`TemporalTransformer`/`STGTModel` | `model.py` |
| `SwarmDataset`, `collate_fn`, `compute_regression_labels` | `dataset.py` |
| `train_one_epoch`, `evaluate`, `train` | `train.py` |
| `predict`/`predict_v2`, `sliding_window_inference`, `build_tactical_context`, `build_llm_prompt` | `inference.py` (one `predict`) |
| `call_llm` | `llm/client.py` (`GroqClient`) |
| `run_full_pipeline` | `llm/pipeline.py` |
| `TEST_CASES`, `INTENT_FAMILIES`, `JUDGE_PROMPT`, `run_llm_evaluation`, `evaluate_ml_model` | `llm/evaluate.py` + `llm/prompts.py` |

## How to run end to end

```bash
pip install -e .            # plus torch + torch-geometric for your platform
pip install -r requirements.txt

python scripts/generate_data.py --per-formation 1000 --transitions 2000
python scripts/train_model.py --classes 8 --epochs 80
# LLM baseline / fine-tune: see llm_finetuning/README.md
```

## Bugs fixed during migration (verify on first run)

- **Reproducibility:** one seeded RNG now threads through ALL sampling. The old
  `generate_swarm_sequence` used an unseeded generator, so datasets were not
  reproducible despite the `seed=42` comment.
- **Normalization:** val/test now reuse TRAIN regression stats and TRAIN
  position mean/std (the old code gave each split its own stats).
- **No more global `device`/`CFG`:** the model infers its device from its
  parameters; graph/threshold are passed explicitly.
- **LLM client:** added HTTP error checking + retry/backoff; `GROQ_API_KEY`
  comes from the environment (no placeholder key in code).
- **Eval:** the judge is now an INDEPENDENT client. The old eval let the same
  model grade itself, which produced 5/5 self-scores while objective intent
  accuracy was ~0. Objective accuracy is now the headline metric. Macro-F1 is
  reported for the classifier to counter the `transitioning`-class imbalance.

## Things I reconstructed (diff against the notebook if exact parity matters)

- `data.generate_transition_sequence` — rebuilt from the documented cosine-ramp
  blend (the original cell body wasn't fully transcribed).
- `inference.predict` transition_from/to — derived from the top-2 non-transition
  class probabilities.

I could not execute anything in this environment (no GPU / torch-geometric), so
**run a smoke test in Colab**: generate a tiny dataset (`--per-formation 20`),
train 1–2 epochs, run one pipeline call. Fix any import/shape mismatch early.

## Still open / decisions for you

- **LICENSE:** none added — for a university capstone the IP may belong to PES
  University. Confirm with your guide, then add the chosen license. The review
  flagged this.
- **RF fingerprinting + EKF** code lives in a teammate's repo and still needs to
  be added under, e.g., `src/swarm_intent/rf/` and `.../tracking/`.
- The synthetic-data and circular-regression-label limitations from
  `CODE_REVIEW.md` are not "fixed" by restructuring — they're research issues to
  tackle next.

## Your LLM fine-tuning plan (short version)

1. **Curate `RULES`** in `build_sft_dataset.py` with your team — this IS the
   ground-truth policy the model learns. Get this right first.
2. **Build ~600 diverse examples** (teacher-distilled prose + rule-clean labels),
   including ambiguous/adversarial cases.
3. **QLoRA fine-tune Qwen2.5-7B-Instruct** on the free T4 (`train_qlora.py`).
4. **Evaluate with an independent judge** and report objective intent/threat
   accuracy vs the Groq baseline (`evaluate_finetuned.py`).
5. **Iterate** on data (the highest-leverage knob), then merge + export for edge.

Full detail: `llm_finetuning/README.md`.
