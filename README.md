# UAV Swarm Threat Assessment — STGT + LLM Tactical Reasoning

> **Team:** Aadhya S Shetty, Abhinav Waddinavar, Alisha Prakash, Sharva Chiradoni
> **Guide:** Dr. Ashok Kumar Patil

A counter-UAV pipeline that turns a swarm's drone-position trajectory into a tactical
assessment: a Spatial-Temporal Graph Transformer (STGT) classifies formation and detects
transitions, a deterministic bridge/rules layer resolves as much of that as it safely can
without a model call, and a fine-tuned LLM handles the genuinely ambiguous remainder — routed
by a measured, not assumed, coverage split.

**What this document is**: an honest, current-state description of what's actually
implemented and measured in this repo, not an aspirational system spec. Where an earlier
version of this README described components that were never built here (RF fingerprinting,
EKF/multilateration tracking, AR dashboard), those are called out explicitly below as **not
in this repo** — see [Scope](#scope-whats-actually-here-vs-not).

---

## Architecture

```
Drone positions (6 drones x 3 coords x 50 timesteps)
        |
        v
   [ STGT ]   src/swarm_intent/model.py -- GATv2 spatial encoder (per timestep)
        |     + Transformer temporal encoder -> formation class + regression heads
        v
   [ Bridge ]  src/swarm_intent/stgt_bridge.py -- reduces a noisy per-window formation
        |      read into one resolved (from, to) transition event, or "unresolved"
        v
   [ Coverage ]  src/swarm_intent/coverage.py -- buckets each case:
        |        A (RULES can answer) / B (must abstain) / C (needs LLM reasoning)
   +----+----+----------------+
   |         |                |
   v         v                v
[ RULES ]  [ abstain ]    [ LLM ]  Qwen2.5-7B-Instruct, QLoRA fine-tuned
   |         |                |    (src/swarm_intent/llm/ -- client.py, pipeline.py)
   +----+----+----------------+
        v
Structured JSON: threat_level / likely_intent / recommended_action / ...
(src/swarm_intent/inference.py's OUTPUT_SCHEMA)
```

`src/swarm_intent/pipeline_v2.py` is the real end-to-end entrypoint implementing this
routing. `docs/DEFENSE.md` has the fully-cited case for why this three-layer split (not a
single model, not a static lookup table) is structurally necessary.

## Scope: what's actually here vs. not

Per `CLAUDE.md` and `CODE_REVIEW.md`: this is a university capstone, and **only two of the
four components sometimes described for this project have code in this repo**:

| Component | Status |
|---|---|
| RF fingerprinting (hardware ID from I/Q signals) | **not in this repo** — a teammate's separate repo, planned integration point `src/swarm_intent/rf/` (see `MIGRATION_GUIDE.md`) |
| EKF / multilateration tracking | **not in this repo** — planned integration point `src/swarm_intent/tracking/` |
| **Swarm behavior model (STGT)** | **implemented here** — classifies formation from drone *positions*, not RF signals |
| **LLM tactical interpretation layer** | **implemented here** — sliding-window inference -> rule-based context -> LLM assessment |
| AR / 3D dashboard | **not in this repo** |

When any other document in this repo and the code disagree, trust the code.

## Current status (2026-08-17)

- **v5-a** (`checkpoints/v5_sft_v5a_PROTECTED/`) is the shipped, hash-locked, read-only
  baseline adapter — QLoRA fine-tuned Qwen2.5-7B-Instruct, trained on the 12,001-row Phase 1
  corpus. It is the model behind the demo (`run_demo.py` / `scripts/demo_web.py`).
- **v5-a's known gap**: 0.0% correct-abstention on structurally unanswerable inputs (genuine
  multi-hop / oscillation trajectories) — it always guesses rather than declining to answer.
  Full diagnosis: `AUDIT.md` sec AK; corpus response: `docs/PHASE3A_ABSTENTION_CORPUS.md`.
- **v5a2** (abstention retrain) is **in progress, not yet scored** — a fresh QLoRA run on
  v5-a's corpus plus 900 new abstention examples, against bars locked *before* training in
  `docs/PREREGISTRATION_V5A2.md`. Do not cite v5a2 numbers until that document's scoring
  script has actually run against a completed checkpoint.

### v5-a's measured numbers (seed=999 population, the current reference population — see
`docs/PREREGISTRATION_V5A2.md` erratum part 3 for why seed=999 superseded seed=4321)

| metric | v5-a | same-population STGT+bridge ceiling |
|---|---|---|
| threat_accuracy (answerable cases) | 78.9% (n=493) | 83.0% |
| pair_accuracy, real/literal (non-proxy) | 63.6% (n=494) | 77.3% |
| correct_abstention_rate (multi_hop / oscillation) | **0.0% / 0.0%** | n/a |
| over_abstention_rate | 0.2% | n/a |
| schema_validity_rate | 100.0% (not discriminative — see `PREREGISTRATION_V5A2.md` bar h) | -- |

## Locked artifacts

Hash-locking datasets, checkpoints, and eval populations *before* they're used to produce a
result is a running discipline in this project (see [Methodology](#methodology-discipline)
below). These are the artifacts currently under that discipline:

| artifact | sha256 | what it is |
|---|---|---|
| `checkpoints/v5_sft_v5a_PROTECTED/adapter_model.safetensors` | `79b71224e2d04a6149adf63ec3fcfc825d58007ce4bfe144e5d1f0e7cb89aad5` | v5-a, the shipped baseline adapter — read-only, never retrained on |
| `data/sft_train_v5_phase3a_merged.jsonl` | `5123a833274a168af2d420cc833f6c51b1493202a3e2b05e06b8e44fd8e2ab6b` | the full 12,901-row v5a2 training pool (12,001 Phase 1 rows + 900 abstention rows) |
| `data/sft_train_v5a2_train.jsonl` | `ce56869c47cfe5666bccc638b26d53d523ef4193d825eaf581cfd45d08359639` | v5a2's actual training split (12,184 rows), stratified from the pool above |
| `data/sft_train_v5a2_val.jsonl` | `c564c0a1cebaa0efc073eb49fe18b677fc9dc9ec8c299fe0acd0213637690f7e` | v5a2's held-out val split (717 rows) |
| `eval_data/LOCKED_seed999_FINAL.json` | `871a9dae4c6fdf08e1aed803592fa7c61b1a852c150693b5819fe2271717b96e` | the eval trajectory population (seed=999) v5a2 will be scored against |
| `docs/PREREGISTRATION_V5A2.md` + `scripts/check_preregistration_v5a2.py` + its two test files, concatenated | `edddf746f41efa45b1e55306d9cee89fe2aa08965fced04341cc5b63dd628ba8` | v5a2's pass/fail bars, locked *before* any v5a2 result exists |

## Known limitations (stated plainly, per `CODE_REVIEW.md`)

- **Synthetic data only.** All STGT training/eval data is simulator-generated (`src/swarm_intent/data.py`); no real drone telemetry has been used anywhere in this repo.
- **Units caveat on regression labels.** `centroid_velocity` etc. are computed on *normalized* positions, not physical units — despite what some worked examples elsewhere imply, this is not m/s. See `CLAUDE.md` / `dataset.py`.
- **v5-a's abstention gap** (above) is real and unresolved as of this document — v5a2 exists specifically to close it, and has not yet been shown to.
- **No CI, no lint config, no committed model weights.** Datasets and checkpoints must be regenerated/retrained locally; see [Reproduction](#reproduction) below.

## Methodology discipline

A distinctive, deliberate part of this project's process, worth surfacing for anyone reading
the repo cold:

- **Rule 0 artifact verification**: before trusting any cited number, re-derive or re-hash
  the artifact live rather than trusting a prior record (`scripts/rule0_*.py`,
  `scripts/phase3a_verify_safety_copy.py`).
- **Preregistration**: pass/fail bars are written down, with numeric targets and reasoning,
  *before* a training run happens — not fit to the result afterward (`docs/PREREGISTRATION.md`,
  `docs/PREREGISTRATION_V5A2.md`). Both documents are append-only past their FINAL lock;
  corrections happen via dated, appended errata, never silent edits.
- **Hash-locking**: training corpora, eval populations, and preregistration documents are
  sha256-locked at the moment they're finalized, so their content can be proven not to have
  been shaped by hindsight.
- **Independent judging**: `src/swarm_intent/llm/evaluate.py`'s `judge_client` must be a
  model independent from the system under test — an earlier version of this project had a
  model grade itself (5/5 self-scores against ~0% objective accuracy); that mistake is now
  guarded against structurally, not just by convention.
- **Non-destructive corrections**: nothing gets silently deleted or rewritten when a mistake
  is found — `AUDIT.md` and the `PREREGISTRATION*.md` docs record corrections as dated,
  appended errata on top of the original text.

Full history of every measurement, dead end, and correction: `AUDIT.md` (the primary lab
notebook, append-only, ~4,000 lines). A summary of how the current model lineage got here:
[`docs/LINEAGE.md`](docs/LINEAGE.md).

## Repository structure

```
src/swarm_intent/      importable package
  model.py, graph.py     STGT: GATv2 spatial encoder + Transformer temporal encoder
  stgt_bridge.py          reduces per-window STGT output to one (from, to) event
  coverage.py             routes each case to RULES / abstain / LLM
  pipeline_v2.py          the real end-to-end entrypoint
  ground_truth_abstention.py   simulator-truth classifier for multi_hop/oscillation labels
  llm/                    client.py (Groq + local HF), pipeline.py, evaluate.py, prompts.py
  stgt/                   a second STGT model/inference implementation (see docs/GAP_DIAGNOSIS.md)

llm_finetuning/         QLoRA fine-tuning + the bulk of this project's diagnostic scripts
  train_sft_v5.py          the v5-a / v5a2 training script (this doc's locked hyperparameters)
  build_sft_dataset.py     RULES dict lives here -- canonical domain decision logic
  evaluate_finetuned.py, eval_sft_v5.py, score_memorization.py, literal_pair_extraction.py, ...

scripts/                data generation, STGT training, demo, and phase0/rule0 diagnostics
  generate_data.py, train_model.py     STGT dataset + training entrypoints
  demo_act1_contrast.py / demo_act2_pipeline.py / demo_act3_live.py / demo_web.py   defense demo
  phase0_*.py, rule0_*.py               historical diagnostic scripts, most superseded by
                                         later AUDIT.md sections but kept for the record
  bench_adapter_hotswap.py              adapter hot-swap vs. full-reload benchmark (not yet
                                         wired into the main eval runners)

data/, evaluation/, eval_data/    training corpora, eval results, locked eval populations
checkpoints/            trained adapters (gitignored -- not committed, see .gitignore)
docs/                    architecture/methodology detail (see below)
tests/                   unit tests

# Historical / superseded, kept for the record rather than deleted (this project's own
# non-destructive convention -- see Methodology above):
swarm_data_backup3_gap2fix_9k/, swarm_data_prefix_backup_20260807/,
swarm_data_prefix_backup2_20260807_gap2/    STGT dataset backups from earlier generator-fix
                                             iterations; swarm_data/ is the current one
checkpoints_variance_diagnosis/             one-off variance-measurement checkpoints, not a
                                             model lineage step
HISTORY.md, PROJECT_HANDOFF.md,
handoff_audit_report.md, ADAPTER_VERSIONS.md   earlier project-state snapshots, superseded
                                                 by AUDIT.md's continuous record
```

### Key docs (all linked, none altered by this branch)

- [`AUDIT.md`](AUDIT.md) — the full measurement history, append-only, every claim cited
- [`docs/CEILING.md`](docs/CEILING.md) — STGT+bridge's own accuracy ceiling, stratified by chain length
- [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) — v5-a's preregistered bars
- [`docs/PREREGISTRATION_V5A2.md`](docs/PREREGISTRATION_V5A2.md) — v5a2's preregistered bars (**FINAL, hash-locked, do not edit**)
- [`docs/PHASE3A_ABSTENTION_CORPUS.md`](docs/PHASE3A_ABSTENTION_CORPUS.md) — how the abstention retrain corpus was built
- [`docs/RULES_EXTENSION_PROPOSAL.md`](docs/RULES_EXTENSION_PROPOSAL.md) — proposed RULES-table coverage extension
- [`docs/DEFENSE.md`](docs/DEFENSE.md) — the cited case for the 3-layer architecture
- [`docs/UPSTREAM_ISSUES.md`](docs/UPSTREAM_ISSUES.md) — defects requested against code outside `src/swarm_intent/`
- [`docs/GAP_DIAGNOSIS.md`](docs/GAP_DIAGNOSIS.md) — diagnostic pass on two accuracy gaps `CEILING.md` flagged
- [`docs/V5_LOG.md`](docs/V5_LOG.md) — running log of V5-era training runs (live during active training)
- [`CODE_REVIEW.md`](CODE_REVIEW.md) — honest assessment of known limitations
- [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) — old-notebook-function -> new-module map

## Reproduction

```bash
# Install (torch + torch-geometric must be installed per-platform — see requirements.txt)
pip install -e .
pip install -r requirements.txt

# 1. Generate synthetic STGT dataset -> swarm_data/
python scripts/generate_data.py --per-formation 1000 --transitions 2000

# 2. Train STGT -> swarm_data/best_model.pt
python scripts/train_model.py --classes 8 --epochs 80

# LLM baseline pipeline requires a Groq key (never hardcode; read from env):
export GROQ_API_KEY=...
```

QLoRA fine-tuning (v5-a / v5a2) runs on a GPU box — see `llm_finetuning/README.md`; deps are
the `finetune` extra: `pip install -e ".[finetune]"`.

### Running the demo

```bash
python run_demo.py                 # terminal walkthrough: Act 1 -> pause -> Act 2 -> pause -> Act 3
python scripts/demo_web.py         # web version, http://127.0.0.1:5000
```

Both re-verify the protected v5-a checkpoint and check free GPU memory before starting,
rather than starting something that would crash mid-presentation.

There is no test suite CI gate, but `tests/` (140+ unit tests) can be run with
`python -m unittest discover -s tests`. No model weights or dataset artifacts are committed —
generate/train locally per the steps above.
