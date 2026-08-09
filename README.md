# RF-Driven Swarm Intelligence

> **Team:** Aadhya S Shetty, Abhinav Waddinavar, Alisha Prakash, Sharva Chiradoni
> **Guide:** Dr. Ashok Kumar Patil

## Abstract

Counter-UAV systems can tell an operator *where* drones are; they rarely tell the operator
*what the swarm is doing* or *what to do about it*. This project attacks the second half of
that problem. Given a sequence of drone positions, a **Spatial-Temporal Graph Transformer
(STGT)** classifies the swarm's formation and detects transitions between formations; a
**bridge/rules layer** reduces that noisy, per-window classifier output into a deterministic
tactical read wherever it safely can; and an **LLM reasoning layer** (a hosted baseline model
plus a QLoRA-fine-tuned local model) handles the residual cases the rules layer structurally
cannot — producing a structured threat-level / likely-intent / recommended-action assessment.
The repository also documents, at length and without cherry-picking, an ongoing empirical
program to find and fix the actual bottlenecks in that pipeline — several of which turned out
to be synthetic-data generation bugs and evaluation-harness bugs, not model capacity.

## Problem

A formation classifier alone is not a tactical assessment. Two failure modes make the gap
between them hard to close honestly:

1. **The reduction problem.** A trained classifier's raw per-window output is noisy near
   formation transitions; naively reducing a window sequence to "the swarm went from A to B"
   throws away resolvable cases and — if done carelessly — can also manufacture false
   confidence on genuinely ambiguous ones.
2. **The LLM-necessity problem.** If a lookup table can answer 100% of realistic inputs, an
   LLM in the loop is decoration. If it can only answer a small, sharply bounded slice, the
   system needs a principled, measured routing policy between deterministic rules, a
   machine-generated "I don't know", and an LLM — not a vague "ask the LLM for everything"
   design.

This repo's `docs/development-history.md` and `AUDIT.md` are, deliberately, a record of
getting both of those wrong first and then measuring exactly how and why, rather than a
retroactively cleaned-up success story.

## Architecture

```
Drone position sequence (6 drones x 3 coords x 50 timesteps)
        |
        v
[STGT] SpatialGAT (2x GATv2Conv + pooling) -> per-timestep embedding
        |
        v
[STGT] Temporal Transformer (4x encoder layers) -> sequence embedding
        |
        v
[STGT] cls_head (7/8-way formation) + reg_head (velocity, approach rate, stability)
        |
        v
[stgt_bridge] sliding-window reduction: per-window predictions -> one (from, to)
              formation pair, or an explicit "not safely resolvable" flag
        |
        v
[coverage] bucket classification: A (dict-resolvable) / B (guardable, abstain) /
           C (structurally unresolvable -- needs real reasoning)
        |
        +--- A --> [RULES] deterministic (from,to) -> threat/intent/action lookup,
        |          LLM used only to narrate, never to decide (decision fields are
        |          validated against RULES and overwritten on any deviation)
        |
        +--- B --> machine-generated abstention, no model call
        |
        +--- C --> [LLM] Groq-hosted baseline, or QLoRA-fine-tuned local model,
                    with a scoped prior correction restricted to provable near-ties
        |
        v
Structured tactical assessment (threat_level, likely_intent, recommended_action,
explanation) -- JSON, schema in src/swarm_intent/inference.py
```

**What this repo does NOT implement**, despite the project's full stated scope (RF sensing
- fingerprinting - localization/tracking - swarm modeling - LLM reasoning - AR dashboard):
RF fingerprinting, multilateration/EKF tracking, ONNX edge export, and the AR/3D dashboard
are a teammate's work, tracked in a separate repository, and are not present here. STGT
consumes drone *positions* directly (as would come out of that upstream tracker), not RF
signals. See `docs/architecture.md` for the implemented/experimental/planned breakdown in
full and `MIGRATION_GUIDE.md` for where the missing pieces are meant to land
(`src/swarm_intent/rf/`, `src/swarm_intent/tracking/`) once integrated.

## Research contributions

Claimed only to the extent the repository actually demonstrates them:

- A measured, reproducible answer to "why not just ship a lookup table instead of an LLM":
  bucket-classified real STGT output shows the deterministic RULES dict alone resolves a
  small, specific minority of real observations (see `docs/evaluation.md`); the rest are
  either safely-abstainable or genuinely need the LLM layer.
- A three-layer routing architecture (`src/swarm_intent/pipeline_v2.py`) with a *provable*
  guarantee that the LLM can never override a deterministic RULES decision in the cases RULES
  actually covers (independently verified in `docs/development-history.md`'s sec AH summary:
  0 deviations across a dedicated verification run).
- Multiple real bottlenecks in the STGT-to-tactical-assessment pipeline, found, audited, and
  root-caused (guard-condition bugs that tested the wrong thing, a data-generator labeling
  regime that never matched realistic long-trajectory evaluation, an evaluation bridge that
  threw away resolvable sequences on brittle unanimity voting) — and, where fixed,
  re-measured with the same protocol before and after, including negative results (one large
  fix, "robust reduction", was implemented, tested, and explicitly **not** shipped as the
  default after measurement showed it traded accuracy for confident silence).
- An explicit chain of custody from a real upstream generator physics bug (shared
  dispersed/converging geometry, no acceleration term) through to its measured effect on
  end-to-end threat-classification ceiling (13.0% to 52.3%/58.7% from one fix), documented
  with the diff that caused it and the commit that ported it.

No claim is made that the current system meets its own stated 70%-pair-accuracy internal
target — it does not, yet. See Current status.

## Current status

| Component | Status |
|---|---|
| STGT formation classifier | **Implemented**, trained on synthetic data only. Chain-length-2 (single real transition) pair accuracy is under active improvement: 18.7% -> 65.8% across the V5 program's fixes so far (docs/development-history.md), short of the program's internally stated 70% floor. Chain-length-1 (steady state) accuracy is ~85-88%. |
| Bridge / reduction logic (`stgt_bridge.py`) | **Implemented**, audited, several guard bugs found and fixed (see docs/development-history.md). A "robust" majority-vote reduction mode exists, is tested, and is **disabled by default** after measurement showed it hurts more than it helps. |
| Coverage routing (`coverage.py`, bucket A/B/C) | **Implemented** and measured against real STGT output, not just templated text cases. |
| RULES deterministic layer | **Implemented**, 49 (from,to) pairs -> threat/intent/action, canonical in `llm_finetuning/build_sft_dataset.py`. |
| LLM layer — hosted baseline | **Implemented** (Groq API, `src/swarm_intent/llm/client.py`). |
| LLM layer — fine-tuned local model | **Implemented**, 5 QLoRA adapter iterations documented in `docs/ADAPTER_VERSIONS.md`; adapter weights are not committed (regenerate via `llm_finetuning/train_qlora.py`). |
| RF fingerprinting | **Not implemented here** — teammate's repo, not integrated. |
| Multilateration / EKF tracking | **Not implemented here** — teammate's repo, not integrated. |
| AR / 3D visualisation dashboard | **Not implemented, planned.** |
| Test suite | 140 unit tests, `python -m unittest discover -s tests`, no GPU required for the bridge/coverage/config tests. |

## Repository structure

```
src/swarm_intent/       importable package (import swarm_intent) -- STGT model, graph
                         construction, synthetic data generation, the stgt_bridge/coverage
                         routing logic, pipeline_v2, and the LLM client layer
scripts/                 data generation, training, and the V5 diagnostic/audit scripts
                         (see scripts/README.md for which are production vs. research tools)
llm_finetuning/          QLoRA fine-tuning pipeline: SFT dataset construction, training,
                         evaluation, and ~50 audit/diagnostic scripts (own README.md)
tests/                   unit tests (unittest, no test framework dependency)
data/                    SFT training/validation sets consumed by llm_finetuning/ (see
                         data/README.md) -- NOT raw RF data; this project's data is synthetic
evaluation/              raw evaluation output (JSON) from the eval scripts above
docs/                    architecture, methodology, development history, evaluation summary,
                         V5 program logs (V5_LOG.md, CEILING.md, etc.), archived stale docs
calibration/             a small calibration artifact consumed by prior-correction tests
visuals/                 example output figures
```

Deliberately unchanged from the working-development layout: the Python package is named
`swarm_intent` (not `rf_swarm`) — renaming it would touch every import across ~150 files for
no functional benefit, so this repository keeps the real, working name rather than a
cosmetic one chosen to match a template.

## Installation

```bash
git clone https://github.com/abhinow03/rf-driven-swarm-intelligence.git
cd rf-driven-swarm-intelligence
pip install -e .
pip install -r requirements.txt   # torch / torch-geometric are platform-specific; see the file
```

LLM fine-tuning has its own, GPU-specific dependency set (`pip install -e ".[finetune]"`) —
see `llm_finetuning/README.md`.

## Data

All data in this project is **synthetic**, generated from `src/swarm_intent/data.py` /
`src/swarm_intent/eval_trajectories.py` — there is no raw RF or flight data of any kind
committed or required. `data/*.jsonl` (SFT training sets for the LLM fine-tuning pipeline) is
committed since it is small (a few MB) and is itself a research artifact with a documented
provenance (see `data/README.md`); it can also be regenerated from scratch. Model
checkpoints, LoRA adapters, and `swarm_data/` (normalization stats, generated splits) are
**not** committed (`.gitignore`) — regenerate them with the commands below.

## Training

```bash
# 1. Generate the synthetic STGT dataset (omit --transitions for the 7-class model)
python scripts/generate_data.py --per-formation 1000 --transitions 2000

# 2. Train STGT
python scripts/train_model.py --classes 8 --epochs 80

# 3. LLM fine-tuning (separate GPU workflow) -- see llm_finetuning/README.md
export GROQ_API_KEY=...   # required for the teacher-distillation and judge steps
python llm_finetuning/build_sft_dataset.py --n 600 --out data/sft_train.jsonl
python llm_finetuning/train_qlora.py ...
```

## Evaluation

The headline metric throughout this project is **end-to-end objective accuracy**
(intent/threat correctness against independently-derived ground truth), not an LLM grading
its own output — an early version of the eval harness had a model self-score 5/5 while
objective accuracy was near 0%; that failure mode is explicitly guarded against now
(`src/swarm_intent/llm/evaluate.py`'s docstring, `docs/methodology.md`). See
`docs/evaluation.md` for the current numbers and what they mean; see
`docs/development-history.md` for how they got there.

## Reproduction

```bash
pip install -e . && pip install -r requirements.txt
python -m unittest discover -s tests -q                      # 140 tests, no GPU needed
python scripts/generate_data.py --per-formation 20 --transitions 40   # smoke-test data gen
python scripts/train_model.py --classes 8 --epochs 2                  # smoke-test training
python scripts/phase0_ceiling.py --n 100                              # V5 ceiling measurement
```

## Experiments

`docs/development-history.md` summarizes the V1-V5 arc; `docs/experiments.md` lists what each
major experiment established, including the ones that didn't pan out (a "robust reduction"
majority-vote fix that was built, tested, and deliberately not shipped after measurement).

## Limitations

Stated plainly, not buried:

- **Synthetic data only.** STGT has never seen real drone telemetry or real RF-derived
  positions; generalization to real sensor noise is untested.
- **Regression labels are in normalized-space, not physical units** (a known bug — see
  `CODE_REVIEW.md`). Do not present `centroid_velocity` output as m/s without rescaling.
- **The pipeline does not yet meet its own internal accuracy target.** `docs/CEILING.md`
  documents the actual measured effect of every attempted fix, including negative results.
- **Chain-length-2 pair accuracy (65.8%) is still below this program's own 70% internal
  target**, and the leading hypothesis (train/eval blend-timing distribution mismatch) is
  diagnosed but not yet fixed as of the latest entry in `docs/V5_LOG.md`.
- **RF fingerprinting, tracking, and the AR dashboard are not in this repository.**

## Citation

No publication exists yet for this work. If you use this code, please cite the repository
directly until a paper citation is available:

```bibtex
@software{rf_driven_swarm_intelligence,
  title  = {RF-Driven Swarm Intelligence},
  author = {Shetty, Aadhya S and Waddinavar, Abhinav and Prakash, Alisha and Chiradoni, Sharva},
  year   = {2026},
  url    = {https://github.com/abhinow03/rf-driven-swarm-intelligence}
}
```

See `CITATION.cff` for the machine-readable version.
