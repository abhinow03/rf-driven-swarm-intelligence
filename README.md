# RF-Driven Swarm Intelligence

A counter-UAV pipeline that turns a swarm's drone-position trajectory into a tactical
assessment: a Spatial-Temporal Graph Transformer (STGT) classifies formation and detects
transitions, a bridge/rules layer resolves as much of that as it safely can deterministically,
and an LLM layer (hosted baseline + QLoRA fine-tuned local model) handles the rest — routed by
a measured, not assumed, coverage split.

## Architecture

```
Drone positions (6 x 3 x 50)
        |
        v
   [ STGT ]  GATv2 spatial encoder + Transformer temporal encoder
        |    -> formation class + transition detection
        v
   [ Bridge ]  stgt_bridge.py: window sequence -> (from, to) pair, or "unresolved"
        |
        v
   [ Coverage ]  bucket A (resolvable) / B (guardable) / C (needs reasoning)
        |
   +----+----+----------------+
   |         |                |
   v         v                v
[RULES]  [abstain]        [ LLM ]  fine-tuned qwen2.5 7B
   |         |                |
   +----+----+----------------+
        v
Threat level / likely intent / recommended action (JSON)
```

## What the System Does

- Classifies UAV swarm formation (7 types) from position trajectories and detects transitions between them
- Reduces noisy per-window model output into a single resolved formation-change event, or an explicit "not resolvable"
- Answers a measured fraction of real observations from a deterministic rules table alone — no model call
- Abstains, rather than guesses, on cases a rules table cannot safely answer
- Routes the remaining, genuinely ambiguous cases to an LLM for tactical reasoning
- Produces a structured JSON tactical assessment: threat level, likely intent, recommended action
- Documents its own accuracy ceiling and what's blocking it, rather than only reporting favorable numbers

## Key Components

```
RF sensing → fingerprinting → localization/tracking → STGT → bridge → RULES → LLM
```

| Component | Role | Status |
|---|---|---|
| RF sensing / fingerprinting / tracking | Turn raw RF into drone positions | external, not in this repo |
| STGT | Classify formation + transitions from positions | implemented |
| Bridge | Reduce noisy window predictions to one event | implemented |
| RULES | Deterministic (from, to) → threat/intent/action | implemented |
| LLM | Reason about cases RULES can't resolve | implemented |

## Current Results

| Metric | Value |
|---|---|
| Chain-length-1 (steady state) pair accuracy | ~87% |
| Chain-length-2 (single transition) pair accuracy | 65.8% (from an 18.7% baseline) |
| End-to-end threat-level ceiling | 52.3–58.7% (from 13.0%) |
| Layer-1 (RULES) decision-override rate | 0 / 60 verified — provably never overridden |

Full methodology and the complete measurement history: `docs/evaluation.md`.

## Repository Structure

```
src/swarm_intent/   importable package — STGT, bridge, coverage routing, LLM client
scripts/            data generation, training, V5 diagnostic tools (see scripts/README.md)
llm_finetuning/     QLoRA fine-tuning pipeline and evaluation scripts
tests/              140 unit tests
data/, evaluation/  SFT datasets and raw evaluation output
docs/               architecture, methodology, development history, evaluation detail
```

## Reproduction

```bash
pip install -e . && pip install -r requirements.txt
python -m unittest discover -s tests -q
python scripts/generate_data.py --per-formation 1000 --transitions 2000
python scripts/train_model.py --classes 8 --epochs 80
```

Full commands (LLM fine-tuning, individual diagnostics): `docs/architecture.md`,
`llm_finetuning/README.md`.

## Research Status

**Implemented:** STGT, bridge/reduction logic, coverage routing, RULES, LLM layer (hosted +
fine-tuned). **Experimental / in progress:** STGT accuracy is below this project's own
internal target — see Limitations. **Not implemented here:** RF fingerprinting,
multilateration/tracking, AR visualisation (external, see `docs/architecture.md`).

## Limitations

- Synthetic training data only — no real drone telemetry has been used
- Regression outputs (velocity, etc.) are in normalized space, not physical units
- Chain-length-2 pair accuracy (65.8%) has not yet reached this project's internal 70% target
- A known majority-vote reduction fix was built, tested, and deliberately **not** shipped after measurement showed it hurt more than it helped (`docs/experiments.md`)

## Citation

No publication exists yet. See `CITATION.cff` to cite this repository directly.
