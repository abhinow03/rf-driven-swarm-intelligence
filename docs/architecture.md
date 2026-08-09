# Architecture

Status labels used throughout: **implemented** (code exists, tested, in this repo),
**experimental** (implemented but under active revision, e.g. the V5 retraining program),
**planned** (designed for, not built), **external** (real, but lives in a teammate's
separate repository, not integrated here).

## Full stated project scope vs. what this repository contains

```
RF Environment
      |
      v
RF Sensing / Signal Processing .......................... external (not in this repo)
      |
      v
RF Fingerprinting ......................................... external (not in this repo)
      |
      v
Anonymous Emitter IDs
      |
      v
Localization ............................................... external (not in this repo)
      |
      v
Trajectory / Multi-target Tracking ......................... external (not in this repo)
      |
      v
STGT Formation Recognition ................................. IMPLEMENTED
      |
      v
Temporal / Transition Interpretation ....................... IMPLEMENTED (stgt_bridge.py)
      |
      v
Coverage routing (bucket A/B/C) ............................ IMPLEMENTED (coverage.py)
      |
      v
Deterministic RULES layer .................................. IMPLEMENTED
      |
      v
LLM Semantic Reasoning ...................................... IMPLEMENTED
      |
      v
Final Tactical Assessment ................................... IMPLEMENTED
      |
      v
Visualization / AR Interface ................................ PLANNED (not started)
```

Per `MIGRATION_GUIDE.md`, the external stages are meant to eventually land under
`src/swarm_intent/rf/` and `src/swarm_intent/tracking/`, contributed by a teammate's
separate repository. STGT's input contract (drone x/y/z positions over time) is exactly
what a working tracker would emit, so the integration point is well-defined even though the
code isn't here.

## STGT: the formation classifier

Input: a fixed-shape `(50, 6, 3)` sequence — 50 timesteps, 6 drones, (x, y, z). This shape is
hardcoded throughout `src/swarm_intent/model.py`, `graph.py`, and `config.py`; changing drone
count or window length requires touching multiple files and retraining, it is not a runtime
parameter.

```
(50, 6, 3) sequence
  -> sequence_to_graphs (graph.py): one proximity graph per timestep.
     nodes = drone position; edges connect drones closer than cfg.edge_threshold
     (in NORMALISED units); edge_attr = (dx, dy, dz, dist); self-loops if isolated.
  -> SpatialGAT: 2x GATv2Conv + global_mean_pool -> (50, d_model) per-timestep embedding
  -> TemporalTransformer: positional encoding + 4x encoder layers + mean-pool -> (d_model,)
  -> shared head -> two heads:
       cls_head: formation classifier (7 base formations, +1 "transitioning" class if
                 the model is trained with --transitions)
       reg_head: 3 regression values [centroid_velocity, approach_rate, stability]
```

`Config` (`config.py`) is the single source of truth for hyperparameters and the formation
vocabulary; `BASE_FORMATIONS`' order *is* the integer label mapping (index = label). The
model infers its compute device from its own parameters — there is no module-level `device`
global, by design (see Conventions below).

**Known caveat, disclosed in `CODE_REVIEW.md`:** regression labels are computed on
*normalised* positions, so `centroid_velocity` is in normalised-space units, not physical
m/s. Do not rescale-free-present this as a real-world velocity.

## stgt_bridge.py: reducing noisy window predictions

STGT runs over a long trajectory via a sliding window (`sliding_window_inference`, default
`window_size=50, stride=10`), producing one classification per window. `stgt_bridge.py`
reduces that sequence of window predictions into either:

- a single, temporally-ordered `(from, to)` formation pair (or `(from, from)` for a steady
  state), or
- an explicit failure to resolve, with a documented reason.

Two reduction modes exist: the **default** (`robust=False`), which requires the
first/last-observed distinct formations to be unambiguous by simple ordering, and an
**experimental "robust" mode** (`robust=True`, majority-vote based, leading/trailing-run
stripping, class-probability aggregation for all-transitioning sequences) that exists,
is fully tested (`tests/test_robust_reduction.py`), but is **not the default** — a dedicated
evaluation (`docs/development-history.md` sec AG) found it recovers more `(from,to)` pairs at
the cost of recovering mostly *wrong* ones, because most recovered cases still trip the
(separately real, separately diagnosed) dispersed/converging ambiguity guard on the way to a
final answer.

## coverage.py: bucket routing

`classify_observation()` / `classify_ctx()` sort every observation into exactly one of:

- **Bucket A (resolvable):** `stgt_bridge` reduces cleanly to a `(from, to)` pair with no
  guard concerns. Routed straight to the deterministic RULES lookup.
- **Bucket B (guardable):** a pair is derivable, but a specific, named guard condition fires
  (a genuinely out-of-vocabulary formation name, a dominant-formation contradiction, a
  dispersed/converging near-tie, or uniformly low confidence). Routed to a machine-generated
  abstention — no model call, deliberately.
- **Bucket C (unresolvable):** no 2-tuple key is derivable at all (multi-hop chains,
  oscillation, an all-unknown/all-transitioning read). Routed to the LLM layer, since this is
  exactly the case class no dict can ever answer regardless of confidence tuning.

Every one of these guard conditions had at least one real bug found in it during the V5
program's guard audit (`docs/development-history.md`) — each is now root-caused, fixed, and
covered by a regression test explaining what the bug was and why the fix is provably correct
(or, for `dominant_history_contradiction`, provably unreachable post-fix, documented as such
rather than silently deleted).

## pipeline_v2.py: the three-layer system

```
Layer 1 (bucket A) -- deterministic RULES[(a,b)] lookup. LLM (if invoked at all) plays a
                       narrator role only: its decision fields are validated against RULES
                       and OVERWRITTEN on any deviation, with the deviation logged. Verified
                       (docs/development-history.md sec AH step 1): 0/60 sampled cases had any
                       decision-field deviation across a dedicated verification run.
Layer 2 (bucket B) -- machine-generated abstention. No model call.
Layer 3 (bucket C) -- LLM layer (Groq baseline or the fine-tuned local model), with a scoped
                       log-p(c) prior correction (src/swarm_intent/llm/prior_correction.py)
                       applied ONLY to medium-argmax/low-runner-up near-ties -- provably never
                       touches a high/critical-threat call.
```

## LLM layer

`src/swarm_intent/llm/client.py` is provider-agnostic: `GroqClient` (hosted baseline,
default `llama-3.3-70b-versatile`) and `LocalHFClient` (a locally-hosted, optionally
QLoRA-fine-tuned HF model) both implement `generate(prompt) -> str`; the shared `complete()`
adds retry/backoff and tolerant JSON extraction against `OUTPUT_SCHEMA`
(`src/swarm_intent/inference.py`, single source of truth for the LLM's required JSON shape).
Swapping the client swaps nothing else in the pipeline.

Five QLoRA adapter iterations exist (`docs/ADAPTER_VERSIONS.md`), fine-tuning
`Qwen/Qwen2.5-7B-Instruct`; weights are not committed (`.gitignore`), regenerate via
`llm_finetuning/train_qlora.py`.

## Conventions worth knowing before reading the code

- **No hidden global state.** `cfg`, `device`, graph thresholds, and normalization stats are
  passed explicitly everywhere — a deliberate migration decision documented in
  `MIGRATION_GUIDE.md`, made after the original notebook code relied on module-level globals.
- **A single seeded RNG is threaded through all of `data.py`'s sampling.** Never call
  `np.random.*` or a fresh `default_rng()` without the threaded `rng` — this is what makes
  every experiment in `docs/development-history.md` reproducible from a stated seed.
- **Secrets come from the environment only** (`GROQ_API_KEY`). There is no hardcoded fallback
  key anywhere in this codebase (verified as part of this repository's own cleanup audit).
