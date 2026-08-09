# Methodology

## Independent ground truth, always

Every evaluation in this project derives ground truth from something the system under test
never produced. For the templated test batteries (`TEST_CASES`, degradation/holdout
batteries), ground truth is the `(formation_a, formation_b)` pair the case was *constructed*
from, looked up directly against RULES — never the system's own read of the case. For real
STGT-output evaluations (`llm_finetuning/measure_coverage.py`,
`scripts/phase0_ceiling.py` and siblings), ground truth is the synthetic data generator's own
known `true_chain` / per-timestep true label — never `stgt_bridge`'s or the LLM's own output
read back as if it were truth. This sounds obvious; it was violated once early in the
project's evaluation history (see below) and the fix is now a standing discipline.

## The LLM-grading-itself failure, and the fix

An early version of the evaluation harness used the system's own model as the judge of its
own output — it self-scored 5/5 while objective, ground-truth-checked accuracy was close to
0%. `src/swarm_intent/llm/evaluate.py::evaluate_llm` now requires `judge_client` to be an
**independent** client/model from the system under test, and its docstring records this
history explicitly so the mistake isn't quietly repeated. `evaluate_llm` reports objective
intent/threat accuracy as the headline metric, not the judge's score, which is reported only
as a secondary signal. `evaluate_ml_model` leads with macro-F1 specifically because the
`transitioning` class is imbalanced and a plain accuracy number would hide that.

## Dev/mining split discipline for any tuned threshold

Any threshold tuned against data (e.g. the "robust reduction" majority-vote confidence
threshold) is tuned on a dedicated dev/mining split, seeded disjointly from every eval-battery
seed, and used **once** — then retired. This was itself violated once (a dev split seeded `1`
was reused across two different tuning sessions weeks apart) and the fix is now a standing
rule, recorded in `docs/V5_LOG.md`'s step-0 entry: cut a fresh, disjoint, single-use seed for
each new tuning question. The reversal this caused is instructive and kept in the log rather
than smoothed over: the first (reused-split) sweep concluded a lower threshold dominated the
shipped default; a properly fresh split reversed that conclusion.

## Chain-length stratification

Pooling all evaluation trajectories into one accuracy number hid a large effect: chain-length-1
(steady state, no transition) and chain-length-2 (one real transition) trajectories have very
different, and very differently-moving, accuracy under the same fix. `docs/V5_LOG.md` step 22
documents the point at which pooled reporting was replaced with a stratified-by-chain-length
"Current state" banner at the top of `docs/CEILING.md`, with a standing project convention:
**correct via a superseding notice, never rewrite a historical dated entry in place.** Every
historical entry below that banner is left exactly as originally written, even when later
superseded — the record of what was believed and measured at each point in time is treated as
itself worth preserving.

## Statistical basis for causal claims

Where this project claims a fix caused an effect (not just "the number changed"), the claim is
backed by a significance test, not vibes — e.g. the chain-3+ false-positive-rate investigation
(`docs/V5_LOG.md` step 26 step 5) used a two-proportion z-test to establish that an apparent
regression was borderline-significant on its own (p=0.062) while the subsequent fix's
improvement was highly significant (p=2.4e-11) and represented a genuine net improvement over
the original baseline (p=4.3e-07), not merely "undoing" the intermediate regression.

## Batched LLM generation, validated equivalent

Large evaluation batteries use a batched generation path (`complete_batch`/`generate_batch`,
~2.78x measured speedup) rather than one request per case. Before being used for anything
load-bearing, batched output was checked for equivalence against the unbatched path on a
shared sample — this validation is what allows the large `n_runs=20` batteries in
`docs/development-history.md` to run in hours rather than days.

## What "robust=True" and its rejection demonstrate about this project's standards

`stgt_bridge.py`'s majority-vote "robust" reduction mode is a real, working, fully-tested
feature — and it is not the default, because measurement said not to ship it. It recovers
more `(from, to)` pairs from noisy window sequences, but a dedicated evaluation
(`docs/development-history.md` sec AG) found precision on the newly-recovered cases far below
what makes the recovery worthwhile, for a specific, diagnosed reason (most recoveries still
trip the dispersed/converging ambiguity guard on the way to a final answer, converting "an
LLM might get this right" into "guaranteed silence"). The code, its tests, and its negative
result are all kept in the repository rather than deleted, because the negative result is
itself a real finding: it directly motivated the V5 program's decision to look for the
problem's *root cause* (a generator physics bug) instead of trying to patch around it in the
reduction logic — which is exactly what the V5 program then found and fixed, with a much
larger effect (13.0% -> 52.3%/58.7% threat-ceiling improvement) than any reduction-logic tweak
achieved.
