# Defense panel document

Every claim below cites the `AUDIT.md` section it was measured in. No claim appears without a
number next to it.

## The two hard questions, answered up front

**Q1. Why do you need an LLM at all — isn't this just a lookup table?**

Because the lookup table can only be trusted on 1.8% of real observations. `AUDIT.md` sec AE
step 2 ran the real trained STGT model over 500 realistic multi-window trajectories and found a
plain `RULES[(a, b)]` dict can confidently and correctly answer only **9/500 (1.8%, 95% CI
[0.9%, 3.4%])** on its own. **60.0% (300/500) has no valid 2-tuple key at all** — multi-hop
chains, terminal ambiguity, oscillation — no dictionary of any size could answer those, by
construction, not by omission. The remaining 38.2% is dict-shaped but requires a hedge a static
lookup has no mechanism to express. `pipeline_v2`'s architecture (sec AE step 3) is built
directly around this split: the dict handles the 1.8% with a verified, zero-exception guarantee
(sec AH step 1, below), and the LLM's judgment is load-bearing on the 60% majority — not
decorative, not a fallback for edge cases.

**Q2. Is the system actually accurate, or are you grading yourself against your own dictionary?**

The first version of this evaluation was a tautology and it was caught and fixed before being
presented anywhere: `pipeline_v2` scored 100% on the project's 55-case clean battery (sec AE
step 4), but sec AF step 1 found that battery is ~100% Layer-1-resolvable by construction — a
dictionary scored against dictionary-derived cases. Replaced with evaluation on real, unforced
STGT model output (sec AF step 2, sec AG step 3, ground truth taken only from the generator's
own known formation chain, never from the model's own noisy classification of itself). The
honest numbers: no system in this project exceeds **48.7%** raw threat accuracy on real,
unconstrained model output (`v2`, sec AH step 2) — `pipeline_v2` trails at **6.7%** unconditional
/ **22.3%** conditional on answering (sec AH step 2), because it is bottlenecked by an upstream,
disclosed data-generation defect (sec AG, finding 6 below), not by a flaw in its own decision
logic (verified separately, finding 8 below). And the dominant real safety failure mode across
this entire project is **under-escalation** — real high/critical threats silently absorbed into
`medium` 11.4-54.3% of the time (sec AD verdict) — not false alarms, which occur only 0-4.3% of
the time.

## The eight validated findings

**1. The low-threat collapse is systematic over-escalation to `medium`, not scattered noise.**
`AUDIT.md` sec M. Measurement: `v3a`/`v3a-nomask` predict `medium` on 15/15 genuinely low-threat
cases; `v2` is unaffected (13/15 correct). *Limitation:* built from majority-vote-per-case labels
across 5 runs, not per-run data (`evaluate_llm` never persisted individual run predictions).

**2. The `medium`-attractor bias is pretraining-inherited, not induced by this project's
fine-tuning.** Sec AA, "decisive test." Measurement: the base model (Qwen2.5-7B-Instruct, zero
exposure to any training row) already predicts `medium` on **73.3%** of low-threat cases; giving
it the same rule table in-context with **zero training** resolves this to **93.3%**.
*Limitation:* fine-tuning does not reliably close this gap on its own — raw greedy accuracy for
`v3a`/`v3b`/`v3b-fix` (0%/20%/40%) never clearly exceeds the untrained base model's own 26.7%.

**3. In-context RULES beats fine-tuning specifically on `threat_level`, not on `likely_intent`.**
Sec AB. Measurement: `rules_in_prompt` greedy low-threat accuracy is 93.3% vs the best fine-tuned
adapter's 40% (`v3b-fix`); `likely_intent` shows no such gap (fine-tuned adapters: 70-100% vs
base's 25.8%). *Limitation:* `rules_in_prompt` has **0%** structural abstention capability — it
is "structurally blind to its own limits," which is exactly why the composite router (sec AB
step 3/4) exists rather than shipping `rules_in_prompt` alone.

**4. The real safety story is under-escalation, not over-escalation.** Sec AD verdict.
Measurement: high/critical cases are silently absorbed into `medium` **11.4-54.3%** of the time
across the four high/critical test cells; the false-alarm direction (routine activity flagged as
crisis) occurs only **0-4.3%** of the time. *Limitation:* the `critical` stratum is **n=2**
throughout every battery-based eval in this project (secs M, AA, AD) — every critical-stratum
number, in either direction, is not statistically meaningful on its own; it is consistently
reproduced, never a large-n result.

**5. Naive prior correction is not a symmetric fix — it helps low-threat and actively hurts
high/critical.** Sec AD step 4. Measurement: the same log-frequency correction technique that
recovered low-threat accuracy for the fine-tuned adapters, applied unscoped to `high`, moves raw
35.7% accuracy to **14.3% (worse)** — the correction's magnitude scales with how rare the target
class is, and `critical` (4.1% of RULES) is far rarer than `low` (26.5%) ever was.
*Limitation:* this is why sec AE step 1 had to explicitly scope the correction to low/medium
only, as a safety fix, before any further work — an unscoped correction would have made the
system less safe, not more.

**6. A plain RULES dict resolves only 1.8% of real STGT output — the rest genuinely needs a
hedge or LLM judgment.** Sec AE step 2 (see Q1 above for the full numbers). *Limitation:* the
number this superseded (pipeline_v2 scoring 100% on the clean battery) turned out to be a
construction artifact, not a real result (sec AF step 1) — this measurement had to be built on
*real* model output specifically because the templated battery could not be trusted to expose
that gap.

**7. The reduction-logic brittleness sec AF exposed is a real, fixable bug — and fixing it does
not solve the actual bottleneck.** Sec AG. Measurement: a robust majority-vote reduction shrinks
the unresolvable bucket from 60.0% to 39.0% of real observations, but Layer-1 (dict-answerable)
firing barely moves (1.8% → 2.4%) because **92/96 (95.8%)** of newly-recovered cases are caught
by the separately-real dispersed/converging geometry ambiguity guard; the fix's own
recovered-pair precision never exceeds **49%** at any threshold, on the dev split or the
held-out set (sec AG step 4 — confirmed genuinely low-precision, not overfit, since dev and
held-out numbers track each other within 2.3 points). *Limitation:* correctly **not shipped** as
the default — 2 of 3 stated success criteria failed at full scale (sec AG step 3: Layer-1 firing
4.0% vs a 40% target; over-abstention 90.9% vs a 25% ceiling).

**8. `pipeline_v2`'s core guarantee holds with zero exceptions — but that is a separate claim
from "the system is accurate."** Sec AH steps 1-2. Measurement: 0/60 sampled Layer-1 units show
any deviation from `RULES[(a, b)]`, and 0/60 log the LLM narrator attempting to alter a decision
field — the architecture is sound. But conditional accuracy (excluding abstentions from the
denominator) is still only **22.3%/24.2%** for `pipeline_v2`/`pipeline_v2-robust`, against
**48.7%/33.5%/35.7%** for `v2`/`v3b-fix`/`rules_in_prompt`. *Limitation:* the guarantee is that
the LLM cannot corrupt a decision it's handed — it says nothing about whether the handed decision
(the recovered `(a, b)` key) is usually correct. Key-selection accuracy is the real bottleneck,
and it traces to finding 6/7's upstream mechanism, not to this pipeline's own code.

## The architecture

`pipeline_v2` (`src/swarm_intent/pipeline_v2.py`) routes every observation through
`coverage.classify_observation` (sec AE step 2) before any LLM touches a decision:

- **Layer 1 (bucket A, dict-resolvable).** `threat_level`/`likely_intent`/`recommended_action`
  come from `RULES[(a, b)]` directly. An LLM is called only to write narrative text around a
  decision it is told is already final; every field it returns is validated against the decision
  and force-overwritten if it disagrees, and the disagreement is logged, never silently allowed
  through (`_finalize_layer1`, `pipeline_v2.py:138-161`). **The guarantee, stated precisely:
  on accepted input (bucket A), the returned decision fields are RULES[(a, b)] by construction —
  verified with zero exceptions across 60 sampled units, temperature 0.3, real narrator client
  (sec AH step 1).**
- **Layer 2 (bucket B, guardable).** No model call. A machine-generated abstention built from
  `coverage.py`'s `guard_reasons` (`oov_name`, `dominant_history_contradiction`,
  `dispersed_converging_ambiguity`, `low_confidence`) — **explicit abstention, not a guess.**
- **Layer 3 (bucket C, unresolvable, 60% of real observations).** Routes to `v3b-fix` with
  scoped prior correction (sec AE step 1). This is where the LLM's actual judgment is
  load-bearing — the majority case, not the exception.

**Stated as one sentence for the panel: correct by construction on accepted input, explicit
abstention otherwise — verified, not assumed, and the two failure modes (a wrong key reaching
Layer 1, or Layer 3's own judgment being wrong) are measured separately and are not confused with
each other in any number in this document.**

## What is NOT fixed

- **Under-escalation on genuinely unresolvable input is not separately re-verified.** Finding 4's
  11.4-54.3% under-escalation rate was measured on battery-tested (resolvable) input. The 251/500
  real sequences with no independently-derivable ground truth (true 3+-hop chains, sec AG step 1)
  are routed to Layer 3 and scored for abstention correctness, but their escalation-DIRECTION
  correctness cannot currently be measured at all — there is no ground truth to check it against.
  This is a disclosed, structural evaluation gap, not a fixed-and-verified property.
- **`critical` is n=2 everywhere in this project.** Every number involving the `critical` stratum,
  in secs M/AA/AD/AB, comes from the same 2 test cases. Consistently reproduced, never
  statistically meaningful on its own.
- **The upstream geometry defect (`docs/UPSTREAM_ISSUES.md` #1) is unfixed.** It is the largest
  single blocker measured in this project — 46.2% of reduction failures (sec AG step 1) — and it
  caps any reduction-logic fix at ~49% precision regardless of threshold (sec AG step 2). Sec AH
  step 3 projects the payoff of fixing it (Layer-1 firing 1.8% → ~12.0%, conservative, single
  stated assumption) but the fix itself has not been made.
- **Train/serve gap on the recalibrated generator.** The scoped prior corrections (sec AD step 4,
  sec AE step 1) and `pipeline_v2`'s Layer-3 class-frequency correction
  (`pipeline_v2.default_class_freq()`, reading `v3b-fix`'s own training file) are both keyed to
  the CURRENT generator's class distribution. If `formations.py` is fixed per the upstream
  request and STGT is retrained, these frequency tables go stale until explicitly recomputed and
  re-tuned — nothing in the pipeline currently re-derives them automatically, and shipping a
  retrained model without redoing this step would silently reintroduce a version of finding 5's
  problem.

## Which model is demoed, and why it differs from the research system

**`pipeline_v2` is not the demo system.** Its contribution is architectural: sec AE step 2 is the
evidence the LLM's role is structurally necessary (1.8% dict-only coverage), and sec AH step 1 is
the evidence its guarantee holds. But its real-world raw accuracy currently trails simpler
systems (6.7% unconditional / 22.3% conditional, sec AH step 2) because it is bottlenecked by the
upstream defect above — not by a flaw in `pipeline_v2`'s own logic, which was independently
verified correct in finding 8.

**The demo runs the composite router** (sec AB steps 3-4). It is the only system in this project
shown to have both properties a live demo needs at once: **100%** abstention on structurally
unanswerable input, AND the best available accuracy when it does answer (**82.3%** low-threat,
**60.6%** overall threat accuracy on the clean battery, sec AB step 4's stratum table). It is not
the more novel architecture — it does not carry `pipeline_v2`'s verified correct-by-construction
guarantee — but it is the system least likely to embarrass itself live, and it is honest about
its own limits rather than answering confidently past them.

`v2` (the bare fine-tuned adapter, no abstention) is deliberately **not** the demo choice despite
having the single highest raw accuracy number measured anywhere in this project — 94.5%
threat / 100% intent on the clean battery (sec AB step 4), and 48.7% on real STGT output, the
best of all five systems measured on real output (sec AH step 2). It structurally never abstains
(0.0% over-abstention everywhere it has been measured) — a real liability specifically because
60% of real inputs (sec AE step 2) have no valid answer at all, and `v2` will confidently produce
one anyway.
