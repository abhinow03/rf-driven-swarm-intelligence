# RULES critical-tier extension — proposal for Dr. Patil

**Status: NOT blocking.** Phase 1 corpus generation proceeded under the fallback
stratification described below without waiting on this. This document exists so the
open question doesn't get lost, not to request an urgent decision.

## Current state

`llm_finetuning/build_sft_dataset.py`'s `RULES` dict has 49 total `(from, to) -> (threat_level,
likely_intent, recommended_action)` entries. Exactly **2 are `critical`**:

| pair | threat_level | likely_intent | recommended_action |
|---|---|---|---|
| `(encirclement, converging)` | critical | encircle | deploy_countermeasure |
| `(converging, encirclement)` | critical | encircle | deploy_countermeasure |

These are both directions of **the same compound-escalation event** — encircling *and*
closing distance simultaneously — not two independent tactical signatures. This is well
below the ~5-pair threshold the project uses as a rule of thumb for "enough distinct
examples to stratify a training corpus over," and it triggered a halt gate at the start of
Phase 1.

## Why the fallback was chosen over extending RULES

Two candidates with genuine tactical justification were identified — both are the
steady-state "ingredients" of the existing critical pair, currently rated `high`:

- **`(converging, converging)`** — sustained closing/converging posture, no transition.
- **`(encirclement, encirclement)`** — sustained encircling posture, no transition.

Reasoning for promotion: a *persistent* converging or encircling posture arguably
represents settled hostile intent rather than a transient one-time event, and could be
judged comparably severe to a single transition into the compound state. This was
**deliberately not applied** — inventing a 3rd/4th/5th pair just to clear a threshold would
be arbitrary, and RULES' critical tier reads as a narrow, deliberate design (one matched
bidirectional pair, one specific compound signature) rather than an oversight that needs
padding.

Instead, the corpus used a **fallback stratification**, capping critical-stratum repetition
at 600 rows/pair (2 pairs × 600 = 1,200 total, ~150 rows per narrative-combination cell at 8
combinations/pair) instead of forcing a uniform 3,000/stratum target that would have meant
1,500 rows per critical pair — far more repetition of the same two scenarios than any other
tier gets of any single pair. The 1,800-row shortfall this created was redistributed
proportionally across the other three tiers (+600 each):

| stratum | uniform target (rejected) | fallback target (used) |
|---|---|---|
| low | 3,000 | 3,600 |
| medium | 3,000 | 3,600 |
| high | 3,000 | 3,600 |
| critical | 3,000 | **1,200** |
| **total** | 12,000 | 12,000 |

RULES itself is untouched under this fallback — no labels changed, no pairs added.

## Open question for Dr. Patil

If `(converging, converging)` and/or `(encirclement, encirclement)` are promoted from
`high` to `critical`, **what should `recommended_action` become?**

- Keep `alert_operator` (only the threat *label* escalates, not the response)?
- Escalate to `deploy_countermeasure`, matching the existing critical pair's action — i.e.
  treat a sustained posture as equally actionable as an active compound maneuver?
- Something narrower than either (e.g. a heightened-surveillance action that doesn't exist
  in the current action vocabulary yet)?

This is a real system-behavior change, not a label edit — `recommended_action` is what
downstream operators/automation actually act on. This project has no domain basis to answer
it and did not attempt to guess; it's raised here as a question, not a proposed answer.

## Not blocking

Phase 1's corpus generation completed under the fallback stratification above regardless of
how this question is resolved. If RULES is extended later, existing rows are unaffected (no
row was labeled under a promoted pair); only future corpus-generation runs would pick up the
change.
