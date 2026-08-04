"""
Routes each case to `rules_in_prompt` (base Qwen2.5-7B-Instruct, no adapter,
RULES.txt pasted as system prompt -- baselines.py's make_rules_in_prompt_run_case
protocol) when a (from, to) formation pair is extractable from its tactical
context via the SAME extraction baselines.py's rules_lookup uses
(`_extract_pair` -- the "Transition detected..." / "No formation transitions
detected." regex), and to a fine-tuned adapter client (v3b-fix) otherwise.

Motivation (AUDIT.md sec AB): step 2 found rules_in_prompt hits 93.3% on
low-threat cases -- beating every fine-tuned adapter -- by using the exact
RULES table in-context, with zero training. But rules_in_prompt NEVER abstains
on unanswerable input (0.0% abstention_rate_when_unanswerable on multi_hop/
terminal_transitioning, evaluation/degradation_rules_in_prompt.json) --
because there is no rule-table entry that says "decline," a base model given
the rules and told to answer just... answers. The fine-tuned adapters
(v3a/v3b/v3b-fix) learned to abstain but pay for it with worse answerable-case
accuracy. The composite's bet: extractability is a cheap, deterministic proxy
for "does a rule apply" -- when it's False (multi_hop, terminal_transitioning),
the case is structurally the kind rules_in_prompt can't be trusted on anyway,
so route to the system that was actually trained to recognize that and
decline. When it's True, route to the system empirically best on answerable
input with an in-context rule table.

Two known edge cases, deliberately NOT special-cased, both from
holdout_shapes.py:

  - `oov_formation` is shaped exactly like an ordinary resolvable transition
    ("A -> phalanx"), so `_extract_pair` returns a pair and this routes to
    rules_in_prompt even though "phalanx" isn't a real formation.
  - `dominant_mismatch` always includes an explicit "Transition detected at
    t=20.0s: A -> B" line (the self-contradiction lives only in the separate
    "Dominant formation" line, which `_extract_pair` never reads), so it ALSO
    routes to rules_in_prompt despite being designed to be unanswerable.

Both are extractability as literally specified (a syntactic check on the
transition line), not semantic or cross-line validity. Whether rules_in_prompt
catches either itself (declines because the rules text has no match / notices
the inconsistency) or hallucinates a confident answer is exactly the kind of
result this composite is built to surface, not hide.

Mirrors the two existing run_case factory conventions in this codebase:

  make_composite_run_case         -- for TEST_CASES (the clean 55-case
                                      battery), which has no precomputed ctx;
                                      builds it via synth_context(), same
                                      shared-rng protocol as
                                      baselines.make_rules_in_prompt_run_case.
  make_composite_battery_run_case -- for degradation.py / holdout_shapes.py
                                      battery cases, which already carry a
                                      precomputed case["ctx"]/case["key_windows"],
                                      same convention as
                                      degradation.make_llm_battery_run_case.

Both accept a `branch_log: dict` (case name -> "rules_in_prompt" | "finetuned")
that they mutate on every call, so callers can report branch-firing rate
after an eval run without re-deriving it.
"""
from __future__ import annotations

import os
import sys
from random import Random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from swarm_intent.inference import build_llm_prompt  # noqa: E402

from baselines import _extract_pair  # noqa: E402
from build_sft_dataset import synth_context  # noqa: E402

RULES_BRANCH = "rules_in_prompt"
FINETUNED_BRANCH = "finetuned"


def _route(ctx: str) -> str:
    return RULES_BRANCH if _extract_pair(ctx) is not None else FINETUNED_BRANCH


def _preds_from_key_windows(key_windows):
    return [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
            "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
            "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
            "role_differentiation": False, "transition_from": kw["from"],
            "transition_to": kw["to"]} for kw in key_windows]


def make_composite_run_case(rules_client, finetuned_client, branch_log: dict, seed: int = 0):
    """For TEST_CASES -- ctx built via synth_context(), seed=0 matches every
    other headline-eval run_case factory in this project (baselines.py,
    logit_inspection.py)."""
    rng = Random(seed)

    def run_case(case):
        ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
        prompt = build_llm_prompt(_preds_from_key_windows(key_windows), ctx, {})
        branch = _route(ctx)
        branch_log[case["name"]] = branch
        client = rules_client if branch == RULES_BRANCH else finetuned_client
        return client.complete(prompt), ctx

    return run_case


def make_composite_battery_run_case(rules_client, finetuned_client, branch_log: dict):
    """For degradation.py / holdout_shapes.py battery cases -- ctx/key_windows
    already precomputed on the case dict, same convention as
    degradation.make_llm_battery_run_case."""
    def run_case(case):
        prompt = build_llm_prompt(_preds_from_key_windows(case["key_windows"]), case["ctx"], {})
        branch = _route(case["ctx"])
        branch_log[case["name"]] = branch
        client = rules_client if branch == RULES_BRANCH else finetuned_client
        return client.complete(prompt), case["ctx"]

    return run_case
