"""
Semantic vocabularies, matching helpers, evaluation test cases, and the judge
prompt. Migrated from the eval cell so they live in one importable place.
"""
from __future__ import annotations

INTENT_FAMILIES = {
    # NOTE: "attack" (a former entry here) was removed — RULES (build_sft_dataset.py)
    # never emits it, so it was unreachable dead vocabulary. If a future RULES change
    # adds a genuine attack-in-progress intent distinct from attack_preparation, restore
    # a family for it here with the corresponding RULES value.
    "approach":     {"approach", "advancing", "closing", "moving toward", "converging"},
    "surveillance": {"surveillance", "observe", "monitor", "scout", "reconnaissance", "recon"},
    "patrol":       {"patrol", "loiter", "holding", "circling", "maintaining"},
    "reposition":   {"repositioning", "reposition", "maneuver", "relocating", "reorganising", "reorganizing"},
    "defensive":    {"defensive", "defend", "shield", "protect", "guarding"},
    "encircle":     {"encirclement", "encircle", "surround", "flanking", "flank"},
    # Families below cover the remaining RULES intents (build_sft_dataset.py),
    # so rule-trained answers are never scored as hallucinations.
    "withdraw":     {"withdraw", "withdrawing", "retreat", "retreating", "disengage"},
    "regroup":      {"regroup", "regrouping", "reform", "reforming"},
    "consolidate":  {"consolidate", "consolidating", "tightening"},
    "transit":      {"transit", "transiting", "en route", "traveling", "travelling"},
    "defensive_transition": {"defensive_transition", "defensive transition"},
    "area_search":  {"area_search", "area search", "search", "sweep", "sweeping"},
    "attack_preparation": {"attack_preparation", "attack preparation", "preparing to attack"},
    "rally":        {"rally", "rallying", "massing", "mustering"},
    # Legal per OUTPUT_SCHEMA but not a RULES output — a correct abstention under
    # degraded/ambiguous input. Without this family, is_hallucination() flagged every
    # correct "unknown" response as a hallucination.
    "unknown":      {"unknown", "unclear", "indeterminate", "insufficient data"},
}
THREAT_FAMILIES = {
    "low":      {"low", "minimal", "none"},
    "medium":   {"medium", "moderate", "elevated"},
    "high":     {"high", "severe", "serious"},
    "critical": {"critical", "extreme", "imminent"},
}
# Covers every RULES-emitted action plus "intercept", which is schema-legal
# (OUTPUT_SCHEMA in inference.py) but never appears in RULES.
ACTION_FAMILIES = {
    "monitor":               {"monitor", "monitoring", "observe", "watch", "no action needed",
                               "continue monitoring", "routine monitoring"},
    "increase_surveillance": {"increase_surveillance", "increase surveillance",
                               "heightened surveillance", "closer watch", "step up surveillance",
                               "enhanced monitoring"},
    "alert_operator":        {"alert_operator", "alert operator", "notify operator",
                               "alert the operator", "flag to operator", "escalate to operator"},
    "deploy_countermeasure": {"deploy_countermeasure", "deploy countermeasure", "countermeasure",
                               "deploy countermeasures", "engage countermeasures",
                               "activate countermeasure"},
    "intercept":             {"intercept", "intercepting", "interception", "engage", "neutralize",
                               "neutralise"},
}
ALL_VALID_INTENT_TOKENS = {t for fam in INTENT_FAMILIES.values() for t in fam}
ALL_VALID_THREAT_TOKENS = {t for fam in THREAT_FAMILIES.values() for t in fam}
ALL_VALID_ACTION_TOKENS = {t for fam in ACTION_FAMILIES.values() for t in fam}


def match_intent(predicted: str, expected_family: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in INTENT_FAMILIES.get(expected_family, set()))


def match_threat(predicted: str, expected_level: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in THREAT_FAMILIES.get(expected_level, set()))


def match_action(predicted: str, expected_family: str) -> bool:
    pred = predicted.lower().strip()
    return any(tok in pred or pred in tok for tok in ACTION_FAMILIES.get(expected_family, set()))


def is_hallucination(predicted_intent: str, predicted_threat: str) -> bool:
    pi, pt = predicted_intent.lower().strip(), predicted_threat.lower().strip()
    intent_ok = any(tok in pi or pi in tok for tok in ALL_VALID_INTENT_TOKENS)
    threat_ok = any(tok in pt or pt in tok for tok in ALL_VALID_THREAT_TOKENS)
    return not (intent_ok and threat_ok)


# Evaluation scenarios. expected_intent/expected_action use INTENT_FAMILIES/
# ACTION_FAMILIES keys. expected_action values are cross-referenced from RULES
# (build_sft_dataset.py) for each case's (formation_a, formation_b) pair — not
# independently chosen — so they agree with the canonical rule table by construction.
# NOTE: expand this list substantially (>=50, incl. ambiguous/adversarial cases)
# before reporting any headline LLM accuracy. Six cases is not enough.
TEST_CASES = [
    {"name": "Converging Attack", "formation_a": "dispersed", "formation_b": "converging",
     "expected_intent": "approach", "expected_threat": "high",
     "expected_action": "alert_operator", "strict": False},
    {"name": "Stable Patrol", "formation_a": "column", "formation_b": "column",
     "expected_intent": "patrol", "expected_threat": "low",
     "expected_action": "monitor", "strict": True},
    {"name": "Encirclement Behavior", "formation_a": "v_shape", "formation_b": "encirclement",
     "expected_intent": "encircle", "expected_threat": "high",
     "expected_action": "alert_operator", "strict": False},
    {"name": "Defensive Shield", "formation_a": "shield", "formation_b": "shield",
     "expected_intent": "defensive", "expected_threat": "medium",
     "expected_action": "monitor", "strict": True},
    {"name": "Area Search", "formation_a": "diamond", "formation_b": "dispersed",
     "expected_intent": "area_search", "expected_threat": "medium",
     "expected_action": "monitor", "strict": False},
    {"name": "Breaking Contact", "formation_a": "converging", "formation_b": "dispersed",
     "expected_intent": "withdraw", "expected_threat": "low",
     "expected_action": "monitor", "strict": False},
]


JUDGE_PROMPT = """\
You are an expert evaluator of UAV swarm tactical analysis systems.
Given the ML tactical context and the LLM assessment below, score each dimension 1-5.

Scoring rubric:
- factual_accuracy    : Does the assessment accurately reflect the ML context? No hallucinations?
- threat_calibration  : Is the threat level reasonable given velocity, formation, stability?
- action_relevance    : Is the recommended_action appropriate?
- reasoning_quality   : Is the reasoning specific and non-generic?
- internal_consistency: Are summary, intent, threat_level, and action mutually consistent?

Return ONLY valid JSON:
{{
  "factual_accuracy": <1-5>,
  "threat_calibration": <1-5>,
  "action_relevance": <1-5>,
  "reasoning_quality": <1-5>,
  "internal_consistency": <1-5>,
  "overall_score": <1.0-5.0>,
  "critique": "<one sentence>"
}}

--- ML TACTICAL CONTEXT ---
{tactical_context}

--- LLM ASSESSMENT ---
{llm_assessment}
"""
