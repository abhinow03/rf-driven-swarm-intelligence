"""
V5a2 preregistration step 2: non-proxy pair_accuracy.

Root cause of the schema gap: OUTPUT_SCHEMA (src/swarm_intent/inference.py) has no field
that states the literal (from, to) formation pair -- likely_intent is a many-to-one function
of the pair via RULES, so matching intent is a lossy proxy, not pair identification.

Fix used here: the free-text fields (situation_summary, key_indicators, threat_reasoning)
DO narrate formation names literally ("maintained a diamond formation... transitioning to a
shield formation"), because build_llm_prompt's key_windows already put formation names in
front of the model. This extracts the model's literally-stated (from, to) pair from
already-generated text -- zero new inference, zero schema change.

Two known text hazards, both handled explicitly (found during hand-validation against 30
known-pair cases, see llm_finetuning/validate_literal_pair_extraction.py):
1. Non-chronological prose: "maintained a shield formation throughout... transitions occurred
   ...from column to diamond and then to shield" states the dominant/current state BEFORE the
   history. Naive first-two-distinct-mention order would misread this as (shield, column).
   Fixed by giving explicit "from A to B (to C...)" / "A ... to B" transition-chain phrasing
   PRIORITY over raw mention order, using the chain's (first, last) as the pair.
2. Vocabulary collision: "converging"/"dispersed" name both a formation type AND a spread-
   dynamics trend descriptor elsewhere in this project's tactical-context template ("Spread
   dynamics: dispersing spread (mean approach_rate=...)"). A mention is only treated as a
   FORMATION claim if "formation"/"configuration"/"shape" appears within 3 words of it;
   unqualified mentions (e.g. "dispersing spread") are excluded, not counted as a formation.
"""
from __future__ import annotations

import re

from swarm_intent.config import BASE_FORMATIONS

_PATTERNS = {
    "v_shape": r"v[\s\-‑‐]?shape\w*",
    "encirclement": r"encircl\w*",
    "column": r"\bcolumn\w*",
    "diamond": r"\bdiamond\w*",
    "dispersed": r"disper\w*",
    "converging": r"converg\w*",
    "shield": r"\bshield\w*",
}
assert set(_PATTERNS.keys()) == set(BASE_FORMATIONS)

_FORMATION_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS.items()), re.IGNORECASE)

# "formation"/"configuration"/"shape" (v_shape's own pattern already contains "shape", so a
# bare v_shape match always self-qualifies) within this many characters counts as qualifying.
_QUALIFIER_WINDOW = 25
_QUALIFIER_RE = re.compile(r"formation|configuration", re.IGNORECASE)

# Connective words linking consecutive formation mentions into one explicit transition chain.
_CONNECTIVE_RE = re.compile(
    r"\b(to|then|transition(?:ed|ing)?|shift(?:ed|ing)?|becam)\b", re.IGNORECASE)
_CONNECTIVE_GAP_MAX_CHARS = 70  # only bridge mentions separated by a short connective clause
_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s+[A-Z]")  # never bridge across a sentence boundary --
# "maintained a diamond formation throughout. Transition from column to diamond" must NOT
# read the first "diamond" as part of the second sentence's chain, even though a connective
# word appears in between (found via a synthetic test shorter than the real corpus text this
# was tuned against, where the intervening clause happened to still be < 70 chars).


def _raw_mentions(text: str) -> list[tuple[str, int, int]]:
    """(name, start, end) for every formation-name match, unfiltered."""
    return [(m.lastgroup, m.start(), m.end()) for m in _FORMATION_RE.finditer(text)]


def _qualified_mentions(text: str) -> list[tuple[str, int, int]]:
    """(name, start, end) for every match that is formation-qualified or self-qualifying
    (v_shape's pattern already includes 'shape'). Excludes matches like 'dispersing spread'/
    'converging spread' -- this project's tactical-context vocabulary reuses these same words
    as a spread-dynamics-trend descriptor, unrelated to formation identity."""
    out = []
    for name, s, e in _raw_mentions(text):
        if name == "v_shape":
            out.append((name, s, e))
            continue
        window = text[max(0, s - _QUALIFIER_WINDOW):e + _QUALIFIER_WINDOW]
        if _QUALIFIER_RE.search(window):
            out.append((name, s, e))
    return out


def _dedup_consecutive(mentions: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    seq = []
    for name, s, e in mentions:
        if not seq or seq[-1][0] != name:
            seq.append((name, s, e))
    return seq


def _longest_connective_chain(text: str, mentions: list[tuple[str, int, int]]) -> list[str]:
    """Longest run of mentions each linked to the next by a short connective gap
    ('X ... to Y ... then to Z'). Returns just the names, or [] if no chain of length >= 2."""
    best: list[str] = []
    current: list[tuple[str, int, int]] = []
    for name, s, e in mentions:
        if current:
            gap = text[current[-1][2]:s]
            if (len(gap) <= _CONNECTIVE_GAP_MAX_CHARS and _CONNECTIVE_RE.search(gap)
                    and not _SENTENCE_BREAK_RE.search(gap)):
                current.append((name, s, e))
                continue
        if len(current) > len(best):
            best = current
        current = [(name, s, e)]
    if len(current) > len(best):
        best = current
    return [n for n, _, _ in best] if len(best) >= 2 else []


def extract_literal_pair(parsed: dict) -> tuple[str, str] | None:
    """The model's literally-stated (from, to) formation pair, or None if no formation
    mention exists anywhere in its free text (extraction failure -- scored separately from
    a wrong answer).

    Priority order:
    1. An explicit connective-linked transition chain among RAW mentions ("Transition from
       column to diamond" doesn't repeat the word "formation" next to "column", so the
       connective-chain search runs on unfiltered mentions, not formation-qualified ones --
       the connective word itself ("from"/"to"/"transition") is the evidence of formation
       identity here).
    2. First two formation-QUALIFIED mentions in order (filters out the spread-dynamics
       vocabulary collision for plain, non-chained narratives).
    3. A single qualified mention, read as a steady state (from, from).
    """
    summary = str(parsed.get("situation_summary", "") or "")
    indicators = " ".join(str(i) for i in (parsed.get("key_indicators") or []))
    reasoning = str(parsed.get("threat_reasoning", "") or "")

    for text in (summary, f"{summary} {indicators} {reasoning}"):
        raw = _dedup_consecutive(_raw_mentions(text))
        chain = _longest_connective_chain(text, raw)
        if len(chain) >= 2:
            return (chain[0], chain[-1])

        qualified = _dedup_consecutive(_qualified_mentions(text))
        if len(qualified) >= 2:
            return (qualified[0][0], qualified[1][0])
        if len(qualified) == 1 and text is not summary:
            # only accept a single-mention steady-state read once the full fallback text has
            # been tried -- summary-only single mentions get one more chance below first
            return (qualified[0][0], qualified[0][0])

    summary_qualified = _dedup_consecutive(_qualified_mentions(summary))
    if len(summary_qualified) == 1:
        return (summary_qualified[0][0], summary_qualified[0][0])
    return None


def true_pair_from_chain(true_chain: list[str]) -> tuple[str, str]:
    """has_ground_truth=True implies len(true_chain) in {1, 2}."""
    assert len(true_chain) in (1, 2), f"expected a has_ground_truth=True chain, got {true_chain}"
    if len(true_chain) == 1:
        return (true_chain[0], true_chain[0])
    return (true_chain[0], true_chain[1])
