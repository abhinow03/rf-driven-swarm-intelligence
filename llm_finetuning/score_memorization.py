"""
Implements the metric committed in docs/PREREGISTRATION.md's "Memorization-vs-
generalization check": verbatim/near-verbatim output-vs-training-target
overlap rate.

For every answered eval case, compute the TF-IDF cosine similarity (word 1-2
grams -- identical method to llm_finetuning/report_distinctness_similarity.py)
between the model's generated situation_summary and every TRAINING-split row's
situation_summary sharing the same (form_a, form_b) RULES pair. The overlap
rate is the fraction of cases whose max similarity to some same-pair training
target is >= NEAR_DUP_THRESHOLD (0.90, same threshold used throughout this
project's distinctness work).

Preregistered interpretation (see docs/PREREGISTRATION.md):
  - null-hypothesis baseline (measured on the real corpus): 0.6%
  - not meaningfully above ~0.6-2%: supports generalization
  - >= 15%: positive memorization signal, reported regardless of accuracy

Usage (once real v5-a outputs exist):
    from score_memorization import build_training_index, overlap_rate
    index = build_training_index("data/sft_train_v5_phase1.jsonl")
    rate, details = overlap_rate(eval_outputs, index)
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent.parent
NEAR_DUP_THRESHOLD = 0.90
MEMORIZATION_SIGNAL_BAR = 0.15  # preregistered in docs/PREREGISTRATION.md
NULL_HYPOTHESIS_BASELINE = 0.006  # measured on the real corpus, same doc


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_training_index(train_path: str) -> dict[tuple, list[str]]:
    """(form_a, form_b) -> list of training situation_summary strings for that
    pair. Requires swarm_intent.coverage on the path (same extraction regex
    used throughout Phase 1's own audit tooling)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from swarm_intent.coverage import _extract_pair_from_ctx

    index: dict[tuple, list[str]] = {}
    for r in load_jsonl(train_path):
        ctx = r["messages"][0]["content"]
        pair = _extract_pair_from_ctx(ctx)
        if pair is None:
            continue
        summary = json.loads(r["messages"][1]["content"])["situation_summary"]
        index.setdefault(pair, []).append(summary)
    return index


def max_similarity_to_pair(summary: str, pair: tuple, index: dict[tuple, list[str]]) -> float:
    """Max TF-IDF cosine similarity of `summary` against every training
    summary sharing `pair`. Returns 0.0 if the pair has no training rows at
    all (should not happen for a RULES pair, but handled defensively)."""
    targets = index.get(pair)
    if not targets:
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([summary] + targets)
    sims = (X[0:1] @ X[1:].T).toarray()[0]
    return float(sims.max()) if len(sims) else 0.0


def overlap_rate(eval_cases: list[dict], index: dict[tuple, list[str]],
                 threshold: float = NEAR_DUP_THRESHOLD) -> tuple[float, list[dict]]:
    """eval_cases: list of {"pair": (form_a, form_b), "situation_summary": str}
    for every ANSWERED case (caller filters out abstentions/unparseable
    outputs before calling this -- this function does not know about
    abstention, only text similarity).

    Returns (rate, details) where details is one dict per case with its own
    max_similarity, for spot-checking which specific cases triggered the
    threshold."""
    details = []
    for case in eval_cases:
        sim = max_similarity_to_pair(case["situation_summary"], case["pair"], index)
        details.append({**case, "max_similarity": sim, "near_duplicate": sim >= threshold})
    n = len(details)
    if n == 0:
        return 0.0, details
    rate = sum(1 for d in details if d["near_duplicate"]) / n
    return rate, details


def interpret(rate: float) -> str:
    """Preregistered interpretation bands, docs/PREREGISTRATION.md."""
    if rate >= MEMORIZATION_SIGNAL_BAR:
        return (f"MEMORIZATION SIGNAL: overlap rate {rate:.1%} >= the preregistered "
               f"{MEMORIZATION_SIGNAL_BAR:.0%} bar -- report explicitly regardless of "
               f"whether accuracy bars are also cleared.")
    if rate <= 0.02:
        return (f"CONSISTENT WITH GENERALIZATION: overlap rate {rate:.1%} is within the "
               f"~0.6-2% chance range (null-hypothesis baseline {NULL_HYPOTHESIS_BASELINE:.1%}).")
    return (f"AMBIGUOUS: overlap rate {rate:.1%} is above the chance baseline "
           f"({NULL_HYPOTHESIS_BASELINE:.1%}) but below the {MEMORIZATION_SIGNAL_BAR:.0%} "
           f"memorization-signal bar -- worth noting, not a clean call either way.")
