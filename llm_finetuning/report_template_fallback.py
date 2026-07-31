"""
Step 3 of the low-threat-collapse diagnosis session (AUDIT.md sec M/N/O).

Tests the "v2's advantage is memorization capacity, not data quality" hypothesis
without retraining. build_sft_dataset.py's gold_assessment() falls back to a fixed
TEMPLATE situation_summary/threat_reasoning ("The swarm is in a {form_a} formation...")
whenever the Groq teacher call fails or is skipped (--no-teacher); a teacher-written
row instead gets free-text prose that varies row-to-row even for the same
(form_a, form_b) pair (different narrative framing each call). So: for a given pair,
if every row's situation_summary is byte-identical, every one of those rows is a
template fallback: N examples but effectively 1 point of information, learnable by
lookup rather than requiring the model to generalize over prose variation.

Cross-references llm_finetuning/check_training_coverage.py's extract_pair() to
recover which (form_a, form_b) pair each row belongs to, matching situation_summary
text per pair. Pure JSON parsing, no model calls.

Usage:
    python llm_finetuning/report_template_fallback.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from check_training_coverage import extract_pair  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402

FILES = {
    "sft_train_v2.jsonl": REPO / "data" / "sft_train_v2.jsonl",
    "sft_train_final.jsonl": REPO / "data" / "sft_train_final.jsonl",
}

TEMPLATE_PREFIX = "The swarm is in a "  # gold_assessment()'s fallback situation_summary


def main():
    for label, path in FILES.items():
        by_pair = defaultdict(list)
        n_rows = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                row = json.loads(line)
                pair = extract_pair(row["messages"][0]["content"])
                if pair is None:
                    continue
                target = json.loads(row["messages"][1]["content"])
                by_pair[pair].append(target["situation_summary"])

        n_pairs = len(by_pair)
        examples_per_pair = [len(v) for v in by_pair.values()]
        distinct_per_pair = [len(set(v)) for v in by_pair.values()]
        template_rows = sum(1 for summaries in by_pair.values() for s in summaries
                            if s.startswith(TEMPLATE_PREFIX))
        pairs_all_template = sum(1 for summaries in by_pair.values()
                                 if all(s.startswith(TEMPLATE_PREFIX) for s in summaries))
        pairs_all_identical = sum(1 for summaries in by_pair.values() if len(set(summaries)) == 1)

        print(f"=== {label} ({n_rows} rows, {n_pairs} distinct pairs) ===")
        print(f"examples-per-pair: min={min(examples_per_pair)} "
              f"mean={sum(examples_per_pair)/n_pairs:.1f} max={max(examples_per_pair)}")
        print(f"distinct situation_summary strings per pair: min={min(distinct_per_pair)} "
              f"mean={sum(distinct_per_pair)/n_pairs:.2f} max={max(distinct_per_pair)}")
        print(f"template-fallback rows (situation_summary starts with "
              f"'{TEMPLATE_PREFIX}...'): {template_rows}/{n_rows} ({template_rows/n_rows:.1%})")
        print(f"pairs where EVERY row is template fallback: {pairs_all_template}/{n_pairs}")
        print(f"pairs where ALL rows share one identical situation_summary "
              f"(1 point of info regardless of row count): {pairs_all_identical}/{n_pairs}")

        print("  is 'low' specifically under-served relative to other threat classes "
              "in THIS file?")
        for threat in ("low", "medium", "high", "critical"):
            pairs = [p for p in by_pair if RULES[p][0] == threat]
            if not pairs:
                continue
            counts = [len(by_pair[p]) for p in pairs]
            distinct = [len(set(by_pair[p])) for p in pairs]
            print(f"    {threat}: n_pairs={len(pairs)} "
                  f"mean_examples/pair={sum(counts)/len(pairs):.2f} "
                  f"mean_distinct_summaries/pair={sum(distinct)/len(pairs):.2f}")
        print()


if __name__ == "__main__":
    main()
