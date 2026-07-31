"""
Step 2 of the low-threat-collapse diagnosis session (AUDIT.md sec M/N).

For each SFT training file, reports the threat_level distribution of the
ASSISTANT TARGETS (not the input contexts) and examples-per-pair, and compares
against RULES' own 15/49 (30.6%) low-threat share -- to tell whether low-threat
rows are underrepresented relative to RULES, or roughly proportional. Pure
JSON parsing, no model calls, runs in well under a second.

Usage:
    python llm_finetuning/report_class_balance.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from build_sft_dataset import RULES  # noqa: E402

FILES = [
    REPO / "data" / "sft_train_v2.jsonl",
    REPO / "data" / "sft_train_final.jsonl",
    REPO / "data" / "sft_train_final_abstain.jsonl",
]

RULES_THREAT_DIST = Counter(threat for threat, _, _ in RULES.values())


def main():
    n_rules = len(RULES)
    print(f"=== RULES' own threat_level distribution (the canonical rule table, "
          f"{n_rules} pairs) ===")
    for threat in ("low", "medium", "high", "critical"):
        n = RULES_THREAT_DIST.get(threat, 0)
        print(f"  {threat}: {n}/{n_rules} ({n / n_rules:.1%})")

    for path in FILES:
        if not path.exists():
            print(f"\n{path.name}: MISSING")
            continue
        threat_counts = Counter()
        n_rows = 0
        n_unparsed = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                row = json.loads(line)
                try:
                    target = json.loads(row["messages"][1]["content"])
                    threat_counts[target["threat_level"]] += 1
                except (json.JSONDecodeError, KeyError):
                    n_unparsed += 1

        print(f"\n=== {path.name} ({n_rows} rows, {n_unparsed} unparsed assistant targets) ===")
        for threat in ("low", "medium", "high", "critical"):
            n = threat_counts.get(threat, 0)
            rules_share = RULES_THREAT_DIST.get(threat, 0) / n_rules
            actual_share = n / (n_rows - n_unparsed) if (n_rows - n_unparsed) else 0
            skew = actual_share - rules_share
            print(f"  {threat}: {n} rows ({actual_share:.1%}) vs RULES share {rules_share:.1%} "
                  f"(delta {skew:+.1%})")


if __name__ == "__main__":
    main()
