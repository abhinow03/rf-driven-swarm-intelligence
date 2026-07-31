"""
Step 1 of the "is the 55-case battery in-distribution?" check (see AUDIT.md sec J).

Parses "Dominant formation: X" / "Formation history: A -> transitioning -> B"
(or "Formation history: A" for steady state, A==B) out of each user-message in
the three SFT training/val files, recovers the exact (form_a, form_b) pair
build_sft_dataset.py sampled for that row, and reports how many of the 49
RULES pairs each file covers plus examples-per-pair.

This is a fast, one-command diagnostic (~1300 rows total) -- no model calls,
no Reporter needed.

Usage:
    python llm_finetuning/check_training_coverage.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402

FILES = [
    REPO / "data" / "sft_train_v2.jsonl",
    REPO / "data" / "sft_train_final.jsonl",
    REPO / "data" / "sft_train_final_abstain.jsonl",
]

DOMINANT_RE = re.compile(r"Dominant formation: (\S+)")
HISTORY_RE = re.compile(r"Formation history: (.+)")

ALL_49_PAIRS = set(RULES.keys())
assert ALL_49_PAIRS == {(a, b) for a in BASE_FORMATIONS for b in BASE_FORMATIONS}, \
    "RULES no longer a complete 7x7 map -- extraction logic below assumes it is"


def extract_pair(user_content: str) -> tuple[str, str] | None:
    """Recover (form_a, form_b) from one row's context block, or None if this
    row's context doesn't match the synth_context() shape at all (e.g. an
    OOV-formation / held-out-shape row that isn't a plain RULES-pair sample)."""
    dom = DOMINANT_RE.search(user_content)
    hist = HISTORY_RE.search(user_content)
    if not dom or not hist:
        return None
    form_a = dom.group(1)
    history_line = hist.group(1).strip()
    if " -> transitioning -> " in history_line:
        parts = history_line.split(" -> transitioning -> ")
        if len(parts) != 2:
            return None
        form_b = parts[1].strip()
    else:
        form_b = history_line
    if form_a not in BASE_FORMATIONS or form_b not in BASE_FORMATIONS:
        return None  # OOV / held-out-shape row, or a >2-hop chain -- not a plain pair
    return (form_a, form_b)


def main():
    print(f"=== training-data RULES-pair coverage check ({len(ALL_49_PAIRS)} possible pairs) ===\n")
    for path in FILES:
        if not path.exists():
            print(f"{path.name}: MISSING\n")
            continue
        counts: Counter[tuple[str, str]] = Counter()
        n_rows = 0
        n_unparsed = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_rows += 1
                row = json.loads(line)
                user_content = row["messages"][0]["content"]
                pair = extract_pair(user_content)
                if pair is None:
                    n_unparsed += 1
                    continue
                counts[pair] += 1

        covered = set(counts.keys())
        n_covered = len(covered & ALL_49_PAIRS)
        print(f"--- {path.name} ({n_rows} rows, {n_unparsed} unparsed/non-pair rows) ---")
        print(f"RULES pairs covered: {n_covered}/{len(ALL_49_PAIRS)} "
              f"({n_covered / len(ALL_49_PAIRS):.0%})")
        if counts:
            examples_per_pair = list(counts.values())
            print(f"examples-per-covered-pair: min={min(examples_per_pair)} "
                  f"max={max(examples_per_pair)} "
                  f"mean={sum(examples_per_pair) / len(examples_per_pair):.1f}")
            missing = ALL_49_PAIRS - covered
            if missing:
                print(f"missing pairs ({len(missing)}): "
                      f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
        print()

    # Cross-file union: what does v2 (the model that scored 100%) actually see,
    # cumulative across every file that could plausibly have contributed?
    v2_counts: Counter[tuple[str, str]] = Counter()
    with open(REPO / "data" / "sft_train_v2.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = extract_pair(row["messages"][0]["content"])
            if pair:
                v2_counts[pair] += 1
    v2_covered = len(set(v2_counts.keys()) & ALL_49_PAIRS)
    print(f"=== qwen-swarm-v2's training file alone: {v2_covered}/{len(ALL_49_PAIRS)} "
          f"({v2_covered / len(ALL_49_PAIRS):.0%}) RULES pairs ===")
    if v2_covered == len(ALL_49_PAIRS):
        print("v2 saw EVERY RULES pair during training. The 55-case battery (49/49 "
              "RULES-pair coverage) is therefore IN-DISTRIBUTION for v2 -- its 100% "
              "accuracy_when_answerable is a rule-table RECALL measurement, not evidence "
              "of generalization, and should not be cited as generalization accuracy.")


if __name__ == "__main__":
    main()
