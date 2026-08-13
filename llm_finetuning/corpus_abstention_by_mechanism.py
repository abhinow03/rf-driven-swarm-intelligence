"""
Hardening audit, Layer-2-gap diagnosis, step 3a: in the 12,001-row v5-a
training corpus (data/sft_train_v5_phase1.jsonl + _val.jsonl), count rows
carrying an abstention target label (assistant likely_intent in
ABSTENTION_TOKENS, same detector prompts.is_abstention uses everywhere else
in this project), broken down by structural mechanism (multi_hop /
terminal_transitioning / oscillation -- the three Layer-3-eligible-by-design
mechanisms from step 1).

Mechanism is parsed directly from each row's own "Formation history: ..."
line (present verbatim in every user prompt) -- the corpus was NOT built
with a stored per-row mechanism tag (checked: data/sft_train_v5_phase1_
provenance.json only records used_teacher/split, not axis), so this reuses
the SAME structural logic classify_observation applies to formation_history
(collapse consecutive repeats, check length, terminal transitioning check),
not a new ad hoc parser -- ported inline since the corpus's ctx text is a
FROZEN artifact, not something classify_ctx's live regex needs to touch.

Usage (fast, no GPU, seconds):
    python llm_finetuning/corpus_abstention_by_mechanism.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402

CORPUS_FILES = ["data/sft_train_v5_phase1.jsonl", "data/sft_train_v5_phase1_val.jsonl",
               "data/sft_train_v5_phase1_mining.jsonl"]  # 10801+600+600=12001, confirmed against
# data/sft_train_v5_phase1_provenance.json's split counts (Counter({'train': 10801, 'val': 600,
# 'mining': 600})) -- the mining file is part of the same 12,001-row corpus V5_LOG.md/
# PREREGISTRATION.md cite, not a separate held-out set; omitting it undercounts by 600 rows.
HISTORY_RE = re.compile(r"Formation history:\s*(.+)")


def parse_mechanism(user_content: str) -> str:
    m = HISTORY_RE.search(user_content)
    if not m:
        return "no_history_line"  # defensive, should be unreachable on real rows
    tokens = [t.strip() for t in m.group(1).split("->")]
    if tokens and tokens[-1] not in BASE_FORMATIONS:
        return "terminal_transitioning"
    known = [t for t in tokens if t in BASE_FORMATIONS]
    known_collapsed = [k for k, _ in groupby(known)]
    if len(known_collapsed) >= 3:
        return "oscillation" if known_collapsed[0] == known_collapsed[-1] else "multi_hop"
    return "resolvable"  # <=2 distinct formations -- not one of the three target mechanisms


def main():
    counts_by_mech = Counter()
    abstained_by_mech = Counter()
    total = 0

    for fname in CORPUS_FILES:
        path = REPO / fname
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                user_content = row["messages"][0]["content"]
                assistant_content = row["messages"][1]["content"]
                try:
                    assistant = json.loads(assistant_content)
                except json.JSONDecodeError:
                    continue
                mechanism = parse_mechanism(user_content)
                abstained = is_abstention(assistant.get("likely_intent", ""))
                total += 1
                counts_by_mech[mechanism] += 1
                if abstained:
                    abstained_by_mech[mechanism] += 1

    total_abstained = sum(abstained_by_mech.values())
    print(f"=== corpus abstention labels, n={total} rows (expected 12001) ===")
    print(f"total rows with an abstention target label: {total_abstained}/{total} ({total_abstained/total:.1%})\n")

    print("| mechanism | rows | abstention-labeled | % of mechanism | % of 12001 |")
    print("|---|---|---|---|---|")
    for mech in ("multi_hop", "terminal_transitioning", "oscillation", "resolvable", "no_history_line"):
        n = counts_by_mech.get(mech, 0)
        k = abstained_by_mech.get(mech, 0)
        pct_mech = f"{k/n:.1%}" if n else "n/a"
        print(f"| {mech} | {n} | {k} | {pct_mech} | {n/total:.1%} |")

    out = {"total_rows": total, "total_abstained": total_abstained,
          "counts_by_mechanism": dict(counts_by_mech), "abstained_by_mechanism": dict(abstained_by_mech)}
    out_path = REPO / "evaluation" / "corpus_abstention_by_mechanism.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
