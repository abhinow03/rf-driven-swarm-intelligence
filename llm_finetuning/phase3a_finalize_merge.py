"""
Phase 3a finalization, step 3: merge the trimmed 900-row abstention corpus into the
existing 12,001-row Phase 1 corpus (sft_train_v5_phase1.jsonl + _val.jsonl +
_mining.jsonl, pooled -- the same three files gate_b in scripts/phase3a_step4_gates.py
treats as "the existing 12,001-row RULES corpus"), writing a NEW file rather than
overwriting any of the three originals. Final size: 12,001 + 900 = 12,901.

Re-runs the dedup check across the FULL merged file (not just new-vs-old, which
sec AO/step 2 already confirmed at 0) -- this additionally checks for any
pre-existing internal duplication within the 12,001-row Phase 1 corpus itself that
a new-vs-old-only check would never have caught.

Usage:
    python llm_finetuning/phase3a_finalize_merge.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXISTING_CORPUS_FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]
NEW_CORPUS_FILE = REPO / "data" / "abstention_corpus_teacher_trimmed900.jsonl"
OUT_PATH = REPO / "data" / "sft_train_v5_phase3a_merged.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    existing_rows = []
    per_file_counts = {}
    for path in EXISTING_CORPUS_FILES:
        rows = load_jsonl(path)
        existing_rows.extend(rows)
        per_file_counts[path.name] = len(rows)
    assert len(existing_rows) == 12001, f"expected 12001 existing rows, got {len(existing_rows)}"

    new_rows = load_jsonl(NEW_CORPUS_FILE)
    assert len(new_rows) == 900, f"expected 900 new rows, got {len(new_rows)}"

    merged = existing_rows + new_rows
    assert len(merged) == 12901

    # Corpus-wide dedup check (not just new-vs-old): exact user-message-string key,
    # same convention as build_sft_dataset.py's --append path / gate_b.
    keys = [r["messages"][0]["content"] for r in merged]
    key_counts = Counter(keys)
    dupes = {k: c for k, c in key_counts.items() if c > 1}
    n_dupe_rows = sum(c - 1 for c in dupes.values())  # rows beyond the first occurrence
    n_unique = len(key_counts)

    OUT_PATH.write_text("\n".join(json.dumps(r) for r in merged) + "\n")

    print("=" * 90)
    print("PHASE 3A FINALIZATION STEP 3: MERGE")
    print("=" * 90)
    print("per-file existing counts:", per_file_counts)
    print(f"existing total: {len(existing_rows)}")
    print(f"new (trimmed abstention) rows: {len(new_rows)}")
    print(f"MERGED TOTAL: {len(merged)}")
    print()
    print(f"corpus-wide dedup check (full {len(merged)}-row file, exact user-message key):")
    print(f"  unique keys: {n_unique}")
    print(f"  duplicate keys found: {len(dupes)}")
    print(f"  total duplicate rows (beyond first occurrence): {n_dupe_rows}")
    zero_dupes = n_dupe_rows == 0
    print(f"  {'ZERO INTERNAL DUPLICATES -- PASS' if zero_dupes else 'DUPLICATES FOUND -- FAIL'}")
    if dupes:
        for k, c in list(dupes.items())[:10]:
            print(f"    x{c}: {k[:100]!r}")

    result = {
        "existing_per_file_counts": per_file_counts,
        "existing_total": len(existing_rows),
        "new_rows": len(new_rows),
        "merged_total": len(merged),
        "n_unique_keys": n_unique,
        "n_duplicate_keys": len(dupes),
        "n_duplicate_rows": n_dupe_rows,
        "zero_internal_duplicates": zero_dupes,
        "out_path": str(OUT_PATH.relative_to(REPO)),
    }
    out_json = REPO / "evaluation" / "phase3a_merge_dedup_results.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nsaved merged corpus: {OUT_PATH}")
    print(f"saved dedup report: {out_json}")

    if not zero_dupes:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
