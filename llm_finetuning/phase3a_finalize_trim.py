"""
Phase 3a finalization, step 2: trim terminal_transitioning out of the teacher corpus.

Per the signed-off decisions (0.0% natural frequency on two independent populations,
sec AO step 3; carry forward the teacher-authored corpus, not the no-teacher one):
drop terminal_transitioning's 100 rows from data/abstention_corpus_teacher.jsonl,
leaving 780 multi_hop + 120 oscillation = 900 rows. Writes a NEW file rather than
overwriting the source corpus (same non-destructive discipline as sec AO step 1's
"neither overwrites the other, both exist on disk").

Then re-runs gates a (row count) and b (zero overlap + zero internal dupes) from
scripts/phase3a_step4_gates.py's logic, adapted: gate a only checks multi_hop/
oscillation against the preregistered targets -- terminal_transitioning is
deliberately absent by decision, not a defect, so it is reported separately as
"intentionally dropped", not scored as a failure.

Usage:
    python llm_finetuning/phase3a_finalize_trim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SOURCE_CORPUS = REPO / "data" / "abstention_corpus_teacher.jsonl"
SOURCE_META = REPO / "data" / "abstention_corpus_teacher_meta.json"
OUT_CORPUS = REPO / "data" / "abstention_corpus_teacher_trimmed900.jsonl"
OUT_META = REPO / "data" / "abstention_corpus_teacher_trimmed900_meta.json"
TARGETS_PATH = REPO / "evaluation" / "phase3a_strata_targets.json"
EXISTING_CORPUS_FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def trim():
    rows = load_jsonl(SOURCE_CORPUS)
    meta = json.loads(SOURCE_META.read_text())
    assert len(rows) == len(meta["rows_detail"]) == meta["n_total"] == 1000

    kept_rows = []
    kept_detail = []
    dropped = 0
    for row, detail in zip(rows, meta["rows_detail"]):
        if detail["mechanism"] == "terminal_transitioning":
            dropped += 1
            continue
        kept_rows.append(row)
        kept_detail.append(detail)

    assert dropped == 100, f"expected to drop exactly 100 terminal_transitioning rows, dropped {dropped}"
    assert len(kept_rows) == 900

    from collections import Counter
    per_mechanism = dict(Counter(d["mechanism"] for d in kept_detail))
    assert per_mechanism == {"multi_hop": 780, "oscillation": 120}

    OUT_CORPUS.write_text("\n".join(json.dumps(r) for r in kept_rows) + "\n")
    trimmed_meta = dict(meta)
    trimmed_meta["n_total"] = 900
    trimmed_meta["per_mechanism"] = per_mechanism
    trimmed_meta["rows_detail"] = kept_detail
    trimmed_meta["n_used_teacher"] = sum(1 for d in kept_detail if d.get("used_teacher"))
    trimmed_meta["dropped_terminal_transitioning"] = 100
    trimmed_meta["source"] = str(SOURCE_CORPUS.relative_to(REPO))
    OUT_META.write_text(json.dumps(trimmed_meta, indent=2))

    print(f"Trimmed corpus written: {OUT_CORPUS.relative_to(REPO)} (900 rows)")
    print(f"  per_mechanism: {per_mechanism}")
    print(f"  n_used_teacher (of 900): {trimmed_meta['n_used_teacher']}")
    return kept_rows, trimmed_meta


def gate_a(meta, targets):
    print("\n=== GATE a (adapted): row counts, terminal_transitioning intentionally dropped ===")
    ok = True
    for mechanism, target in targets["strata_targets"].items():
        actual = meta["per_mechanism"].get(mechanism, 0)
        if mechanism == "terminal_transitioning":
            print(f"  {mechanism}: actual=0 target={target} -- INTENTIONALLY DROPPED "
                  f"(sec AO step 3: 0.0% natural frequency on two populations), not scored")
            continue
        passed = actual > 0 and actual >= target
        ok &= passed
        print(f"  {mechanism}: actual={actual} target={target} -- {'PASS' if passed else 'FAIL'}")
    print(f"GATE a: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_b(new_rows):
    print("\n=== GATE b: zero overlap with existing 12,001-row RULES corpus + zero internal dupes ===")
    existing_keys = set()
    for path in EXISTING_CORPUS_FILES:
        for row in load_jsonl(path):
            existing_keys.add(row["messages"][0]["content"])
    n_existing = len(existing_keys)
    assert n_existing == 12001, f"expected 12001 existing keys, found {n_existing}"

    new_keys = [r["messages"][0]["content"] for r in new_rows]
    overlap = [k for k in new_keys if k in existing_keys]
    internal_dupes = len(new_keys) - len(set(new_keys))

    ok = len(overlap) == 0 and internal_dupes == 0
    print(f"  existing corpus keys: {n_existing}")
    print(f"  trimmed corpus rows: {len(new_keys)}")
    print(f"  overlap with existing corpus: {len(overlap)}")
    print(f"  internal duplicates within trimmed corpus: {internal_dupes}")
    print(f"GATE b: {'PASS' if ok else 'FAIL'}")
    return ok, len(overlap), internal_dupes


def main():
    kept_rows, trimmed_meta = trim()
    targets = json.loads(TARGETS_PATH.read_text())

    ok_a = gate_a(trimmed_meta, targets)
    ok_b, n_overlap, n_internal_dupes = gate_b(kept_rows)

    overall = ok_a and ok_b
    print("\n" + "=" * 80)
    print(f"STEP 2 GATES (trimmed 900-row corpus): {'PASS' if overall else 'FAIL -- STOP'}")
    print(f"  gate a: {'PASS' if ok_a else 'FAIL'}")
    print(f"  gate b: {'PASS' if ok_b else 'FAIL'} (overlap={n_overlap}, internal_dupes={n_internal_dupes})")
    print("=" * 80)

    out = {
        "final_composition": trimmed_meta["per_mechanism"],
        "final_total": trimmed_meta["n_total"],
        "gate_a_pass": ok_a,
        "gate_b_pass": ok_b,
        "gate_b_overlap": n_overlap,
        "gate_b_internal_dupes": n_internal_dupes,
        "overall_pass": overall,
        "corpus_file": str(OUT_CORPUS.relative_to(REPO)),
        "meta_file": str(OUT_META.relative_to(REPO)),
    }
    out_path = REPO / "evaluation" / "phase3a_step4_gates_results_trimmed900.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
