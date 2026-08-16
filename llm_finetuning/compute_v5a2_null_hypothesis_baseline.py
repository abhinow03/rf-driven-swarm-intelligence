"""
V5a2 preregistration step 3j: re-measure the memorization null-hypothesis baseline fresh on
data/sft_train_v5_phase3a_merged.jsonl (12,901 rows), rather than reusing v5-a's 0.6% figure
(measured on the original 12,001-row Phase 1 corpus) -- the user's explicit instruction,
since the training corpus changed.

Same methodology as PREREGISTRATION.md's original baseline: leave-one-out TF-IDF cosine
similarity (score_memorization.py's max_similarity_to_pair, unmodified), >= 0.90 threshold,
among TEACHER-authored rows sharing the same (form_a, form_b) RULES pair. TEACHER-authored
filtering uses data/sft_train_v5_phase1_provenance.json (sha256(messages[0].content) ->
{split, used_teacher}), the same provenance record used_teacher was defined against --
confirmed necessary by a first pass that included template-fallback rows and got a much
higher (and wrong) 7.4% rate, because templated same-pair rows share boilerplate phrasing by
construction, which is not memorization signal.

Also confirms empirically (not by assumption) whether the 900 new abstention rows contribute
any pair-keyed rows at all -- they describe multi_hop/oscillation mechanisms, not a single
RULES pair, so build_training_index's _extract_pair_from_ctx should find none.

Usage:
    python llm_finetuning/compute_v5a2_null_hypothesis_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))
sys.path.insert(0, str(REPO / "src"))

from score_memorization import build_training_index, max_similarity_to_pair, NEAR_DUP_THRESHOLD  # noqa: E402

MERGED_CORPUS = REPO / "data" / "sft_train_v5_phase3a_merged.jsonl"
ORIGINAL_POOLED_FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]
PROVENANCE_PATH = REPO / "data" / "sft_train_v5_phase1_provenance.json"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def teacher_authored_keys() -> set[str]:
    prov = json.loads(PROVENANCE_PATH.read_text())
    assert prov["key"] == "sha256(messages[0].content)"
    return {k for k, v in prov["rows"].items() if v.get("used_teacher")}


def build_teacher_only_index(corpus_path: Path, teacher_keys: set[str]) -> dict:
    """Same pair-keying as build_training_index, restricted to rows whose
    sha256(messages[0].content) is in teacher_keys."""
    full_index = build_training_index(str(corpus_path))
    rows = load_jsonl(corpus_path)
    teacher_summary_by_pair: dict = {}
    sys.path.insert(0, str(REPO / "src"))
    from swarm_intent.coverage import _extract_pair_from_ctx
    for r in rows:
        content = r["messages"][0]["content"]
        key = hashlib.sha256(content.encode()).hexdigest()
        if key not in teacher_keys:
            continue
        pair = _extract_pair_from_ctx(content)
        if pair is None:
            continue
        summary = json.loads(r["messages"][1]["content"])["situation_summary"]
        teacher_summary_by_pair.setdefault(pair, []).append(summary)
    return teacher_summary_by_pair


def leave_one_out_baseline(index: dict) -> tuple[float, int, int]:
    n_near_dup = 0
    n_scored = 0
    for pair, summaries in index.items():
        if len(summaries) < 2:
            continue
        for i, s in enumerate(summaries):
            others = summaries[:i] + summaries[i + 1:]
            sim = max_similarity_to_pair(s, pair, {pair: others})
            n_scored += 1
            if sim >= NEAR_DUP_THRESHOLD:
                n_near_dup += 1
    rate = n_near_dup / n_scored if n_scored else 0.0
    return rate, n_near_dup, n_scored


def main():
    teacher_keys = teacher_authored_keys()
    print(f"teacher-authored rows per provenance record: {len(teacher_keys)}/12001")

    merged_rows = load_jsonl(MERGED_CORPUS)
    pooled_original_rows = []
    for p in ORIGINAL_POOLED_FILES:
        pooled_original_rows.extend(load_jsonl(p))
    assert len(pooled_original_rows) == 12001
    print(f"merged corpus: {len(merged_rows)} rows; pooled original Phase 1 corpus: "
          f"{len(pooled_original_rows)} rows")

    index_merged_all = build_training_index(str(MERGED_CORPUS))
    n_pairkeyed_merged_all = sum(len(v) for v in index_merged_all.values())
    print(f"pair-keyed rows (any provenance) in merged corpus: {n_pairkeyed_merged_all} "
          f"(expected 12001 -- confirms the 900 new abstention rows contribute 0 pair keys)")
    assert n_pairkeyed_merged_all == 12001

    teacher_index_merged = build_teacher_only_index(MERGED_CORPUS, teacher_keys)
    n_teacher_pairkeyed = sum(len(v) for v in teacher_index_merged.values())
    print(f"TEACHER-authored, pair-keyed rows: {n_teacher_pairkeyed}")

    rate, n_near_dup, n_scored = leave_one_out_baseline(teacher_index_merged)
    print(f"\nFRESH null-hypothesis baseline (merged 12,901-row corpus, TEACHER-authored rows "
          f"only, leave-one-out, >= {NEAR_DUP_THRESHOLD} cosine similarity):")
    print(f"  {n_near_dup}/{n_scored} = {rate:.4%}")
    print(f"\nFor reference, v5-a's original figure (same methodology, pre-merge corpus): "
          f"0.6% (56/10,080)")

    out = {
        "merged_corpus_rows": len(merged_rows),
        "pooled_original_corpus_rows": len(pooled_original_rows),
        "teacher_authored_rows_total": len(teacher_keys),
        "pairkeyed_rows_any_provenance_merged": n_pairkeyed_merged_all,
        "new_abstention_rows_contributing_a_pair_key": n_pairkeyed_merged_all - 12001,
        "teacher_authored_pairkeyed_rows": n_teacher_pairkeyed,
        "null_hypothesis_baseline_fresh": rate,
        "n_near_duplicate": n_near_dup,
        "n_scored": n_scored,
        "old_v5a_baseline_for_reference_only": {"rate": 0.006, "n_near_dup": 56, "n_scored": 10080},
        "note": "computed on data/sft_train_v5_phase3a_merged.jsonl (12,901 rows), teacher-"
                "authored rows only (data/sft_train_v5_phase1_provenance.json), NOT reused "
                "from v5-a's original 0.6% figure -- see module docstring.",
    }
    out_path = REPO / "evaluation" / "v5a2_null_hypothesis_baseline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
