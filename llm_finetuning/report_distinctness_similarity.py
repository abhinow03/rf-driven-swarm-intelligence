"""
AUDIT.md V5 Phase 1 step-3-followup: distinctness gate tightening check.

The distinct-summary gate (step 2, 93.6% overall) only checks EXACT string
uniqueness -- it would pass two near-identical templated-sounding sentences
with one word swapped ("A group of UAVs..." vs "A small group of UAVs...")
as two distinct rows. This script measures near-duplication directly via
TF-IDF cosine similarity (word 1-2 grams) over every teacher-authored
situation_summary in the corpus (combined across train/val/mining, matching
step 2's own combined-set methodology exactly so the before/after numbers
are apples-to-apples), then re-computes the distinct-summary gate using a
STRICT definition: connected-components clustering at cosine similarity
>= 0.90 (i.e. two summaries within 0.90 cosine similarity of each other
count as ONE cluster/one "distinct" unit, not two). Template-fallback rows
are still counted via exact-dedup only (they're already known to collapse
onto the 49 RULES-pair template strings; a similarity check adds nothing
there and would be expensive for no reason -- see step 1/2's report).

Usage:
    python llm_finetuning/report_distinctness_similarity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_FOLLOWUP = "Monitor formation and approach rate over the next window."
NEAR_DUP_THRESHOLD = 0.90

FILES = [
    REPO / "data" / "sft_train_v5_phase1.jsonl",
    REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pairwise_max_sim(X, chunk=1000):
    """Returns (max_sim[n], nearest_idx[n]) -- each row's highest cosine
    similarity to any OTHER row, and which row that is. Chunked so the dense
    similarity block stays memory-bounded regardless of corpus size."""
    n = X.shape[0]
    max_sim = np.zeros(n, dtype=np.float32)
    nearest_idx = np.zeros(n, dtype=np.int64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = (X[start:end] @ X.T).toarray()
        for local_i, global_i in enumerate(range(start, end)):
            block[local_i, global_i] = -1.0
        max_sim[start:end] = block.max(axis=1)
        nearest_idx[start:end] = block.argmax(axis=1)
    return max_sim, nearest_idx


def cluster_count(X, threshold, chunk=1000):
    """Connected-components cluster count at the given cosine-similarity
    threshold -- the STRICT distinct count (a cluster of 3 near-identical
    summaries counts as 1, not 3)."""
    n = X.shape[0]
    rows_idx, cols_idx = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = (X[start:end] @ X.T).toarray()
        for local_i, global_i in enumerate(range(start, end)):
            block[local_i, global_i] = 0.0
        r_local, c_local = np.where(block >= threshold)
        rows_idx.extend(r_local + start)
        cols_idx.extend(c_local)
    adj = csr_matrix((np.ones(len(rows_idx)), (rows_idx, cols_idx)), shape=(n, n))
    n_clusters, _ = connected_components(adj, directed=False)
    return int(n_clusters)


def main():
    all_rows = []
    for path in FILES:
        all_rows.extend(load_jsonl(path))
    n_total = len(all_rows)

    teacher_summaries, fallback_summaries = [], []
    for r in all_rows:
        a = json.loads(r["messages"][1]["content"])
        if a["follow_up_watch"] == TEMPLATE_FOLLOWUP:
            fallback_summaries.append(a["situation_summary"])
        else:
            teacher_summaries.append(a["situation_summary"])

    print(f"n_total={n_total}  teacher={len(teacher_summaries)}  fallback={len(fallback_summaries)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(teacher_summaries)
    max_sim, nearest_idx = pairwise_max_sim(X)

    print("\n" + "=" * 90)
    print("near-duplicate rate at various cosine-similarity thresholds (teacher rows only)")
    print("=" * 90)
    for thresh in (0.99, 0.95, 0.90, 0.80, 0.70):
        k = int((max_sim >= thresh).sum())
        print(f"  >= {thresh:.2f}: {k}/{len(teacher_summaries)} ({k/len(teacher_summaries):.1%})")
    print(f"mean nearest-neighbor similarity: {max_sim.mean():.3f}, median: {np.median(max_sim):.3f}")

    print("\n" + "=" * 90)
    print("10 most similar non-identical pairs (manual inspection)")
    print("=" * 90)
    order = np.argsort(-max_sim)
    shown = 0
    for i in order:
        if max_sim[i] >= 0.999:
            continue
        j = nearest_idx[i]
        print(f"\nsim={max_sim[i]:.3f}\n  A: {teacher_summaries[i]}\n  B: {teacher_summaries[j]}")
        shown += 1
        if shown >= 10:
            break

    # ---- gate re-computation: exact-dedup (original) vs strict near-dup clustering ----
    exact_teacher_distinct = len(set(teacher_summaries))
    exact_fallback_distinct = len(set(fallback_summaries))
    exact_overlap = len(set(teacher_summaries) & set(fallback_summaries))
    exact_total_distinct = exact_teacher_distinct + exact_fallback_distinct - exact_overlap

    strict_teacher_clusters = cluster_count(X, NEAR_DUP_THRESHOLD)
    strict_total_distinct = strict_teacher_clusters + exact_fallback_distinct

    print("\n" + "=" * 90)
    print(f"distinct-summary gate: exact-dedup (original, step 2) vs. STRICT (>= {NEAR_DUP_THRESHOLD} cosine cluster)")
    print("=" * 90)
    print(f"  exact-dedup:  {exact_total_distinct}/{n_total} = {exact_total_distinct/n_total:.1%}")
    print(f"  strict:       {strict_total_distinct}/{n_total} = {strict_total_distinct/n_total:.1%}  "
         f"(target >=90%: {'PASS' if strict_total_distinct/n_total >= 0.90 else 'FAIL'})")


if __name__ == "__main__":
    main()
