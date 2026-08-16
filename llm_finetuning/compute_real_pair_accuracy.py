"""
V5a2 preregistration step 2: compute the REAL (non-proxy) pair_accuracy for v5-a on the
full has_ground_truth=True population (n=498), using literal_pair_extraction.py, validated
by hand against 30 known-pair cases first (llm_finetuning/validate_literal_pair_extraction.py,
19/30 raw agreement, 0 extraction bugs found in the mismatches -- every mismatch traced to a
genuine model narration error, not a regex miss).

Zero new inference: re-scores evaluation/phase4_v5a_results.json's already-cached
parsed_by_case output against evaluation/phase4_eval_set.json's true_chain.

This replaces the old proxy (likely_intent match against RULES[true_pair]'s intent, 63.6%,
AUDIT.md sec ~CC "step2_pair_accuracy_fix") with a literal pair-identification number, for
use as the v5-a baseline in v5a2's regression_vs_v5a bar.

Usage:
    python llm_finetuning/compute_real_pair_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from literal_pair_extraction import extract_literal_pair, true_pair_from_chain  # noqa: E402


def main():
    phase4 = json.loads((REPO / "evaluation" / "phase4_eval_set.json").read_text())
    results = json.loads((REPO / "evaluation" / "phase4_v5a_results.json").read_text())
    parsed_by_case = results["parsed_by_case"]

    gt_true_items = [it for it in phase4["items"] if it["has_ground_truth"]]
    n = len(gt_true_items)

    n_correct = 0
    n_extraction_failed = 0
    per_case = []
    for it in gt_true_items:
        name = it["name"]
        true_pair = true_pair_from_chain(it["true_chain"])
        parsed = parsed_by_case.get(name)
        if parsed is None:
            per_case.append({"name": name, "status": "MISSING_FROM_CACHE"})
            continue
        extracted = extract_literal_pair(parsed)
        if extracted is None:
            n_extraction_failed += 1
        correct = extracted == true_pair
        n_correct += correct
        per_case.append({
            "name": name, "true_pair": list(true_pair),
            "extracted_pair": list(extracted) if extracted else None, "correct": correct,
        })

    n_scored = sum(1 for c in per_case if c.get("status") != "MISSING_FROM_CACHE")
    pair_accuracy = n_correct / n_scored

    print("=" * 90)
    print("REAL (non-proxy) pair_accuracy -- v5-a, has_ground_truth=True population")
    print("=" * 90)
    print(f"n (has_ground_truth=True): {n}")
    print(f"n scored (found in cache): {n_scored}")
    print(f"extraction failures (no formation mention found): {n_extraction_failed}")
    print(f"n correct (literal pair match): {n_correct}")
    print(f"PAIR_ACCURACY (literal, non-proxy): {pair_accuracy:.4f} ({100*pair_accuracy:.1f}%)")
    print()
    print("Comparison to the old proxy (likely_intent match against RULES[true_pair]'s "
          f"intent): 0.6358 (63.6%, n=497). Real literal-pair accuracy: {pair_accuracy:.4f} "
          f"({100*pair_accuracy:.1f}%, n={n_scored}).")

    out = {
        "metric": "pair_accuracy_pooled_when_answerable (literal, non-proxy)",
        "n": n, "n_scored": n_scored, "n_correct": n_correct,
        "n_extraction_failed": n_extraction_failed,
        "pair_accuracy": pair_accuracy,
        "old_proxy_pair_accuracy": 0.6358,
        "old_proxy_n": 497,
        "extraction_method": "llm_finetuning/literal_pair_extraction.py",
        "validation": "llm_finetuning/validate_literal_pair_extraction.py, 19/30 (63.3%) raw "
                      "agreement on hand sample, 0 extraction bugs found among the 11 "
                      "mismatches (all traced to genuine model narration errors)",
        "per_case": per_case,
    }
    out_path = REPO / "evaluation" / "v5a_real_pair_accuracy.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
