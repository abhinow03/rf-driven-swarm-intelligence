"""
V5a2 preregistration step 2: hand-validate literal_pair_extraction against a sample of
KNOWN pairs before trusting it at scale, per the explicit instruction not to trust the
extractor on the first run. Prints every sampled case's true pair, extracted pair, and the
source text so each can be eyeballed, then reports raw agreement + a breakdown of
disagreement reasons (extraction miss vs genuine model error) for the sample.

Usage:
    python llm_finetuning/validate_literal_pair_extraction.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from literal_pair_extraction import extract_literal_pair, true_pair_from_chain  # noqa: E402

N_SAMPLE = 30
SEED = 7


def main():
    phase4 = json.loads((REPO / "evaluation" / "phase4_eval_set.json").read_text())
    results = json.loads((REPO / "evaluation" / "phase4_v5a_results.json").read_text())
    parsed_by_case = results["parsed_by_case"]

    gt_true_items = [it for it in phase4["items"] if it["has_ground_truth"]]
    rng = random.Random(SEED)
    sample = rng.sample(gt_true_items, N_SAMPLE)

    rows = []
    for it in sample:
        name = it["name"]
        true_pair = true_pair_from_chain(it["true_chain"])
        parsed = parsed_by_case.get(name)
        if parsed is None:
            rows.append({"name": name, "true_pair": true_pair, "status": "MISSING_FROM_CACHE"})
            continue
        extracted = extract_literal_pair(parsed)
        rows.append({
            "name": name, "true_chain": it["true_chain"], "true_pair": list(true_pair),
            "extracted_pair": list(extracted) if extracted else None,
            "match": extracted == true_pair,
            "situation_summary": parsed.get("situation_summary", ""),
        })

    print("=" * 100)
    print(f"HAND-VALIDATION SAMPLE (n={N_SAMPLE}, seed={SEED})")
    print("=" * 100)
    n_match = 0
    n_extract_fail = 0
    for r in rows:
        if r.get("status") == "MISSING_FROM_CACHE":
            print(f"\n[{r['name']}] MISSING FROM CACHE")
            continue
        tag = "MATCH" if r["match"] else "MISMATCH"
        if r["match"]:
            n_match += 1
        if r["extracted_pair"] is None:
            n_extract_fail += 1
        print(f"\n[{r['name']}] true={r['true_pair']}  extracted={r['extracted_pair']}  [{tag}]")
        print(f"  summary: {r['situation_summary']}")

    n_scored = sum(1 for r in rows if r.get("status") != "MISSING_FROM_CACHE")
    print("\n" + "=" * 100)
    print(f"raw agreement on hand-checked sample: {n_match}/{n_scored} "
          f"({100*n_match/n_scored:.1f}%)")
    print(f"extraction failures (no formation name found at all): {n_extract_fail}/{n_scored}")
    print("=" * 100)
    print("\nMANUAL REVIEW REQUIRED: read each [MISMATCH] above and confirm whether it's (a) a "
          "genuine model error (correctly extracted, model just got the pair wrong -- this is "
          "exactly what the metric should catch) or (b) an extraction bug (the summary text "
          "states the correct pair but the regex missed it, or picked up a spurious mention, "
          "e.g. from follow_up_watch speculation like 'watch for a shift to converging'). "
          "Only proceed to the full-scale run once (b)-type errors are at or near zero.")

    out_path = REPO / "evaluation" / "literal_pair_extraction_validation_sample.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
