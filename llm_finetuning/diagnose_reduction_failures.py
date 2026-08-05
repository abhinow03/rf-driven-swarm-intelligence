"""
AUDIT.md sec AG step 1: sec AF found pipeline_v2's bucket classification
recognizes only 9/249 (3.6%) of REAL sequences whose generator ground truth
IS a clean (a,b) pair as Layer-1 resolvable. This script categorizes WHY,
for every one of those 249 sequences, using the CURRENT (unanimity-based,
robust=False) stgt_bridge.bridge_predictions / coverage.classify_observation
-- exactly the code path sec AF measured, unmodified.

Regenerates the identical 500 sequences (same seed=0, same sample_chain /
build_long_sequence, bit-for-bit reproducible -- verified in sec AF step 4).

Categories (priority order, mutually exclusive):
  already_resolved         -- bucket A already (the 9 sec AF found)
  all_windows_transitioning -- bucket C, subtype=all_unknown: every window
                               read outside BASE_FORMATIONS
  trailing_transitioning_run -- bucket C, subtype=terminal_unknown: the
                               sequence's LAST window(s) read transitioning/
                               unknown while earlier windows were clean --
                               sec AE/AF step 4's "windowing artefact"
  formation_name_mismatch  -- bucket C, subtype in (multi_hop, oscillation),
                               AND the extra "hop" is a REAL but WRONG
                               BASE_FORMATIONS class (not a transitioning/
                               unknown read) -- genuine model misclassification,
                               not a reduction-logic brittleness; majority
                               voting cannot fix this
  interior_noisy_window    -- bucket C multi_hop/oscillation caused by an
                               INTERIOR unknown window splitting an otherwise-
                               clean history into 3+ groups, OR bucket B via
                               oov_name (an isolated unknown blip amid an
                               otherwise 2-formation history)
  dispersed_converging_ambiguity -- bucket B, guarded by the dispersed/
                               converging near-tie (a real upstream defect,
                               not brittleness -- kept as-is per step 2d)
  other                     -- bucket B via dominant_history_contradiction/
                               low_confidence alone, or anything unclassified

Usage (run inside tmux):
    python llm_finetuning/diagnose_reduction_failures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 500
SEED = 0


def ground_truth_pair(true_chain: list):
    if len(true_chain) == 1:
        return (true_chain[0], true_chain[0])
    if len(true_chain) == 2:
        return (true_chain[0], true_chain[1])
    return None


def categorize_failure(bucket_info: dict, predictions: list) -> str:
    bucket = bucket_info["bucket"]
    if bucket == BUCKET_A:
        return "already_resolved"

    if bucket == BUCKET_C:
        subtype = bucket_info["subtype"]
        if subtype == "all_unknown":
            return "all_windows_transitioning"
        if subtype == "terminal_unknown":
            return "trailing_transitioning_run"
        if subtype in ("multi_hop", "oscillation"):
            history = bucket_info["summary"]["formation_history"]
            real_formations = {f for f in history if f in BASE_FORMATIONS}
            if len(real_formations) >= 3:
                return "formation_name_mismatch"
            return "interior_noisy_window"
        return "other"

    # bucket == BUCKET_B
    reasons = bucket_info["guard_reasons"]
    if "dispersed_converging_ambiguity" in reasons:
        return "dispersed_converging_ambiguity"
    if "oov_name" in reasons:
        return "interior_noisy_window"
    return "other"


def main():
    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(SEED)
    reporter = Reporter("diagnose_reduction_failures", N_SEQUENCES, rate_hint=8.0)

    records = []
    for i in range(N_SEQUENCES):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        gt_pair = ground_truth_pair(chain)
        if gt_pair is None:
            reporter.update(1, item=f"seq {i}")
            continue  # not one of the 249 GT-clean sequences

        bucket_info = classify_observation(predictions)
        category = categorize_failure(bucket_info, predictions)
        records.append({"i": i, "true_chain": chain, "gt_pair": list(gt_pair),
                        "bucket": bucket_info["bucket"], "subtype": bucket_info["subtype"],
                        "guard_reasons": bucket_info["guard_reasons"], "category": category})
        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    out_path = REPO / "evaluation" / "reduction_failure_diagnosis.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nsaved {out_path}")

    n = len(records)
    print(f"\n=== reduction failure diagnosis, n={n} (all 249 GT-clean sequences) ===")
    print("| category | n | % |")
    print("|---|---|---|")
    cat_counts = Counter(r["category"] for r in records)
    for cat, k in cat_counts.most_common():
        print(f"| {cat} | {k} | {k/n:.1%} |")

    failures = [r for r in records if r["category"] != "already_resolved"]
    print(f"\n=== same breakdown, as % of FAILURES only (n={len(failures)}) ===")
    fail_counts = Counter(r["category"] for r in failures)
    for cat, k in fail_counts.most_common():
        print(f"  {cat}: {k}/{len(failures)} ({k/len(failures):.1%})")


if __name__ == "__main__":
    main()
