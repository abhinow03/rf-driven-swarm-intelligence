"""
AUDIT.md sec AG step 2 (final part): Layer-1 firing rate before vs after
robust reduction, on the SAME 500 held-out real sequences (seed=0) sec AE/AF
measured -- NOT the seed=1 dev split step 2's threshold tuning used. STGT-only
(no LLM calls), fast.

Usage (run inside tmux):
    python llm_finetuning/report_robust_reduction_firing_rate.py
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

from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.stgt_bridge import DEFAULT_ROBUST_THRESHOLD  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 500
SEED = 0


def ground_truth_pair(true_chain):
    if len(true_chain) == 1:
        return (true_chain[0], true_chain[0])
    if len(true_chain) == 2:
        return (true_chain[0], true_chain[1])
    return None


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
    reporter = Reporter("report_robust_reduction_firing_rate", N_SEQUENCES, rate_hint=8.0)

    records = []
    for i in range(N_SEQUENCES):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        gt_pair = ground_truth_pair(chain)

        before = classify_observation(predictions, robust=False)
        after = classify_observation(predictions, robust=True, robust_threshold=DEFAULT_ROBUST_THRESHOLD)

        record = {"i": i, "true_chain": chain, "gt_pair": list(gt_pair) if gt_pair else None,
                 "bucket_before": before["bucket"], "bucket_after": after["bucket"],
                 "rules_key_after": list(after["rules_key"]) if after["rules_key"] else None,
                 "robust_recovery": after.get("robust_recovery")}
        if gt_pair is not None and after["bucket"] == BUCKET_A:
            record["after_correct"] = (tuple(after["rules_key"]) == gt_pair)
        records.append(record)
        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    out_path = REPO / "evaluation" / "robust_reduction_firing_rate.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nsaved {out_path}")

    n = len(records)
    before_counts = Counter(r["bucket_before"] for r in records)
    after_counts = Counter(r["bucket_after"] for r in records)
    print(f"\n=== Layer-1 (bucket A) firing rate, n={n} real sequences (seed=0, same as sec AE/AF) ===")
    print("| bucket | before (unanimity) | after (robust) |")
    print("|---|---|---|")
    for b in (BUCKET_A, BUCKET_B, BUCKET_C):
        bk, ak = before_counts.get(b, 0), after_counts.get(b, 0)
        print(f"| {b} | {bk}/{n} ({bk/n:.1%}) | {ak}/{n} ({ak/n:.1%}) |")

    gt_records = [r for r in records if r["gt_pair"] is not None]
    n_gt = len(gt_records)
    gt_a_before = sum(1 for r in gt_records if r["bucket_before"] == BUCKET_A)
    gt_a_after = sum(1 for r in gt_records if r["bucket_after"] == BUCKET_A)
    print(f"\n=== of the {n_gt} GT-determinable sequences specifically ===")
    print(f"  bucket A before: {gt_a_before}/{n_gt} ({gt_a_before/n_gt:.1%})")
    print(f"  bucket A after:  {gt_a_after}/{n_gt} ({gt_a_after/n_gt:.1%})")

    after_a_gt = [r for r in gt_records if r["bucket_after"] == BUCKET_A]
    if after_a_gt:
        correct = sum(1 for r in after_a_gt if r.get("after_correct"))
        print(f"  precision of the NEWLY-robust bucket-A cases against ground truth: "
             f"{correct}/{len(after_a_gt)} ({correct/len(after_a_gt):.1%})")

    moved_to_a = [r for r in records if r["bucket_before"] != BUCKET_A and r["bucket_after"] == BUCKET_A]
    print(f"\ncases that moved from B/C -> A: {len(moved_to_a)}/{n} ({len(moved_to_a)/n:.1%})")
    print(f"cases with a robust_recovery attempt at all (recovered or not): "
         f"{sum(1 for r in records if r['robust_recovery'] is not None)}/{n}")


if __name__ == "__main__":
    main()
