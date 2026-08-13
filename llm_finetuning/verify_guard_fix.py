"""
AUDIT.md sec AL step 2 verification: confirms the new
MAX_WINDOWS_PER_SINGLE_TRANSITION guard (src/swarm_intent/coverage.py) does
EXACTLY what it should and nothing else -- regenerates the locked seed=4321
population and diffs the new bucket/guard_reasons against the BEFORE
snapshot (evaluation/categorize_unanswerable_502.json, captured before this
fix), case by case, for all 1000 sequences (not just the 502 unanswerable
ones -- also checks the 498 has_ground_truth=True cases for accidental
regressions).

Usage:
    python llm_finetuning/verify_guard_fix.py
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

from swarm_intent.coverage import classify_observation  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
TARGET_9 = {105, 291, 292, 314, 385, 520, 574, 611, 982}
BEFORE_PATH = REPO / "evaluation" / "categorize_unanswerable_502.json"


def ground_truth_from_true_chain(true_chain: list):
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    return RULES[pair]


def main():
    import torch
    import swarm_intent.stgt.model as stgt_model_module
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    device = torch.device("cpu")
    stgt_model_module.device = device
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    before = json.loads(BEFORE_PATH.read_text())
    before_by_i = {r["i"]: r for r in before["unanswerable_records"]}

    rng = np.random.default_rng(SEED)
    changed_within_502 = {}
    target_9_status = {}
    unchanged_502_by_category = Counter()
    bucket_tally_after = Counter()
    n_gt_true_flipped_to_B = []

    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)  # robust=False, default, unchanged
        true_chain = [str(f) for f in chain]
        has_gt = ground_truth_from_true_chain(true_chain) is not None
        bucket_tally_after[bucket_info["bucket"]] += 1

        if has_gt and bucket_info["bucket"] == "B" \
                and "observation_too_long_for_reduced_state_count" in bucket_info["guard_reasons"]:
            n_gt_true_flipped_to_B.append(i)

        prior = before_by_i.get(i)
        if prior is None:
            continue  # a has_ground_truth=True case -- was never in the 502
        new_bucket, new_reasons = bucket_info["bucket"], bucket_info["guard_reasons"]
        old_bucket, old_reasons = prior["bucket"], prior["guard_reasons"]
        if i in TARGET_9:
            target_9_status[i] = {"old": (old_bucket, old_reasons), "new": (new_bucket, new_reasons),
                                  "true_chain": true_chain}
        elif (new_bucket, new_reasons) != (old_bucket, old_reasons):
            changed_within_502[i] = {"old": (old_bucket, old_reasons), "new": (new_bucket, new_reasons),
                                     "true_chain": true_chain}
        else:
            unchanged_502_by_category[old_bucket] += 1

    print("=== the 9 target cases (must ALL now be bucket B with the new guard reason) ===")
    all_9_fixed = True
    for i in sorted(TARGET_9):
        st = target_9_status.get(i)
        if st is None:
            print(f"  seq {i}: NOT FOUND in this run -- integrity problem")
            all_9_fixed = False
            continue
        ok = (st["new"][0] == "B" and "observation_too_long_for_reduced_state_count" in st["new"][1])
        all_9_fixed &= ok
        print(f"  seq {i}: {st['old']} -> {st['new']}  true_chain={st['true_chain']}  "
             f"{'OK' if ok else 'NOT FIXED'}")
    print(f"\nALL 9 FIXED: {all_9_fixed}")

    print(f"\n=== the other 493 has_ground_truth=False cases (492 Layer-3-eligible + 1 "
         f"dispersed_converging_ambiguity) -- must be UNCHANGED ===")
    print(f"unchanged: {sum(unchanged_502_by_category.values())} ({dict(unchanged_502_by_category)})")
    print(f"changed (unexpected): {len(changed_within_502)}")
    for i, d in changed_within_502.items():
        print(f"  seq {i}: {d['old']} -> {d['new']}  true_chain={d['true_chain']}")

    print(f"\n=== full 1000-sequence bucket tally, and has_ground_truth=True false-positive check ===")
    print(f"bucket tally (n=1000): {dict(bucket_tally_after)}")
    print(f"has_ground_truth=True cases newly flagged by the guard (should be 0 -- these would be "
         f"FALSE POSITIVES on genuinely-resolvable cases): {len(n_gt_true_flipped_to_B)} {n_gt_true_flipped_to_B}")

    out = {"target_9_status": {str(k): v for k, v in target_9_status.items()},
          "all_9_fixed": all_9_fixed, "changed_within_502_unexpected": changed_within_502,
          "unchanged_502_by_category": dict(unchanged_502_by_category),
          "bucket_tally_after_n1000": dict(bucket_tally_after),
          "false_positives_on_gt_true": n_gt_true_flipped_to_B}
    out_path = REPO / "evaluation" / "verify_guard_fix.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
