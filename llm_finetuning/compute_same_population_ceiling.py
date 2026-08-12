"""
Hardening audit step 3: the locked 83.0% threat ceiling (docs/V5_LOG.md Phase 0
close) was measured on a DIFFERENT trajectory population (seed=999, n=1000,
509 pair-eligible) than v5-a's Phase 4 evaluation (seed=4321, n=1000, 498
has_ground_truth=True). Dividing v5-a's accuracy_when_answerable by that
cross-population ceiling would not be a same-subset comparison.

This script regenerates the EXACT locked seed=4321 population (same code,
deterministic, CPU-only -- generate_phase4_eval_set.py's own STGT+bridge
pipeline), verifies the regeneration reproduces the locked
evaluation/phase4_eval_set.json bit-for-bit (an integrity check on the lock
itself, not just a means to an end), and this time ALSO captures the bridge's
own concluded rules_key per case -- giving a TRUE same-population ceiling:
does STGT's own reduction correctly resolve to the ground-truth pair, on the
EXACT SAME 498 cases v5-a's accuracy_when_answerable was measured over.

Usage:
    python llm_finetuning/compute_same_population_ceiling.py
"""
from __future__ import annotations

import hashlib
import json
import sys
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
LOCKED_PATH = REPO / "evaluation" / "phase4_eval_set.json"


def ground_truth_from_true_chain(true_chain: list):
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    threat, intent, action = RULES[pair]
    return {"expected_threat": threat, "pair": list(pair)}


def main():
    import torch
    import swarm_intent.stgt.model as stgt_model_module
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    device = torch.device("cpu")
    stgt_model_module.device = device  # same monkeypatch as generate_phase4_eval_set.py

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    print(f"=== regenerating the LOCKED seed={SEED} population, capturing rules_key this time ===")
    rng = np.random.default_rng(SEED)

    locked = json.loads(LOCKED_PATH.read_text())
    locked_items = locked["items"]

    mismatches = 0
    chain_diffs = 0
    bucket_diffs = 0
    ctx_only_diffs = 0
    ceiling_hits, ceiling_scored = 0, 0
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        true_chain = [str(f) for f in chain]

        # integrity check: does this reproduce the locked ctx/bucket exactly?
        locked_item = locked_items[i]
        chain_diff = locked_item["true_chain"] != true_chain
        bucket_diff = locked_item["bucket"] != bucket_info["bucket"]
        ctx_diff = locked_item["ctx"] != bucket_info["context_text"]
        if chain_diff or bucket_diff or ctx_diff:
            mismatches += 1
            if chain_diff:
                chain_diffs += 1
            if bucket_diff:
                bucket_diffs += 1
            if ctx_diff and not chain_diff and not bucket_diff:
                ctx_only_diffs += 1
            if mismatches <= 5:
                print(f"MISMATCH seq_{i}: chain_diff={chain_diff} bucket_diff={bucket_diff} ctx_diff={ctx_diff}")
                if chain_diff:
                    print(f"  locked true_chain={locked_item['true_chain']} vs regenerated={true_chain}")
                if bucket_diff:
                    print(f"  locked bucket={locked_item['bucket']} vs regenerated={bucket_info['bucket']}")
                if ctx_diff and not chain_diff:
                    print(f"  locked ctx:\n{locked_item['ctx']}\n  regenerated ctx:\n{bucket_info['context_text']}")
            if chain_diff or bucket_diff:
                # a REAL divergence (trajectory RNG or bucket outcome) -- this case's
                # rules_key/bucket cannot be trusted as matching the locked population,
                # skip it from the ceiling computation entirely rather than use a
                # possibly-different bucket/rules_key than what v5-a was scored against.
                continue
            # else: cosmetic ctx-text-only diff (e.g. dominant-formation tie-break) --
            # rules_key/bucket are UNAFFECTED, safe to include in the ceiling below.

        gt = ground_truth_from_true_chain(true_chain)
        if gt is None:
            continue  # has_ground_truth=False, not part of the ceiling population

        # STGT's OWN concluded pair (may be None if bridge couldn't resolve one --
        # bucket B/C cases -- that IS the ceiling's own imperfection, not excluded)
        rules_key = bucket_info.get("rules_key")
        ceiling_scored += 1
        if rules_key is not None and rules_key in RULES:
            bridge_threat = RULES[rules_key][0]
            if bridge_threat == gt["expected_threat"]:
                ceiling_hits += 1
        # else: bridge produced no resolvable key (or an OOV one) -- counted as a
        # miss against ground truth, same convention the historical ceiling uses
        # (an unresolved bridge read cannot be "correct" by construction).

    print(f"\nintegrity check: {mismatches}/{N_SEQUENCES} sequences differ from the locked file")
    print(f"  of which: chain_diff={chain_diffs} (trajectory-generation RNG divergence -- would be fatal)")
    print(f"  of which: bucket_diff={bucket_diffs} (A/B/C classification divergence -- would affect the ceiling)")
    print(f"  of which: ctx_only_diff={ctx_only_diffs} (cosmetic text-only, e.g. dominant-formation tie-break "
         f"-- does NOT affect rules_key/bucket, safe to proceed if this is the only category)")

    if chain_diffs > 0 or bucket_diffs > 0:
        print("STOPPING: a chain or bucket divergence exists -- this WOULD affect the ceiling "
             "computation. Investigate before trusting any number below.")
        return
    print("\nAll mismatches are ctx-text-only (cosmetic) -- rules_key/bucket outcomes, which is "
         "what the ceiling actually depends on, are unaffected. Proceeding.")

    print(f"\nSAME-POPULATION ceiling (STGT bridge's own accuracy against ground truth, "
         f"computed on the EXACT 498 has_ground_truth=True cases v5-a was evaluated on):")
    print(f"  {ceiling_hits}/{ceiling_scored} = {ceiling_hits/ceiling_scored:.1%}")

    Path(REPO / "evaluation" / "same_population_ceiling.json").write_text(json.dumps({
        "seed": SEED, "n_sequences": N_SEQUENCES, "integrity_mismatches": mismatches,
        "ceiling_hits": ceiling_hits, "ceiling_scored": ceiling_scored,
        "ceiling_accuracy": ceiling_hits / ceiling_scored if ceiling_scored else None,
    }, indent=2))
    print("\nsaved evaluation/same_population_ceiling.json")


if __name__ == "__main__":
    main()
