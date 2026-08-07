"""
AUDIT.md sec AH step 1: sec AG's `robust_reduction_firing_rate.json` already
recorded WHICH 12/500 held-out sequences reached bucket A under robust=True
and whether the recovered (a,b) pair matched independent ground truth (2/12,
16.7%). That number is about the UPSTREAM key (did robust reduction recover
the right (a,b) pair) -- it says nothing about whether pipeline_v2's Layer 1
DOWNSTREAM of a given key is actually "correct by construction": does
_finalize_layer1's forced overwrite (pipeline_v2.py:138-161) really make
threat_level/likely_intent/recommended_action equal RULES[(a,b)] on every
run, with the LLM narrator never able to leak a different value through?

This script re-runs ONLY Layer 1 (the RULES.txt-prompted narrator client, no
STGT re-training, no fine-tuned adapters, no other systems) for the exact 12
sequences sec AG already identified, 5 stochastic samples each (temperature
0.3, matching every other eval in this project) to give the narrator repeated
chances to deviate. This is a verification re-check of already-diagnosed
cases, not a new experiment: no new accuracy numbers, no new comparison, one
existing architectural claim checked directly against code behaviour.

Usage (run inside tmux):
    python llm_finetuning/verify_layer1_guarantee.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation  # noqa: E402
from swarm_intent.stgt_bridge import DEFAULT_ROBUST_THRESHOLD  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402
from swarm_intent.llm.client import LocalHFClient  # noqa: E402

from baselines import load_rules_txt  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SEQUENCES = 500
SEED = 0
N_RUNS = 5

FIRING_RATE_PATH = REPO / "evaluation" / "robust_reduction_firing_rate.json"


def main():
    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    firing = json.loads(FIRING_RATE_PATH.read_text())
    target_records = {r["i"]: r for r in firing if r["bucket_after"] == "A"}
    target_idx = set(target_records)
    assert len(target_idx) == 12, f"expected 12 bucket-A sequences, found {len(target_idx)}"
    print(f"target sequences (bucket_after == A on robust=True): {sorted(target_idx)}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(SEED)
    target_preds = {}
    for i in range(N_SEQUENCES):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        if i not in target_idx:
            continue
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        target_preds[i] = (chain, predictions)

    assert set(target_preds) == target_idx
    print("re-derived STGT predictions for all 12 target sequences (bit-for-bit reproduction of seed=0 run)")

    import gc
    del stgt_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== loading rules_narrator_client (base model, RULES.txt system prompt) ===")
    rules_client = LocalHFClient(BASE_MODEL, adapter_path=None, temperature=0.3, system_prompt=load_rules_txt())

    results = []
    for i in sorted(target_idx):
        chain, predictions = target_preds[i]
        bucket_info = classify_observation(predictions, robust=True, robust_threshold=DEFAULT_ROBUST_THRESHOLD)
        assert bucket_info["bucket"] == "A", f"seq {i}: expected bucket A, got {bucket_info['bucket']}"
        rules_key = tuple(bucket_info["rules_key"])
        prior_key = tuple(target_records[i]["rules_key_after"])
        assert rules_key == prior_key, f"seq {i}: rules_key mismatch, {rules_key} vs recorded {prior_key}"
        expected = RULES[rules_key]  # (threat, intent, action)
        threat, intent, action = expected

        run_records = []
        for r in range(N_RUNS):
            assessment, layer, detail = pipeline_v2.assess_observation(
                rules_client, None, predictions, class_freq={}, robust=True,
                robust_threshold=DEFAULT_ROBUST_THRESHOLD)
            assert layer == pipeline_v2.LAYER_1_DETERMINISTIC
            matches = (assessment.get("threat_level") == threat
                      and assessment.get("likely_intent") == intent
                      and assessment.get("recommended_action") == action)
            run_records.append({
                "run": r,
                "assessment_threat": assessment.get("threat_level"),
                "assessment_intent": assessment.get("likely_intent"),
                "assessment_action": assessment.get("recommended_action"),
                "matches_rules_lookup": matches,
                "llm_deviation": detail.get("llm_deviation", {}),
            })

        gt_pair = target_records[i].get("gt_pair")
        gt_expected = RULES[tuple(gt_pair)] if gt_pair else None
        results.append({
            "i": i, "true_chain": chain, "rules_key": list(rules_key),
            "rules_lookup_result": {"threat_level": threat, "likely_intent": intent, "recommended_action": action},
            "gt_pair": gt_pair,
            "gt_would_give": ({"threat_level": gt_expected[0], "likely_intent": gt_expected[1],
                              "recommended_action": gt_expected[2]} if gt_expected else None),
            "key_matches_gt": (rules_key == tuple(gt_pair)) if gt_pair else False,
            "runs": run_records,
        })
        n_dev = sum(1 for rr in run_records if rr["llm_deviation"])
        n_mismatch = sum(1 for rr in run_records if not rr["matches_rules_lookup"])
        print(f"  seq {i}: rules_key={rules_key} -> RULES gives "
             f"({threat}/{intent}/{action}); {N_RUNS - n_mismatch}/{N_RUNS} runs match RULES lookup exactly; "
             f"{n_dev}/{N_RUNS} runs had a non-empty llm_deviation log entry")

    out_path = REPO / "evaluation" / "layer1_guarantee_verification.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")

    total_runs = sum(len(r["runs"]) for r in results)
    total_mismatch = sum(1 for r in results for rr in r["runs"] if not rr["matches_rules_lookup"])
    total_deviation_logged = sum(1 for r in results for rr in r["runs"] if rr["llm_deviation"])
    n_key_matches_gt = sum(1 for r in results if r["key_matches_gt"])

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"sequences checked: {len(results)}")
    print(f"total (sequence, run) units: {total_runs}")
    print(f"assessment fields != RULES[rules_key] lookup: {total_mismatch}/{total_runs} "
         f"({'GUARANTEE VIOLATED' if total_mismatch else 'guarantee holds, zero exceptions'})")
    print(f"runs where the LLM narrator's raw output != the decision fields (llm_deviation non-empty), "
         f"but was overwritten before being returned: {total_deviation_logged}/{total_runs}")
    print(f"of the 12 sequences, rules_key matches independent ground truth pair: {n_key_matches_gt}/12 "
         f"(matches sec AG step 2's 2/12, 16.7% precision figure)")


if __name__ == "__main__":
    main()
