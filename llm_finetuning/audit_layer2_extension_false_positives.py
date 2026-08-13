"""
AUDIT.md sec AN step 4: MANDATORY false-positive audit for the new structural multi-hop/
oscillation routing (src/swarm_intent/pipeline_v2.py's _structural_chain_reason). Hard gate,
per the LOCKED CONFIG: zero tolerance for any of the 498 has_ground_truth=True (answerable)
cases newly routing to Layer 2 because of the new "structurally_unresolvable_multihop"/
"structurally_unresolvable_oscillation" reasons. A real answerable case is, by construction,
a chain of length 1 or 2 -- known_history should never reach length 3 for these UNLESS the
classifier's own noisy read inflates it (e.g. a genuinely 2-formation trajectory misread as
column->diamond->column due to oscillating misclassification). This script checks that
empirically rather than assuming it.

CPU-only, deterministic -- regenerates the locked seed=4321 population exactly as
categorize_unanswerable_502.py / investigate_layer2_early_returns.py do (same 9 expected
bucket mismatches from AUDIT.md sec AK/AL's guard-logic fix), then evaluates
pipeline_v2._structural_chain_reason() against every has_ground_truth=True case's real
bucket_info.

Usage (CPU-only, ~9 min):
    python llm_finetuning/audit_layer2_extension_false_positives.py
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
from swarm_intent.pipeline_v2 import _structural_chain_reason  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
LOCKED_PATH = REPO / "evaluation" / "phase4_eval_set.json"
EXPECTED_BUCKET_MISMATCHES = {105, 291, 292, 314, 385, 520, 574, 611, 982}


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
    stgt_model_module.device = device

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    locked_items = json.loads(LOCKED_PATH.read_text())["items"]

    print("=== regenerating locked seed=4321 population ===")
    rng = np.random.default_rng(SEED)
    chain_mismatches = 0
    bucket_mismatch_indices = []
    answerable_records = []
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        true_chain = [str(f) for f in chain]

        locked = locked_items[i]
        if locked["true_chain"] != true_chain:
            chain_mismatches += 1
        if locked["bucket"] != bucket_info["bucket"]:
            bucket_mismatch_indices.append(i)

        gt = ground_truth_from_true_chain(true_chain)
        if gt is None:
            continue  # not one of the 498 has_ground_truth=True cases -- out of scope for this gate

        structural_reason = _structural_chain_reason(bucket_info)
        answerable_records.append({
            "i": i, "true_chain": true_chain, "expected_threat": gt["expected_threat"],
            "expected_pair": gt["pair"], "original_bucket": bucket_info["bucket"],
            "original_subtype": bucket_info["subtype"], "original_rules_key": bucket_info["rules_key"],
            "original_guard_reasons": bucket_info["guard_reasons"],
            "formation_history": bucket_info["summary"].get("formation_history"),
            "structural_reason": structural_reason,
            "would_be_new_false_positive": structural_reason is not None,
        })

    unexpected = set(bucket_mismatch_indices) - EXPECTED_BUCKET_MISMATCHES
    missing = EXPECTED_BUCKET_MISMATCHES - set(bucket_mismatch_indices)
    print(f"integrity: chain_mismatches={chain_mismatches}, "
         f"bucket_mismatches={len(bucket_mismatch_indices)} (9 expected), "
         f"unexpected={sorted(unexpected)}, missing={sorted(missing)}")
    if chain_mismatches or unexpected or missing:
        print("STOPPING: a real, unexplained divergence exists -- do not trust the audit below.")
        return

    print(f"\nhas_ground_truth=True cases: {len(answerable_records)} (expected 498)")

    false_positives = [r for r in answerable_records if r["would_be_new_false_positive"]]
    n_fp = len(false_positives)

    print("\n" + "=" * 100)
    print(f"STEP 4 RESULT -- NEW FALSE POSITIVES ON THE 498 ANSWERABLE CASES: {n_fp}")
    print("=" * 100)
    if n_fp > 0:
        print("\nDETAIL -- each false-positive case, its true answer, and why it triggered:")
        for r in false_positives:
            print(f"\n  seq {r['i']}")
            print(f"    true_chain={r['true_chain']}  expected_pair={r['expected_pair']}  "
                 f"expected_threat={r['expected_threat']}")
            print(f"    original routing: bucket={r['original_bucket']} subtype={r['original_subtype']} "
                 f"rules_key={r['original_rules_key']} guard_reasons={r['original_guard_reasons']}")
            print(f"    formation_history={r['formation_history']}")
            print(f"    NEW structural_reason that would misroute it: {r['structural_reason']}")
    else:
        print("\nZero false positives. The new routing is safe to ship against this gate.")

    out = {
        "n_answerable": len(answerable_records), "n_new_false_positives": n_fp,
        "false_positive_cases": false_positives,
        "all_answerable_records": answerable_records,
    }
    out_path = REPO / "evaluation" / "layer2_extension_false_positive_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
