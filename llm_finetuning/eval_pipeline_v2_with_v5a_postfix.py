"""
AUDIT.md sec AL step 3: re-runs eval_pipeline_v2_with_v5a.py's exact protocol
AFTER the MAX_WINDOWS_PER_SINGLE_TRANSITION guard fix (src/swarm_intent/
coverage.py, sec AK/AL) -- the ONLY change from that script is the integrity
check, which now KNOWINGLY expects exactly 9 bucket differences against the
stale phase4_eval_set.json (generated before the fix): the 9 cases sec AK
identified and sec AL's fix targets, verified case-by-case in
verify_guard_fix.py before this ran. Any OTHER divergence (chain mismatches,
or bucket mismatches outside that named set) still halts the run --
this is a narrowed acceptance, not a disabled check.

Usage:
    python llm_finetuning/eval_pipeline_v2_with_v5a_postfix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402
from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402
from eval_sft_v5 import resolve_adapter_path  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V5A_ADAPTER_DIR = str(REPO / "checkpoints" / "v5_sft")
BATCH_SIZE = 8
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


def regenerate_items_with_full_bucket_info():
    """CPU-only, deterministic. Returns items list matching pipeline_v2's
    expected shape, plus true_chain/ground truth per item, plus an integrity
    mismatch count (chain/bucket only -- ctx-text-only cosmetic diffs from the
    known dominant-formation tie-break are expected and ignored, see
    compute_same_population_ceiling.py for the full diagnosis)."""
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

    rng = np.random.default_rng(SEED)
    items, chain_mismatches, bucket_mismatch_indices = [], 0, []
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
        items.append({
            "name": f"phase4_seq_{i}", "ctx": bucket_info["context_text"],
            "key_windows": bucket_info["key_windows"], "bucket_info": bucket_info,
            "true_chain": true_chain, "has_ground_truth": gt is not None,
            **(gt or {}),
        })

    return items, chain_mismatches, bucket_mismatch_indices


# sec AK's 9 identified bucket_A_misrouted cases -- the ONLY bucket differences this
# run is allowed to have against the stale (pre-fix) phase4_eval_set.json, verified
# case-by-case in verify_guard_fix.py before this ran.
EXPECTED_BUCKET_MISMATCHES = {105, 291, 292, 314, 385, 520, 574, 611, 982}


def main():
    print("=== regenerating full bucket_info on the locked seed=4321 population (CPU-only) ===")
    items, chain_mismatches, bucket_mismatch_indices = regenerate_items_with_full_bucket_info()
    unexpected = set(bucket_mismatch_indices) - EXPECTED_BUCKET_MISMATCHES
    missing = EXPECTED_BUCKET_MISMATCHES - set(bucket_mismatch_indices)
    print(f"integrity: chain_mismatches={chain_mismatches}, "
         f"bucket_mismatches={len(bucket_mismatch_indices)} (9 expected -- sec AK's "
         f"guard-logic fix, verified in verify_guard_fix.py), unexpected={sorted(unexpected)}, "
         f"missing={sorted(missing)}")
    if chain_mismatches > 0 or unexpected or missing:
        print("STOPPING: a real, unexplained divergence exists -- do not trust pipeline_v2 routing below.")
        return

    bucket_tally = {}
    for it in items:
        bucket_tally[it["bucket_info"]["bucket"]] = bucket_tally.get(it["bucket_info"]["bucket"], 0) + 1
    print(f"bucket split: A={bucket_tally.get(BUCKET_A,0)} B={bucket_tally.get(BUCKET_B,0)} "
         f"C={bucket_tally.get(BUCKET_C,0)}")

    print("\n=== loading rules_narrator client (base, no adapter) + v5-a (Layer 3) ===")
    rules_client = LocalHFClient(BASE_MODEL, adapter_path=None, temperature=0.0)
    v5a_path = resolve_adapter_path(V5A_ADAPTER_DIR)
    finetuned_client = LocalHFClient(BASE_MODEL, adapter_path=v5a_path, temperature=0.0)
    class_freq = pipeline_v2.default_class_freq()

    print("=== running pipeline_v2._resolve_batched ===")
    pipeline_v2._resolve_batched(items, rules_client, finetuned_client, class_freq, BATCH_SIZE)

    layer_counts = {}
    for it in items:
        layer_counts[it["layer"]] = layer_counts.get(it["layer"], 0) + 1
    print(f"\nLayer firing rates (n=1000): "
         f"Layer1={layer_counts.get(pipeline_v2.LAYER_1_DETERMINISTIC,0)}, "
         f"Layer2={layer_counts.get(pipeline_v2.LAYER_2_GUARD,0)}, "
         f"Layer3={layer_counts.get(pipeline_v2.LAYER_3_LLM,0)}")

    # correct-abstention on unanswerable (has_ground_truth=False) cases, broken
    # down by WHICH layer actually handled them -- this is the honest version of
    # "Layer 2's job, not the LLM's": report what fraction of unanswerable cases
    # even REACH Layer 2 in the first place.
    unanswerable = [it for it in items if not it["has_ground_truth"]]
    layer_of_unanswerable = {}
    correct_abstention_by_layer = {}
    for it in unanswerable:
        layer = it["layer"]
        layer_of_unanswerable[layer] = layer_of_unanswerable.get(layer, 0) + 1
        abstained = is_abstention(it["assessment"].get("likely_intent", "")) if isinstance(it["assessment"], dict) else False
        if abstained:
            correct_abstention_by_layer[layer] = correct_abstention_by_layer.get(layer, 0) + 1

    print(f"\nUnanswerable cases (n={len(unanswerable)}) by layer reached:")
    for layer in (pipeline_v2.LAYER_1_DETERMINISTIC, pipeline_v2.LAYER_2_GUARD, pipeline_v2.LAYER_3_LLM):
        n_layer = layer_of_unanswerable.get(layer, 0)
        n_correct = correct_abstention_by_layer.get(layer, 0)
        pct_correct = f"{n_correct/n_layer:.1%}" if n_layer else "n/a"
        print(f"  {layer}: {n_layer} cases, {n_correct} correctly abstained ({pct_correct})")

    total_correct_abstention = sum(correct_abstention_by_layer.values())
    print(f"\nEND-TO-END correct-abstention-on-unanswerable rate (pipeline_v2+v5a): "
         f"{total_correct_abstention}/{len(unanswerable)} = {total_correct_abstention/len(unanswerable):.1%}")

    # end-to-end threat accuracy on has_ground_truth=True cases
    gt_items = [it for it in items if it["has_ground_truth"]]
    hits, scored, abstained_gt = 0, 0, 0
    for it in gt_items:
        a = it["assessment"]
        if not isinstance(a, dict):
            continue
        if is_abstention(a.get("likely_intent", "")):
            abstained_gt += 1
            continue
        scored += 1
        if a.get("threat_level", "").strip().lower() == it["expected_threat"]:
            hits += 1
    print(f"\nEnd-to-end threat accuracy (has_ground_truth=True, n={len(gt_items)}): "
         f"{hits}/{scored} = {hits/scored:.1%} (accuracy_when_answerable), "
         f"{abstained_gt}/{len(gt_items)} = {abstained_gt/len(gt_items):.1%} abstained (over-abstention)")

    Path(REPO / "evaluation" / "pipeline_v2_v5a_results_postfix.json").write_text(json.dumps({
        "seed": SEED, "n_sequences": N_SEQUENCES,
        "bucket_split": bucket_tally,
        "layer_firing": layer_counts,
        "unanswerable_by_layer": layer_of_unanswerable,
        "correct_abstention_by_layer": correct_abstention_by_layer,
        "end_to_end_correct_abstention_rate": total_correct_abstention / len(unanswerable),
        "end_to_end_accuracy_when_answerable": hits / scored if scored else None,
        "end_to_end_over_abstention_rate": abstained_gt / len(gt_items),
        "items": [{k: v for k, v in it.items() if k not in ("bucket_info", "key_windows")} for it in items],
    }, indent=2))
    print("\nsaved evaluation/pipeline_v2_v5a_results_postfix.json")


if __name__ == "__main__":
    main()
