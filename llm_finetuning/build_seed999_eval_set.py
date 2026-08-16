"""
V5a2 preregistration erratum part 2, step 1/2: builds a phase4_eval_set.json-SHAPED file from
eval_data/LOCKED_seed999_FINAL.json's already-existing raw trajectories (chain/positions/
true_labels), so bars f can be scored on this population without a new generation/inference
pipeline -- only the ALREADY-PROVEN building blocks generate_phase4_eval_set.py and
scripts/rule0_am_reproduce_headline.py already use are reused here: sliding_window_inference
+ classify_observation run directly on the locked file's own persisted `positions` (no new
trajectory generation -- unlike phase4_eval_set.json, which generates fresh seed=4321
trajectories, this script LOADS seed=999's trajectories, which already exist, verbatim).

Same has_ground_truth convention as generate_phase4_eval_set.py: len(true_chain) in {1,2} ->
True (RULES-pair answerable); >=3 -> False (abstention is correct behavior).

Integrity-gated: refuses to run if the locked file's sha256 has drifted from the cited value
(same discipline as scripts/rule0_am_reproduce_headline.py).

CPU-only, no training, no LLM call -- this only produces the STGT+bridge half (ctx/key_windows/
bucket/has_ground_truth) that a later LLM generation pass would consume, exactly mirroring
phase4_eval_set.json's own two-stage structure (generate_phase4_eval_set.py produces the
eval set; a separate v5a-generation run then consumes it).

Usage:
    python llm_finetuning/build_seed999_eval_set.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from build_sft_dataset import RULES  # noqa: E402

LOCKED_PATH = REPO / "eval_data" / "LOCKED_seed999_FINAL.json"
CITED_SHA = "871a9dae4c6fdf08e1aed803592fa7c61b1a852c150693b5819fe2271717b96e"
DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"
OUT_PATH = REPO / "evaluation" / "seed999_eval_set.json"


def ground_truth_from_true_chain(true_chain: list):
    """Identical convention to generate_phase4_eval_set.py."""
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    threat, intent, action = RULES[pair]
    return {"expected_threat": threat, "expected_intent": intent, "expected_action": action, "pair": list(pair)}


def main():
    raw = LOCKED_PATH.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    assert actual_sha == CITED_SHA, (
        f"INTEGRITY FAILURE: {LOCKED_PATH} sha256 {actual_sha} != cited {CITED_SHA} -- "
        f"refusing to build an eval set from a file that may have drifted since it was locked.")
    print(f"integrity gate: locked file sha256 matches cited hash ({actual_sha[:16]}...) -- OK")

    data = json.loads(raw)
    records = data["records"]
    assert data["seed"] == 999 and data["n"] == 1000 and len(records) == 1000

    import torch
    import swarm_intent.stgt.model as stgt_model_module
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    device = torch.device("cpu")
    stgt_model_module.device = device  # same monkeypatch as generate_phase4_eval_set.py

    ckpt_bytes = CHECKPOINT.read_bytes()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
    bridge_path = REPO / "src" / "swarm_intent" / "stgt_bridge.py"
    bridge_sha256 = hashlib.sha256(bridge_path.read_bytes()).hexdigest()
    print(f"checkpoint sha256: {ckpt_sha256}")
    print(f"stgt_bridge.py sha256: {bridge_sha256}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    print(f"=== scoring {len(records)} EXISTING seed=999 trajectories (no new generation) ===")
    reporter = Reporter("build_seed999_eval_set", len(records), rate_hint=4.0)
    items = []
    bucket_tally = Counter()
    n_gt = 0
    for rec in records:
        true_chain = [str(f) for f in rec["chain"]]
        long_seq = np.array(rec["positions"], dtype=np.float64)  # ALREADY EXISTS, not generated
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        bucket_tally[bucket_info["bucket"]] += 1
        gt = ground_truth_from_true_chain(true_chain)
        item = {
            "name": f"seed999_seq_{rec['i']}", "true_chain": true_chain,
            "spread": rec["spread"], "noise_std": rec["noise_std"],
            "ctx": bucket_info["context_text"], "key_windows": bucket_info["key_windows"],
            "bucket": bucket_info["bucket"], "has_ground_truth": gt is not None,
        }
        if gt is not None:
            item.update(gt)
            n_gt += 1
        items.append(item)
        reporter.update(1, item=f"seq {rec['i']}")
    reporter.status = "done"
    reporter._write()

    print(f"sequences with determinable ground truth: {n_gt}/{len(records)} ({n_gt/len(records):.1%})")
    print(f"bucket split: A={bucket_tally.get(BUCKET_A, 0)} B={bucket_tally.get(BUCKET_B, 0)} "
         f"C={bucket_tally.get(BUCKET_C, 0)}")

    payload = {
        "n_sequences": len(records), "seed": 999,
        "source_locked_file": str(LOCKED_PATH.relative_to(REPO)),
        "source_locked_file_sha256": actual_sha,
        "checkpoint_sha256": ckpt_sha256, "stgt_bridge_sha256": bridge_sha256,
        "items": items,
    }
    payload_json = json.dumps(payload, indent=2)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(payload_json)
    file_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
    print(f"\nsaved {OUT_PATH}")
    print(f"sha256 = {file_sha256}")


if __name__ == "__main__":
    main()
