"""
Generates and LOCKS the Phase 4 preregistered eval set (docs/PREREGISTRATION.md):
1,000 real STGT trajectories, fresh seed, ground truth from the generator's
true chain only. Depends ONLY on the frozen STGT checkpoint (sha256
18fc201d...) and the generator -- NOT on v5-a, so it can be built and locked
before any v5-a number exists to tempt tuning against it.

FORCED CPU: v5-a training is running in a separate tmux session on the only
GPU. STGT itself is small enough that CPU inference is entirely practical for
1,000 trajectories, and running it there guarantees zero GPU contention with
training -- torch.device("cpu") is passed explicitly below, overriding
swarm_intent.stgt.config's own cuda-if-available default.

SEED=4321: fresh, disjoint from every seed already used elsewhere in this
project's history (0 coverage-measurement/eval_real_stgt_output, 1 retired
dev split, 2024 mining-sweep, 7/10/11 generator diagnostics, 42 training
split, 999 ceiling, 5001 val/mining split).

Ground-truth derivation: identical convention to
eval_real_stgt_output.py's ground_truth_from_true_chain -- RULES looked up on
the TRUE chain the generator was told to build, captured before any model
sees the sequence, NEVER derived from the STGT model's own bridge output.
len(true_chain) in {1,2} -> has_ground_truth=True; >=3 -> False (abstention is
correct behavior).

Usage:
    python llm_finetuning/generate_phase4_eval_set.py
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
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
OUT_PATH = REPO / "evaluation" / "phase4_eval_set.json"


def ground_truth_from_true_chain(true_chain: list):
    """Identical convention to eval_real_stgt_output.py -- see module docstring."""
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    threat, intent, action = RULES[pair]
    return {"expected_threat": threat, "expected_intent": intent, "expected_action": action, "pair": list(pair)}


def main():
    import torch
    import swarm_intent.stgt.model as stgt_model_module
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    device = torch.device("cpu")  # forced -- see module docstring, no GPU contention with v5-a training
    assert not torch.cuda.is_available() or torch.cuda.memory_allocated() == 0, \
        "unexpected GPU memory already allocated before this CPU-only script started"

    # DISCOVERED BUG (flagged, not silently patched): stgt/model.py line ~194
    # (`batched = Batch.from_data_list(all_graphs).to(device)`) imports and uses
    # config.device (cuda-if-available) directly, NOT the device the model's own
    # parameters were actually placed on -- contradicting CLAUDE.md's documented
    # invariant "the model infers its device from its own parameters (no global
    # device)". On a machine with a visible GPU, this means a graph batch gets
    # forced onto cuda:0 even when the model itself is on CPU, crashing with a
    # device-mismatch RuntimeError (confirmed: this crashed on first run before
    # this patch was added). Monkeypatching the name bound in that module's own
    # namespace here (not editing the shared source file while v5-a training is
    # running elsewhere) is the safe workaround for THIS script; the real fix
    # (batch device should be derived from the model's own parameters, not the
    # global import) is flagged in V5_LOG.md as a follow-up, not applied here.
    stgt_model_module.device = device

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

    print(f"=== generating {N_SEQUENCES} fresh trajectories (seed={SEED}, CPU-only) ===")
    rng = np.random.default_rng(SEED)
    reporter = Reporter("generate_phase4_eval_set", N_SEQUENCES, rate_hint=4.0)
    items = []
    bucket_tally = Counter()
    n_gt = 0
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        bucket_tally[bucket_info["bucket"]] += 1
        true_chain = [str(f) for f in chain]
        gt = ground_truth_from_true_chain(true_chain)
        item = {
            "name": f"phase4_seq_{i}", "true_chain": true_chain,
            "spread": spread, "noise_std": noise_std,
            "ctx": bucket_info["context_text"], "key_windows": bucket_info["key_windows"],
            "bucket": bucket_info["bucket"], "has_ground_truth": gt is not None,
        }
        if gt is not None:
            item.update(gt)
            n_gt += 1
        items.append(item)
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()

    print(f"sequences with determinable ground truth: {n_gt}/{N_SEQUENCES} ({n_gt/N_SEQUENCES:.1%})")
    print(f"bucket split: A={bucket_tally.get(BUCKET_A, 0)} B={bucket_tally.get(BUCKET_B, 0)} "
         f"C={bucket_tally.get(BUCKET_C, 0)}")

    payload = {
        "n_sequences": N_SEQUENCES, "seed": SEED,
        "checkpoint_sha256": ckpt_sha256, "stgt_bridge_sha256": bridge_sha256,
        "items": items,
    }
    payload_json = json.dumps(payload, indent=2)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(payload_json)
    file_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
    print(f"\nsaved {OUT_PATH}")
    print(f"LOCK: {OUT_PATH.name} sha256 = {file_sha256}")
    print("Record this hash in docs/V5_STATE.json's post_training_prep key -- this file must "
         "not change after being locked (same discipline as the checkpoint-freshness lock).")

    post_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert post_alloc == 0, f"this CPU-only script somehow allocated {post_alloc} bytes of GPU memory"
    print("Confirmed: torch.cuda.memory_allocated() == 0 (this script never touched the GPU).")


if __name__ == "__main__":
    main()
