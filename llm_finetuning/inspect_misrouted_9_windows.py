"""
Diagnostic (not committed as part of the fix): regenerates the locked
seed=4321 population and, for the 9 bucket_A_misrouted sequences identified
in AUDIT.md sec AK, prints EVERY window's formation_type + full
class_probabilities -- to find an OBSERVABLE signal (available from
`predictions` alone, no ground truth) that a distinct intermediate state
was dropped, before designing coverage.py's fix.

Usage:
    python llm_finetuning/inspect_misrouted_9_windows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_A  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
TARGET_INDICES = {105, 291, 292, 314, 385, 520, 574, 611, 982}


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

    rng = np.random.default_rng(SEED)
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        if i not in TARGET_INDICES:
            continue

        bucket_info = classify_observation(predictions)
        assert bucket_info["bucket"] == BUCKET_A, f"seq {i} is not bucket A anymore -- regime changed?"
        true_chain = [str(f) for f in chain]
        key = bucket_info["rules_key"]
        print(f"\n=== seq {i} | true_chain={true_chain} | rules_key={key} | n_windows={len(predictions)} ===")
        for w, p in enumerate(predictions):
            cp = p.get("class_probabilities", {})
            ranked = sorted(cp.items(), key=lambda kv: kv[1], reverse=True)
            top3 = ", ".join(f"{f}={v:.3f}" for f, v in ranked[:3])
            print(f"  w{w:2d} t={p['time_start_s']:>6.1f}s argmax={p['formation_type']:14s} "
                 f"conf={p['formation_confidence']:.3f}  top3=[{top3}]")


if __name__ == "__main__":
    main()
