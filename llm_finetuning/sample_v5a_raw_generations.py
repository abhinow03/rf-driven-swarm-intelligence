"""
Hardening audit, Layer-2-gap diagnosis, step 3b: samples 20 raw v5-a
generations on REAL eval cases from the Layer-3-eligible has_ground_truth=
False subset (multi_hop/terminal_transitioning/oscillation -- the 492
cases step 1 confirmed are architecture-scoped to Layer 3, not a guard
bug), prints them verbatim, and checks specifically for the same bug CLASS
step 1 of the prior hardening session found (a parseable signal existing
in the raw text that the eval harness's own field-based parser misses) --
here: does v5-a ever produce abstention-like PROSE ("insufficient
information", "cannot determine", etc.) while its likely_intent field
still names a concrete (non-abstention) intent?

Same locked seed=4321 regeneration as categorize_unanswerable_502.py
(CPU-only, deterministic, 0/0 integrity mismatches already confirmed
twice) -- this run additionally keeps context_text/key_windows for the
20 sampled cases (dropped in the prior script to keep its output small).

Usage:
    python llm_finetuning/sample_v5a_raw_generations.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_C  # noqa: E402
from swarm_intent.inference import build_llm_prompt  # noqa: E402
from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402
from eval_sft_v5 import resolve_adapter_path  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
N_SAMPLE = 20
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V5A_ADAPTER_DIR = str(REPO / "checkpoints" / "v5_sft")

ABSTENTION_PHRASES = ("insufficient information", "insufficient data", "cannot determine",
                     "unable to determine", "unable to resolve", "cannot be determined",
                     "not enough information", "cannot confidently", "unclear from",
                     "indeterminate", "no reliable", "not possible to determine")


def ground_truth_from_true_chain(true_chain: list):
    if len(true_chain) >= 3:
        return None
    pair = (true_chain[0], true_chain[0]) if len(true_chain) == 1 else (true_chain[0], true_chain[1])
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

    print("=== regenerating locked seed=4321 population, sampling Layer-3-eligible cases ===")
    rng = np.random.default_rng(SEED)
    sampled = []
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        true_chain = [str(f) for f in chain]
        has_gt = ground_truth_from_true_chain(true_chain) is not None

        if (not has_gt and bucket_info["bucket"] == BUCKET_C
                and bucket_info["subtype"] in ("multi_hop", "oscillation", "terminal_unknown")
                and len(sampled) < N_SAMPLE):
            sampled.append({"i": i, "true_chain": true_chain, "subtype": bucket_info["subtype"],
                            "ctx": bucket_info["context_text"], "key_windows": bucket_info["key_windows"]})

    print(f"sampled {len(sampled)} Layer-3-eligible cases (target {N_SAMPLE})")

    del stgt_model
    import gc
    gc.collect()

    print("\n=== loading v5-a adapter, greedy decode (matches preregistered headline-number protocol) ===")
    v5a_path = resolve_adapter_path(V5A_ADAPTER_DIR)
    client = LocalHFClient(BASE_MODEL, adapter_path=v5a_path, temperature=0.0)

    prompts = [build_llm_prompt(pipeline_v2._preds_from_key_windows(s["key_windows"]), s["ctx"], {})
              for s in sampled]
    raw_texts = client.generate_batch(prompts, batch_size=8)

    results = []
    n_mismatch = 0
    for s, text in zip(sampled, raw_texts):
        try:
            parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            parsed = None
        formal_abstain = is_abstention(parsed.get("likely_intent", "")) if isinstance(parsed, dict) else False
        text_lower = text.lower()
        raw_abstain_signal = [p for p in ABSTENTION_PHRASES if p in text_lower]
        mismatch = bool(raw_abstain_signal) and not formal_abstain
        if mismatch:
            n_mismatch += 1
        results.append({"i": s["i"], "true_chain": s["true_chain"], "subtype": s["subtype"],
                        "raw_text": text, "parsed": parsed, "formal_abstain": formal_abstain,
                        "raw_abstain_signal": raw_abstain_signal, "mismatch": mismatch})

    print(f"\n=== {len(results)} verbatim generations ===\n")
    for r in results:
        print(f"--- seq {r['i']} | true_chain={r['true_chain']} | subtype={r['subtype']} ---")
        print(r["raw_text"])
        print(f"[formal_abstain={r['formal_abstain']} | raw_abstain_phrases={r['raw_abstain_signal']} "
             f"| MISMATCH={r['mismatch']}]\n")

    print(f"=== SUMMARY: {n_mismatch}/{len(results)} cases show raw-text abstention language "
         f"NOT captured by the formal likely_intent parser ===")

    out_path = REPO / "evaluation" / "sample_v5a_raw_generations.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
