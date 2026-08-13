"""
AUDIT.md sec AN, steps 1-2: Layer-2 multi-hop/oscillation extension, investigation phase.
No fixes here -- pure diagnosis, per the LOCKED CONFIG's "investigate before touching."

Step 1: confirms the existing structural-mechanism classifier (multi_hop / oscillation /
terminal_transitioning / OOV / contradiction / dispersed_converging) is
llm_finetuning/categorize_unanswerable_502.py's categorize() function, built ON TOP of
src/swarm_intent/coverage.py's classify_observation() output (specifically: subtype ==
"multi_hop"/"oscillation" can ONLY ever be produced by coverage.py's
`if len(known_history) >= 3: return _c(...)` early return, coverage.py:121-122). This is a
STANDALONE diagnostic script -- its categorize() function is never imported by pipeline_v2.py
or coverage.py, so it is not reachable from the live pipeline today.

Step 2: regenerates the locked seed=4321 population WITH raw summary/formation_history
retained (categorize_unanswerable_502.py's saved JSON strips "summary" before writing), then
for EVERY has_ground_truth=False case independently recomputes known_history directly from
summary["formation_history"] -- the SAME collapse coverage.py itself performs at line 119,
just run a second time here regardless of which early return coverage.py actually took.
Reports:
  (a) confirmation that ALL subtype=="multi_hop"/"oscillation" cases hit the len>=3 early
      return (tautological -- that subtype value has no other source -- but verified by
      recomputing known_history for every one of them and checking length>=3 holds).
  (b) any OTHER case (terminal_unknown / all_unknown) that ALSO has known_history>=3 hidden
      underneath -- i.e. coverage.py's early-return ORDER (terminal_unknown's
      `history[-1]==UNKNOWN_FORMATION` check runs BEFORE the len>=3 check) silently discarded
      real multi-hop/oscillation signal for that case.

No code changes in this script -- diagnosis only.

Usage (CPU-only, ~1-2 min):
    python llm_finetuning/investigate_layer2_early_returns.py
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

from swarm_intent.coverage import classify_observation, _collapse_consecutive, BUCKET_A, BUCKET_C  # noqa: E402
from swarm_intent.stgt_bridge import UNKNOWN_FORMATION  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 1000
SEED = 4321
LOCKED_PATH = REPO / "evaluation" / "phase4_eval_set.json"

# AUDIT.md sec AK/AL's guard-logic fix (MAX_WINDOWS_PER_SINGLE_TRANSITION) landed AFTER
# evaluation/phase4_eval_set.json was locked -- these 9 cases are EXPECTED to now bucket
# differently (A->B) against that stale file, verified case-by-case in verify_guard_fix.py.
# Same narrowed acceptance as llm_finetuning/eval_pipeline_v2_with_v5a_postfix.py.
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


def recompute_known_history(summary: dict):
    history = (summary or {}).get("formation_history") or []
    return _collapse_consecutive([f for f in history if f != UNKNOWN_FORMATION])


def structural_label(known_history: list):
    if len(known_history) < 3:
        return None
    return "oscillation" if known_history[0] == known_history[-1] else "multi_hop"


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

    print("=== regenerating locked seed=4321 population, keeping RAW summary this time ===")
    rng = np.random.default_rng(SEED)
    records = []
    chain_mismatches = 0
    bucket_mismatch_indices = []
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
        known_history_recomputed = recompute_known_history(bucket_info["summary"])
        records.append({
            "i": i, "true_chain": true_chain, "has_ground_truth": gt is not None,
            "bucket": bucket_info["bucket"], "subtype": bucket_info["subtype"],
            "formation_history": bucket_info["summary"].get("formation_history"),
            "known_history_recomputed": known_history_recomputed,
            "structural_label": structural_label(known_history_recomputed),
        })

    unexpected = set(bucket_mismatch_indices) - EXPECTED_BUCKET_MISMATCHES
    missing = EXPECTED_BUCKET_MISMATCHES - set(bucket_mismatch_indices)
    print(f"integrity: chain_mismatches={chain_mismatches}, "
         f"bucket_mismatches={len(bucket_mismatch_indices)} (9 expected -- sec AK/AL's "
         f"guard-logic fix, verified in verify_guard_fix.py), unexpected={sorted(unexpected)}, "
         f"missing={sorted(missing)}")
    if chain_mismatches or unexpected or missing:
        print("STOPPING: a real, unexplained divergence exists -- do not trust the investigation below.")
        return

    unanswerable = [r for r in records if not r["has_ground_truth"]]
    print(f"\nhas_ground_truth=False cases: {len(unanswerable)} (expected 502)")

    print("\n=== STEP 2a: does subtype==multi_hop/oscillation always match the recomputed "
         "structural_label? (tautology check) ===")
    mismatch = [r for r in unanswerable if r["subtype"] in ("multi_hop", "oscillation")
               and r["structural_label"] != r["subtype"]]
    n_mh_os = sum(1 for r in unanswerable if r["subtype"] in ("multi_hop", "oscillation"))
    print(f"subtype in (multi_hop, oscillation): {n_mh_os} cases; "
         f"structural_label mismatch: {len(mismatch)} (expected 0 -- these subtypes have no "
         f"other source than the len(known_history)>=3 early return)")

    print("\n=== STEP 2b: cases where subtype != multi_hop/oscillation, but the structural "
         "check STILL finds known_history>=3 underneath -- signal masked by early-return ORDER ===")
    masked = [r for r in unanswerable if r["subtype"] not in ("multi_hop", "oscillation")
             and r["structural_label"] is not None]
    print(f"masked cases: {len(masked)}")
    for r in masked:
        print(f"  seq {r['i']:4d}  true_chain={r['true_chain']}  subtype={r['subtype']!r}  "
             f"formation_history={r['formation_history']}  "
             f"recomputed_known_history={r['known_history_recomputed']}  "
             f"-> structurally {r['structural_label']}")

    print("\n=== STEP 2c: full category breakdown (subtype vs recomputed structural_label) ===")
    subtype_counts = Counter(r["subtype"] for r in unanswerable)
    print("subtype counts:", dict(subtype_counts))
    structural_counts = Counter(r["structural_label"] or "none" for r in unanswerable)
    print("recomputed structural_label counts:", dict(structural_counts))

    out = {
        "n_unanswerable": len(unanswerable),
        "n_subtype_multihop_or_oscillation": n_mh_os,
        "n_tautology_mismatches": len(mismatch),
        "n_masked_by_early_return_order": len(masked),
        "masked_cases": [{"i": r["i"], "true_chain": r["true_chain"], "subtype": r["subtype"],
                          "formation_history": r["formation_history"],
                          "known_history_recomputed": r["known_history_recomputed"],
                          "structural_label": r["structural_label"]} for r in masked],
        "subtype_counts": dict(subtype_counts),
        "structural_label_counts": dict(structural_counts),
        "all_records": records,
    }
    out_path = REPO / "evaluation" / "layer2_early_return_investigation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
