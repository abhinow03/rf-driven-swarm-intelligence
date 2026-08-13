"""
Hardening audit, Layer-2-gap diagnosis, steps 1-2: step 5 found only 1/502
has_ground_truth=False cases reach Layer 2 (492 fall to Layer 3 and inherit
v5-a's 0.0% correct-abstention). This categorizes ALL 502 by structural
mechanism (step 1), then pulls the actual guard code path for the subset
that structurally should have been Layer-2-eligible but wasn't (step 2).

CPU-only, deterministic -- regenerates the exact locked seed=4321
population the same way eval_pipeline_v2_with_v5a.py / compute_same_
population_ceiling.py already did (0/0 mismatches confirmed twice), this
time keeping the FULL bucket_info (bucket/subtype/guard_reasons/summary)
per case instead of stripping it before saving.

Step 1 categories, six named by the instruction plus two the code can
actually produce that aren't on that list (reported, not hidden):
  Layer-2-eligible (guard code path WAS reached: len(known_history)<=2,
  not terminal/all-unknown -- see classify_observation's control flow):
    oov_name, dominant_history_contradiction, dispersed_converging_ambiguity,
    low_confidence (a 4th real guard condition, not named in the instruction
    but produced by the same code -- reported for completeness)
    -- PLUS bucket_A_misrouted: guard code path WAS reached, NO guard fired
       at all (this is step 2's subject)
  Layer-3-eligible (guard code path NEVER reached -- an early `return`
  inside classify_observation exits before any guard_reasons check runs):
    multi_hop, oscillation, terminal_unknown (== "terminal_transitioning"),
    all_unknown (a 4th real subtype, not named in the instruction -- the
    entire window set was unclassifiable, reported for completeness)

Step 2: for every bucket_A_misrouted case (guard code reached, nothing
fired), prints the exact summary fields each guard condition reads and
states in words which specific boolean evaluated False and why.

Usage (fast, ~1-2 min, CPU only):
    python llm_finetuning/categorize_unanswerable_502.py
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

from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
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


def categorize(bucket_info: dict) -> str:
    bucket = bucket_info["bucket"]
    if bucket == BUCKET_A:
        return "bucket_A_misrouted"
    if bucket == BUCKET_C:
        subtype = bucket_info["subtype"]
        return {"terminal_unknown": "terminal_transitioning"}.get(subtype, subtype)
    # bucket == BUCKET_B -- report the PRIMARY reason (same priority order as
    # AUDIT.md sec AG step 1: ambiguity dominates when it co-occurs, since it's
    # the one guard NEVER suppressed by robust recovery and the one most often
    # cited as the real upstream defect)
    reasons = bucket_info["guard_reasons"]
    for primary in ("dispersed_converging_ambiguity", "oov_name",
                    "dominant_history_contradiction", "low_confidence"):
        if primary in reasons:
            return primary
    return "other_bucket_B"  # defensive, should be unreachable (guard_reasons non-empty implies bucket B)


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

    print("=== regenerating the locked seed=4321 population, CPU-only, keeping full bucket_info ===")
    rng = np.random.default_rng(SEED)
    records = []
    chain_mismatches = bucket_mismatches = 0
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
            bucket_mismatches += 1

        gt = ground_truth_from_true_chain(true_chain)
        records.append({
            "i": i, "true_chain": true_chain, "has_ground_truth": gt is not None,
            "bucket": bucket_info["bucket"], "subtype": bucket_info["subtype"],
            "guard_reasons": bucket_info["guard_reasons"], "rules_key": bucket_info["rules_key"],
            "summary": bucket_info["summary"], "n_windows": len(predictions),
        })

    print(f"integrity: chain_mismatches={chain_mismatches} bucket_mismatches={bucket_mismatches} "
         f"(0/0 expected, matches step 3/5's confirmation)")
    if chain_mismatches or bucket_mismatches:
        print("STOPPING: a real divergence exists -- do not trust the categorization below.")
        return

    unanswerable = [r for r in records if not r["has_ground_truth"]]
    print(f"\nhas_ground_truth=False cases: {len(unanswerable)} (expected 502)")

    for r in unanswerable:
        r["category"] = categorize(
            {"bucket": r["bucket"], "subtype": r["subtype"], "guard_reasons": r["guard_reasons"]})

    print(f"\n=== STEP 1: category breakdown, n={len(unanswerable)} ===")
    print("| category | layer-eligible-by-design | n | % of 502 |")
    print("|---|---|---|---|")
    layer2_eligible = {"oov_name", "dominant_history_contradiction", "dispersed_converging_ambiguity",
                       "low_confidence"}
    layer3_eligible = {"multi_hop", "oscillation", "terminal_transitioning", "all_unknown"}
    cat_counts = Counter(r["category"] for r in unanswerable)
    for cat, k in cat_counts.most_common():
        elig = ("Layer 2 (guard code reached)" if cat in layer2_eligible
               else "Layer 3 (guard code NEVER reached)" if cat in layer3_eligible
               else "N/A -- bug (guard code reached, nothing fired)")
        print(f"| {cat} | {elig} | {k} | {k/len(unanswerable):.1%} |")

    n_layer2_eligible = sum(k for c, k in cat_counts.items() if c in layer2_eligible)
    n_layer3_eligible = sum(k for c, k in cat_counts.items() if c in layer3_eligible)
    n_misrouted = cat_counts.get("bucket_A_misrouted", 0)
    print(f"\nLayer-2-eligible total (landed in bucket B): {n_layer2_eligible}/{len(unanswerable)} "
         f"({n_layer2_eligible/len(unanswerable):.1%})")
    print(f"Layer-3-eligible-by-design total (landed in bucket C): {n_layer3_eligible}/{len(unanswerable)} "
         f"({n_layer3_eligible/len(unanswerable):.1%})")
    print(f"bucket_A_misrouted (genuine bug, guard code reached but nothing fired): "
         f"{n_misrouted}/{len(unanswerable)} ({n_misrouted/len(unanswerable):.1%})")

    print(f"\n=== STEP 2: bucket_A_misrouted cases -- guard code path pulled, exact condition that failed ===")
    misrouted = [r for r in unanswerable if r["category"] == "bucket_A_misrouted"]
    print(f"n={len(misrouted)}\n")
    print("Guard code (src/swarm_intent/coverage.py:100-135), evaluated for each case below:")
    print("""
    if summary["n_genuinely_oov_windows"] > 0 and not robust_recovered:
        guard_reasons.append("oov_name")
    if len(known_history) == 2 and not robust_recovered:
        if summary["dominant_formation"] not in (key[0], key[1]):
            guard_reasons.append("dominant_history_contradiction")
    if summary["n_ambiguous_dispersed_converging_windows"] > 0:
        guard_reasons.append("dispersed_converging_ambiguity")
    if predictions and all(p["formation_confidence"] < 0.6 for p in predictions):
        guard_reasons.append("low_confidence")
    """)
    for r in misrouted:
        s = r["summary"]
        n_oov = s.get("n_genuinely_oov_windows", "MISSING")
        n_amb = s.get("n_ambiguous_dispersed_converging_windows", "MISSING")
        dom = s.get("dominant_formation")
        key = tuple(r["rules_key"]) if r["rules_key"] else None
        print(f"  seq {r['i']:4d}  true_chain={r['true_chain']}  rules_key={key}")
        print(f"    n_genuinely_oov_windows={n_oov} (oov_name needs >0 -- "
             f"{'FIRED' if isinstance(n_oov,int) and n_oov>0 else 'condition False: 0 windows read as a genuinely OOV name'})")
        dom_contra_note = ("condition False: dominant IS one of the key formations "
                          "(provably always true here, per the guard's own comment)")
        print(f"    dominant_formation={dom!r}, key={key} -- dominant_history_contradiction needs "
             f"dominant NOT IN key -- {'FIRED' if key and dom not in key else dom_contra_note}")
        print(f"    n_ambiguous_dispersed_converging_windows={n_amb} -- "
             f"{'FIRED' if isinstance(n_amb,int) and n_amb>0 else 'condition False: 0 windows had a near-tie dispersed/converging read'}")
        low_conf = s.get("low_conf_windows"), s.get("n_windows")
        print(f"    low_conf_windows={low_conf[0]}/{low_conf[1]} -- low_confidence needs ALL windows <0.6 -- "
             f"{'FIRED' if low_conf[0]==low_conf[1] and low_conf[1] else 'condition False: not every window was low-confidence'}")
        print(f"    -> the model's OWN read of this true {len(r['true_chain'])}-formation sequence "
             f"collapsed to known_history={key} with NONE of the 4 guard conditions true; "
             f"real answer was thrown away as a confident-looking bucket-A pair, not caught anywhere.\n")

    out = {"n_unanswerable": len(unanswerable), "category_counts": dict(cat_counts),
          "layer2_eligible_total": n_layer2_eligible, "layer3_eligible_total": n_layer3_eligible,
          "bucket_A_misrouted_total": n_misrouted,
          "unanswerable_records": [{k: v for k, v in r.items() if k != "summary"} for r in unanswerable],
          "misrouted_detail": [{"i": r["i"], "true_chain": r["true_chain"], "rules_key": r["rules_key"],
                                "summary": r["summary"]} for r in misrouted]}
    out_path = REPO / "evaluation" / "categorize_unanswerable_502.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
