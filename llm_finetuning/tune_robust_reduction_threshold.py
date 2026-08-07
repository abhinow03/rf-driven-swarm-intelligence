"""
AUDIT.md sec AG step 2: tunes the majority-plurality threshold the robust
reduction (src/swarm_intent/stgt_bridge.py, `robust=True`) uses, on a
SEPARATE dev split -- seed=1, NEVER seed=0 (the eval seed sec AE/AF/AG's
500-sequence held-out set uses throughout this whole thread). This script
is the ONLY place the threshold is chosen; every other script imports the
frozen result as a constant. See AUDIT.md sec AG step 4 for the standing
check that dev and held-out accuracy don't diverge (the threshold-overfitting
guard sec AG step 4 explicitly requires).

The robust-reduction algorithm implemented here is EXACTLY what gets ported
into stgt_bridge.py's `_robust_reduce`/`_robust_all_unknown_fallback` once a
threshold is chosen -- this script is the tuning harness, not a different
approximation of the real logic, so "tuned on dev" and "shipped in
production" are provably the same code.

Usage (run inside tmux):
    python llm_finetuning/tune_robust_reduction_threshold.py
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

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

DEV_SEED = 1  # NEVER 0 -- 0 is the held-out eval seed used throughout secs AE/AF/AG
N_DEV_SEQUENCES = 300
UNKNOWN_FORMATION = "unknown"
ALL_UNKNOWN_FALLBACK_MARGIN = 0.05  # fixed, not swept -- see module docstring in stgt_bridge.py
THRESHOLD_SWEEP = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def _validate_formation(name) -> str:
    return name if name in BASE_FORMATIONS else UNKNOWN_FORMATION


def robust_reduce(formation_seq, predictions, threshold):
    """The exact algorithm ported into stgt_bridge.py's _robust_reduce. Returns
    (dominant, formation_history, transitions, info) or None."""
    n = len(formation_seq)
    start = 0
    while start < n and formation_seq[start] == UNKNOWN_FORMATION:
        start += 1
    end = n
    while end > start and formation_seq[end - 1] == UNKNOWN_FORMATION:
        end -= 1
    stripped = formation_seq[start:end]
    n_lead, n_trail = start, n - end

    if not stripped:
        return _all_unknown_fallback(predictions, n_lead, n_trail)

    if len(stripped) == 1:
        f = stripped[0]
        return (f, [f], [], {"stripped_leading": n_lead, "stripped_trailing": n_trail,
                             "mode": "single_window", "recovered": True, "low_confidence": True})

    mid = len(stripped) // 2
    first_half, second_half = stripped[:mid], stripped[mid:]

    def modal(seq):
        """Modal NON-unknown formation -- UNKNOWN_FORMATION must never itself win
        the vote (a half dominated by transitioning/OOV windows should FAIL the
        threshold, not get reported as if "unknown" were a real formation).
        Denominator is the full half length (unknowns count against the
        plurality fraction), not just the non-unknown subset -- otherwise the
        threshold couldn't reject a mostly-noisy half at all."""
        counts = Counter(f for f in seq if f != UNKNOWN_FORMATION)
        if not counts:
            return None, 0.0
        f, count = counts.most_common(1)[0]
        return f, count / len(seq)

    f1, frac1 = modal(first_half)
    f2, frac2 = modal(second_half)
    if f1 is None or f2 is None or frac1 < threshold or frac2 < threshold:
        return None

    info = {"stripped_leading": n_lead, "stripped_trailing": n_trail, "mode": "majority_halves",
           "frac_first_half": round(frac1, 4), "frac_second_half": round(frac2, 4),
           "recovered": True, "low_confidence": False}
    if f1 == f2:
        return (f1, [f1], [], info)
    dominant = f1 if frac1 >= frac2 else f2
    return (dominant, [f1, f2], [{"from": f1, "to": f2}], info)


def _all_unknown_fallback(predictions, n_lead, n_trail):
    sums = {f: 0.0 for f in BASE_FORMATIONS}
    counts = 0
    for p in predictions:
        cp = p.get("class_probabilities", {})
        if not cp:
            continue
        counts += 1
        for f in BASE_FORMATIONS:
            sums[f] += cp.get(f, 0.0)
    if counts == 0:
        return None
    means = {f: sums[f] / counts for f in BASE_FORMATIONS}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    top1, top1_p = ranked[0]
    top2, top2_p = ranked[1]
    margin = top1_p - top2_p
    if margin < ALL_UNKNOWN_FALLBACK_MARGIN:
        return None
    info = {"stripped_leading": n_lead, "stripped_trailing": n_trail,
           "mode": "all_unknown_probability_fallback", "top1": top1, "top1_p": round(top1_p, 4),
           "top2": top2, "top2_p": round(top2_p, 4), "margin": round(margin, 4),
           "recovered": True, "low_confidence": True}
    return (top1, [top1], [], info)


def ground_truth_pair(true_chain):
    if len(true_chain) == 1:
        return (true_chain[0], true_chain[0])
    if len(true_chain) == 2:
        return (true_chain[0], true_chain[1])
    return None


def main():
    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(DEV_SEED)
    reporter = Reporter("tune_robust_reduction_threshold", N_DEV_SEQUENCES, rate_hint=8.0)

    dev_cases = []
    for i in range(N_DEV_SEQUENCES):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        gt = ground_truth_pair(chain)
        formation_seq = [_validate_formation(p["formation_type"]) for p in predictions]
        dev_cases.append({"i": i, "true_chain": chain, "gt_pair": list(gt) if gt else None,
                          "formation_seq": formation_seq,
                          "predictions_probs": [p.get("class_probabilities", {}) for p in predictions]})
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    gt_cases = [c for c in dev_cases if c["gt_pair"] is not None]
    print(f"\ndev set: {len(dev_cases)} sequences, {len(gt_cases)} with GT (len(true_chain)<=2)")

    print("\n=== threshold sweep (dev set ONLY) ===")
    print("| threshold | recovered | recovery_rate | correct | precision |")
    print("|---|---|---|---|---|")
    sweep_results = []
    for threshold in THRESHOLD_SWEEP:
        recovered, correct = 0, 0
        for c in gt_cases:
            # reconstruct minimal "predictions" list shape robust_reduce/_all_unknown_fallback need
            fake_preds = [{"class_probabilities": p} for p in c["predictions_probs"]]
            result = robust_reduce(c["formation_seq"], fake_preds, threshold)
            if result is None:
                continue
            recovered += 1
            _, history, _, _ = result
            recovered_pair = (history[0], history[0]) if len(history) == 1 else (history[0], history[1])
            if recovered_pair == tuple(c["gt_pair"]):
                correct += 1
        recovery_rate = recovered / len(gt_cases)
        precision = correct / recovered if recovered else 0.0
        sweep_results.append({"threshold": threshold, "recovered": recovered, "n": len(gt_cases),
                              "recovery_rate": recovery_rate, "correct": correct, "precision": precision})
        print(f"| {threshold} | {recovered}/{len(gt_cases)} | {recovery_rate:.1%} | "
             f"{correct}/{recovered if recovered else 1} | {precision:.1%} |")

    # Selection rule, stated up front: the LOWEST threshold (= highest recovery) whose
    # precision among recovered cases is still >= 90% -- prioritizes NOT trading real
    # answers for wrong ones (sec AG step 3's explicit failure mode to avoid), while
    # recovering as much as that constraint allows.
    PRECISION_FLOOR = 0.90
    eligible = [r for r in sweep_results if r["precision"] >= PRECISION_FLOOR]
    if eligible:
        chosen = min(eligible, key=lambda r: r["threshold"])
    else:
        chosen = max(sweep_results, key=lambda r: r["precision"])
        print(f"\nWARNING: no threshold reached {PRECISION_FLOOR:.0%} precision -- "
             f"falling back to the highest-precision threshold measured.")

    print(f"\n=== CHOSEN THRESHOLD: {chosen['threshold']} ===")
    print(f"dev recovery_rate={chosen['recovery_rate']:.1%}, dev precision={chosen['precision']:.1%} "
         f"(selection rule: lowest threshold with precision >= {PRECISION_FLOOR:.0%})")

    # Diagnose the precision ceiling: is it the dispersed/converging defect (2d says
    # leave alone) or something else, at the chosen threshold?
    DISPERSED_CONVERGING_MARGIN = 0.15
    mismatch_kinds = Counter()
    for c in gt_cases:
        fake_preds = [{"class_probabilities": p} for p in c["predictions_probs"]]
        result = robust_reduce(c["formation_seq"], fake_preds, chosen["threshold"])
        if result is None:
            continue
        _, history, _, _ = result
        recovered_pair = (history[0], history[0]) if len(history) == 1 else (history[0], history[1])
        if recovered_pair == tuple(c["gt_pair"]):
            continue
        involves_dc = {"dispersed", "converging"} & (set(recovered_pair) | set(c["gt_pair"]))
        mismatch_kinds["involves_dispersed_converging" if involves_dc else "other_formation_confusion"] += 1
    n_mismatch = sum(mismatch_kinds.values())
    print(f"\n=== mismatch diagnosis at chosen threshold (n_wrong={n_mismatch}) ===")
    for kind, k in mismatch_kinds.most_common():
        print(f"  {kind}: {k}/{n_mismatch} ({k/n_mismatch:.1%})" if n_mismatch else f"  {kind}: 0")

    # step 2d: the dispersed/converging ambiguity guard is UNCONDITIONAL and applies
    # on top of robust recovery -- a case with an ambiguous window routes to bucket B
    # regardless of what robust_reduce recovered. Precision AS MEASURED ABOVE ignores
    # this downstream guard entirely (worst-case bound); precision AFTER the guard is
    # the number that actually reaches Layer 1 in production. Recompute it here.
    def has_ambiguous_window(probs_list):
        for cp in probs_list:
            d, c = cp.get("dispersed"), cp.get("converging")
            if d is not None and c is not None and abs(d - c) < DISPERSED_CONVERGING_MARGIN:
                return True
        return False

    would_be_a, would_be_a_correct = 0, 0
    would_be_b = 0
    for c in gt_cases:
        fake_preds = [{"class_probabilities": p} for p in c["predictions_probs"]]
        result = robust_reduce(c["formation_seq"], fake_preds, chosen["threshold"])
        if result is None:
            continue
        if has_ambiguous_window(c["predictions_probs"]):
            would_be_b += 1
            continue
        would_be_a += 1
        _, history, _, _ = result
        recovered_pair = (history[0], history[0]) if len(history) == 1 else (history[0], history[1])
        if recovered_pair == tuple(c["gt_pair"]):
            would_be_a_correct += 1

    print(f"\n=== precision AFTER the (unconditional) dispersed/converging ambiguity guard ===")
    print(f"  of {chosen['recovered']} robustly-recovered cases: {would_be_b} would route to bucket B "
         f"(ambiguous window present, guard still applies per step 2d)")
    print(f"  {would_be_a} would actually reach bucket A -- precision there: "
         f"{would_be_a_correct}/{would_be_a} ({would_be_a_correct/would_be_a:.1%})" if would_be_a
         else "  0 would reach bucket A")

    out = {"dev_seed": DEV_SEED, "n_dev_sequences": N_DEV_SEQUENCES, "n_dev_gt_cases": len(gt_cases),
          "sweep": sweep_results, "chosen_threshold": chosen["threshold"],
          "precision_floor": PRECISION_FLOOR, "chosen_dev_recovery_rate": chosen["recovery_rate"],
          "chosen_dev_precision": chosen["precision"]}
    out_path = REPO / "evaluation" / "robust_reduction_threshold_tuning.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
