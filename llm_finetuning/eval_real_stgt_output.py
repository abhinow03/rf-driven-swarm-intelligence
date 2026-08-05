"""
AUDIT.md sec AF step 2: the headline evaluation. Sec AE's clean-battery
figures are a construction artefact (sec AF step 1) -- this is the number
that actually bears on generalization.

Regenerates the IDENTICAL 500 sequences from sec AE step 2
(llm_finetuning/measure_coverage.py, same seed=0, same sample_chain /
build_long_sequence functions, imported not duplicated), runs the real
trained STGT + stgt_bridge exactly as that script did, and evaluates
pipeline_v2, v2, rules_in_prompt, and v3b-fix against it.

GROUND-TRUTH DERIVATION (read this before trusting any number below):
Ground truth threat_level/likely_intent/recommended_action is looked up from
RULES ONLY using each sequence's TRUE, KNOWN formation chain -- the exact
list of formations `measure_coverage.py`'s generator was TOLD to build (`
chain` in `sample_chain`/`build_long_sequence`), captured BEFORE any model
ever sees the sequence. It is NEVER derived from stgt_bridge's own
`bridge_predictions` output (dominant_formation / formation_history / the
bucket's own `rules_key`) -- that output comes from the SAME noisy STGT
classification every system under test also consumes, so using it as an
answer key would silently launder classifier error into the ground truth
and make every system's score partly a measure of agreement with its own
input rather than with reality. RULES itself is not part of the code path
under test (it is a static domain-policy lookup, consulted here on the true
chain, not on any model's read of it) -- this is the exact same "ground
truth = RULES on the TRUE formation_a/formation_b" convention every existing
battery in this project (TEST_CASES, the degradation battery) already uses;
this section just applies it to real STGT output instead of templated text.
  - len(true_chain) == 1 (steady state): has_ground_truth=True,
    expected = RULES[(f, f)].
  - len(true_chain) == 2 (one real transition): has_ground_truth=True,
    expected = RULES[(chain[0], chain[1])].
  - len(true_chain) >= 3: has_ground_truth=False -- RULES has no 3+-tuple
    key and none is claimed to exist; correct behaviour is abstention, and
    it is scored that way (the SAME has_ground_truth=False convention
    evaluate_llm/degradation.py already use), not silently dropped or
    scored against an invented multi-hop label.

Usage (run inside tmux):
    python llm_finetuning/eval_real_stgt_output.py
"""
from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.prompts import is_abstention  # noqa: E402
from swarm_intent.inference import build_llm_prompt  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402

from baselines import load_rules_txt  # noqa: E402
from build_sft_dataset import RULES  # noqa: E402
from measure_coverage import sample_chain, build_long_sequence, DATA_DIR, CHECKPOINT  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V2_ADAPTER = "adapters/qwen-swarm-v2"
V3B_FIX_ADAPTER = "adapters/qwen-swarm-v3b-fix"
BATCH_SIZE = 8
N_SEQUENCES = 500
SEED = 0

THREAT_ORDER = ("low", "medium", "high", "critical")


def wilson_ci95(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def normalize_threat(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for level in ("low", "medium", "high", "critical"):
        if level in raw:
            return level
    return "unparsed"


def ground_truth_from_true_chain(true_chain: list):
    """The ONLY place ground truth is derived -- see module docstring. Never
    touches bridge_predictions/classify_observation's output."""
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    threat, intent, action = RULES[pair]
    return {"expected_threat": threat, "expected_intent": intent, "expected_action": action, "pair": pair}


def main():
    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stgt_model = STGTModel(ckpt["cfg"]).to(device)
    stgt_model.load_state_dict(ckpt["model_state_dict"])
    stgt_model.eval()
    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    print("=== regenerating the identical 500 sequences (seed=0) and running STGT + bridge ===")
    rng = np.random.default_rng(SEED)
    reporter = Reporter("eval_real_stgt_output_gen", N_SEQUENCES, rate_hint=8.0)
    items = []  # one dict per sequence: name, ctx, key_windows, bucket_info, has_gt, expected_*
    bucket_tally = Counter()
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_info = classify_observation(predictions)
        bucket_tally[bucket_info["bucket"]] += 1
        gt = ground_truth_from_true_chain([str(f) for f in chain])
        item = {"name": f"seq_{i}", "true_chain": [str(f) for f in chain],
               "ctx": bucket_info["context_text"], "key_windows": bucket_info["key_windows"],
               "bucket_info": bucket_info, "has_ground_truth": gt is not None}
        if gt is not None:
            item.update(gt)
        items.append(item)
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()

    n_gt = sum(1 for it in items if it["has_ground_truth"])
    print(f"sequences with independently-determinable ground truth: {n_gt}/{N_SEQUENCES} "
         f"({n_gt/N_SEQUENCES:.1%}) -- len(true_chain) in {{1,2}}")
    print(f"bucket split on this real-observation set (cross-check vs sec AE step 2): "
         f"A={bucket_tally.get('A',0)} B={bucket_tally.get('B',0)} C={bucket_tally.get('C',0)}")

    del stgt_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== loading 3 model clients (shared across systems, same as sec AE step 4) ===")
    rules_client = LocalHFClient(BASE_MODEL, adapter_path=None, temperature=0.3, system_prompt=load_rules_txt())
    ft_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    v2_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V2_ADAPTER), temperature=0.3)
    class_freq = pipeline_v2.default_class_freq()

    print("=== batched generation, 4 systems ===")
    results = {}
    prompts = [build_llm_prompt(pipeline_v2._preds_from_key_windows(it["key_windows"]), it["ctx"], {})
              for it in items]

    for label, client in (("v2", v2_client), ("rules_in_prompt", rules_client), ("v3b-fix", ft_client)):
        print(f"  {label} ...")
        raw = client.complete_batch(prompts, batch_size=BATCH_SIZE)
        results[label] = {it["name"]: a for it, a in zip(items, raw)}

    print("  pipeline_v2 ...")
    p2_items = [{"ctx": it["ctx"], "key_windows": it["key_windows"], "bucket_info": it["bucket_info"]}
               for it in items]
    pipeline_v2._resolve_batched(p2_items, rules_client, ft_client, class_freq, BATCH_SIZE)
    results["pipeline_v2"] = {it["name"]: p2it["assessment"] for it, p2it in zip(items, p2_items)}
    layer_log = {it["name"]: p2it["layer"] for it, p2it in zip(items, p2_items)}

    del rules_client, ft_client, v2_client
    gc.collect()
    torch.cuda.empty_cache()

    out_path = REPO / "evaluation" / "eval_real_stgt_output.json"
    out_path.write_text(json.dumps({
        "n_sequences": N_SEQUENCES, "n_ground_truth": n_gt,
        "bucket_tally": dict(bucket_tally), "items": [{k: v for k, v in it.items() if k != "bucket_info"}
                                                       for it in items],
        "results": results, "pipeline_v2_layer_log": layer_log,
    }, indent=2))
    print(f"\nsaved {out_path}")

    # ================= SCORING =================
    print("\n" + "=" * 100)
    print("STEP 2: per-class threat accuracy on REAL STGT output, Wilson 95% CI")
    print("=" * 100)
    gt_items = [it for it in items if it["has_ground_truth"]]
    print("| system | stratum | n | accuracy | 95% CI |")
    print("|---|---|---|---|---|")
    for label in ("v2", "rules_in_prompt", "v3b-fix", "pipeline_v2"):
        for stratum in THREAT_ORDER:
            names = [it["name"] for it in gt_items if it["expected_threat"] == stratum]
            if not names:
                continue
            hits, scored = 0, 0
            for name in names:
                a = results[label][name]
                if is_abstention(a.get("likely_intent", "")):
                    continue
                scored += 1
                gt = next(it for it in gt_items if it["name"] == name)
                if normalize_threat(a.get("threat_level", "")) == gt["expected_threat"]:
                    hits += 1
            if scored:
                p, lo, hi = wilson_ci95(hits, scored)
                print(f"| {label} | {stratum} | {scored} | {p:.1%} | [{lo:.1%}, {hi:.1%}] |")
            else:
                print(f"| {label} | {stratum} | 0 | n/a (all abstained) | n/a |")

    print("\n" + "=" * 100)
    print("STEP 2: abstention / over-abstention, escalation direction")
    print("=" * 100)
    print("| system | abstention_rate (all n=500) | over_abstention (on n={} w/ GT) | "
         "correct | under_esc | over_esc | abstained (GT subset) |".format(n_gt))
    print("|---|---|---|---|---|---|---|")
    for label in ("v2", "rules_in_prompt", "v3b-fix", "pipeline_v2"):
        all_abst = sum(1 for it in items if is_abstention(results[label][it["name"]].get("likely_intent", "")))
        over_abst = sum(1 for it in gt_items
                        if is_abstention(results[label][it["name"]].get("likely_intent", "")))
        correct = under_esc = over_esc = abst_gt = 0
        for it in gt_items:
            a = results[label][it["name"]]
            if is_abstention(a.get("likely_intent", "")):
                abst_gt += 1
                continue
            pred = normalize_threat(a.get("threat_level", ""))
            expected = it["expected_threat"]
            if pred == expected:
                correct += 1
            elif pred in THREAT_ORDER and THREAT_ORDER.index(pred) < THREAT_ORDER.index(expected):
                under_esc += 1
            elif pred in THREAT_ORDER:
                over_esc += 1
        print(f"| {label} | {all_abst}/{N_SEQUENCES} ({all_abst/N_SEQUENCES:.1%}) | "
             f"{over_abst}/{n_gt} ({over_abst/n_gt:.1%}) | {correct}/{n_gt} ({correct/n_gt:.1%}) | "
             f"{under_esc}/{n_gt} ({under_esc/n_gt:.1%}) | {over_esc}/{n_gt} ({over_esc/n_gt:.1%}) | "
             f"{abst_gt}/{n_gt} ({abst_gt/n_gt:.1%}) |")

    print("\n" + "=" * 100)
    print("STEP 2: pipeline_v2 layer-firing rate on REAL STGT output (cross-check vs sec AE step 2's 1.8/38.2/60.0)")
    print("=" * 100)
    layer_counts = Counter(layer_log.values())
    for layer in ("layer1_deterministic", "layer2_guard", "layer3_llm"):
        n = layer_counts.get(layer, 0)
        print(f"  {layer}: {n}/{N_SEQUENCES} ({n/N_SEQUENCES:.1%})")


if __name__ == "__main__":
    main()
