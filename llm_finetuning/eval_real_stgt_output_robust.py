"""
AUDIT.md sec AG step 3: re-evaluate on real STGT output, same protocol and
same independently-derived ground truth as sec AF (llm_finetuning/
eval_real_stgt_output.py -- ground truth from each sequence's TRUE formation
chain, never from bridge_predictions' own output on the model's noisy
classification), but at n_runs=20 on the volatile strata (low/high/critical,
per sec AD/AE/AF's established variance finding) and n_runs=5 elsewhere
(medium, and the non-GT sequences where correct behaviour is abstention).
Five systems: v2, rules_in_prompt, v3b-fix, pipeline_v2 (original,
unanimity reduction), and pipeline_v2-robust (sec AG step 2's fix,
robust=True, threshold=0.7 -- tuned on a separate dev split, never this
eval's seed=0 sequences).

STGT inference and bucket classification are each computed ONCE per
sequence (both deterministic given fixed weights/threshold) -- only the
downstream LLM calls are resampled n_runs times per sequence.

SUCCESS CRITERIA, stated in advance (per this session's instructions):
Layer-1 firing above 40% and over-abstention below 25%, with escalation
error no worse than sec AF's 20.5%. Sec AG step 2 already measured Layer-1
firing at only 2.4% (vs sec AF's 1.8%) on this exact 500-sequence set --
this evaluation reports whether that translates into any other system-level
improvement, and states plainly if it does not.

Usage (run inside tmux):
    python llm_finetuning/eval_real_stgt_output_robust.py
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
from swarm_intent.stgt_bridge import DEFAULT_ROBUST_THRESHOLD  # noqa: E402
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
STRATIFIED_N_RUNS = 20
DEFAULT_N_RUNS = 5
THREAT_ORDER = ("low", "medium", "high", "critical")

SUCCESS_LAYER1_FLOOR = 0.40
SUCCESS_OVER_ABSTENTION_CEILING = 0.25
SUCCESS_ESCALATION_ERROR_CEILING = 0.205  # sec AF's measured baseline


def t_ci95(values):
    from scipy import stats
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    return float(stats.t.ppf(0.975, df=n - 1) * se)


def normalize_threat(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for level in THREAT_ORDER:
        if level in raw:
            return level
    return "unparsed"


def ground_truth_from_true_chain(true_chain: list):
    if len(true_chain) == 1:
        pair = (true_chain[0], true_chain[0])
    elif len(true_chain) == 2:
        pair = (true_chain[0], true_chain[1])
    else:
        return None
    threat, intent, action = RULES[pair]
    return {"expected_threat": threat, "expected_intent": intent, "expected_action": action, "pair": pair}


def n_runs_for(gt) -> int:
    if gt is not None and gt["expected_threat"] in ("low", "high", "critical"):
        return STRATIFIED_N_RUNS
    return DEFAULT_N_RUNS


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

    print("=== regenerating the identical 500 sequences (seed=0), STGT + bridge (both reduction modes) ===")
    rng = np.random.default_rng(SEED)
    reporter = Reporter("eval_real_stgt_output_robust_gen", N_SEQUENCES, rate_hint=8.0)
    seqs = []
    for i in range(N_SEQUENCES):
        chain = [str(f) for f in sample_chain(rng)]
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)
        predictions = sliding_window_inference(stgt_model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        bucket_default = classify_observation(predictions, robust=False)
        bucket_robust = classify_observation(predictions, robust=True, robust_threshold=DEFAULT_ROBUST_THRESHOLD)
        gt = ground_truth_from_true_chain(chain)
        n_runs = n_runs_for(gt)
        seqs.append({"name": f"seq_{i}", "true_chain": chain, "has_ground_truth": gt is not None,
                    "n_runs": n_runs, "bucket_default": bucket_default, "bucket_robust": bucket_robust,
                    **(gt or {})})
        reporter.update(1, item=f"seq {i}")
    reporter.status = "done"
    reporter._write()

    n_gt = sum(1 for s in seqs if s["has_ground_truth"])
    total_units = sum(s["n_runs"] for s in seqs) * 5
    print(f"sequences with ground truth: {n_gt}/{N_SEQUENCES}; total case-run units across 5 systems: {total_units}")

    del stgt_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== loading 3 model clients ===")
    rules_client = LocalHFClient(BASE_MODEL, adapter_path=None, temperature=0.3, system_prompt=load_rules_txt())
    ft_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    v2_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V2_ADAPTER), temperature=0.3)
    class_freq = pipeline_v2.default_class_freq()

    gen_reporter = Reporter("eval_real_stgt_output_robust_gen_llm", total_units, rate_hint=0.9)

    # -------- flat (sequence, run) expansion for the 3 always-generate systems --------
    flat = []  # one entry per (seq, run)
    for s in seqs:
        for r in range(s["n_runs"]):
            flat.append(s)
    prompts = [build_llm_prompt(pipeline_v2._preds_from_key_windows(s["bucket_default"]["key_windows"]),
                                s["bucket_default"]["context_text"], {}) for s in flat]

    results = {}
    for label, client in (("v2", v2_client), ("rules_in_prompt", rules_client), ("v3b-fix", ft_client)):
        print(f"  {label} ({len(prompts)} generations) ...")
        raw = client.complete_batch(prompts, batch_size=BATCH_SIZE)
        by_seq = defaultdict(list)
        for s, a in zip(flat, raw):
            by_seq[s["name"]].append(a)
        results[label] = by_seq
        gen_reporter.update(len(prompts))

    # -------- pipeline_v2 (default) and pipeline_v2-robust --------
    for label, bucket_key in (("pipeline_v2", "bucket_default"), ("pipeline_v2-robust", "bucket_robust")):
        print(f"  {label} ...")
        items = []
        for s in seqs:
            for r in range(s["n_runs"]):
                bi = s[bucket_key]
                items.append({"ctx": bi["context_text"], "key_windows": bi["key_windows"],
                             "bucket_info": bi, "_seq": s})
        pipeline_v2._resolve_batched(items, rules_client, ft_client, class_freq, BATCH_SIZE)
        by_seq = defaultdict(list)
        layer_by_seq = defaultdict(list)
        for item in items:
            by_seq[item["_seq"]["name"]].append(item["assessment"])
            layer_by_seq[item["_seq"]["name"]].append(item["layer"])
        results[label] = by_seq
        results[f"{label}__layers"] = layer_by_seq
        gen_reporter.update(sum(s["n_runs"] for s in seqs))

    gen_reporter.status = "done"
    gen_reporter._write()

    del rules_client, ft_client, v2_client
    gc.collect()
    torch.cuda.empty_cache()

    out_path = REPO / "evaluation" / "eval_real_stgt_output_robust.json"
    out_path.write_text(json.dumps({
        "n_sequences": N_SEQUENCES, "n_ground_truth": n_gt,
        "seqs": [{k: v for k, v in s.items() if k not in ("bucket_default", "bucket_robust")} for s in seqs],
        "results": {k: v for k, v in results.items()},
    }, indent=2))
    print(f"\nsaved {out_path}")

    # ================= SCORING =================
    SYSTEMS = ("v2", "rules_in_prompt", "v3b-fix", "pipeline_v2", "pipeline_v2-robust")
    gt_seqs = [s for s in seqs if s["has_ground_truth"]]

    print("\n" + "=" * 100)
    print("STEP 3: per-class threat accuracy, mean +/- 95% CI (t-dist where n_runs>1)")
    print("=" * 100)
    print("| system | low | medium | high | critical |")
    print("|---|---|---|---|---|")
    for label in SYSTEMS:
        cells = []
        for stratum in THREAT_ORDER:
            strat_seqs = [s for s in gt_seqs if s["expected_threat"] == stratum]
            if not strat_seqs:
                cells.append("n/a")
                continue
            run_accs = []
            for r in range(strat_seqs[0]["n_runs"]):
                hits, scored = 0, 0
                for s in strat_seqs:
                    a = results[label][s["name"]][r]
                    if is_abstention(a.get("likely_intent", "")):
                        continue
                    scored += 1
                    if normalize_threat(a.get("threat_level", "")) == s["expected_threat"]:
                        hits += 1
                if scored:
                    run_accs.append(hits / scored)
            if run_accs:
                cells.append(f"{np.mean(run_accs):.1%}+/-{t_ci95(run_accs):.1%}")
            else:
                cells.append("n/a (all abstained)")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 100)
    print("STEP 3: abstention / over-abstention / escalation direction (run-level, averaged)")
    print("=" * 100)
    print("| system | over_abstention | correct | under_esc | over_esc | escalation_error |")
    print("|---|---|---|---|---|---|")
    summary_rows = {}
    for label in SYSTEMS:
        over_abst_runs, correct_runs, under_runs, over_runs = [], [], [], []
        for s in gt_seqs:
            for r in range(s["n_runs"]):
                a = results[label][s["name"]][r]
                if is_abstention(a.get("likely_intent", "")):
                    over_abst_runs.append(1)
                    correct_runs.append(0); under_runs.append(0); over_runs.append(0)
                    continue
                over_abst_runs.append(0)
                pred = normalize_threat(a.get("threat_level", ""))
                expected = s["expected_threat"]
                is_correct = pred == expected
                is_under = pred in THREAT_ORDER and THREAT_ORDER.index(pred) < THREAT_ORDER.index(expected)
                is_over = pred in THREAT_ORDER and THREAT_ORDER.index(pred) > THREAT_ORDER.index(expected)
                correct_runs.append(int(is_correct)); under_runs.append(int(is_under)); over_runs.append(int(is_over))
        over_abst = np.mean(over_abst_runs)
        correct = np.mean(correct_runs)
        under = np.mean(under_runs)
        over = np.mean(over_runs)
        esc_err = under + over
        summary_rows[label] = {"over_abstention": over_abst, "correct": correct, "under_esc": under,
                               "over_esc": over, "escalation_error": esc_err}
        print(f"| {label} | {over_abst:.1%} | {correct:.1%} | {under:.1%} | {over:.1%} | {esc_err:.1%} |")

    print("\n" + "=" * 100)
    print("STEP 3: layer-firing rates, pipeline_v2 vs pipeline_v2-robust (all 500 sequences)")
    print("=" * 100)
    layer_rates = {}
    for label in ("pipeline_v2", "pipeline_v2-robust"):
        all_layers = [layer for s in seqs for layer in results[f"{label}__layers"][s["name"]]]
        counts = Counter(all_layers)
        total = len(all_layers)
        layer1_rate = counts.get("layer1_deterministic", 0) / total
        layer_rates[label] = layer1_rate
        print(f"\n-- {label} (n={total} case-run units) --")
        for layer in ("layer1_deterministic", "layer2_guard", "layer3_llm"):
            n = counts.get(layer, 0)
            print(f"  {layer}: {n}/{total} ({n/total:.1%})")

    print("\n" + "=" * 100)
    print("STEP 3: SUCCESS CRITERIA CHECK (stated in advance)")
    print("=" * 100)
    robust_layer1 = layer_rates["pipeline_v2-robust"]
    robust_over_abst = summary_rows["pipeline_v2-robust"]["over_abstention"]
    robust_esc_err = summary_rows["pipeline_v2-robust"]["escalation_error"]
    print(f"Layer-1 firing > {SUCCESS_LAYER1_FLOOR:.0%}: {robust_layer1:.1%} -> "
         f"{'PASS' if robust_layer1 > SUCCESS_LAYER1_FLOOR else 'FAIL'}")
    print(f"over-abstention < {SUCCESS_OVER_ABSTENTION_CEILING:.0%}: {robust_over_abst:.1%} -> "
         f"{'PASS' if robust_over_abst < SUCCESS_OVER_ABSTENTION_CEILING else 'FAIL'}")
    print(f"escalation error <= {SUCCESS_ESCALATION_ERROR_CEILING:.1%}: {robust_esc_err:.1%} -> "
         f"{'PASS' if robust_esc_err <= SUCCESS_ESCALATION_ERROR_CEILING else 'FAIL'}")


if __name__ == "__main__":
    main()
