"""
AUDIT.md sec AC step 3: rebuild the sec AB composite comparison table with
run-level std across n_runs=5, for all four systems (v2, rules_in_prompt,
v3b-fix, composite) -- only after sec AC step 1 resolved why the three
rules_in_prompt point-estimate numbers (93.3%/65.3%/82.3%) disagreed
(answer: single-token-argmax-vs-full-generation "reasoning drift", plus
substantial temperature=0.3 run-to-run sampling variance, both real and
quantified -- see reconcile_low_threat_accuracy.py).

evaluate_llm() does NOT expose per-run raw predictions in its return value
(only case-level MEANS across runs), so there is no way to compute a
stratum-restricted (e.g. "low-threat only") run-level std from the
eval_expanded_*.json files already on disk -- not even for v2, which has
never been re-run with capture. This script re-runs v2 and v3b-fix through
the same capturing wrapper reconcile_low_threat_accuracy.py already used for
rules_in_prompt/composite (reusing THOSE two systems' already-captured raw
data from evaluation/reconcile_c_sampled_standalone.json / reconcile_d_
sampled_composite.json -- no GPU calls needed for them), then computes,
for every system and every threat stratum: for each of the n_runs
independent runs, the accuracy across that stratum's cases in THAT run,
then reports mean +/- std of those n_runs run-level numbers -- genuine
run-to-run variance, not case-to-case spread.

Usage (run inside tmux):
    python llm_finetuning/rebuild_composite_table.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES, is_abstention, match_intent, match_threat  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
THREAT_ORDER = ("low", "medium", "high", "critical")
NAME_TO_CASE = {c["name"]: c for c in TEST_CASES}


def _capturing(run_case, log: dict):
    def wrapped(case):
        assessment, ctx = run_case(case)
        log[case["name"]].append(assessment)
        return assessment, ctx
    return wrapped


def run_and_capture(label: str, adapter_path, n_runs: int, reporter: Reporter, out_dir: Path):
    print(f"\n=== {label}: full 55-case battery, n_runs={n_runs}, temp=0.3 ===")
    client = LocalHFClient(BASE_MODEL, adapter_path=adapter_path, temperature=0.3)
    log: dict = defaultdict(list)
    run_case = _capturing(make_rules_in_prompt_run_case(client, seed=0), log)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=None, n_runs=n_runs,
                       progress_reporter=reporter)
    (out_dir / f"reconcile_{label}.json").write_text(json.dumps(
        {"aggregate": res["aggregate"], "per_case": res["per_case"], "raw": log}, indent=2))
    del client
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return log


def run_level_stats(raw: dict, n_runs: int, metric: str):
    """For each threat stratum + overall, compute per-run accuracy (across
    that stratum's cases, for run index r) then mean/std across the n_runs
    run-level numbers. metric: 'threat' or 'intent'."""
    match_fn = match_threat if metric == "threat" else match_intent
    field = "threat_level" if metric == "threat" else "likely_intent"
    expected_key = "expected_threat" if metric == "threat" else "expected_intent"

    out = {}
    for stratum in THREAT_ORDER + ("overall",):
        names = [n for n in raw if (stratum == "overall" or NAME_TO_CASE[n]["expected_threat"] == stratum)]
        run_accs = []
        for r in range(n_runs):
            hits, scored = 0, 0
            for name in names:
                case = NAME_TO_CASE[name]
                if expected_key not in case:
                    continue
                assessment = raw[name][r]
                if is_abstention(assessment.get("likely_intent", "")):
                    continue  # excluded from accuracy, same as evaluate_llm
                scored += 1
                if match_fn(assessment.get(field, ""), case[expected_key]):
                    hits += 1
            run_accs.append(hits / scored if scored else None)
        valid = [a for a in run_accs if a is not None]
        out[stratum] = (float(np.mean(valid)) if valid else None,
                        float(np.std(valid)) if valid else None, len(names))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reporter = Reporter("rebuild_composite_table", len(TEST_CASES) * args.n_runs * 2, rate_hint=0.25)

    raw = {}
    raw["v2"] = run_and_capture("v2", str(REPO / "adapters/qwen-swarm-v2"), args.n_runs, reporter, out_dir)
    raw["v3b-fix"] = run_and_capture("v3b-fix2", str(REPO / "adapters/qwen-swarm-v3b-fix"),
                                     args.n_runs, reporter, out_dir)

    reporter.status = "done"
    reporter._write()

    # reuse already-captured raw data from the reconciliation run -- no GPU calls
    raw["rules_in_prompt"] = json.loads(
        (out_dir / "reconcile_c_sampled_standalone.json").read_text())["raw"]
    raw["composite"] = json.loads(
        (out_dir / "reconcile_d_sampled_composite.json").read_text())["raw"]

    print("\n\n=== STEP 3: composite comparison table, run-level mean +/- std (n_runs={}) ===".format(args.n_runs))
    print("\n--- threat accuracy, per stratum ---")
    print("| system | low | medium | high | critical | overall |")
    print("|---|---|---|---|---|---|")
    table = {}
    for label, r in raw.items():
        stats = run_level_stats(r, args.n_runs, "threat")
        table[label] = stats
        row = " | ".join(
            f"{stats[s][0]:.1%}±{stats[s][1]:.1%}" if stats[s][0] is not None else "N/A"
            for s in THREAT_ORDER + ("overall",))
        print(f"| {label} | {row} |")

    print("\n--- intent accuracy, per stratum ---")
    print("| system | low | medium | high | critical | overall |")
    print("|---|---|---|---|---|---|")
    for label, r in raw.items():
        stats = run_level_stats(r, args.n_runs, "intent")
        row = " | ".join(
            f"{stats[s][0]:.1%}±{stats[s][1]:.1%}" if stats[s][0] is not None else "N/A"
            for s in THREAT_ORDER + ("overall",))
        print(f"| {label} | {row} |")

    (out_dir / "composite_comparison_table.json").write_text(json.dumps(
        {label: run_level_stats(r, args.n_runs, "threat") for label, r in raw.items()}, indent=2))
    print(f"\nwrote {out_dir / 'composite_comparison_table.json'}")


if __name__ == "__main__":
    main()
