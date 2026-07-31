"""
Step 3 of the throughput-optimization session (AUDIT.md sec V): run
qwen-swarm-v3a on the 55-case TEST_CASES battery TWICE, --n-runs 5 -- once
unbatched (--batch-size 1, byte-identical to every prior eval_expanded_v3a.json
run), once batched (batch size from step 1's memory bench) -- and compare
per-class accuracy + std. Batching must be adopted nowhere else in this project
until this check passes.

Usage:
    python llm_finetuning/run_batching_equivalence_check.py --batch-size 8
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
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case, make_batched_run_case  # noqa: E402

NAME_TO_THREAT = {c["name"]: c["expected_threat"] for c in TEST_CASES}


def per_class_threat_accuracy(res):
    by_threat = defaultdict(list)
    for c in res["per_case"]:
        t = NAME_TO_THREAT.get(c["name"])
        if t:
            by_threat[t].append(c["threat_accuracy"])
    return {t: (float(np.mean(v)), float(np.std(v))) for t, v in by_threat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-swarm-v3a")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    n_cases = len(TEST_CASES)
    total = n_cases * args.n_runs
    out_dir = Path(args.out_dir)

    results = {}
    for mode, bs in (("unbatched", 1), ("batched", args.batch_size)):
        print(f"\n=== {mode} (batch_size={bs}) ===")
        client = LocalHFClient(args.base, adapter_path=str(REPO / args.adapter), temperature=0.3)
        reporter = Reporter(f"equivalence_check_{mode}", total, rate_hint=0.25)
        if bs > 1:
            run_case = make_batched_run_case(client, TEST_CASES, args.n_runs, bs, seed=0)
        else:
            run_case = make_rules_in_prompt_run_case(client, seed=0)
        res = evaluate_llm(run_case, TEST_CASES, judge_client=None, n_runs=args.n_runs,
                           progress_reporter=reporter)
        (out_dir / f"eval_expanded_v3a_equivcheck_{mode}.json").write_text(json.dumps(res, indent=2))
        results[mode] = res
        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    print("\n\n=== EQUIVALENCE CHECK: unbatched vs batched (qwen-swarm-v3a, 55-case battery) ===")
    print("| threat class | unbatched threat_acc | batched threat_acc | |delta| | within 1 unbatched-std? |")
    print("|---|---|---|---|---|")
    unb = per_class_threat_accuracy(results["unbatched"])
    bat = per_class_threat_accuracy(results["batched"])
    any_violation = False
    for threat in ("low", "medium", "high", "critical"):
        um, us = unb.get(threat, (None, None))
        bm, bs_ = bat.get(threat, (None, None))
        if um is None or bm is None:
            continue
        delta = abs(um - bm)
        within = delta <= max(us, 1e-9)
        any_violation = any_violation or not within
        print(f"| {threat} | {um:.1%} (std {us:.1%}) | {bm:.1%} (std {bs_:.1%}) | "
              f"{delta:.1%} | {'yes' if within else 'NO -- VIOLATION'} |")

    agg_u, agg_b = results["unbatched"]["aggregate"], results["batched"]["aggregate"]
    print(f"\noverall accuracy_when_answerable: unbatched={agg_u['accuracy_when_answerable_mean_across_runs']:.2%} "
          f"+/- {agg_u['accuracy_when_answerable_std_across_runs']:.2%} | "
          f"batched={agg_b['accuracy_when_answerable_mean_across_runs']:.2%} "
          f"+/- {agg_b['accuracy_when_answerable_std_across_runs']:.2%}")

    if any_violation:
        print("\nSTOP: batched and unbatched results differ by more than one unbatched-std "
              "in at least one threat class. Do NOT adopt batching elsewhere until this "
              "is understood.")
    else:
        print("\nPASS: batched and unbatched results agree within one std in every threat "
              "class. Batching is safe to adopt.")


if __name__ == "__main__":
    main()
