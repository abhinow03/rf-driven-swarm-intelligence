"""
AUDIT.md sec AB step 3: evaluates the composite router (composite.py) against
its two components (rules_in_prompt alone, v3b-fix alone) and v2, on both the
clean 55-case battery and the full 5-axis degradation battery.

Also produces a same-protocol (temperature=0.3, n_runs=5, unbatched --
matching the existing eval_expanded_v2.json / eval_expanded_rules_in_prompt.json
on disk, run_headline_eval.py's default --batch-size=1) v3b-fix clean-battery
result, since the only v3b-fix result so far (evaluation/
eval_expanded_v3b-fix_greedy.json) is greedy/n_runs=1 and not directly
comparable to v2/rules_in_prompt's sampled numbers.

Writes:
  evaluation/eval_expanded_v3b-fix.json       (n_runs=5, temp=0.3, unbatched)
  evaluation/eval_expanded_composite.json     (+ branch_log)
  evaluation/degradation_composite.json       (+ branch_log)

Usage (run inside tmux -- see AUDIT.md sec AB step 3):
    python llm_finetuning/eval_composite.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import GroqClient, LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES, ORIGINAL_TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case, load_rules_txt  # noqa: E402
from composite import make_composite_run_case, make_composite_battery_run_case  # noqa: E402
from degradation import build_battery  # noqa: E402
from run_degradation_eval import group_by_severity, _fmt  # noqa: E402

V3B_FIX_ADAPTER = "adapters/qwen-swarm-v3b-fix"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    judge = None
    if os.environ.get("GROQ_API_KEY"):
        judge = GroqClient(model="llama-3.3-70b-versatile")
        print("judge: llama-3.3-70b-versatile (advisory only)")
    else:
        print("GROQ_API_KEY not set — running WITHOUT a judge; objective metrics unaffected")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_clean_gen = len(TEST_CASES) * args.n_runs
    degradation_battery = build_battery(ORIGINAL_TEST_CASES)
    n_degrad_gen = sum(len(cases) for cases in degradation_battery.values()) * args.n_runs
    total = n_clean_gen * 2 + n_degrad_gen  # v3b-fix clean + composite clean + composite degradation
    reporter = Reporter("eval_composite", total, rate_hint=0.25)

    # --- 1. v3b-fix, clean battery, same protocol as eval_expanded_v2/rules_in_prompt.json ---
    print(f"\n=== v3b-fix: clean 55-case battery (n_runs={args.n_runs}, temp=0.3, unbatched) ===")
    ft_client = LocalHFClient(args.base, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    run_case = make_rules_in_prompt_run_case(ft_client, seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs,
                       progress_reporter=reporter)
    (out_dir / "eval_expanded_v3b-fix.json").write_text(json.dumps(res, indent=2))
    agg = res["aggregate"]
    print(f"  accuracy_when_answerable={_fmt(agg['accuracy_when_answerable'])} "
          f"threat_acc={_fmt(agg['mean_threat_accuracy'])}")

    # --- 2. composite, clean battery ---
    print(f"\n=== composite: clean 55-case battery (n_runs={args.n_runs}, temp=0.3, unbatched) ===")
    rules_client = LocalHFClient(args.base, adapter_path=None, temperature=0.3,
                                 system_prompt=load_rules_txt())
    branch_log_clean: dict = {}
    run_case = make_composite_run_case(rules_client, ft_client, branch_log_clean, seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs,
                       progress_reporter=reporter)
    res["branch_log"] = branch_log_clean
    res["branch_counts"] = dict(Counter(branch_log_clean.values()))
    (out_dir / "eval_expanded_composite.json").write_text(json.dumps(res, indent=2))
    agg = res["aggregate"]
    print(f"  accuracy_when_answerable={_fmt(agg['accuracy_when_answerable'])} "
          f"threat_acc={_fmt(agg['mean_threat_accuracy'])} branches={res['branch_counts']}")

    # --- 3. composite, degradation battery (full 5 axes, matches degradation_v2/v3b-fix/rules_in_prompt.json) ---
    print(f"\n=== composite: degradation battery (n_runs={args.n_runs}) ===")
    branch_log_degrad: dict = {}
    run_case = make_composite_battery_run_case(rules_client, ft_client, branch_log_degrad)
    out = {"system": "composite", "n_runs": args.n_runs, "axes": {}}
    for axis, cases in degradation_battery.items():
        axis_results = []
        for severity, sev_cases in group_by_severity(cases):
            res = evaluate_llm(run_case, sev_cases, judge_client=judge, n_runs=args.n_runs,
                               progress_reporter=reporter)
            agg = res["aggregate"]
            axis_results.append({"severity": severity, "aggregate": agg, "per_case": res["per_case"]})
            n_gt, n_no_gt = agg["n_cases_with_ground_truth"], agg["n_cases_without_ground_truth"]
            gt = f"{n_gt}gt/{n_no_gt}no-gt" if (n_gt and n_no_gt) else ("gt" if n_gt else "no-gt")
            print(f"    {axis} sev={severity} ({gt}): "
                  f"acc_when_answerable={_fmt(agg['accuracy_when_answerable'])} "
                  f"abstain_when_unanswerable={_fmt(agg['abstention_rate_when_unanswerable'])} "
                  f"over_abstain={_fmt(agg['over_abstention_rate'])}", flush=True)
        out["axes"][axis] = axis_results
    out["branch_log"] = branch_log_degrad
    out["branch_counts"] = dict(Counter(branch_log_degrad.values()))
    print(f"  branch counts (degradation battery): {out['branch_counts']}")
    (out_dir / "degradation_composite.json").write_text(json.dumps(out, indent=2))

    reporter.status = "done"
    reporter._write()

    del ft_client, rules_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print(f"\nwrote eval_expanded_v3b-fix.json, eval_expanded_composite.json, degradation_composite.json to {out_dir}")


if __name__ == "__main__":
    main()
