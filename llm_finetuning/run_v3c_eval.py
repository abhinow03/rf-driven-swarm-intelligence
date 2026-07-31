"""
Step 2 (second half) of the "under-training vs data-diversity" session.

Evaluates qwen-swarm-v3c (epoch-matched control: same data/hyperparameters as
v3a, but trained to ~306 optimizer steps to match v2's step count, then rolled
back to its best-eval-loss checkpoint via --load-best-model) on the 55-case
TEST_CASES battery, --n-runs 5, same protocol as run_headline_eval.py, so it's
directly comparable to the existing eval_expanded_v3a.json.

Usage:
    python llm_finetuning/run_v3c_eval.py --n-runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import GroqClient, LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-swarm-v3c")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "eval_expanded_v3c.json"))
    ap.add_argument("--rate-hint", type=float, default=0.21)
    args = ap.parse_args()

    n_cases = len(TEST_CASES)
    total = n_cases * args.n_runs
    print(f"=== qwen-swarm-v3c: {n_cases} cases x {args.n_runs} runs = {total} generations ===")
    print(f"estimated runtime at --rate-hint={args.rate_hint}/s: ~{total / args.rate_hint / 60:.1f} min")

    judge = None
    if os.environ.get("GROQ_API_KEY"):
        judge = GroqClient(model="llama-3.3-70b-versatile")

    reporter = Reporter("run_v3c_eval", total, rate_hint=args.rate_hint)
    client = LocalHFClient(args.base, adapter_path=str(REPO / args.adapter), temperature=0.3)
    run_case = make_rules_in_prompt_run_case(client, seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs,
                       progress_reporter=reporter)
    Path(args.out).write_text(json.dumps(res, indent=2))
    reporter.status = "done"
    reporter._write()

    agg = res["aggregate"]
    print(f"\naccuracy_when_answerable={agg['accuracy_when_answerable_mean_across_runs']:.2%} "
          f"+/- {agg['accuracy_when_answerable_std_across_runs']:.2%}")
    print(f"mean_threat_accuracy={agg['mean_threat_accuracy']:.2%} "
          f"mean_action_accuracy={agg['mean_action_accuracy']:.2%}")


if __name__ == "__main__":
    main()
