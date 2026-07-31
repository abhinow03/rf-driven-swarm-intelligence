"""
4-way eval: rules_lookup (no model), base Qwen2.5-7B-Instruct (no adapter),
rules_in_prompt (base + RULES.txt pasted into a system prompt), and
adapters/qwen-swarm-v2 — all through the same evaluate_llm() harness, same
TEST_CASES, same n_runs, so the headline numbers are directly comparable.

This produces the first-ever objective evaluation numbers for this project
(evaluation/finetuned_eval.json never existed before this session — see AUDIT.md
sec A). rules_lookup is the control: a 49-entry dict with zero model involved. It
is EXPECTED to score at or near 100% on these 6 clean synthetic cases — that's the
ceiling every LLM-backed system is being measured against, not a failure.

Judge (llama-3.3-70b-versatile, GroqClient) is ADVISORY ONLY: the headline metrics
(intent/threat/action accuracy, hallucination rate, abstention rate) are all
computed independent of it. If GROQ_API_KEY isn't set, runs without a judge rather
than blocking.

Usage:
    export GROQ_API_KEY=...   # optional, judge only
    python llm_finetuning/run_4way_eval.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
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

from baselines import (make_rules_lookup_run_case,  # noqa: E402
                        make_rules_in_prompt_run_case, load_rules_txt)


def pct_or_na(x) -> str:
    return f"{x:.2%}" if x is not None else "N/A"


def print_summary(name: str, res: dict):
    agg = res["aggregate"]
    print(f"  {name}: intent={pct_or_na(agg['mean_intent_accuracy'])} "
          f"threat={pct_or_na(agg['mean_threat_accuracy'])} "
          f"action={pct_or_na(agg['mean_action_accuracy'])} "
          f"halluc={pct_or_na(agg['mean_hallucination_rate'])} "
          f"abstain={pct_or_na(agg['mean_abstention_rate'])}")


def judge_mean_str(res: dict) -> str:
    scores = [c["judge_overall_mean"] for c in res["per_case"] if c["judge_overall_mean"] is not None]
    if not scores:
        return "N/A"
    return f"{sum(scores)/len(scores):.2f}/5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    judge = None
    if os.environ.get("GROQ_API_KEY"):
        judge = GroqClient(model="llama-3.3-70b-versatile")
        print("judge: llama-3.3-70b-versatile (advisory only)")
    else:
        print("GROQ_API_KEY not set — running WITHOUT a judge; "
              "objective headline metrics are unaffected")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("\n=== rules_lookup (no model call) ===")
    run_case = make_rules_lookup_run_case(seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    (out_dir / "eval_rules.json").write_text(json.dumps(res, indent=2))
    all_results["rules_lookup"] = res
    print_summary("rules_lookup", res)

    print("\n=== base (no adapter) ===")
    base_client = LocalHFClient(args.base, adapter_path=None, temperature=0.3)
    run_case = make_rules_in_prompt_run_case(base_client, seed=0)  # generic client-runner
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    (out_dir / "eval_base.json").write_text(json.dumps(res, indent=2))
    all_results["base"] = res
    print_summary("base", res)

    print("\n=== rules_in_prompt (base + RULES.txt system prompt) ===")
    base_client.system_prompt = load_rules_txt()
    run_case = make_rules_in_prompt_run_case(base_client, seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    (out_dir / "eval_rules_in_prompt.json").write_text(json.dumps(res, indent=2))
    all_results["rules_in_prompt"] = res
    print_summary("rules_in_prompt", res)

    del base_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print("\n=== adapters/qwen-swarm-v2 ===")
    v2_client = LocalHFClient(args.base, adapter_path=str(REPO / "adapters/qwen-swarm-v2"),
                               temperature=0.3)
    run_case = make_rules_in_prompt_run_case(v2_client, seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    (out_dir / "eval_v2.json").write_text(json.dumps(res, indent=2))
    all_results["qwen-swarm-v2"] = res
    print_summary("qwen-swarm-v2", res)

    print("\n\n=== 4-WAY COMPARISON (n_runs={}, {} test cases) ===".format(
        args.n_runs, len(TEST_CASES)))
    print("| system | intent acc | threat acc | action acc | hallucination | abstention | judge (advisory) |")
    print("|---|---|---|---|---|---|---|")
    for name, res in all_results.items():
        agg = res["aggregate"]
        print(f"| {name} | {pct_or_na(agg['mean_intent_accuracy'])} | {pct_or_na(agg['mean_threat_accuracy'])} | "
              f"{pct_or_na(agg['mean_action_accuracy'])} | {pct_or_na(agg['mean_hallucination_rate'])} | "
              f"{pct_or_na(agg['mean_abstention_rate'])} | {judge_mean_str(res)} |")


if __name__ == "__main__":
    main()
